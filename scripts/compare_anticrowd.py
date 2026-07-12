"""策略层/组合层反拥挤优化对比（§51）。

因子层已 10 重墙穷尽，本脚本测策略层/组合层 overlay。复用 rolling_validate.py
产出的 data/rolling_90d_result.pkl（361 天 OOS pred/label），叠加 5 个方案后用同一
hand-rolled top-k backtester 回测对比。

方案清单：
  baseline : topk10 equal-weight 满仓
  G7       : 波动率目标仓位（§48 已验证 Calmar 4.88→6.20，此处复现）
  CS       : 横截面分散度择时（cheap-falsify 先行，PASS 才回测）
  CONCEPT  : 概念分散反拥挤（cheap-falsify 先行，PASS 才回测）
  4a       : 动态 topk（信号集中缩 topk，发散扩 topk）
  4b       : score rank-weight（非 equal-weight）

Run:
  conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
      scripts/compare_anticrowd.py
"""
from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from qlib_ifind_beta.config import OVERLAY_ROOT, ROLLING_STEP

RESULT_PKL = Path("data/rolling_90d_result.pkl")
OPEN_COST = 0.0005
CLOSE_COST = 0.0015
BENCHMARK = "SH000300"

# ─── shared utilities ─────────────────────────────────────────────────────────

def _load_data():
    """Load rolling pred/label, return (pred Series, label Series, sorted dates)."""
    with open(RESULT_PKL, "rb") as f:
        d = pickle.load(f)
    pred = d["pred"].copy()
    label = d["label"].copy()
    dates = sorted(pred.index.get_level_values(0).unique())
    return pred, label, dates


def _bench_returns(dates):
    """Load benchmark daily returns aligned to dates."""
    import qlib
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib.data import D

    start = dates[0].strftime("%Y-%m-%d")
    end = dates[-1].strftime("%Y-%m-%d")
    bench = D.features([BENCHMARK], ["$close"], start_time=start, end_time=end)
    bench_ret = bench["$close"].pct_change().groupby(level="datetime").first()
    # Align to our dates
    ret = pd.Series(index=pd.DatetimeIndex(dates), dtype=float)
    for d in dates:
        if d in bench_ret.index:
            ret[d] = bench_ret[d]
    return ret


def _daily_topk(pred, label, date, topk=10):
    """Get topk stocks for a given date. Returns (codes, scores, label_vals)."""
    dp = pred.xs(date, level=0).dropna()
    dl = label.xs(date, level=0).dropna()
    common = dp.index.intersection(dl.index)
    if len(common) < topk:
        return None, None, None
    top = dp.reindex(common).sort_values(ascending=False).head(topk)
    codes = list(top.index)
    scores = top.values
    label_vals = dl.reindex(codes).values
    return codes, scores, label_vals


def _calc_metrics(returns_net, returns_gross, bench_ret):
    """Calculate excess return, IR, max drawdown, Calmar.

    Excess = portfolio net - benchmark (same as rolling_validate.py convention).
    Calmar = annualized_excess / |max_drawdown|.
    """
    s_net = pd.Series(returns_net)
    s_gross = pd.Series(returns_gross)
    bench = bench_ret.reindex(s_net.index).fillna(0)

    excess_net = s_net.values - bench.values
    excess_gross = s_gross.values - bench.values

    cum = np.cumprod(1 + excess_net)
    if len(cum) == 0:
        return {"excess_net": 0, "IR": 0, "max_dd": 0, "Calmar": 0}
    max_dd = (cum / np.maximum.accumulate(cum) - 1).min()

    ann_excess = (cum[-1] - 1) * 100
    std_excess = np.std(excess_net)
    ann_ir = np.mean(excess_net) / std_excess * np.sqrt(250) if std_excess > 0 else 0
    calmar = ann_excess / abs(max_dd * 100) if max_dd < 0 else float("inf")

    return {
        "excess_net": ann_excess,
        "excess_gross": (np.cumprod(1 + np.nan_to_num(excess_gross))[-1] - 1) * 100,
        "IR": ann_ir,
        "max_dd": max_dd * 100,
        "Calmar": calmar,
    }


# ─── baseline + generic weighted backtest ─────────────────────────────────────

def backtest_weighted(pred, label, dates, bench_ret, weights_fn, topk=10):
    """Generic backtest with custom weight function.

    weights_fn(codes, scores, date) -> np.array of weights (sums to 1).
    Returns daily net/gross portfolio returns.
    """
    rets_net, rets_gross = [], []
    idx = []
    for d in dates:
        codes, scores, label_vals = _daily_topk(pred, label, d, topk)
        if codes is None:
            continue
        w = weights_fn(codes, scores, d)
        if w is None:
            continue
        r = label_vals
        rets_gross.append(np.sum(w * r))
        rets_net.append(np.sum(w * ((1 + r) * (1 - CLOSE_COST) / (1 + OPEN_COST) - 1)))
        idx.append(d)
    return pd.Series(rets_net, index=idx), pd.Series(rets_gross, index=idx)


def _equal_weight(codes, scores, date):
    n = len(codes)
    return np.ones(n) / n


# ─── G7: vol-target position sizing ───────────────────────────────────────────

def backtest_g7(pred, label, dates, bench_ret, topk=10, vol_window=10):
    """G7: position = median_vol / current_vol when vol > median, else 1.0.

    Trailing vol_window-day strategy absolute return volatility. 'median' =
    full-sample median of daily trailing vols. Position scales portfolio return;
    no look-ahead (position for day T uses returns T-vol_window..T-1).
    """
    # Baseline equal-weight daily returns (same as backtest_weighted with _equal_weight)
    s_net, _ = backtest_weighted(pred, label, dates, bench_ret, _equal_weight, topk)

    # Trailing vol per day
    daily_vols = {}
    vol_dates = list(s_net.index)
    for i, d in enumerate(vol_dates):
        if i >= vol_window:
            daily_vols[d] = s_net.iloc[i - vol_window:i].std()
        else:
            daily_vols[d] = np.nan
    med_vol = np.nanmedian(list(daily_vols.values()))

    # Position-scaled returns
    mod_net = s_net.copy()
    mod_gross = pd.Series(index=s_net.index, dtype=float)
    for d in vol_dates:
        vol = daily_vols.get(d, np.nan)
        if np.isfinite(vol) and vol > med_vol and med_vol > 0:
            position = med_vol / vol
        else:
            position = 1.0
        mod_net[d] = position * s_net[d]
        mod_gross[d] = position * s_net[d]  # approximate gross (cost-neutral)

    return mod_net, mod_gross


# ─── CS: cross-sectional dispersion timing ────────────────────────────────────

def cheap_falsify_cs_dispersion(pred, label, dates, bench_ret):
    """Cheap-falsify: does pool cross-sectional return dispersion predict next-day top10 return?

    dispersion = daily label std / |daily label mean|  (cross-sectional, dimensionless)
    Test: Spearman corr(dispersion[T], top10_return[T+1]) + group comparison.
    """
    from scipy.stats import spearmanr, mannwhitneyu

    # Build daily top10 returns
    top10_rets = []
    dispersions = []
    for i, d in enumerate(dates):
        dp = pred.xs(d, level=0).dropna()
        dl = label.xs(d, level=0).dropna()
        common = dp.index.intersection(dl.index)
        if len(common) < 10:
            continue
        top = dp.reindex(common).sort_values(ascending=False).head(10)
        r = dl.reindex(top.index).values
        top10_rets.append(np.mean(r))

        # Cross-sectional dispersion of ALL stocks' labels that day
        all_labels = dl.values
        mu = np.mean(all_labels)
        sd = np.std(all_labels)
        if abs(mu) > 1e-10:
            dispersions.append(sd / abs(mu))
        else:
            dispersions.append(np.nan)

    top10_rets = np.array(top10_rets)
    dispersions = np.array(dispersions)

    # Dispersion[T] vs next-day top10 return[T+1]
    valid = np.isfinite(dispersions[:-1]) & np.isfinite(top10_rets[1:])
    rho, p = spearmanr(dispersions[:-1][valid], top10_rets[1:][valid])

    # Group comparison: low dispersion (bottom 25%) vs high dispersion (top 25%)
    med = np.nanmedian(dispersions)
    low_mask = dispersions < np.nanpercentile(dispersions, 25)
    high_mask = dispersions > np.nanpercentile(dispersions, 75)

    # Low dispersion days' NEXT day return
    low_next = []
    high_next = []
    for i in range(len(dispersions) - 1):
        if np.isnan(dispersions[i]) or np.isnan(top10_rets[i + 1]):
            continue
        if low_mask[i]:
            low_next.append(top10_rets[i + 1])
        elif high_mask[i]:
            high_next.append(top10_rets[i + 1])

    low_next = np.array(low_next)
    high_next = np.array(high_next)

    if len(low_next) > 5 and len(high_next) > 5:
        u_stat, u_p = mannwhitneyu(low_next, high_next, alternative="less")
    else:
        u_stat, u_p = 0, 1.0

    print("\n=== 方案 2: 横截面分散度择时 cheap-falsify ===")
    print(f"样本: {len(dispersions)} 天, 中位分散度: {med:.3f}")
    print(f"Spearman corr(disp[T], ret[T+1]): {rho:.4f} (p={p:.4f})")
    print(f"低分散日次日收益 mean: {np.mean(low_next)*100:.4f}% (n={len(low_next)})")
    print(f"高分散日次日收益 mean: {np.mean(high_next)*100:.4f}% (n={len(high_next)})")
    print(f"Mann-Whitney U p-value: {u_p:.4f}")
    print(f"低分散日占比: {np.sum(low_mask)/len(dispersions)*100:.1f}%")

    pass_bar = abs(rho) > 0.05 and u_p < 0.1
    print(f"\n判定 bar: |corr|>0.05 AND p<0.1 → {'✓ PASS' if pass_bar else '✗ STOP'}")

    return {
        "pass": pass_bar,
        "corr": rho,
        "corr_p": p,
        "low_mean": np.mean(low_next) if len(low_next) > 0 else np.nan,
        "high_mean": np.mean(high_next) if len(high_next) > 0 else np.nan,
        "u_p": u_p,
        "median_disp": med,
    }


def backtest_cs_dispersion(pred, label, dates, bench_ret, topk=10):
    """CS dispersion timing: low dispersion → reduce position.

    Applies position = disp / med_disp when disp < med_disp (reduces exposure
    during low cross-sectional dispersion periods).
    """
    s_net, _ = backtest_weighted(pred, label, dates, bench_ret, _equal_weight, topk)

    # Precompute dispersions
    dispersions = {}
    for d in dates:
        dl = label.xs(d, level=0).dropna()
        mu = np.mean(dl.values)
        sd = np.std(dl.values)
        dispersions[d] = sd / abs(mu) if abs(mu) > 1e-10 else np.nan

    med_disp = np.nanmedian(list(dispersions.values()))

    mod_net = s_net.copy()
    mod_gross = pd.Series(index=s_net.index, dtype=float)
    for d in s_net.index:
        disp = dispersions.get(d, np.nan)
        if np.isnan(disp) or disp >= med_disp or med_disp <= 0:
            position = 1.0
        else:
            position = disp / med_disp  # low dispersion → reduce
        mod_net[d] = position * s_net[d]
        mod_gross[d] = position * s_net[d]

    return mod_net, mod_gross


# ─── CONCEPT: concept crowding anti-resonance ────────────────────────────────

def _load_concepts():
    """Load all concept membership: dict code -> set of concept codes."""
    import qlib
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")

    concept_dir = Path("/home/zxh/qlib_data/instruments")
    code2concepts = defaultdict(set)
    concept_count = 0
    for f in sorted(concept_dir.glob("concept_tzt_*.txt")):
        concept_id = f.stem  # e.g. concept_tzt_993075
        concept_count += 1
        with open(f) as fp:
            for line in fp:
                parts = line.strip().split("\t")
                if len(parts) >= 1 and parts[0]:
                    code2concepts[parts[0]].add(concept_id)
    print(f"  loaded {concept_count} concepts, {len(code2concepts)} stocks mapped")
    return dict(code2concepts)


def cheap_falsify_concept(pred, label, dates, bench_ret, code2concepts):
    """Cheap-falsify: does top10 concept concentration predict return?

    Concentration = max single-concept occupancy in topk10.
    Test: high-concentration days vs low-concentration days return comparison.
    """
    from scipy.stats import mannwhitneyu

    concentrations = []
    top10_rets = []
    max_concept_names = []

    for d in dates:
        codes, scores, label_vals = _daily_topk(pred, label, d, 10)
        if codes is None:
            continue

        # Count concept frequency in top10
        concept_freq = defaultdict(int)
        for c in codes:
            for concept in code2concepts.get(c, set()):
                concept_freq[concept] += 1

        if concept_freq:
            max_concept, max_count = max(concept_freq.items(), key=lambda x: x[1])
            concentration = max_count / len(codes)  # 0.1 = perfectly dispersed, 1.0 = all same concept
        else:
            max_concept, max_count, concentration = "none", 0, 0.1

        concentrations.append(concentration)
        max_concept_names.append(max_concept)
        top10_rets.append(np.mean(label_vals))

    concentrations = np.array(concentrations)
    top10_rets = np.array(top10_rets)

    med_conc = np.median(concentrations)
    high_threshold = np.percentile(concentrations, 75)
    low_threshold = np.percentile(concentrations, 25)

    high_mask = concentrations >= high_threshold
    low_mask = concentrations <= low_threshold

    high_rets = top10_rets[high_mask]
    low_rets = top10_rets[low_mask]

    if len(high_rets) > 5 and len(low_rets) > 5:
        u_stat, u_p = mannwhitneyu(high_rets, low_rets, alternative="less")
    else:
        u_stat, u_p = 0, 1.0

    print("\n=== 方案 3: 概念分散反拥挤 cheap-falsify ===")
    print(f"样本: {len(concentrations)} 天")
    print(f"top10 最大概念占比: median={med_conc:.2f}, mean={np.mean(concentrations):.2f}, max={np.max(concentrations):.2f}")
    print(f"  (0.1=完全分散, 1.0=全部同一概念)")
    print(f"高集中日占比: {np.sum(high_mask)/len(concentrations)*100:.1f}%")
    print(f"高集中日收益 mean: {np.mean(high_rets)*100:.4f}% (n={len(high_rets)})")
    print(f"低集中日收益 mean: {np.mean(low_rets)*100:.4f}% (n={len(low_rets)})")
    print(f"Mann-Whitney U p-value (高<低): {u_p:.4f}")

    # Top concepts appearing
    concept_count = defaultdict(int)
    for c in max_concept_names:
        if c != "none":
            concept_count[c] += 1
    top5 = sorted(concept_count.items(), key=lambda x: -x[1])[:5]
    print(f"最高频主导概念 top5: {top5}")

    pass_bar = u_p < 0.1 and np.sum(high_mask) / len(concentrations) > 0.1
    print(f"\n判定 bar: p<0.1 AND 高集中占比>10% → {'✓ PASS' if pass_bar else '✗ STOP'}")

    return {"pass": pass_bar, "u_p": u_p, "median_conc": med_conc}


# ─── 4a: dynamic topk ────────────────────────────────────────────────────────

def backtest_dynamic_topk(pred, label, dates, bench_ret, low_topk=5, high_topk=15):
    """Dynamic topk: high CV (concentrated signal) → small topk; low CV → large topk.

    CV = std(pred) / |mean(pred)| per day. CV > 1.5 → topk=5; CV < 0.8 → topk=15;
    else topk=10. Same backtest_weighted framework but per-day topk varies.
    """
    # Precompute daily CVs
    daily_cv = {}
    for d in dates:
        dp = pred.xs(d, level=0).dropna()
        mu = dp.mean()
        daily_cv[d] = dp.std() / abs(mu) if abs(mu) > 1e-10 else 0

    # Precompute per-day returns at each topk level, then pick by CV
    topk_returns = {k: {} for k in [5, 10, 15]}
    for d in dates:
        dp = pred.xs(d, level=0).dropna()
        dl = label.xs(d, level=0).dropna()
        common = dp.index.intersection(dl.index)
        if len(common) < 15:
            continue
        top_scores = dp.reindex(common).sort_values(ascending=False)
        for k in [5, 10, 15]:
            top = top_scores.head(k)
            r = dl.reindex(top.index).values
            topk_returns[k][d] = (np.mean(r), np.mean((1 + r) * (1 - CLOSE_COST) / (1 + OPEN_COST) - 1))

    rets_net, rets_gross = [], []
    idx = []
    for d in dates:
        if d not in topk_returns[15]:
            continue
        cv = daily_cv.get(d, 1.0)
        k = 5 if cv > 1.5 else (15 if cv < 0.8 else 10)
        rets_gross.append(topk_returns[k][d][0])
        rets_net.append(topk_returns[k][d][1])
        idx.append(d)

    return pd.Series(rets_net, index=idx), pd.Series(rets_gross, index=idx)


# ─── 4b: score rank-weight ───────────────────────────────────────────────────

def backtest_rank_weight(pred, label, dates, bench_ret, topk=10):
    """Rank-weighted: weight ∝ rank position (rank 1 gets highest weight)."""
    def _rank_weight(codes, scores, date):
        n = len(codes)
        # Linear rank weight: rank 0 (highest score) gets n, rank n-1 gets 1
        ranks = np.arange(n, 0, -1).astype(float)
        return ranks / ranks.sum()

    return backtest_weighted(pred, label, dates, bench_ret, _rank_weight, topk)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print("▶ 策略层/组合层反拥挤优化对比（§51）")
    print("  复用 data/rolling_90d_result.pkl（361 天 OOS, HFLGBModel 90 天滚动重训）\n")

    pred, label, dates = _load_data()
    bench_ret = _bench_returns(dates)
    print(f"  pred: {len(pred)} 行, {len(dates)} 天")
    print(f"  benchmark: {BENCHMARK}, {bench_ret.notna().sum()}/{len(dates)} 天有数据\n")

    results = {}

    # --- baseline ---
    print("▶ [1/6] baseline (topk10 equal-weight 满仓)")
    s_net, s_gross = backtest_weighted(pred, label, dates, bench_ret, _equal_weight, 10)
    results["baseline"] = _calc_metrics(s_net.values, s_gross.values, bench_ret.reindex(s_net.index))
    print(f"  超额={results['baseline']['excess_net']:.1f}% IR={results['baseline']['IR']:.2f} "
          f"DD={results['baseline']['max_dd']:.1f}% Calmar={results['baseline']['Calmar']:.2f}")

    # --- G7 ---
    print("\n▶ [2/6] G7 波动率目标仓位")
    s_net, s_gross = backtest_g7(pred, label, dates, bench_ret, topk=10, vol_window=10)
    results["G7"] = _calc_metrics(s_net.values, s_gross.values, bench_ret.reindex(s_net.index))
    print(f"  超额={results['G7']['excess_net']:.1f}% IR={results['G7']['IR']:.2f} "
          f"DD={results['G7']['max_dd']:.1f}% Calmar={results['G7']['Calmar']:.2f}")

    # --- CS dispersion cheap-falsify ---
    print("\n▶ [3/6] 横截面分散度择时")
    cs_result = cheap_falsify_cs_dispersion(pred, label, dates, bench_ret)
    if cs_result["pass"]:
        print("  → PASS, 回测中...")
        s_net, s_gross = backtest_cs_dispersion(pred, label, dates, bench_ret, topk=10)
        results["CS"] = _calc_metrics(s_net.values, s_gross.values, bench_ret.reindex(s_net.index))
        print(f"  超额={results['CS']['excess_net']:.1f}% IR={results['CS']['IR']:.2f} "
              f"DD={results['CS']['max_dd']:.1f}% Calmar={results['CS']['Calmar']:.2f}")
    else:
        print("  → STOP, 不回测")
        results["CS"] = {"excess_net": None, "IR": None, "max_dd": None, "Calmar": None,
                         "note": f"falsified: corr={cs_result['corr']:.4f}, u_p={cs_result['u_p']:.4f}"}

    # --- Concept crowding cheap-falsify ---
    print("\n▶ [4/6] 概念分散反拥挤")
    code2concepts = _load_concepts()
    concept_result = cheap_falsify_concept(pred, label, dates, bench_ret, code2concepts)
    if concept_result["pass"]:
        print("  → PASS（本版仅诊断，概念上限回测待后续实现）")
        results["CONCEPT"] = {"excess_net": None, "IR": None, "max_dd": None, "Calmar": None,
                              "note": "falsify PASS, backtest TBD"}
    else:
        print("  → STOP")
        results["CONCEPT"] = {"excess_net": None, "IR": None, "max_dd": None, "Calmar": None,
                              "note": f"falsified: u_p={concept_result['u_p']:.4f}"}

    # --- 4a dynamic topk ---
    print("\n▶ [5/6] 4a 动态 topk")
    s_net, s_gross = backtest_dynamic_topk(pred, label, dates, bench_ret)
    results["4a"] = _calc_metrics(s_net.values, s_gross.values, bench_ret.reindex(s_net.index))
    print(f"  超额={results['4a']['excess_net']:.1f}% IR={results['4a']['IR']:.2f} "
          f"DD={results['4a']['max_dd']:.1f}% Calmar={results['4a']['Calmar']:.2f}")

    # --- 4b rank weight ---
    print("\n▶ [6/6] 4b score rank-weight")
    s_net, s_gross = backtest_rank_weight(pred, label, dates, bench_ret, topk=10)
    results["4b"] = _calc_metrics(s_net.values, s_gross.values, bench_ret.reindex(s_net.index))
    print(f"  超额={results['4b']['excess_net']:.1f}% IR={results['4b']['IR']:.2f} "
          f"DD={results['4b']['max_dd']:.1f}% Calmar={results['4b']['Calmar']:.2f}")

    # --- summary table ---
    print("\n" + "=" * 80)
    print("=== §51 对比表（361 天 OOS, HFLGBModel 90 天滚动重训）===")
    print(f"{'方案':<20} {'超额(net)':>12} {'IR':>8} {'最大回撤':>10} {'Calmar':>8}")
    print("-" * 80)
    for name, m in results.items():
        if m["excess_net"] is not None:
            print(f"{name:<20} {m['excess_net']:>11.1f}% {m['IR']:>8.2f} "
                  f"{m['max_dd']:>9.1f}% {m['Calmar']:>8.2f}")
        else:
            print(f"{name:<20} {m.get('note', '—')}")
    print("=" * 80)

    # Best by Calmar
    valid = {k: v for k, v in results.items() if v["excess_net"] is not None}
    if valid:
        best = max(valid, key=lambda k: valid[k]["Calmar"])
        print(f"\n最优 Calmar: {best} = {valid[best]['Calmar']:.2f}")


if __name__ == "__main__":
    main()
