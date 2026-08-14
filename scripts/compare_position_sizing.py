"""Position sizing optimization: 10 variants vs baseline (§55).

§51 已证 G7/CS/dynamic-topk/rank-weight 在 HFLGBModel pred 上全不如 baseline。
本脚本测试 3 大新方向（A 改进 G7 / B 滚动 IC regime / C 基准趋势连续仓位）共 10 个变体。

复用 data/rolling_90d_result.pkl（HFLGBModel 361 天 OOS pred/label），hand-rolled
top-k equal-weight backtester + 仓位 overlay。

Run:
  conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
      scripts/compare_position_sizing.py
"""
from __future__ import annotations

import os, sys, pickle
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qlib_ifind_beta.config import OVERLAY_ROOT

RESULT_PKL = Path("data/rolling_90d_result.pkl")
OPEN_COST = 0.0005
CLOSE_COST = 0.0015
BENCHMARK = "SH000300"
TOPK = 10

# ─── shared utilities ─────────────────────────────────────────────────────────

def _load_data():
    with open(RESULT_PKL, "rb") as f:
        d = pickle.load(f)
    pred = d["pred"].copy()
    label = d["label"].copy()
    dates = sorted(pred.index.get_level_values(0).unique())
    return pred, label, dates


def _bench_returns(dates):
    import qlib
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib.data import D
    # Fetch pre-history so pct_change for the first evaluation day is valid.
    start = (pd.Timestamp(dates[0]) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end = dates[-1].strftime("%Y-%m-%d")
    bench = D.features([BENCHMARK], ["$close"], start_time=start, end_time=end)
    bench_ret = bench["$close"].pct_change().groupby(level="datetime").first()
    ret = pd.Series(index=pd.DatetimeIndex(dates), dtype=float)
    for d in dates:
        if d in bench_ret.index:
            ret[d] = bench_ret[d]
    return ret


def _baseline_returns(pred, label, dates):
    """Compute baseline top10 equal-weight daily net returns."""
    rets = []
    idx = []
    for d in dates:
        dp = pred.xs(d, level=0).dropna()
        dl = label.xs(d, level=0).dropna()
        common = dp.index.intersection(dl.index)
        if len(common) < TOPK:
            continue
        top = dp.reindex(common).sort_values(ascending=False).head(TOPK)
        r = dl.reindex(top.index).values
        net = np.mean((1 + r) * (1 - CLOSE_COST) / (1 + OPEN_COST) - 1)
        rets.append(net)
        idx.append(d)
    return pd.Series(rets, index=idx)


def _pool_mean_returns(pred, label, dates):
    """Daily pool mean label (for alpha = top10 - pool)."""
    rets = []
    idx = []
    for d in dates:
        dl = label.xs(d, level=0).dropna()
        if len(dl) < TOPK:
            continue
        rets.append(dl.mean())
        idx.append(d)
    return pd.Series(rets, index=idx)


def _daily_ic(pred, label, dates):
    """Daily cross-sectional Spearman IC."""
    ics = []
    idx = []
    for d in dates:
        dp = pred.xs(d, level=0).dropna()
        dl = label.xs(d, level=0).dropna()
        common = dp.index.intersection(dl.index)
        if len(common) < 5:
            continue
        rho, _ = spearmanr(dp.reindex(common), dl.reindex(common))
        if np.isfinite(rho):
            ics.append(rho)
            idx.append(d)
    return pd.Series(ics, index=idx)


def _calc_metrics(s_net, bench_ret):
    """Standard absolute and benchmark-relative portfolio metrics."""
    s_net, bench = s_net.align(bench_ret, join="inner")
    valid = s_net.notna() & bench.notna()
    s_net, bench = s_net[valid], bench[valid]
    if s_net.empty:
        return {"cum_return": 0, "excess": 0, "IR": 0, "DD": 0, "Calmar": 0}
    nav = (1 + s_net).cumprod()
    bench_nav = (1 + bench).cumprod()
    max_dd = (nav / nav.cummax() - 1).min()
    ann_ret = nav.iloc[-1] ** (250.0 / len(nav)) - 1
    active = s_net - bench
    active_std = active.std(ddof=1)
    ir = active.mean() / active_std * np.sqrt(250) if active_std > 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else float("inf")
    geometric_excess = nav.iloc[-1] / bench_nav.iloc[-1] - 1
    return {
        "cum_return": (nav.iloc[-1] - 1) * 100,
        "excess": geometric_excess * 100,
        "IR": ir,
        "DD": max_dd * 100,
        "Calmar": calmar,
    }


def _apply_position(s_net, positions):
    """Scale baseline returns by position series (same index)."""
    aligned_pos = positions.reindex(s_net.index).fillna(1.0)
    return s_net * aligned_pos


# ─── Direction A: improved G7 vol-target ──────────────────────────────────────

def scheme_A1_vol_window(s_net, bench_ret, windows=(5, 10, 15, 20, 30)):
    """A1: vol_window sweep. Returns dict of {name: scaled_net_series}."""
    results = {}
    for w in windows:
        daily_vols = s_net.shift(1).rolling(w).std()
        med_vol = daily_vols.expanding(min_periods=w).median()
        position = (med_vol / daily_vols).clip(upper=1.0).fillna(1.0)
        position[daily_vols <= med_vol] = 1.0
        results[f"A1_w{w}"] = _apply_position(s_net, position)
    return results


def scheme_A2_expanding_median(s_net, bench_ret, vol_window=10):
    """A2: expanding median (no look-ahead), vol_window=10."""
    daily_vols = s_net.shift(1).rolling(vol_window).std()
    med_vol = daily_vols.expanding(min_periods=vol_window).median()
    position = (med_vol / daily_vols).clip(upper=1.0).fillna(1.0)
    position[daily_vols <= med_vol] = 1.0
    return {"A2": _apply_position(s_net, position)}


def scheme_A3_excess_vol(s_net, bench_ret, vol_window=10):
    """A3: use excess-return vol (portfolio - benchmark) instead of absolute."""
    bench = bench_ret.reindex(s_net.index).fillna(0)
    excess = (s_net - bench).shift(1)
    daily_vols = excess.rolling(vol_window).std()
    med_vol = daily_vols.expanding(min_periods=vol_window).median()
    position = (med_vol / daily_vols).clip(upper=1.0).fillna(1.0)
    position[daily_vols <= med_vol] = 1.0
    return {"A3": _apply_position(s_net, position)}


def scheme_A4_asymmetric(s_net, bench_ret, vol_window=10, down_mult=0.5, up_clip=1.0):
    """A4: asymmetric — de-leverage harder in high vol, don't add leverage."""
    daily_vols = s_net.shift(1).rolling(vol_window).std()
    med_vol = daily_vols.expanding(min_periods=vol_window).median()
    # Only reduce position when vol > median; never exceed 1.0
    position = pd.Series(1.0, index=s_net.index)
    high_vol = daily_vols > med_vol
    # Asymmetric: reduce to down_mult * (med/vol) instead of med/vol
    position[high_vol] = (down_mult + (1 - down_mult) * (med_vol[high_vol] / daily_vols[high_vol])).clip(lower=0.3, upper=1.0)
    return {"A4": _apply_position(s_net, position)}


# ─── Direction B: rolling IC regime timing ────────────────────────────────────

def scheme_B1_ic_threshold(s_net, bench_ret, ic_series, window=10, threshold=0.03):
    """B1: rolling IC < threshold → reduce position proportionally."""
    rolling_ic = ic_series.shift(1).rolling(window).mean()
    # Position = clip(rolling_ic / threshold, 0.3, 1.0)
    position = (rolling_ic / threshold).clip(lower=0.3, upper=1.0).fillna(1.0)
    return {"B1": _apply_position(s_net, position)}


def scheme_B2_alpha_signal(s_net, bench_ret, pool_mean, window=10):
    """B2: rolling top10-pool alpha < 0 → reduce."""
    alpha = (s_net - pool_mean.reindex(s_net.index).fillna(0)).shift(1)
    rolling_alpha = alpha.rolling(window).mean()
    # If alpha > 0 → full position; if < 0 → scale down
    position = pd.Series(1.0, index=s_net.index)
    neg_alpha = rolling_alpha < 0
    # Scale: alpha=0 → 0.5, alpha=-0.01 → 0.25, alpha=-0.02 → 0
    scale = (1.0 + rolling_alpha[neg_alpha] / 0.01).clip(lower=0.0, upper=0.5) + 0.5
    position[neg_alpha] = scale.clip(lower=0.0, upper=0.5)
    return {"B2": _apply_position(s_net, position)}


def scheme_B3_ic_zscore(s_net, bench_ret, ic_series, window=20):
    """B3: IC z-score regime — reduce when IC drops below expanding mean - 1σ."""
    known_ic = ic_series.shift(1)
    rolling_mean = known_ic.rolling(window).mean()
    rolling_std = known_ic.rolling(window).std()
    z = (known_ic - rolling_mean) / rolling_std
    # Position: z > 0 → 1.0, z in [-1, 0] → 0.5~1.0, z < -1 → 0.25
    position = pd.Series(1.0, index=s_net.index)
    mild = (z < 0) & (z >= -1)
    position[mild] = 0.5 + 0.5 * (1 + z[mild])  # z=0 → 1.0, z=-1 → 0.5
    severe = z < -1
    position[severe] = 0.25
    return {"B3": _apply_position(s_net, position)}


# ─── Direction C: benchmark trend continuous position ─────────────────────────

def scheme_C1_bench_momentum(s_net, bench_ret, window=20, threshold=0.02, floor=0.3):
    """C1: position = clip(benchmark 20d momentum / threshold, floor, 1.0)."""
    bench_mom = bench_ret.shift(1).rolling(window).apply(lambda x: (1 + x).prod() - 1)
    position = (bench_mom / threshold).clip(lower=floor, upper=1.0).fillna(1.0)
    return {"C1": _apply_position(s_net, position)}


def scheme_C2_bench_drawdown(s_net, bench_ret, window=5, max_dd_pct=0.10):
    """C2: position = 1 - max(0, benchmark trailing drawdown / max_dd_pct)."""
    bench_cum = (1 + bench_ret.shift(1)).cumprod()
    bench_dd = bench_cum / bench_cum.cummax() - 1  # negative or 0
    position = (1 + bench_dd / max_dd_pct).clip(lower=0.0, upper=1.0).fillna(1.0)
    return {"C2": _apply_position(s_net, position)}


def scheme_C3_combined(s_net, bench_ret, vol_window=10):
    """C3: vol-target position × benchmark momentum position, take min."""
    # Vol target
    daily_vols = s_net.shift(1).rolling(vol_window).std()
    med_vol = daily_vols.expanding(min_periods=vol_window).median()
    pos_vol = (med_vol / daily_vols).clip(upper=1.0).fillna(1.0)
    pos_vol[daily_vols <= med_vol] = 1.0
    # Bench momentum
    bench_mom = bench_ret.shift(1).rolling(20).apply(lambda x: (1 + x).prod() - 1)
    pos_bench = (bench_mom / 0.02).clip(lower=0.3, upper=1.0).fillna(1.0)
    # Combined = min (more conservative)
    position = pd.concat([pos_vol, pos_bench.reindex(pos_vol.index).fillna(1.0)], axis=1).min(axis=1)
    return {"C3": _apply_position(s_net, position)}


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print("▶ 仓位层策略优化对比（§55）")
    print("  HFLGBModel 361 天 OOS, baseline Calmar = 4.88\n")

    pred, label, dates = _load_data()
    bench_ret = _bench_returns(dates)
    s_net = _baseline_returns(pred, label, dates)
    pool_mean = _pool_mean_returns(pred, label, dates)
    ic_series = _daily_ic(pred, label, dates)

    # Align indices
    common_idx = s_net.index.intersection(bench_ret.index)

    # Baseline
    baseline = _calc_metrics(s_net, bench_ret)
    print(f"baseline: return={baseline['cum_return']:.1f}% excess={baseline['excess']:.1f}% "
          f"active_IR={baseline['IR']:.2f} "
          f"DD={baseline['DD']:.1f}% Calmar={baseline['Calmar']:.2f}\n")

    # Run all schemes
    all_schemes = {}

    # Direction A
    print("▶ Direction A: improved G7 vol-target")
    for name, series in scheme_A1_vol_window(s_net, bench_ret).items():
        all_schemes[name] = series
    for name, series in scheme_A2_expanding_median(s_net, bench_ret).items():
        all_schemes[name] = series
    for name, series in scheme_A3_excess_vol(s_net, bench_ret).items():
        all_schemes[name] = series
    for name, series in scheme_A4_asymmetric(s_net, bench_ret).items():
        all_schemes[name] = series

    # Direction B
    print("▶ Direction B: rolling IC regime timing")
    for name, series in scheme_B1_ic_threshold(s_net, bench_ret, ic_series).items():
        all_schemes[name] = series
    for name, series in scheme_B2_alpha_signal(s_net, bench_ret, pool_mean).items():
        all_schemes[name] = series
    for name, series in scheme_B3_ic_zscore(s_net, bench_ret, ic_series).items():
        all_schemes[name] = series

    # Direction C
    print("▶ Direction C: benchmark trend continuous position")
    for name, series in scheme_C1_bench_momentum(s_net, bench_ret).items():
        all_schemes[name] = series
    for name, series in scheme_C2_bench_drawdown(s_net, bench_ret).items():
        all_schemes[name] = series
    for name, series in scheme_C3_combined(s_net, bench_ret).items():
        all_schemes[name] = series

    # Results table
    print("\n" + "=" * 80)
    print(f"{'Scheme':<12} {'Return':>10} {'Excess':>10} {'Act.IR':>8} {'MaxDD':>10} {'Calmar':>8} {'vs base':>8}")
    print("-" * 60)
    print(f"{'baseline':<12} {baseline['cum_return']:>9.1f}% {baseline['excess']:>9.1f}% {baseline['IR']:>8.2f} "
          f"{baseline['DD']:>9.1f}% {baseline['Calmar']:>8.2f} {'—':>8}")

    results = {"baseline": baseline}
    best_calmar = baseline["Calmar"]
    best_name = "baseline"

    for name in sorted(all_schemes.keys()):
        series = all_schemes[name]
        m = _calc_metrics(series, bench_ret)
        results[name] = m
        delta = m["Calmar"] - baseline["Calmar"]
        marker = " ★" if m["Calmar"] > baseline["Calmar"] else ""
        print(f"{name:<12} {m['cum_return']:>9.1f}% {m['excess']:>9.1f}% {m['IR']:>8.2f} "
              f"{m['DD']:>9.1f}% {m['Calmar']:>8.2f} {delta:>+8.2f}{marker}")
        if m["Calmar"] > best_calmar:
            best_calmar = m["Calmar"]
            best_name = name

    print("=" * 80)
    print(f"\n最优 Calmar: {best_name} = {best_calmar:.2f} (baseline = {baseline['Calmar']:.2f})")

    if best_name != "baseline":
        print(f"  提升: {(best_calmar/baseline['Calmar']-1)*100:+.1f}%")
        # Position stats for the best scheme
        best_series = all_schemes[best_name]
        # Recompute positions to show stats
        print(f"\n  ⚠️ 需要进一步验证非过拟合（子时段稳定性检查）")

    return results


if __name__ == "__main__":
    main()
