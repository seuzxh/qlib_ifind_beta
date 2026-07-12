"""Drawdown chasing diagnosis: is the -34% drawdown caused by momentum chasing?

Analyze the baseline top10 equal-weight returns (from rolling_90d_result.pkl)
to answer:
  1. Where exactly is the max drawdown? Which dates?
  2. During drawdown, are top10 picks more "overbought" (higher recent return)?
  3. Does extreme momentum predict next-day reversal (chasing crash hypothesis)?
  4. Is there a momentum threshold above which the strategy loses money?

This is a pure read-only diagnostic. No new factors materialized.
"""
from __future__ import annotations

import os, sys, pickle
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

from qlib_ifind_beta.config import OVERLAY_ROOT

OPEN_COST, CLOSE_COST, TOPK = 0.0005, 0.0015, 10


def main():
    import qlib
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib.data import D

    with open("data/rolling_90d_result.pkl", "rb") as f:
        dd = pickle.load(f)
    pred = dd["pred"]
    label = dd["label"]
    dates = sorted(pred.index.get_level_values(0).unique())
    all_codes = sorted(set(pred.index.get_level_values(1)))

    # ── 1. Baseline daily returns + drawdown identification ─────────────────
    print("=" * 80)
    print("=== Part 1: Drawdown identification ===")

    daily_data = []  # (date, top10_codes, top10_labels, top10_ret, all_labels)
    for d in dates:
        dp = pred.xs(d, level=0).dropna()
        dl = label.xs(d, level=0).dropna()
        common = dp.index.intersection(dl.index)
        if len(common) < TOPK:
            continue
        top = dp.reindex(common).sort_values(ascending=False).head(TOPK)
        top_codes = list(top.index)
        top_labels = dl.reindex(top_codes).values
        top_ret = np.mean(top_labels)
        net_ret = np.mean((1 + top_labels) * (1 - CLOSE_COST) / (1 + OPEN_COST) - 1)
        daily_data.append({
            "date": d,
            "codes": top_codes,
            "labels": top_labels,
            "gross_ret": top_ret,
            "net_ret": net_ret,
            "all_labels": dl.values,
            "universe_mean": np.mean(dl.values),
        })

    df_daily = pd.DataFrame([{"date": d["date"], "net_ret": d["net_ret"],
                              "gross_ret": d["gross_ret"],
                              "universe_mean": d["universe_mean"]} for d in daily_data])
    df_daily = df_daily.set_index("date")

    # Cumulative and drawdown
    cum = (1 + df_daily["net_ret"]).cumprod()
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    # Find the drawdown period (peak to trough)
    peak_date = cum.loc[:max_dd_date].idxmax()
    print(f"  Max drawdown: {max_dd*100:.2f}%")
    print(f"  Peak date: {peak_date.strftime('%Y-%m-%d')} (cum={cum[peak_date]:.4f})")
    print(f"  Trough date: {max_dd_date.strftime('%Y-%m-%d')} (cum={cum[max_dd_date]:.4f})")
    dd_days = (max_dd_date - peak_date).days
    print(f"  Duration: {dd_days} calendar days")

    # Identify drawdown segments (>10% DD)
    dd_mask = drawdown < -0.05
    normal_mask = drawdown >= -0.02
    print(f"\n  Days with DD < -5%: {dd_mask.sum()} / {len(df_daily)}")
    print(f"  Days with DD >= -2% (normal): {normal_mask.sum()} / {len(df_daily)}")

    # ── 2. Load daily close for momentum calculation ────────────────────────
    print("\n" + "=" * 80)
    print("=== Part 2: Top10 momentum during drawdown vs normal ===")

    close_data = D.features(all_codes, ["$close"], start_time="2024-12-01",
                            end_time=str(dates[-1].date()))
    close_df = close_data["$close"].unstack(level=0)
    mom_1d = close_df.pct_change(1)
    mom_3d = close_df.pct_change(3)
    mom_5d = close_df.pct_change(5)
    mom_10d = close_df.pct_change(10)

    # Volume data for turnover/crowding
    vol_data = D.features(all_codes, ["$volume"], start_time="2024-12-01",
                          end_time=str(dates[-1].date()))
    vol_df = vol_data["$volume"].unstack(level=0)
    vol_5d = vol_df.rolling(5).mean()
    vol_ratio_5d = vol_df / vol_5d  # today volume / 5d avg

    # For each day, compute top10's momentum stats
    daily_mom = []
    for d in daily_data:
        dt = d["date"]
        codes = d["codes"]
        if dt not in mom_5d.index:
            daily_mom.append({"date": dt, "top10_mom5d": np.nan, "top10_mom3d": np.nan,
                              "top10_mom1d": np.nan, "top10_mom10d": np.nan,
                              "top10_vol_ratio": np.nan})
            continue

        m5 = mom_5d.loc[dt].reindex(codes)
        m3 = mom_3d.loc[dt].reindex(codes)
        m1 = mom_1d.loc[dt].reindex(codes)
        m10 = mom_10d.loc[dt].reindex(codes)
        vr = vol_ratio_5d.loc[dt].reindex(codes) if dt in vol_ratio_5d.index else pd.Series(dtype=float)

        daily_mom.append({
            "date": dt,
            "top10_mom5d": np.nanmean(m5.values),
            "top10_mom3d": np.nanmean(m3.values),
            "top10_mom1d": np.nanmean(m1.values),
            "top10_mom10d": np.nanmean(m10.values),
            "top10_vol_ratio": np.nanmean(vr.values),
        })

    df_mom = pd.DataFrame(daily_mom).set_index("date")
    df_mom["drawdown"] = drawdown
    df_mom["net_ret"] = df_daily["net_ret"]

    print(f"\n  Top10 5d momentum stats:")
    print(f"    Drawdown days (DD<-5%): mean={df_mom.loc[dd_mask, 'top10_mom5d'].mean()*100:.2f}%")
    print(f"    Normal days (DD>-2%):   mean={df_mom.loc[normal_mask, 'top10_mom5d'].mean()*100:.2f}%")

    u_stat, u_p = mannwhitneyu(
        df_mom.loc[dd_mask & df_mom["top10_mom5d"].notna(), "top10_mom5d"],
        df_mom.loc[normal_mask & df_mom["top10_mom5d"].notna(), "top10_mom5d"],
        alternative="greater")
    print(f"    Mann-Whitney U (DD>Normal): p={u_p:.4f}")

    for col, lbl in [("top10_mom1d", "1d"), ("top10_mom3d", "3d"),
                        ("top10_mom5d", "5d"), ("top10_mom10d", "10d"),
                        ("top10_vol_ratio", "vol_ratio")]:
        dd_mean = df_mom.loc[dd_mask, col].mean()
        nl_mean = df_mom.loc[normal_mask, col].mean()
        print(f"  {lbl:>12}: DD={dd_mean*100 if 'mom' in col else dd_mean:>+.4f}%  "
              f"Normal={nl_mean*100 if 'mom' in col else nl_mean:>+.4f}%  "
              f"Δ={dd_mean-nl_mean:>+.4f}")

    # ── 3. Momentum quintile analysis: does extreme momentum predict loss? ──
    print("\n" + "=" * 80)
    print("=== Part 3: Momentum quintile → next-day return ===")
    print("  (Is there a nonlinear threshold where extreme momentum → reversal?)")

    # For each day, compute all stocks' 5d momentum, quintile them,
    # then compute each quintile's label return
    quintile_stats = {q: [] for q in [1, 2, 3, 4, 5]}
    label_dates = set(label.index.get_level_values(0))
    for d in dates:
        dts = pd.Timestamp(d)
        if dts not in mom_5d.index or dts not in label_dates:
            continue
        dl = label.xs(dts, level=0).dropna()
        m5 = mom_5d.loc[dts].dropna()
        common = dl.index.intersection(m5.index)
        if len(common) < 20:
            continue
        m5_c = m5.reindex(common)
        dl_c = dl.reindex(common)

        # Rank into quintiles by momentum
        ranks = m5_c.rank(pct=True)
        for q in [1, 2, 3, 4, 5]:
            lo = (q - 1) / 5
            hi = q / 5
            mask = (ranks > lo) & (ranks <= hi)
            if mask.sum() > 0:
                quintile_stats[q].append(np.mean(dl_c[mask].values))

    print(f"\n  {'Quintile':>10} {'Mean Label':>12} {'Description':>20}")
    print("  " + "-" * 50)
    for q in [1, 2, 3, 4, 5]:
        mean_lbl = np.mean(quintile_stats[q]) if quintile_stats[q] else np.nan
        desc = ["lowest mom", "", "", "", "highest mom"][q - 1]
        print(f"  Q{q} ({desc:>12}) {mean_lbl*100:>11.4f}%")

    # Focus on Q5 (extreme high momentum) — is the label negative?
    q5_mean = np.mean(quintile_stats[5])
    q4_mean = np.mean(quintile_stats[4])
    print(f"\n  Q5 (extreme high mom) label: {q5_mean*100:.4f}%")
    print(f"  Q4 label: {q4_mean*100:.4f}%")
    print(f"  Q5 < Q4? {'YES — chasing crash at extreme' if q5_mean < q4_mean else 'NO — momentum monotonic'}")

    # ── 4. Within top10: does the most overbought subset lose more? ─────────
    print("\n" + "=" * 80)
    print("=== Part 4: Within top10, momentum-split return ===")
    print("  (Split top10 into 'high-mom top5' vs 'low-mom top5' by 5d momentum)")

    high_mom_rets = []
    low_mom_rets = []
    for d in daily_data:
        dt = d["date"]
        codes = d["codes"]
        labels = d["labels"]
        if dt not in mom_5d.index:
            continue
        m5 = mom_5d.loc[dt].reindex(codes)
        if m5.dropna().shape[0] < 10:
            continue
        # Split top10 by momentum: high5 = top5 by mom5d, low5 = bottom5
        sorted_idx = m5.sort_values(ascending=False).index
        high5 = sorted_idx[:5]
        low5 = sorted_idx[5:]
        lbl_map = dict(zip(codes, labels))
        high_mom_rets.append(np.mean([lbl_map[c] for c in high5 if c in lbl_map]))
        low_mom_rets.append(np.mean([lbl_map[c] for c in low5 if c in lbl_map]))

    high_mom_rets = np.array(high_mom_rets)
    low_mom_rets = np.array(low_mom_rets)
    print(f"  High-mom top5 daily mean: {np.mean(high_mom_rets)*100:.4f}%")
    print(f"  Low-mom top5 daily mean:  {np.mean(low_mom_rets)*100:.4f}%")
    print(f"  Low - High: {(np.mean(low_mom_rets) - np.mean(high_mom_rets))*100:.4f}%/day")
    u, p = mannwhitneyu(low_mom_rets, high_mom_rets, alternative="greater")
    print(f"  Mann-Whitney U (Low>High): p={p:.4f}")

    # ── 5. Momentum threshold: conditional return ───────────────────────────
    print("\n" + "=" * 80)
    print("=== Part 5: Top10 conditional return by 5d momentum level ===")
    print("  (If top10 avg 5d mom > threshold, does next-day return deteriorate?)")

    df_mom_valid = df_mom.dropna(subset=["top10_mom5d"])
    thresholds = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    print(f"\n  {'Threshold':>10} {'Days above':>12} {'Mean net ret':>14} {'Days below':>12} {'Mean net ret':>14} {'p-value':>8}")
    print("  " + "-" * 80)
    for t in thresholds:
        above = df_mom_valid[df_mom_valid["top10_mom5d"] > t]
        below = df_mom_valid[df_mom_valid["top10_mom5d"] <= t]
        above_ret = above["net_ret"].mean() if len(above) > 0 else np.nan
        below_ret = below["net_ret"].mean() if len(below) > 0 else np.nan
        if len(above) > 5 and len(below) > 5:
            u, p = mannwhitneyu(below["net_ret"].values, above["net_ret"].values, alternative="greater")
        else:
            p = np.nan
        print(f"  {t*100:>9.0f}% {len(above):>12} {above_ret*100:>13.4f}% {len(below):>12} "
              f"{below_ret*100:>13.4f}% {p:>8.4f}")

    # ── 6. Drawdown period deep dive ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("=== Part 6: Drawdown period daily detail ===")
    dd_period = df_mom.loc[peak_date:max_dd_date]
    print(f"  Period: {peak_date.strftime('%Y-%m-%d')} → {max_dd_date.strftime('%Y-%m-%d')}")
    print(f"  Days: {len(dd_period)}")
    print(f"  Mean net ret: {dd_period['net_ret'].mean()*100:.4f}%")
    print(f"  Mean top10 5d mom: {dd_period['top10_mom5d'].mean()*100:.2f}%")
    print(f"  Mean top10 1d mom: {dd_period['top10_mom1d'].mean()*100:.2f}%")
    print(f"  Mean universe ret: {df_daily.loc[peak_date:max_dd_date, 'universe_mean'].mean()*100:.4f}%")

    print(f"\n  First 15 days of drawdown:")
    print(f"  {'Date':>12} {'net_ret':>10} {'mom5d':>10} {'mom1d':>10} {'DD':>8}")
    for d, row in dd_period.head(15).iterrows():
        print(f"  {d.strftime('%Y-%m-%d')} {row['net_ret']*100:>9.3f}% "
              f"{row['top10_mom5d']*100:>9.2f}% {row['top10_mom1d']*100:>9.2f}% "
              f"{row['drawdown']*100:>7.1f}%")


if __name__ == "__main__":
    main()
