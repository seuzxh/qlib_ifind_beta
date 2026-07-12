"""Drawdown winners analysis — CORRECTED: per-day time-varying universe.

§54 original analysis was wrong: it used pool-union (4661 stocks across 361 days)
instead of the daily time-varying universe (~90 stocks/day, 89% daily turnover).

Correct approach: for each trading day T in the drawdown period, take the
DAILY universe (~90 stocks), use label[T] (= T-day 9:41 buy → T+1 close return)
to identify winners/losers WITHIN THAT DAY'S POOL. Then analyze cross-sectional
characteristics of winners vs losers.

This is the selection set the strategy actually faces each day.
"""
from __future__ import annotations

import os, sys, pickle
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from qlib_ifind_beta.config import OVERLAY_ROOT

DD_START = "2025-02-24"
DD_END = "2025-04-07"
NORMAL_START = "2025-05-01"  # normal period for comparison
NORMAL_END = "2025-06-30"


def main():
    import qlib
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib.data import D

    with open("data/rolling_90d_result.pkl", "rb") as f:
        dd = pickle.load(f)
    pred = dd["pred"]
    label = dd["label"]
    dates = sorted(pred.index.get_level_values(0).unique())

    # ── 1. Per-day universe analysis ────────────────────────────────────────
    print("=" * 80)
    print("=== Part 1: Per-day universe during drawdown (corrected) ===")
    print(f"  Label = T-day 9:41 buy → T+1 close return (the actual holding period)")
    print(f"  Universe = daily high-beta pool (~90 stocks, time-varying)")

    dd_dates = [d for d in dates if pd.Timestamp(DD_START) <= d <= pd.Timestamp(DD_END)]
    nl_dates = [d for d in dates if pd.Timestamp(NORMAL_START) <= d <= pd.Timestamp(NORMAL_END)]

    print(f"  Drawdown period: {len(dd_dates)} trading days ({DD_START} → {DD_END})")
    print(f"  Normal period:   {len(nl_dates)} trading days ({NORMAL_START} → {NORMAL_END})")

    # For each day, compute: universe size, % winners (label>0), top10 ret, universe mean
    dd_daily = []
    nl_daily = []
    for d in dd_dates:
        dl = label.xs(d, level=0).dropna()
        dp = pred.xs(d, level=0).dropna()
        common = dl.index.intersection(dp.index)
        if len(common) < 10:
            continue
        labels = dl.reindex(common)
        preds = dp.reindex(common)
        top10 = preds.sort_values(ascending=False).head(10)
        top10_ret = labels.reindex(top10.index).mean()
        win_rate = (labels > 0).mean()
        dd_daily.append({
            "date": d, "pool_size": len(common),
            "win_rate": win_rate,
            "pool_mean": labels.mean(),
            "top10_ret": top10_ret,
            "labels": labels,
            "preds": preds,
            "codes": list(common),
        })

    for d in nl_dates:
        dl = label.xs(d, level=0).dropna()
        dp = pred.xs(d, level=0).dropna()
        common = dl.index.intersection(dp.index)
        if len(common) < 10:
            continue
        labels = dl.reindex(common)
        preds = dp.reindex(common)
        top10 = preds.sort_values(ascending=False).head(10)
        top10_ret = labels.reindex(top10.index).mean()
        win_rate = (labels > 0).mean()
        nl_daily.append({
            "date": d, "pool_size": len(common),
            "win_rate": win_rate,
            "pool_mean": labels.mean(),
            "top10_ret": top10_ret,
            "labels": labels,
            "preds": preds,
            "codes": list(common),
        })

    dd_wr = np.array([d["win_rate"] for d in dd_daily])
    dd_pm = np.array([d["pool_mean"] for d in dd_daily])
    dd_tr = np.array([d["top10_ret"] for d in dd_daily])
    nl_wr = np.array([d["win_rate"] for d in nl_daily])
    nl_pm = np.array([d["pool_mean"] for d in nl_daily])
    nl_tr = np.array([d["top10_ret"] for d in nl_daily])

    print(f"\n  {'Metric':<25} {'Drawdown':>12} {'Normal':>12}")
    print(f"  {'-'*52}")
    print(f"  {'Pool size (avg)':<25} {np.mean([d['pool_size'] for d in dd_daily]):>12.1f} {np.mean([d['pool_size'] for d in nl_daily]):>12.1f}")
    print(f"  {'Daily win rate':<25} {dd_wr.mean()*100:>11.1f}% {nl_wr.mean()*100:>11.1f}%")
    print(f"  {'Pool mean label':<25} {dd_pm.mean()*100:>+11.4f}% {nl_pm.mean()*100:>+11.4f}%")
    print(f"  {'Top10 mean label':<25} {dd_tr.mean()*100:>+11.4f}% {nl_tr.mean()*100:>+11.4f}%")
    print(f"  {'Top10 - Pool alpha':<25} {(dd_tr.mean()-dd_pm.mean())*100:>+11.4f}% {(nl_tr.mean()-nl_pm.mean())*100:>+11.4f}%")

    # ── 2. Within-day winner/loser characteristics (pooled) ────────────────
    print("\n" + "=" * 80)
    print("=== Part 2: Within-day winner vs loser cross-sectional features ===")
    print("  (Each day, split pool into label>0 'winners' vs label<=0 'losers')")

    # Load daily close + volume for momentum features
    all_codes = sorted(set(pred.index.get_level_values(1)))
    close_vol = D.features(all_codes, ["$close", "$volume"], start_time="2025-01-01",
                           end_time=str(dates[-1].date()))
    close_df = close_vol["$close"].unstack(level=0)
    vol_df = close_vol["$volume"].unstack(level=0)
    mom_5d = close_df.pct_change(5)
    mom_1d = close_df.pct_change(1)
    mom_10d = close_df.pct_change(10)
    vol_ratio = vol_df / vol_df.rolling(5).mean()

    # Pool all drawdown days: each stock-day is one observation
    # Features measured at T (pre-trade), label is T-day holding return
    dd_rows = []
    for d_info in dd_daily:
        d = d_info["date"]
        if d not in mom_5d.index:
            continue
        for code in d_info["codes"]:
            lbl = d_info["labels"].get(code, np.nan)
            if np.isnan(lbl):
                continue
            dd_rows.append({
                "date": d, "code": code, "label": lbl,
                "mom_5d": mom_5d.loc[d].get(code, np.nan),
                "mom_1d": mom_1d.loc[d].get(code, np.nan),
                "mom_10d": mom_10d.loc[d].get(code, np.nan),
                "vol_ratio": vol_ratio.loc[d].get(code, np.nan) if d in vol_ratio.index else np.nan,
                "pred": d_info["preds"].get(code, np.nan),
                "price": close_df.loc[d].get(code, np.nan) if d in close_df.index else np.nan,
            })
    dd_pooled = pd.DataFrame(dd_rows).dropna(subset=["label"])

    # Split within each day: label > median → "winners", < median → "losers"
    # Actually, split by label > 0 vs label <= 0 (simple winner/loser)
    dd_win = dd_pooled[dd_pooled["label"] > 0]
    dd_lose = dd_pooled[dd_pooled["label"] <= 0]

    print(f"\n  Pooled stock-days (drawdown period): {len(dd_pooled)}")
    print(f"  Winners (label>0): {len(dd_win)} ({len(dd_win)/len(dd_pooled)*100:.1f}%)")
    print(f"  Losers  (label<=0): {len(dd_lose)} ({len(dd_lose)/len(dd_pooled)*100:.1f}%)")

    print(f"\n  {'Feature':<15} {'Winners':>12} {'Losers':>12} {'p-value':>8} {'Direction':>12}")
    print(f"  {'-'*62}")
    for feat in ["mom_1d", "mom_5d", "mom_10d", "vol_ratio", "pred", "price"]:
        w_vals = dd_win[feat].dropna()
        l_vals = dd_lose[feat].dropna()
        if len(w_vals) < 10 or len(l_vals) < 10:
            continue
        u, p = mannwhitneyu(w_vals, l_vals, alternative="two-sided")
        direction = "W>L" if w_vals.mean() > l_vals.mean() else "W<L"
        if feat == "price":
            print(f"  {feat:<15} {w_vals.mean():>12.1f} {l_vals.mean():>12.1f} {p:>8.4f} {direction:>12}")
        else:
            print(f"  {feat:<15} {w_vals.mean():>12.4f} {l_vals.mean():>12.4f} {p:>8.4f} {direction:>12}")

    # ── 3. Rank IC of features vs label within drawdown ────────────────────
    print("\n" + "=" * 80)
    print("=== Part 3: Feature rank IC vs label (within drawdown days) ===")

    for feat in ["mom_1d", "mom_5d", "mom_10d", "vol_ratio", "pred"]:
        ics = []
        for d_info in dd_daily:
            d = d_info["date"]
            if d not in mom_5d.index:
                continue
            feat_vals = pd.Series()
            if feat == "pred":
                feat_vals = d_info["preds"]
            elif feat == "mom_1d":
                feat_vals = mom_1d.loc[d]
            elif feat == "mom_5d":
                feat_vals = mom_5d.loc[d]
            elif feat == "mom_10d":
                feat_vals = mom_10d.loc[d]
            elif feat == "vol_ratio":
                feat_vals = vol_ratio.loc[d] if d in vol_ratio.index else pd.Series(dtype=float)

            common = d_info["labels"].index.intersection(feat_vals.dropna().index)
            if len(common) < 5:
                continue
            rho, _ = spearmanr(feat_vals.reindex(common), d_info["labels"].reindex(common))
            if np.isfinite(rho):
                ics.append(rho)
        ics = np.array(ics)
        print(f"  {feat:<15} IC={ics.mean():>+.4f}  IC>0={((ics>0).mean()*100):>5.1f}%  n={len(ics)}")

    # ── 4. Strategy hit rate within each day's pool ────────────────────────
    print("\n" + "=" * 80)
    print("=== Part 4: Strategy top10 hit rate within daily pool ===")

    # For each day: what fraction of top10 are in the top-quartile of labels?
    hit_rates = []
    for d_info in dd_daily:
        labels = d_info["labels"]
        preds = d_info["preds"]
        top10_codes = preds.sort_values(ascending=False).head(10).index
        # Within-pool rank of label: top quartile = "winners"
        q75 = labels.quantile(0.75)
        top_quartile = set(labels[labels >= q75].index)
        hits = len(set(top10_codes) & top_quartile)
        hit_rates.append(hits / 10)

    hit_rates = np.array(hit_rates)
    print(f"  Top10 in top-quartile-label: mean={hit_rates.mean()*100:.1f}% (random=25%)")
    print(f"  Enrichment: {hit_rates.mean()/0.25:.2f}x over random")
    print(f"  Days analyzed: {len(hit_rates)}")

    # Same for normal period
    nl_hit_rates = []
    for d_info in nl_daily:
        labels = d_info["labels"]
        preds = d_info["preds"]
        top10_codes = preds.sort_values(ascending=False).head(10).index
        q75 = labels.quantile(0.75)
        top_quartile = set(labels[labels >= q75].index)
        hits = len(set(top10_codes) & top_quartile)
        nl_hit_rates.append(hits / 10)
    nl_hit_rates = np.array(nl_hit_rates)
    print(f"\n  Normal period comparison:")
    print(f"  Top10 in top-quartile: mean={nl_hit_rates.mean()*100:.1f}% (random=25%)")
    print(f"  Enrichment: {nl_hit_rates.mean()/0.25:.2f}x over random")
    print(f"  → Strategy alpha in DD: {hit_rates.mean()/0.25:.2f}x vs Normal: {nl_hit_rates.mean()/0.25:.2f}x")

    # ── 5. Within-day winner correlation ──────────────────────────────────
    print("\n" + "=" * 80)
    print("=== Part 5: Within-day winner characteristics summary ===")

    # Daily winner rate distribution
    print(f"  Daily win rate distribution (drawdown period):")
    print(f"    min: {dd_wr.min()*100:.1f}%, max: {dd_wr.max()*100:.1f}%")
    print(f"    days with <10% winners: {(dd_wr<0.1).sum()}/{len(dd_wr)}")
    print(f"    days with >30% winners: {(dd_wr>0.3).sum()}/{len(dd_wr)}")

    # Is win_rate related to pool mean?
    rho, p = spearmanr(dd_wr, dd_pm)
    print(f"\n  Win rate vs pool mean label: corr={rho:.4f} (p={p:.4f})")


if __name__ == "__main__":
    main()
