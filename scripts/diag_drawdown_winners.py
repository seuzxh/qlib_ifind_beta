"""Drawdown period winners analysis: who survived 2025-02-24 → 04-07?

Identify stocks that had positive returns during the max drawdown period,
analyze their concept clustering and style characteristics vs losers.
"""
from __future__ import annotations

import os, sys, pickle
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from qlib_ifind_beta.config import OVERLAY_ROOT

DD_START = "2025-02-24"
DD_END = "2025-04-07"


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

    # ── 1. Compute per-stock return during drawdown period ─────────────────
    print("=" * 80)
    print("=== Part 1: Per-stock return during drawdown (2025-02-24 → 04-07) ===")

    # Use daily close to compute period return per stock
    close_data = D.features(all_codes, ["$close"], start_time=DD_START, end_time=DD_END)
    # Period return = close[end] / close[start] - 1 per stock
    period_ret = {}
    for code in all_codes:
        try:
            s = close_data.loc[code, "$close"]
        except KeyError:
            continue
        if len(s) < 2:
            continue
        ret = s.iloc[-1] / s.iloc[0] - 1
        if np.isfinite(ret):
            period_ret[code] = ret

    ret_series = pd.Series(period_ret)
    print(f"  Stocks with full period data: {len(ret_series)}")
    print(f"  Period return distribution:")
    print(f"    mean:   {ret_series.mean()*100:+.2f}%")
    print(f"    median: {ret_series.median()*100:+.2f}%")
    print(f"    % positive: {(ret_series > 0).mean()*100:.1f}%")
    print(f"    % >+10%:   {(ret_series > 0.10).mean()*100:.1f}%")
    print(f"    % >+20%:   {(ret_series > 0.20).mean()*100:.1f}%")
    print(f"    min:    {ret_series.min()*100:+.2f}%")
    print(f"    max:    {ret_series.max()*100:+.2f}%")

    # Winners (positive return) vs losers
    winners = ret_series[ret_series > 0].sort_values(ascending=False)
    big_winners = ret_series[ret_series > 0.10].sort_values(ascending=False)
    losers = ret_series[ret_series <= 0].sort_values()

    print(f"\n  Winners (ret>0):  {len(winners)} stocks ({len(winners)/len(ret_series)*100:.1f}%)")
    print(f"  Big winners (>10%): {len(big_winners)} stocks ({len(big_winners)/len(ret_series)*100:.1f}%)")
    print(f"  Losers (ret<=0):  {len(losers)} stocks ({len(losers)/len(ret_series)*100:.1f}%)")

    print(f"\n  Top 30 big winners:")
    print(f"  {'Code':<12} {'Period Ret':>12}")
    print(f"  {'-'*25}")
    for code, ret in big_winners.head(30).items():
        print(f"  {code:<12} {ret*100:>+11.2f}%")

    # ── 2. Concept clustering of winners ───────────────────────────────────
    print("\n" + "=" * 80)
    print("=== Part 2: Concept clustering of winners ===")

    # Load concepts
    concept_dir = Path("/home/zxh/qlib_data/instruments")
    code2concepts = defaultdict(set)
    for f in sorted(concept_dir.glob("concept_tzt_*.txt")):
        concept_id = f.stem
        with open(f) as fp:
            for line in fp:
                parts = line.strip().split("\t")
                if parts and parts[0]:
                    code2concepts[parts[0]].add(concept_id)

    # Count concept frequency among big winners vs all stocks
    winner_concepts = defaultdict(int)
    winner_in_concept = defaultdict(list)
    for code in big_winners.index:
        for c in code2concepts.get(code, set()):
            winner_concepts[c] += 1
            winner_in_concept[c].append(code)

    all_concepts = defaultdict(int)
    for code in all_codes:
        for c in code2concepts.get(code, set()):
            all_concepts[c] += 1

    # Enrichment: winner_share / universe_share
    print(f"\n  Concept enrichment (big winners > 10%):")
    print(f"  {'Concept':<25} {'Winners':>8} {'Universe':>10} {'Enrichment':>12}")
    print(f"  {'-'*60}")
    enrichments = []
    for c, w_count in sorted(winner_concepts.items(), key=lambda x: -x[1])[:25]:
        u_count = all_concepts.get(c, 1)
        winner_rate = w_count / len(big_winners)
        universe_rate = u_count / len(all_codes)
        enr = winner_rate / universe_rate if universe_rate > 0 else 0
        enrichments.append((c, w_count, u_count, enr))
        print(f"  {c:<25} {w_count:>8} {u_count:>10} {enr:>11.2f}x")

    # ── 3. Winner style characteristics ────────────────────────────────────
    print("\n" + "=" * 80)
    print("=== Part 3: Winner vs Loser style characteristics ===")
    print("  (Style measured at DD start = 2025-02-24, using pre-DD data)")

    # Pre-DD 5d and 10d momentum, volume ratio
    style_data = D.features(all_codes, ["$close", "$volume"],
                            start_time="2025-01-20", end_time=DD_START)
    close_pre = style_data["$close"].unstack(level=0)
    vol_pre = style_data["$volume"].unstack(level=0)

    # Momentum at DD start
    mom_5d_start = close_pre.pct_change(5).loc[DD_START] if DD_START in close_pre.index else pd.Series(dtype=float)
    mom_10d_start = close_pre.pct_change(10).loc[DD_START] if DD_START in close_pre.index else pd.Series(dtype=float)
    vol_ratio_5d = (vol_pre / vol_pre.rolling(5).mean()).loc[DD_START] if DD_START in vol_pre.index else pd.Series(dtype=float)

    # Absolute price level (proxy for market cap)
    price_at_start = close_pre.loc[DD_START] if DD_START in close_pre.index else pd.Series(dtype=float)

    winner_codes = set(big_winners.index)
    loser_codes = set(losers.index)

    print(f"\n  {'Metric':<20} {'Winners':>12} {'Losers':>12} {'p-value':>8}")
    print(f"  {'-'*55}")
    for name, series in [("5d mom", mom_5d_start), ("10d mom", mom_10d_start),
                          ("vol_ratio_5d", vol_ratio_5d), ("price_level", price_at_start)]:
        w_vals = series.reindex(list(winner_codes)).dropna()
        l_vals = series.reindex(list(loser_codes)).dropna()
        if len(w_vals) < 5 or len(l_vals) < 5:
            print(f"  {name:<20} {'N/A':>12} {'N/A':>12}")
            continue
        u, p = mannwhitneyu(w_vals, l_vals, alternative="two-sided")
        fmt = "{:.4f}" if "mom" in name or "ratio" in name else "{:.1f}"
        print(f"  {name:<20} {fmt.format(w_vals.mean()):>12} {fmt.format(l_vals.mean()):>12} {p:>8.4f}")

    # ── 4. Did the strategy pick any winners? ──────────────────────────────
    print("\n" + "=" * 80)
    print("=== Part 4: Did champion strategy pick winners during drawdown? ===")

    # For each day in DD period, check if top10 picks overlapped with eventual winners
    dd_dates = [d for d in dates if pd.Timestamp(DD_START) <= d <= pd.Timestamp(DD_END)]
    winner_picks = 0
    total_picks = 0
    all_top10_picks = []

    for d in dd_dates:
        dp = pred.xs(d, level=0).dropna()
        dl = label.xs(d, level=0).dropna()
        common = dp.index.intersection(dl.index)
        if len(common) < 10:
            continue
        top = dp.reindex(common).sort_values(ascending=False).head(10)
        top_codes = list(top.index)
        all_top10_picks.extend(top_codes)
        total_picks += len(top_codes)
        for c in top_codes:
            if c in winner_codes:
                winner_picks += 1

    # Also check unique picks
    unique_picks = set(all_top10_picks)
    unique_winner_picks = unique_picks & winner_codes

    print(f"  Total top10 pick-slots during DD: {total_picks}")
    print(f"  Winner pick-slots: {winner_picks} ({winner_picks/total_picks*100:.1f}%)")
    print(f"  Unique stocks picked: {len(unique_picks)}")
    print(f"  Unique winners picked: {len(unique_winner_picks)} ({len(unique_winner_picks)/len(unique_picks)*100:.1f}%)")
    print(f"  Total big winners in pool: {len(winner_codes)}")
    print(f"  → Strategy picked {len(unique_winner_picks)}/{len(winner_codes)} big winners ({len(unique_winner_picks)/len(winner_codes)*100:.1f}%)")

    # ── 5. Correlation among winners during drawdown ──────────────────────
    print("\n" + "=" * 80)
    print("=== Part 5: Winner return correlation during drawdown ===")

    # Daily returns of big winners during DD period
    winner_daily = close_data.loc[list(big_winners.index[:30]), "$close"].unstack(level=0).pct_change().dropna()
    if len(winner_daily.columns) > 1:
        corr_matrix = winner_daily.corr()
        # Upper triangle average
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_corr = corr_matrix.values[mask].mean()
        med_corr = np.median(corr_matrix.values[mask])
        print(f"  Top 30 winners daily return correlation:")
        print(f"    mean pairwise corr: {avg_corr:.4f}")
        print(f"    median pairwise corr: {med_corr:.4f}")
        print(f"    >0.5 pairs: {(corr_matrix.values[mask] > 0.5).sum()}/{mask.sum()}")

        # Compare with losers
        loser_daily = close_data.loc[list(losers.index[:30]), "$close"].unstack(level=0).pct_change().dropna()
        lcorr_matrix = loser_daily.corr()
        lmask = np.triu(np.ones_like(lcorr_matrix, dtype=bool), k=1)
        lavg_corr = lcorr_matrix.values[lmask].mean()
        print(f"\n  Top 30 losers daily return correlation:")
        print(f"    mean pairwise corr: {lavg_corr:.4f}")
        print(f"    → Winners {'more' if avg_corr > lavg_corr else 'less'} correlated than losers")


if __name__ == "__main__":
    main()
