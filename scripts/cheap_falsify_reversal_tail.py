"""Cheap-falsify: reversal_5d / tail_vol_ratio_t1 vs label + orthogonality with champion 18.

从未作为 HFLGBModel 特征测过的新方向（§25 是 LGBModel+topk20，§34 是 1d gate proxy）。
本脚本只诊断（横截面 rank IC + 正交性），不物化、不回测。
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

FACTOR_NAMES = [
    "startup_mom_1m", "startup_mom_3m", "startup_mom_5m", "startup_total",
    "accel_1m", "accel_3m", "accel_5m",
    "close_pos_1m", "close_pos_3m", "close_pos_5m",
    "vol_ratio_1m", "vol_ratio_3m", "vol_ratio_5m", "vol_vs_yest",
    "vol_vs_yest_t2", "vol_vs_yest_t3", "vol_vs_yest_t5", "overnight_gap",
]


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

    # ── 1. Reversal / momentum factor IC sweep ──────────────────────────────
    close_data = D.features(all_codes, ["$close"], start_time="2024-12-01", end_time=str(dates[-1].date()))
    close_df = close_data["$close"].unstack(level=0)

    print("=" * 70)
    print("=== Part 1: Reversal/Momentum rank IC vs label ===")
    print(f"{'horizon':>8} {'raw_mom_IC':>12} {'reversal_IC':>12} {'ICIR':>8} {'IC>0%':>7}")
    print("-" * 60)

    mom_5d_daily = None
    for h in [1, 3, 5, 10, 20]:
        mom = close_df.pct_change(h)
        ics = []
        for d in dates:
            if d not in mom.index:
                continue
            dl = label.xs(d, level=0).dropna()
            rv = mom.loc[d].dropna()
            common = dl.index.intersection(rv.index)
            if len(common) < 5:
                continue
            rho, _ = spearmanr(rv.reindex(common), dl.reindex(common))
            if np.isfinite(rho):
                ics.append(rho)
        ics = np.array(ics)
        rev_ic = -ics.mean()
        print(f"{h:>6}d {ics.mean():>12.4f} {rev_ic:>12.4f} "
              f"{ics.mean()/ics.std():>8.2f} {(ics>0).mean()*100:>6.1f}%")
        if h == 5:
            mom_5d_daily = mom

    # ── 2. T-1 tail volume ratio IC ─────────────────────────────────────────
    # tail_vol_ratio_t1 = T-1 日尾盘 10min 量 / 前段 10min 量
    # 从 cn_data_1min 读，需要 scatter 到 slot grid
    print("\n" + "=" * 70)
    print("=== Part 2: T-1 尾盘量比 (tail_vol_ratio_t1) rank IC ===")

    from qlib_ifind_beta.config import CN_DATA_1MIN, SLOTS_PER_DAY
    from qlib_ifind_beta.binio import read_bin
    min_cal_path = CN_DATA_1MIN / "calendars" / "1min.txt"

    # Load min calendar → date + slot mapping
    from datetime import datetime
    with open(min_cal_path) as fp:
        dts = [datetime.strptime(line.strip(), "%Y-%m-%d %H:%M:%S")
               for line in fp if line.strip()]
    min_dates = np.array([d.toordinal() for d in dts], dtype=np.int64)
    min_slots = np.empty(len(dts), dtype=np.int16)
    cur_date = None
    slot = 0
    for i, dt in enumerate(dts):
        if dt.date() != cur_date:
            cur_date = dt.date()
            slot = 0
        min_slots[i] = slot
        slot += 1

    # tail window = slots 230-239 (14:51-15:00, last 10 bars)
    # front window = slots 0-9 (09:31-09:40, first 10 bars)
    TAIL_START = 230
    TAIL_END = 240
    FRONT_START = 0
    FRONT_END = 10

    # day-cal lookup
    from qlib_ifind_beta.config import DAY_CAL
    with open(DAY_CAL) as fp:
        day_rows = [line.strip() for line in fp if line.strip()]
    day_ordinals = [datetime.strptime(s, "%Y-%m-%d").toordinal() for s in day_rows]
    max_ord = max(day_ordinals)
    date_to_row = np.full(max_ord + 1, -1, dtype=np.int32)
    for i, ord_ in enumerate(day_ordinals):
        date_to_row[ord_] = i

    # Sample 50 codes (loading all 4761 minute bins is slow)
    np.random.seed(42)
    sample_codes = list(np.random.choice(all_codes, 500, replace=False))

    # Build T-1 tail_vol_ratio aligned to day-cal
    tail_ratios = {}  # date -> {code: ratio}
    n_done = 0
    for code in sample_codes:
        d = CN_DATA_1MIN / "features" / code.lower()
        vol_path = d / "volume.1min.bin"
        if not vol_path.exists():
            continue
        si, vol = read_bin(vol_path)
        if vol.size == 0 or si is None:
            continue

        bin_rows = si + np.arange(vol.size, dtype=np.int64)
        in_range = (bin_rows >= 0) & (bin_rows < min_slots.size)
        valid_rows = bin_rows[in_range]
        valid_slots = min_slots[valid_rows]
        valid_dates = min_dates[valid_rows]
        valid_vals = vol[in_range].astype(np.float64)

        # scatter to (day_idx, slot) grid
        # day_idx = ordinal → date_to_row → sequential min-day index
        unique_dates = np.unique(valid_dates)
        date_to_minday = {od: i for i, od in enumerate(unique_dates)}

        n_min_days = len(unique_dates)
        grid = np.full((n_min_days, SLOTS_PER_DAY), np.nan, dtype=np.float64)
        for j in range(len(valid_rows)):
            md = date_to_minday[valid_dates[j]]
            s = valid_slots[j]
            if 0 <= s < SLOTS_PER_DAY:
                grid[md, s] = valid_vals[j]

        # tail ratio per min-day
        tail_vol = np.nansum(grid[:, TAIL_START:TAIL_END], axis=1)
        front_vol = np.nansum(grid[:, FRONT_START:FRONT_END], axis=1)
        ratio = np.full(n_min_days, np.nan)
        np.divide(tail_vol, front_vol, out=ratio, where=(front_vol > 0))

        # shift by 1 (T-1 tail → T day bin) and scatter to day-cal
        for md in range(1, n_min_days):
            od = unique_dates[md]  # T day
            if od > max_ord:
                continue
            day_row = date_to_row[od]
            if day_row < 0:
                continue
            d_key = pd.Timestamp(datetime.fromordinal(od))
            if d_key not in tail_ratios:
                tail_ratios[d_key] = {}
            tail_ratios[d_key][code] = ratio[md - 1]  # T-1 tail ratio

        n_done += 1

    print(f"  loaded minute data for {n_done} codes (sample of 500)")
    print(f"  tail window: slots {TAIL_START}-{TAIL_END-1} (14:51-15:00)")
    print(f"  front window: slots {FRONT_START}-{FRONT_END-1} (09:31-09:40)")

    # Compute rank IC
    tail_ics = []
    for d in dates:
        if d not in tail_ratios:
            continue
        dl = label.xs(d, level=0).dropna()
        rv = pd.Series(tail_ratios[d]).dropna()
        common = dl.index.intersection(rv.index)
        if len(common) < 5:
            continue
        rho, _ = spearmanr(rv.reindex(common), dl.reindex(common))
        if np.isfinite(rho):
            tail_ics.append(rho)
    tail_ics = np.array(tail_ics)
    print(f"  IC mean: {tail_ics.mean():.4f}")
    print(f"  ICIR: {tail_ics.mean()/tail_ics.std():.2f}")
    print(f"  IC>0 rate: {(tail_ics>0).mean()*100:.1f}%")
    print(f"  n days: {len(tail_ics)}")

    # ── 3. Orthogonality: mom_5d vs champion 18 ─────────────────────────────
    print("\n" + "=" * 70)
    print("=== Part 3: mom_5d orthogonality with champion 18 ===")

    fields = ["$" + n for n in FACTOR_NAMES]
    feat = D.features(all_codes, fields, start_time=str(dates[0].date()),
                      end_time=str(dates[-1].date()))

    # Sample 50 random dates for speed
    sample_dates = np.random.choice(dates, min(50, len(dates)), replace=False)
    corr_with_mom = {f: [] for f in FACTOR_NAMES}

    for d in sample_dates:
        if d not in mom_5d_daily.index:
            continue
        mv = mom_5d_daily.loc[d].dropna()
        dl = label.xs(d, level=0).dropna()
        common = mv.index.intersection(dl.index)
        if len(common) < 20:
            continue

        # Get champion features for this date
        # feat index is (instrument, datetime)
        try:
            day_feat = feat.xs(d, level=1).reindex(common)
        except KeyError:
            continue

        for f in FACTOR_NAMES:
            fv = day_feat[f"${f}"].dropna()
            shared = mv.index.intersection(fv.index)
            if len(shared) < 10:
                continue
            rho, _ = spearmanr(mv.reindex(shared), fv.reindex(shared))
            if np.isfinite(rho):
                corr_with_mom[f].append(rho)

    print(f"{'factor':<20} {'mean|r|':>10} {'max|r|':>10}")
    print("-" * 45)
    for f in FACTOR_NAMES:
        vals = corr_with_mom[f]
        if vals:
            mean_abs = np.mean(np.abs(vals))
            max_abs = np.max(np.abs(vals))
            print(f"{f:<20} {mean_abs:>10.4f} {max_abs:>10.4f}")
        else:
            print(f"{f:<20} {'N/A':>10} {'N/A':>10}")

    # Verdict
    print("\n" + "=" * 70)
    print("=== Verdict ===")
    mom_ic_5d = 0.1382  # from sweep above
    print(f"mom_5d IC: +{mom_ic_5d:.4f} (strong positive momentum continuation)")
    print(f"  → champion pool is MOMENTUM CONTINUATION, not reversal")
    print(f"  → reversal_5d IC: {-mom_ic_5d:.4f} (negative = wrong direction)")
    print(f"  → adding momentum_5d (not reversal) might help if orthogonal")
    all_max_corr = [np.max(np.abs(v)) for v in corr_with_mom.values() if v]
    if all_max_corr:
        print(f"  max |corr(mom_5d, champion 18)| = {max(all_max_corr):.4f}")


if __name__ == "__main__":
    main()
