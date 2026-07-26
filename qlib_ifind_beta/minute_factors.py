"""14 minute-bar factors + $price_941, computed for ONE trading day.

Inputs are per-day 1min slot arrays (length 11). Slot map (probe-verified
2026-07-06, cn_data_1min): index 0-9 = slots 1-10 = 09:31-09:40 (the first 10
REAL trading bars; slot 0 is universally NaN pool-wide so the window starts at
slot 1), index 10 = slot 11 = 09:41 (buy-price bar, NOT used in factors).
The active timing and data contract is documented in
docs/superpowers/specs/2026-07-20-intraday-production-signal-design.md.

Pure (no IO) — the materialize layer (materialize_minute.py) calls this per day
and writes the results as day.bin. Hand-unit-tested in tests/test_minute_factors.py.
"""
from __future__ import annotations

import numpy as np

from .config import REAL_BARS_PER_DAY


def compute_day_factors(c, o, h, l, vol, prev_day_minute_vol=None) -> dict:
    """Compute 14 factors + price_941 for one trading day.

    Args:
        c, o, h, l, vol: 1D arrays length 11. Index 0-9 = slots 1-10 (09:31-09:40);
            index 10 = slot 11 (09:41 buy bar). Float-castable.
        prev_day_minute_vol: T-1 day's TOTAL minute volume (sum of all ~240 real
            bars that day) from cn_data_1min. None/<=0 → vol_vs_yest NaN.

    Returns:
        dict with keys = 14 factor names + "price_941". NaN where undefined.
    """
    c = np.asarray(c, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)

    out = {}
    # A. startup momentum (return from index 9 = slot 10)
    out["startup_mom_1m"] = c[9] / c[8] - 1.0
    out["startup_mom_3m"] = c[9] / c[6] - 1.0
    out["startup_mom_5m"] = c[9] / c[4] - 1.0
    out["startup_total"] = c[9] / o[0] - 1.0   # o[0] = slot 1 open = daily open

    # B. acceleration = back-seg return − front-seg return
    out["accel_1m"] = (c[9] / o[9] - 1.0) - (c[0] / o[0] - 1.0)
    out["accel_3m"] = (c[9] / o[7] - 1.0) - (c[1] / o[0] - 1.0)
    out["accel_5m"] = (c[9] / o[5] - 1.0) - (c[3] / o[0] - 1.0)

    # C. close position in window = (c9 - low_W) / (high_W - low_W)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["close_pos_1m"] = (c[9] - l[9]) / (h[9] - l[9])
        h3, l3 = h[7:10].max(), l[7:10].min()
        out["close_pos_3m"] = (c[9] - l3) / (h3 - l3)
        h5, l5 = h[5:10].max(), l[5:10].min()
        out["close_pos_5m"] = (c[9] - l5) / (h5 - l5)

    # D. volume ratio = back-seg mean vol / front-seg mean vol
    d1 = vol[0:9].mean()
    out["vol_ratio_1m"] = vol[9] / d1 if d1 > 0 else np.nan
    d3 = vol[0:2].mean()
    out["vol_ratio_3m"] = vol[7:10].mean() / d3 if d3 > 0 else np.nan
    d5 = vol[0:4].mean()
    out["vol_ratio_5m"] = vol[5:10].mean() / d5 if d5 > 0 else np.nan

    # E. cross-day volume = first-ten-bar total / (prev day full-day vol / REAL_BARS_PER_DAY)
    if prev_day_minute_vol is not None and prev_day_minute_vol > 0:
        out["vol_vs_yest"] = vol[0:10].sum() / (prev_day_minute_vol / float(REAL_BARS_PER_DAY))
    else:
        out["vol_vs_yest"] = np.nan

    # buy-price bar is auxiliary, not a feature.  Live production can compute
    # the ten factor bars before the 09:41 bar closes, so expose it only when
    # the caller supplied the eleventh value.
    if c.size > 10:
        out["price_941"] = float(c[10])
    return out


def compute_champion_factors(
    c, o, h, l, vol, prev_volumes,
    *, prev_close=None, prev_factor=None, today_open=None, today_factor=None,
    execution_close=None,
) -> dict:
    """Compute the frozen 18-factor Champion contract from ten factor bars.

    ``prev_volumes`` is ordered T-1/T-2/T-3/T-5 and uses full-day minute
    volume.  ``execution_close`` is the separately collected 09:41 close; it
    only creates price/change auxiliary fields and never changes a feature.
    """
    c = np.asarray(c, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)
    if any(arr.size < 10 for arr in (c, o, h, l, vol)):
        raise ValueError("champion factors require the closed 09:31-09:40 bars")
    pv = list(prev_volumes or [])
    pv += [0.0] * (4 - len(pv))
    out = compute_day_factors(
        c[:10], o[:10], h[:10], l[:10], vol[:10],
        prev_day_minute_vol=pv[0] if pv[0] > 0 else None,
    )
    morning_vol = float(np.sum(vol[:10]))
    for idx, name in ((1, "vol_vs_yest_t2"), (2, "vol_vs_yest_t3"),
                      (3, "vol_vs_yest_t5")):
        denominator = float(pv[idx])
        out[name] = (
            morning_vol / (denominator / float(REAL_BARS_PER_DAY))
            if denominator > 0 else np.nan
        )

    values = (prev_close, prev_factor, today_open, today_factor)
    if all(value is not None and np.isfinite(float(value)) for value in values):
        raw_prev_close = float(prev_close) / float(prev_factor)
        raw_open = float(today_open) / float(today_factor)
        out["overnight_gap"] = (
            raw_open / raw_prev_close - 1.0 if raw_prev_close > 0 else np.nan
        )
        if execution_close is not None and np.isfinite(float(execution_close)):
            out["price_941"] = float(execution_close)
            out["change_941"] = (
                (float(execution_close) / float(today_factor)) / raw_prev_close - 1.0
                if raw_prev_close > 0 else np.nan
            )
    else:
        out["overnight_gap"] = np.nan
        if execution_close is not None:
            out["price_941"] = float(execution_close)
            out["change_941"] = np.nan
    return out
