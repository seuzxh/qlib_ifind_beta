"""14 minute-bar factors + $price_941, computed for ONE trading day.

Inputs are per-day 1min slot arrays (length 11). Slot map (probe-verified
2026-07-06, cn_data_1min): index 0-9 = slots 1-10 = 09:30-09:40 (the first 10
REAL trading bars; slot 0 is universally NaN pool-wide so the window starts at
slot 1), index 10 = slot 11 = 09:41 (buy-price bar, NOT used in factors).
See docs/superpowers/specs/2026-07-06-minute-factors-design.md.

Pure (no IO) — the materialize layer (materialize_minute.py) calls this per day
and writes the results as day.bin. Hand-unit-tested in tests/test_minute_factors.py.
"""
from __future__ import annotations

import numpy as np


def compute_day_factors(c, o, h, l, vol, prev_day_minute_vol=None) -> dict:
    """Compute 14 factors + price_941 for one trading day.

    Args:
        c, o, h, l, vol: 1D arrays length 11. Index 0-9 = slots 1-10 (09:30-09:40);
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

    # E. cross-day volume = first-ten-bar total / (prev day full-day vol / 240)
    if prev_day_minute_vol is not None and prev_day_minute_vol > 0:
        out["vol_vs_yest"] = vol[0:10].sum() / (prev_day_minute_vol / 240.0)
    else:
        out["vol_vs_yest"] = np.nan

    # buy-price bar (index 10 = slot 11 = 09:41 close)
    out["price_941"] = float(c[10])
    return out
