"""14 minute-bar factors + $price_941, computed for ONE trading day.

Inputs are per-day 1min slot arrays (length >= 12). Slot map (probe-verified,
cn_data_1min): index 0-10 = 9:30-9:40 (factor input), index 11 = 9:41 (buy-price
bar, NOT used in factors). See docs/superpowers/specs/2026-07-06-minute-factors-design.md.

Pure (no IO) — the materialize layer (materialize_minute.py) calls this per day
and writes the results as day.bin. Hand-unit-tested in tests/test_minute_factors.py.
"""
from __future__ import annotations

import numpy as np


def compute_day_factors(c, o, h, l, vol, prev_day_volume=None) -> dict:
    """Compute 14 factors + price_941 for one trading day.

    Args:
        c, o, h, l, vol: 1D arrays length >= 12 (slots 0-11). Float-castable.
        prev_day_volume: previous trading day's TOTAL daily volume (scalar), used
            as the vol_vs_yest denominator (Ref($volume,1)/240). None/<=0 → NaN
            (first trading day or missing daily bin).

    Returns:
        dict with keys = 14 factor names + "price_941". NaN where undefined.
    """
    c = np.asarray(c, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)

    out = {}
    # A. startup momentum (last-N-bar return from slot 10)
    out["startup_mom_1m"] = c[10] / c[9] - 1.0
    out["startup_mom_3m"] = c[10] / c[7] - 1.0
    out["startup_mom_5m"] = c[10] / c[5] - 1.0
    out["startup_total"] = c[10] / o[0] - 1.0

    # B. acceleration = back-seg return − front-seg return (seg = seg-open → seg-close)
    out["accel_1m"] = (c[10] / o[10] - 1.0) - (c[0] / o[0] - 1.0)
    out["accel_3m"] = (c[10] / o[8] - 1.0) - (c[2] / o[0] - 1.0)
    out["accel_5m"] = (c[10] / o[6] - 1.0) - (c[4] / o[0] - 1.0)

    # C. close position in window = (c10 - low_W) / (high_W - low_W)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["close_pos_1m"] = (c[10] - l[10]) / (h[10] - l[10])
        h3, l3 = h[8:11].max(), l[8:11].min()
        out["close_pos_3m"] = (c[10] - l3) / (h3 - l3)
        h5, l5 = h[6:11].max(), l[6:11].min()
        out["close_pos_5m"] = (c[10] - l5) / (h5 - l5)

    # D. volume ratio = back-seg mean vol / front-seg mean vol
    d1 = vol[0:10].mean()
    out["vol_ratio_1m"] = vol[10] / d1 if d1 > 0 else np.nan
    d3 = vol[0:3].mean()
    out["vol_ratio_3m"] = vol[8:11].mean() / d3 if d3 > 0 else np.nan
    d5 = vol[0:5].mean()
    out["vol_ratio_5m"] = vol[6:11].mean() / d5 if d5 > 0 else np.nan

    # E. cross-day volume = first-11-bar total / (prev day total vol / 240)
    if prev_day_volume is not None and prev_day_volume > 0:
        out["vol_vs_yest"] = vol[0:11].sum() / (prev_day_volume / 240.0)
    else:
        out["vol_vs_yest"] = np.nan

    # buy-price bar (slot 11) — not a feature, materialized as $price_941
    out["price_941"] = float(c[11])
    return out
