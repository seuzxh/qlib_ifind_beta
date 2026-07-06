"""14 minute-factor formulas — hand-computed expected values.

Length-11 array: index 0-9 = slots 1-10 = 09:30-09:40 (the first 10 REAL factor
bars; slot 0 (09:30) is universally NaN pool-wide so the window starts at slot 1
— probe 2026-07-06), index 10 = slot 11 = 09:41 (price_941, NOT a factor).
See spec §因子集 for formula derivation.
"""
import numpy as np
import pytest

from qlib_ifind_beta.minute_factors import compute_day_factors


def _synthetic():
    """11-slot arrays with a clean linear ramp so hand-math is exact."""
    c = np.array([10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0])
    o = np.array([9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9])
    h = c + 0.05        # high = close + 0.05
    l = o - 0.05        # low  = open - 0.05
    vol = np.array([100.0] * 9 + [50.0, 50.0])   # index 9=50 (last factor bar), index 10=50 (price_941, irrelevant)
    return c, o, h, l, vol


def test_startup_momentum():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_minute_vol=24000.0)
    # return from index 9 (slot 10, last factor bar)
    assert f["startup_mom_1m"] == pytest.approx(10.9 / 10.8 - 1)
    assert f["startup_mom_3m"] == pytest.approx(10.9 / 10.6 - 1)
    assert f["startup_mom_5m"] == pytest.approx(10.9 / 10.4 - 1)
    assert f["startup_total"] == pytest.approx(10.9 / 9.9 - 1)   # o[0] = slot 1 open


def test_acceleration():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_minute_vol=24000.0)
    assert f["accel_1m"] == pytest.approx((10.9 / 10.8 - 1) - (10.0 / 9.9 - 1))
    assert f["accel_3m"] == pytest.approx((10.9 / 10.6 - 1) - (10.1 / 9.9 - 1))
    assert f["accel_5m"] == pytest.approx((10.9 / 10.4 - 1) - (10.3 / 9.9 - 1))


def test_close_position():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_minute_vol=24000.0)
    # h = c+0.05 (increasing), l = o-0.05 (increasing); min/max over the window
    # pick the lowest/highest index respectively.
    # window index 9: h9=10.95, l9=10.75
    assert f["close_pos_1m"] == pytest.approx((10.9 - 10.75) / (10.95 - 10.75))
    # window index 7-9: max h = h9 = 10.95, min l = l7 = 10.55
    assert f["close_pos_3m"] == pytest.approx((10.9 - 10.55) / (10.95 - 10.55))
    # window index 5-9: max h = h9 = 10.95, min l = l5 = 10.35
    assert f["close_pos_5m"] == pytest.approx((10.9 - 10.35) / (10.95 - 10.35))


def test_volume_ratio():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_minute_vol=24000.0)
    # vol[9]=50, mean(vol[0:9])=100
    assert f["vol_ratio_1m"] == pytest.approx(50.0 / 100.0)
    # mean(vol[7:10])=(100+100+50)/3, mean(vol[0:2])=100
    assert f["vol_ratio_3m"] == pytest.approx(((100 + 100 + 50) / 3) / 100.0)
    # mean(vol[5:10])=(100*4+50)/5=90, mean(vol[0:4])=100
    assert f["vol_ratio_5m"] == pytest.approx(90.0 / 100.0)


def test_vol_vs_yest():
    c, o, h, l, vol = _synthetic()
    # prev_day_minute_vol=24000 → per-min denom = 100; numerator = sum(vol[0:10]) = 9*100+50 = 950
    f = compute_day_factors(c, o, h, l, vol, prev_day_minute_vol=24000.0)
    assert f["vol_vs_yest"] == pytest.approx(950.0 / 100.0)
    # no prev volume → NaN (first trading day)
    f0 = compute_day_factors(c, o, h, l, vol, prev_day_minute_vol=None)
    assert np.isnan(f0["vol_vs_yest"])


def test_price_941():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_minute_vol=24000.0)
    assert f["price_941"] == pytest.approx(11.0)   # c[10] = slot 11 = 09:41 close


def test_edge_cases_nan_guards():
    """Verify NaN guards: vol_ratio front-seg=0 → NaN, close_pos h==l → NaN, vol_vs_yest denom=0 → NaN."""
    n = 11
    idx = np.arange(n, dtype=np.float64)

    # --- Case 1: vol_ratio_* denominator = 0 → NaN (not inf) ---
    # vol[0:9] all zero so d1=d3=d5 are 0; vol[9]=100, vol[10]=50 (non-zero back-seg)
    vol_zero = np.zeros(n)
    vol_zero[9] = 100.0
    vol_zero[10] = 50.0
    c1 = 10.0 + idx * 0.1
    o1 = c1 - 0.05
    h1 = c1 + 0.05
    l1 = o1 - 0.05
    f1 = compute_day_factors(c1, o1, h1, l1, vol_zero)
    assert np.isnan(f1["vol_ratio_1m"])
    assert np.isnan(f1["vol_ratio_3m"])
    assert np.isnan(f1["vol_ratio_5m"])

    # --- Case 2: close_pos_* when h == l (limit-up/down flat bar) → NaN ---
    flat = np.full(n, 10.0)  # h == l == c == o everywhere
    vol_flat = np.full(n, 100.0)
    f2 = compute_day_factors(flat, flat, flat, flat, vol_flat)
    assert np.isnan(f2["close_pos_1m"])
    assert np.isnan(f2["close_pos_3m"])
    assert np.isnan(f2["close_pos_5m"])

    # --- Case 3: vol_vs_yest when prev_day_minute_vol == 0 → NaN ---
    c3, o3, h3, l3, vol3 = _synthetic()
    f3 = compute_day_factors(c3, o3, h3, l3, vol3, prev_day_minute_vol=0.0)
    assert np.isnan(f3["vol_vs_yest"])
