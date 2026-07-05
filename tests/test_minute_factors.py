"""14 minute-factor formulas — hand-computed expected values.

Slots 0-10 = 9:30-9:40 (factor input), slot 11 = 9:41 (price_941, NOT a factor).
See spec §因子集 for formula derivation.
"""
import numpy as np
import pytest

from qlib_ifind_beta.minute_factors import compute_day_factors


def _synthetic():
    """12-slot arrays with a clean linear ramp so hand-math is exact."""
    c = np.array([10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1])
    o = np.array([9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0])
    h = c + 0.05        # high = close + 0.05
    l = o - 0.05        # low  = open - 0.05
    vol = np.array([100.0] * 10 + [50.0, 50.0])   # slot 10=50 (last factor bar, distinct), slot 11=50 (price_941, irrelevant)
    return c, o, h, l, vol


def test_startup_momentum():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_volume=24000.0)
    assert f["startup_mom_1m"] == pytest.approx(11.0 / 10.9 - 1)
    assert f["startup_mom_3m"] == pytest.approx(11.0 / 10.7 - 1)
    assert f["startup_mom_5m"] == pytest.approx(11.0 / 10.5 - 1)
    assert f["startup_total"] == pytest.approx(11.0 / 9.9 - 1)


def test_acceleration():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_volume=24000.0)
    assert f["accel_1m"] == pytest.approx((11.0 / 10.9 - 1) - (10.0 / 9.9 - 1))
    assert f["accel_3m"] == pytest.approx((11.0 / 10.7 - 1) - (10.2 / 9.9 - 1))
    assert f["accel_5m"] == pytest.approx((11.0 / 10.5 - 1) - (10.4 / 9.9 - 1))


def test_close_position():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_volume=24000.0)
    # window index10
    assert f["close_pos_1m"] == pytest.approx((11.0 - 10.85) / (11.05 - 10.85))
    # window index8-10: max h = 11.05, min l = 10.65
    assert f["close_pos_3m"] == pytest.approx((11.0 - 10.65) / (11.05 - 10.65))
    # window index6-10: max h = 11.05, min l = 10.45
    assert f["close_pos_5m"] == pytest.approx((11.0 - 10.45) / (11.05 - 10.45))


def test_volume_ratio():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_volume=24000.0)
    # vol[10]=50, mean(vol[0:10])=100
    assert f["vol_ratio_1m"] == pytest.approx(50.0 / 100.0)
    # mean(vol[8:11])=(100+100+50)/3, mean(vol[0:3])=100
    assert f["vol_ratio_3m"] == pytest.approx(((100 + 100 + 50) / 3) / 100.0)
    # mean(vol[6:11])=(100*4+50)/5=90, mean(vol[0:5])=100
    assert f["vol_ratio_5m"] == pytest.approx(90.0 / 100.0)


def test_vol_vs_yest():
    c, o, h, l, vol = _synthetic()
    # prev_day_volume=24000 → per-min denom = 100; numerator = sum(vol[0:11]) = 10*100+50 = 1050
    f = compute_day_factors(c, o, h, l, vol, prev_day_volume=24000.0)
    assert f["vol_vs_yest"] == pytest.approx(1050.0 / 100.0)
    # no prev volume → NaN (first trading day)
    f0 = compute_day_factors(c, o, h, l, vol, prev_day_volume=None)
    assert np.isnan(f0["vol_vs_yest"])


def test_price_941():
    c, o, h, l, vol = _synthetic()
    f = compute_day_factors(c, o, h, l, vol, prev_day_volume=24000.0)
    assert f["price_941"] == pytest.approx(11.1)   # c[11]
