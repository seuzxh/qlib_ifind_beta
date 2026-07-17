from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _index(n=25):
    return pd.MultiIndex.from_product(
        [[pd.Timestamp("2026-07-02")], [f"S{i:02d}" for i in range(n)]],
        names=["datetime", "instrument"],
    )


def test_chase_risk_requires_prior_trend_and_same_day_acceleration():
    from qlib_ifind_beta.risk_overlay import compute_chase_risk

    index = _index(3)
    frame = pd.DataFrame({
        "mom_5d": [.20, .20, -.05],
        "positive_days_5": [1.0, 1.0, 1.0],
        "overnight_gap": [.03, .03, .03],
        "startup_total": [.02, .02, .02],
        "accel_5m": [.02, -.01, .02],
    }, index=index)
    risk = compute_chase_risk(frame)
    assert risk.iloc[0] > 0
    assert risk.iloc[1] == 0
    assert risk.iloc[2] == 0


def test_tracking_risk_penalizes_beta_deviation_low_corr_and_idio_vol():
    from qlib_ifind_beta.risk_overlay import compute_tracking_risk

    frame = pd.DataFrame({
        "basket_beta_10m": [1.0, 2.0, .9],
        "basket_resonance_corr": [.9, -.5, .8],
        "basket_idio_vol": [.01, .10, .02],
    }, index=_index(3))
    risk = compute_tracking_risk(frame)
    assert risk.iloc[1] > risk.iloc[2] > risk.iloc[0]


def test_penalty_operates_only_inside_fixed_top20_pool():
    from qlib_ifind_beta.risk_overlay import apply_top_pool_penalty

    index = _index()
    score = pd.Series(np.arange(25, dtype=float), index=index, name="score")
    risk = pd.Series(0.0, index=index)
    risk.iloc[-1] = 1.0
    adjusted = apply_top_pool_penalty(score, risk, .75, .15, top_pool=20)
    selected = set(adjusted.nlargest(10).index)
    original_top20 = set(score.nlargest(20).index)
    assert selected <= original_top20
    assert adjusted.iloc[-1] < adjusted.iloc[-2]


def test_penalty_missing_risk_is_safe_and_zero_weight_is_zero_drift():
    from qlib_ifind_beta.risk_overlay import apply_top_pool_penalty

    index = _index()
    score = pd.Series(np.arange(25, dtype=float), index=index)
    pd.testing.assert_series_equal(
        apply_top_pool_penalty(score, pd.Series(dtype=float), .85, 0), score
    )
    adjusted = apply_top_pool_penalty(score, pd.Series(np.nan, index=index), .85, .1)
    assert adjusted.notna().all()


@pytest.mark.parametrize("threshold", [0, 1, -1, 2])
def test_penalty_rejects_invalid_threshold(threshold):
    from qlib_ifind_beta.risk_overlay import apply_top_pool_penalty

    score = pd.Series([1.0], index=_index(1))
    with pytest.raises(ValueError):
        apply_top_pool_penalty(score, score, threshold, .1)
