import numpy as np
import pandas as pd

from scripts.validate_index_stage_joint_models import (
    VARIANTS,
    build_interaction_features,
    full_gate,
)


def _features():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05", "2026-01-06"]), ["A", "B"]],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame(
        {
            "online_stage": ["rebound_start", "uptrend", "boiling", "pullback_continuation"],
            "relative_return_3": [0.1, 0.2, 0.3, 0.4],
            "relative_return_5": [0.2, 0.1, 0.4, 0.3],
            "accel_5m": [0.1, 0.2, 0.3, 0.4],
            "minute_path_max_drawdown": [-0.2, -0.1, -0.4, -0.3],
            "minute_path_slope": [0.3, 0.1, 0.4, 0.2],
            "basket_idio_vol": [0.1, 0.2, 0.4, 0.3],
        },
        index=index,
    )


def test_interactions_are_finite_and_only_active_in_their_regime():
    result = build_interaction_features(_features())
    assert set(result.columns) == {field for fields in VARIANTS.values() for field in fields}
    assert np.isfinite(result.to_numpy()).all()
    assert result["stage_rebound"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert result["stage_overheat"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert result["stage_pullback"].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert (result.loc[result["stage_rebound"] == 0, "rebound_x_path_slope"] == 0).all()
    assert (result.loc[result["stage_overheat"] == 0, "overheat_x_idio_vol"] == 0).all()
    assert (result.loc[result["stage_pullback"] == 0, "pullback_x_joint_confirmation"] == 0).all()


def test_combined_variant_contains_both_regime_feature_sets():
    combined = set(VARIANTS["rebound_pullback_joint"])
    assert set(VARIANTS["rebound_joint"]) <= combined
    assert set(VARIANTS["pullback_joint"]) <= combined


def test_full_gate_rejects_any_drawdown_deterioration():
    baseline = {
        "IC": 0.05,
        "RankIC": 0.06,
        "absolute_annualized_return": 0.50,
        "absolute_max_drawdown": -0.20,
        "excess_annualized_return": 0.30,
        "excess_max_drawdown": -0.15,
    }
    candidate = {key: value + 0.01 for key, value in baseline.items()}
    assert full_gate(baseline, candidate)
    candidate["absolute_max_drawdown"] = -0.21
    assert not full_gate(baseline, candidate)


def test_full_gate_rejects_return_or_signal_quality_deterioration():
    baseline = {
        "IC": 0.05,
        "RankIC": 0.06,
        "absolute_annualized_return": 0.50,
        "absolute_max_drawdown": -0.20,
        "excess_annualized_return": 0.30,
        "excess_max_drawdown": -0.15,
    }
    candidate = dict(baseline)
    candidate["IC"] = 0.049
    assert not full_gate(baseline, candidate)
