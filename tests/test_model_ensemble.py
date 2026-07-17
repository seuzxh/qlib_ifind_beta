from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _series(values):
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2026-07-02")], [f"S{i}" for i in range(len(values))]],
        names=["datetime", "instrument"],
    )
    return pd.Series(values, index=index, dtype=float)


def test_blend_scores_matches_frozen_cross_sectional_formula():
    from qlib_ifind_beta.model_ensemble import blend_scores, cross_sectional_zscore

    hflgb = _series([1, 2, 3, 4])
    xgb = _series([4, 1, 3, 2])
    actual = blend_scores(hflgb, xgb, 0.25)
    expected = 0.75 * cross_sectional_zscore(hflgb) + 0.25 * cross_sectional_zscore(xgb)
    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_series_equal(blend_scores(hflgb, xgb, 0), hflgb)


@pytest.mark.parametrize(
    "candidate,expected",
    [
        ({"IC": .11, "RankIC": .21, "excess_annualized_return": .31}, True),
        ({"IC": .09, "RankIC": .21, "excess_annualized_return": .31}, False),
        ({"IC": .11, "RankIC": .19, "excess_annualized_return": .31}, False),
        ({"IC": .11, "RankIC": .21, "excess_annualized_return": .29}, False),
        ({"IC": .11, "RankIC": .21, "excess_annualized_return": np.nan}, False),
    ],
)
def test_gate_requires_all_three_metrics(candidate, expected):
    from qlib_ifind_beta.model_ensemble import gate_passes

    baseline = {"IC": .10, "RankIC": .20, "excess_annualized_return": .30}
    assert gate_passes(baseline, candidate) is expected


def test_gate_metadata_is_date_bound_and_fail_closed():
    from qlib_ifind_beta.model_ensemble import make_gate_metadata, validate_gate_metadata

    baseline = {"IC": .10, "RankIC": .20, "excess_annualized_return": .30}
    candidate = {"IC": .11, "RankIC": .21, "excess_annualized_return": .31}
    metadata = make_gate_metadata(
        baseline, candidate, "xgb-id", ["2026-07-02", "2026-07-02"],
        ["2026-06-02", "2026-06-30"],
    )
    assert validate_gate_metadata(metadata, "2026-07-02")["weight"] == .25
    with pytest.raises(ValueError, match="does not cover"):
        validate_gate_metadata(metadata, "2026-07-03")

    tampered = dict(metadata, weight=.5)
    with pytest.raises(ValueError, match="unexpected ensemble weight"):
        validate_gate_metadata(tampered, "2026-07-02")
    with pytest.raises(ValueError, match="label embargo"):
        validate_gate_metadata(dict(metadata, embargo_days=0), "2026-07-02")


def test_missing_xgb_recorder_can_never_enable_gate():
    from qlib_ifind_beta.model_ensemble import make_gate_metadata

    metrics = {"IC": .10, "RankIC": .20, "excess_annualized_return": .30}
    metadata = make_gate_metadata(
        metrics, metrics, "", ["2026-07-02", "2026-07-02"],
        ["2026-06-02", "2026-06-30"],
    )
    assert metadata["gate_passed"] is False
    assert metadata["weight"] == 0


class _InnerModel:
    def predict(self, values):
        return np.asarray(values).sum(axis=1)


class _HFLGBModel:
    def __init__(self):
        self.model = _InnerModel()


def test_predict_feature_matrix_preserves_index():
    from qlib_ifind_beta.model_ensemble import predict_feature_matrix

    features = pd.DataFrame([[1, 2], [3, 4]], index=["A", "B"])
    result = predict_feature_matrix(_HFLGBModel(), features)
    pd.testing.assert_series_equal(result, pd.Series([3, 7], index=features.index))


class _Recorder:
    def __init__(self, objects):
        self.objects = objects

    def load_object(self, name):
        value = self.objects[name]
        if isinstance(value, Exception):
            raise value
        return value


def test_load_bundle_uses_date_matched_pair(monkeypatch):
    import qlib
    qlib.init(provider_uri="data/qlib_root", region="cn")
    from qlib.workflow import R
    from qlib.workflow.online import utils
    from qlib_ifind_beta.config import ROLLING_GATE_ARTIFACT
    from qlib_ifind_beta.model_ensemble import load_model_bundle, make_gate_metadata

    metrics = {"IC": .10, "RankIC": .20, "excess_annualized_return": .30}
    gate = make_gate_metadata(
        metrics, metrics, "xgb-id", ["2026-07-02", "2026-07-02"],
        ["2026-06-02", "2026-06-30"],
    )
    online = _Recorder({
        "task": {"dataset": {"kwargs": {"segments": {
            "valid": ["2026-06-02", "2026-06-30"],
            "test": ["2026-07-02", "2026-07-02"]
        }}}},
        "params.pkl": "hf-model",
        ROLLING_GATE_ARTIFACT: gate,
    })
    xgb = _Recorder({"params.pkl": "xgb-model"})

    class _Tool:
        def __init__(self, *args, **kwargs):
            pass

        def online_models(self, **kwargs):
            return [online]

    monkeypatch.setattr(utils, "OnlineToolR", _Tool)
    monkeypatch.setattr(R, "get_recorder", lambda **kwargs: xgb)
    bundle = load_model_bundle("2026-07-02")
    assert bundle.source == "rolling_ensemble"
    assert bundle.hflgb == "hf-model"
    assert bundle.xgb == "xgb-model"
    assert bundle.weight == .25


def test_load_bundle_task_gate_mismatch_falls_back_to_online_hflgb(monkeypatch):
    import qlib
    qlib.init(provider_uri="data/qlib_root", region="cn")
    from qlib.workflow import R
    from qlib.workflow.online import utils
    from qlib_ifind_beta.config import ROLLING_GATE_ARTIFACT
    from qlib_ifind_beta.model_ensemble import load_model_bundle, make_gate_metadata

    metrics = {"IC": .10, "RankIC": .20, "excess_annualized_return": .30}
    gate = make_gate_metadata(
        metrics, metrics, "xgb-id", ["2026-07-02", "2026-07-02"],
        ["2026-06-02", "2026-06-30"],
    )
    online = _Recorder({
        "task": {"dataset": {"kwargs": {"segments": {
            "valid": ["2026-06-03", "2026-07-01"],
            "test": ["2026-07-02", "2026-07-02"],
        }}}},
        "params.pkl": "hf-model",
        ROLLING_GATE_ARTIFACT: gate,
    })

    class _Tool:
        def __init__(self, *args, **kwargs):
            pass

        def online_models(self, **kwargs):
            return [online]

    monkeypatch.setattr(utils, "OnlineToolR", _Tool)
    monkeypatch.setattr(R, "get_recorder", lambda **kwargs: pytest.fail("must not load XGB"))
    bundle = load_model_bundle("2026-07-02")
    assert bundle.hflgb == "hf-model"
    assert bundle.weight == 0
    assert "ensemble_fallback:ValueError" in bundle.source


def test_load_bundle_bad_artifact_keeps_online_hflgb(monkeypatch):
    import qlib
    qlib.init(provider_uri="data/qlib_root", region="cn")
    from qlib.workflow import R
    from qlib.workflow.online import utils
    from qlib_ifind_beta.config import ROLLING_GATE_ARTIFACT
    from qlib_ifind_beta.model_ensemble import load_model_bundle

    online = _Recorder({
        "task": {"dataset": {"kwargs": {"segments": {
            "test": ["2026-07-02", "2026-07-02"]
        }}}},
        "params.pkl": "hf-model",
        ROLLING_GATE_ARTIFACT: ValueError("corrupt"),
    })

    class _Tool:
        def __init__(self, *args, **kwargs):
            pass

        def online_models(self, **kwargs):
            return [online]

    monkeypatch.setattr(utils, "OnlineToolR", _Tool)
    monkeypatch.setattr(R, "get_recorder", lambda **kwargs: pytest.fail("no fallback needed"))
    bundle = load_model_bundle("2026-07-02")
    assert bundle.hflgb == "hf-model"
    assert bundle.weight == 0
    assert bundle.source.startswith("rolling_hflgb:ensemble_fallback")
