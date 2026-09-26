"""Production helpers for the validation-gated HFLGB/XGBoost ensemble.

The candidate weight is frozen at 25%.  A rolling model may use it only when
the preceding purged validation segment improves Pearson IC, Spearman RankIC,
and exact TD0 annualized excess return. A one-session embargo separates that
validation data from the frozen test block. Missing or malformed metadata
always degrades to the HFLGB score rather than guessing a weight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CHAMPION_EXPERIMENT,
    CHAMPION_RECORDER_ID,
    ROLLING_EMBARGO_DAYS,
    ROLLING_ENSEMBLE_WEIGHT,
    ROLLING_EXPERIMENT,
    ROLLING_GATE_ARTIFACT,
    ROLLING_XGB_EXPERIMENT,
)

GATE_SCHEMA_VERSION = 2
REQUIRED_METRICS = ("IC", "RankIC", "excess_annualized_return")


def cross_sectional_zscore(score: pd.Series) -> pd.Series:
    """Z-score each prediction date using pandas' sample standard deviation."""
    if "datetime" not in score.index.names:
        mean, std = score.mean(), score.std()
        return (score - mean) / (std if pd.notna(std) and std != 0 else np.nan)
    grouped = score.groupby(level="datetime")
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return (score - mean) / std


def blend_scores(hflgb: pd.Series, xgb: pd.Series, weight: float) -> pd.Series:
    """Blend model scores on their common, finite cross-section."""
    if weight == 0:
        return hflgb.copy()
    if not 0 <= weight <= 1:
        raise ValueError(f"ensemble weight must be in [0,1], got {weight}")
    common = hflgb.dropna().index.intersection(xgb.dropna().index)
    return ((1 - weight) * cross_sectional_zscore(hflgb.loc[common])
            + weight * cross_sectional_zscore(xgb.loc[common]))


def correlation_metrics(score: pd.Series, label: pd.Series,
                        min_instruments: int = 20) -> dict[str, float]:
    """Mean daily Pearson IC and Spearman RankIC on aligned observations."""
    common = score.index.intersection(label.index)
    score, label = score.loc[common], label.loc[common]
    values: dict[str, list[float]] = {"IC": [], "RankIC": []}
    if "datetime" in score.index.names:
        dates = score.index.get_level_values("datetime").unique()
        slices = ((score.xs(date, level="datetime"),
                   label.xs(date, level="datetime")) for date in dates)
    else:
        slices = ((score, label),)
    for daily_score, daily_label in slices:
        daily_common = daily_score.dropna().index.intersection(daily_label.dropna().index)
        if len(daily_common) < min_instruments:
            continue
        pearson = daily_score.loc[daily_common].corr(daily_label.loc[daily_common])
        spearman = daily_score.loc[daily_common].corr(daily_label.loc[daily_common], method="spearman")
        if pd.notna(pearson):
            values["IC"].append(float(pearson))
        if pd.notna(spearman):
            values["RankIC"].append(float(spearman))
    return {key: float(np.mean(item)) if item else float("nan")
            for key, item in values.items()}


def gate_passes(baseline: dict[str, float], candidate: dict[str, float]) -> bool:
    """Return True only when all three frozen validation gates improve."""
    for key in REQUIRED_METRICS:
        base = baseline.get(key)
        cand = candidate.get(key)
        if base is None or cand is None or not np.isfinite(base) or not np.isfinite(cand):
            return False
        if cand < base:
            return False
    return True


def make_gate_metadata(
    baseline: dict[str, float],
    candidate: dict[str, float],
    xgb_recorder_id: str,
    test_segment: list[str] | tuple[str, str],
    validation_segment: list[str] | tuple[str, str],
) -> dict[str, Any]:
    """Build the only artifact format accepted by online inference."""
    passed = bool(xgb_recorder_id) and gate_passes(baseline, candidate)
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "embargo_days": ROLLING_EMBARGO_DAYS,
        "weight": ROLLING_ENSEMBLE_WEIGHT if passed else 0.0,
        "xgb_recorder_id": xgb_recorder_id,
        "xgb_experiment": ROLLING_XGB_EXPERIMENT,
        "test_segment": list(test_segment),
        "validation_segment": list(validation_segment),
        "validation_baseline": baseline,
        "validation_candidate": candidate,
        "gate_passed": passed,
    }


def validate_gate_metadata(metadata: Any, target_date: str | None = None) -> dict[str, Any]:
    """Fail closed on stale, incomplete, or tampered ensemble metadata."""
    if not isinstance(metadata, dict) or metadata.get("schema_version") != GATE_SCHEMA_VERSION:
        raise ValueError("unsupported or missing ensemble gate schema")
    if metadata.get("embargo_days") != ROLLING_EMBARGO_DAYS:
        raise ValueError("ensemble metadata does not prove the label embargo")
    weight = metadata.get("weight")
    if weight not in (0.0, ROLLING_ENSEMBLE_WEIGHT):
        raise ValueError(f"unexpected ensemble weight {weight}")
    if bool(metadata.get("gate_passed")) != (weight > 0):
        raise ValueError("gate_passed and weight disagree")
    segment = metadata.get("test_segment")
    if not isinstance(segment, (list, tuple)) or len(segment) != 2:
        raise ValueError("invalid test segment")
    if target_date is not None and not (str(segment[0]) <= target_date <= str(segment[1])):
        raise ValueError(f"online model segment {segment} does not cover {target_date}")
    valid_segment = metadata.get("validation_segment")
    if not isinstance(valid_segment, (list, tuple)) or len(valid_segment) != 2:
        raise ValueError("invalid validation segment")
    if weight > 0 and not metadata.get("xgb_recorder_id"):
        raise ValueError("positive ensemble weight requires an XGBoost recorder")
    return metadata


def validate_recorder_provenance(recorder: Any, metadata: dict[str, Any]) -> None:
    """Prove that recorder task, gate artifact and one-session purge agree."""
    from qlib.data import D

    segments = recorder.load_object("task")["dataset"]["kwargs"]["segments"]
    if list(segments["test"]) != list(metadata["test_segment"]):
        raise ValueError("gate test segment disagrees with recorder task")
    if list(segments["valid"]) != list(metadata["validation_segment"]):
        raise ValueError("gate validation segment disagrees with recorder task")
    valid_end = str(segments["valid"][1])
    test_start = str(segments["test"][0])
    gap = list(D.calendar(start_time=valid_end, end_time=test_start, freq="day"))
    if len(gap) != ROLLING_EMBARGO_DAYS + 2:
        raise ValueError("recorder task does not contain the required label embargo")


def predict_feature_matrix(model: Any, features: pd.DataFrame) -> pd.Series:
    """Predict a raw feature matrix with either Qlib HFLGBModel or XGBModel."""
    inner = getattr(model, "model", model)
    module = type(model).__module__.lower() + "." + type(inner).__module__.lower()
    if "xgboost" in module or type(model).__name__ == "XGBModel":
        import xgboost as xgb
        values = inner.predict(xgb.DMatrix(features.values))
    else:
        values = inner.predict(features.values)
    return pd.Series(np.asarray(values).reshape(-1), index=features.index)


@dataclass(frozen=True)
class OnlineModelBundle:
    hflgb: Any
    xgb: Any | None
    weight: float
    source: str
    metadata: dict[str, Any] | None = None


def _latest_covering_recorder(recorders: list[Any], target_date: str) -> Any | None:
    covering = []
    for recorder in recorders:
        try:
            task = recorder.load_object("task")
            segment = task["dataset"]["kwargs"]["segments"]["test"]
            if str(segment[0]) <= target_date <= str(segment[1]):
                covering.append((str(segment[1]), recorder))
        except Exception:
            continue
    return max(covering, key=lambda item: item[0])[1] if covering else None


def load_model_bundle(target_date: str, use_online: bool = True) -> OnlineModelBundle:
    """Load the date-matched online pair, with a frozen champion fallback.

    Any online metadata/XGBoost failure keeps the valid online HFLGB model at
    weight zero.  Absence of a date-matched online HFLGB falls back to the
    frozen production champion.
    """
    from qlib.workflow import R

    if not use_online:
        rec = R.get_recorder(recorder_id=CHAMPION_RECORDER_ID,
                             experiment_name=CHAMPION_EXPERIMENT)
        return OnlineModelBundle(rec.load_object("params.pkl"), None, 0.0, "frozen")

    from qlib.workflow.online.utils import OnlineToolR
    tool = OnlineToolR(ROLLING_EXPERIMENT)
    recorder = _latest_covering_recorder(
        list(tool.online_models(exp_name=ROLLING_EXPERIMENT)), target_date
    )
    if recorder is None:
        rec = R.get_recorder(recorder_id=CHAMPION_RECORDER_ID,
                             experiment_name=CHAMPION_EXPERIMENT)
        return OnlineModelBundle(rec.load_object("params.pkl"), None, 0.0,
                                 "frozen:no_date_matched_online")

    hflgb = recorder.load_object("params.pkl")
    try:
        metadata = validate_gate_metadata(
            recorder.load_object(ROLLING_GATE_ARTIFACT), target_date=target_date
        )
        validate_recorder_provenance(recorder, metadata)
        if metadata["weight"] == 0:
            return OnlineModelBundle(hflgb, None, 0.0, "rolling_hflgb", metadata)
        xgb_rec = R.get_recorder(
            recorder_id=metadata["xgb_recorder_id"],
            experiment_name=metadata.get("xgb_experiment", ROLLING_XGB_EXPERIMENT),
        )
        return OnlineModelBundle(hflgb, xgb_rec.load_object("params.pkl"),
                                 float(metadata["weight"]), "rolling_ensemble", metadata)
    except Exception as exc:
        return OnlineModelBundle(hflgb, None, 0.0,
                                 f"rolling_hflgb:ensemble_fallback:{type(exc).__name__}")


def predict_bundle_matrix(bundle: OnlineModelBundle, features: pd.DataFrame) -> pd.Series:
    hflgb = predict_feature_matrix(bundle.hflgb, features)
    if bundle.weight == 0 or bundle.xgb is None:
        return hflgb
    xgb = predict_feature_matrix(bundle.xgb, features)
    return blend_scores(hflgb, xgb, bundle.weight)
