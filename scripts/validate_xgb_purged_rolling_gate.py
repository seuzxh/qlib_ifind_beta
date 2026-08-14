"""Leakage-safe 19-step walk-forward validation of the HFLGB/XGBoost gate.

The label for feature date T is only known at T+1 close.  Therefore a model
used at the first test date may use validation labels only through T-2.  This
script inserts one trading-session embargo between the 20-day validation
segment and each frozen next-test segment, then retrains both models.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qlib_ifind_beta.config import OVERLAY_ROOT, ROLLING_ENSEMBLE_WEIGHT
from qlib_ifind_beta.model_ensemble import (
    blend_scores, correlation_metrics, gate_passes, predict_feature_matrix,
)
from scripts.validate_factor_challengers import _backtest, _task

HF_EXPERIMENT = "rolling_90d_purged_hflgb"
XGB_EXPERIMENT = "rolling_90d_purged_xgb"
RESULT_PATH = ROOT / "data" / "xgb_purged_rolling_gate_ab.json"
TRAIN_DAYS = 90
VALID_DAYS = 20
EMBARGO_DAYS = 1


def purged_segments(calendar: list[pd.Timestamp],
                    test_segment: list[str] | tuple[str, str]) -> dict[str, list[str]]:
    """Return 90 train / 20 valid / 1 embargo / frozen test boundaries."""
    start, end = map(pd.Timestamp, test_segment)
    pos = calendar.index(start)
    required = TRAIN_DAYS + VALID_DAYS + EMBARGO_DAYS
    if pos < required:
        raise ValueError(f"not enough history before {start.date()}")
    valid_end_pos = pos - EMBARGO_DAYS - 1
    valid_start_pos = valid_end_pos - VALID_DAYS + 1
    train_end_pos = valid_start_pos - 1
    train_start_pos = train_end_pos - TRAIN_DAYS + 1
    fmt = lambda value: value.strftime("%Y-%m-%d")
    return {
        "train": [fmt(calendar[train_start_pos]), fmt(calendar[train_end_pos])],
        "valid": [fmt(calendar[valid_start_pos]), fmt(calendar[valid_end_pos])],
        "embargo": [fmt(calendar[pos - EMBARGO_DAYS]), fmt(calendar[pos - 1])],
        "test": [fmt(start), fmt(end)],
    }


def build_task(segments: dict[str, list[str]], variant: str) -> dict:
    model_segments = {key: segments[key] for key in ("train", "valid", "test")}
    return _task(
        "MinuteEnhancedHandler", "qlib_ifind_beta.minute_enhanced_handler",
        model_segments, variant,
    )


def _existing_by_test(experiment_name: str) -> dict[tuple[str, str], object]:
    from qlib.workflow import R

    try:
        recorders = R.get_exp(experiment_name=experiment_name, create=False).list_recorders().values()
    except Exception:
        return {}
    result = {}
    for recorder in recorders:
        try:
            task = recorder.load_object("task")
            segment = tuple(task["dataset"]["kwargs"]["segments"]["test"])
            recorder.load_object("params.pkl")
            recorder.load_object("pred.pkl")
            recorder.load_object("label.pkl")
            result[segment] = recorder
        except Exception:
            continue
    return result


def _validation_scores(hf_recorder, xgb_recorder, task):
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config

    dataset = init_instance_by_config(task["dataset"])
    frame = dataset.prepare(
        "valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I
    ).dropna()
    features, label = frame["feature"], frame["label"].iloc[:, 0]
    hf = predict_feature_matrix(hf_recorder.load_object("params.pkl"), features)
    xgb = predict_feature_matrix(xgb_recorder.load_object("params.pkl"), features)
    return hf, xgb, label


def _metrics(score: pd.Series, label: pd.Series,
             start: str, end: str) -> dict[str, float]:
    result = correlation_metrics(score, label)
    result.update(_backtest(score.to_frame("score"), start, end))
    return result


def main() -> None:
    import qlib
    from qlib.config import C
    from qlib.data import D
    from qlib.model.trainer import task_train

    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(ROOT / "mlruns")
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn", exp_manager=exp_manager)
    calendar = [pd.Timestamp(date) for date in D.calendar(freq="day")]
    original = json.loads((ROOT / "data" / "xgb_rolling_gate_ab.json").read_text())
    test_segments = [tuple(row["test_segment"]) for row in original["segments"]]

    hf_existing = _existing_by_test(HF_EXPERIMENT)
    xgb_existing = _existing_by_test(XGB_EXPERIMENT)
    baseline_tests, adaptive_tests, fixed_tests, labels, rows = [], [], [], [], []
    for number, test_segment in enumerate(test_segments, 1):
        segments = purged_segments(calendar, test_segment)
        hf = hf_existing.get(test_segment)
        if hf is None:
            hf = task_train(build_task(segments, "champion18"), experiment_name=HF_EXPERIMENT)
            hf_existing[test_segment] = hf
        xgb = xgb_existing.get(test_segment)
        if xgb is None:
            xgb = task_train(build_task(segments, "xgb18"), experiment_name=XGB_EXPERIMENT)
            xgb_existing[test_segment] = xgb

        task = hf.load_object("task")
        hf_valid, xgb_valid, valid_label = _validation_scores(hf, xgb, task)
        candidate_valid = blend_scores(hf_valid, xgb_valid, ROLLING_ENSEMBLE_WEIGHT)
        base_metrics = _metrics(hf_valid, valid_label, *segments["valid"])
        candidate_metrics = _metrics(candidate_valid, valid_label, *segments["valid"])
        chosen = ROLLING_ENSEMBLE_WEIGHT if gate_passes(base_metrics, candidate_metrics) else 0.0

        hf_obj = hf.load_object("pred.pkl")
        hf_test = hf_obj.iloc[:, 0] if isinstance(hf_obj, pd.DataFrame) else hf_obj
        xgb_obj = xgb.load_object("pred.pkl")
        xgb_test = xgb_obj.iloc[:, 0] if isinstance(xgb_obj, pd.DataFrame) else xgb_obj
        label_obj = hf.load_object("label.pkl")
        label = label_obj.iloc[:, 0] if isinstance(label_obj, pd.DataFrame) else label_obj
        baseline_tests.append(hf_test)
        adaptive_tests.append(blend_scores(hf_test, xgb_test, chosen))
        fixed_tests.append(blend_scores(hf_test, xgb_test, ROLLING_ENSEMBLE_WEIGHT))
        labels.append(label)
        rows.append({
            "segments": segments, "chosen_weight": chosen,
            "validation_baseline": base_metrics,
            "validation_candidate": candidate_metrics,
            "hflgb_recorder_id": getattr(hf, "recorder_id", None) or getattr(hf, "id", None),
            "xgb_recorder_id": getattr(xgb, "recorder_id", None) or getattr(xgb, "id", None),
        })
        RESULT_PATH.write_text(json.dumps({"embargo_days": EMBARGO_DAYS, "segments": rows},
                                          ensure_ascii=False, indent=2) + "\n")
        print(f"{number:02d}/{len(test_segments)} test={test_segment[0]}..{test_segment[1]} "
              f"chosen={chosen:.2f} IC={base_metrics['IC']:+.4f}->{candidate_metrics['IC']:+.4f} "
              f"RankIC={base_metrics['RankIC']:+.4f}->{candidate_metrics['RankIC']:+.4f}",
              flush=True)

    baseline = pd.concat(baseline_tests).sort_index()
    adaptive = pd.concat(adaptive_tests).sort_index()
    fixed = pd.concat(fixed_tests).sort_index()
    label = pd.concat(labels).sort_index().reindex(baseline.index)
    start = baseline.index.get_level_values("datetime").min().strftime("%Y-%m-%d")
    end = baseline.index.get_level_values("datetime").max().strftime("%Y-%m-%d")
    summary = {}
    for name, score in (("baseline", baseline), ("adaptive", adaptive), ("fixed_025", fixed)):
        summary[name] = _metrics(score, label, start, end)
        print(name, json.dumps(summary[name], ensure_ascii=False), flush=True)
    result = {"embargo_days": EMBARGO_DAYS, "segments": rows, "summary": summary}
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
