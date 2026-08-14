"""Full model-level A/B: champion18 versus champion18 + stage interactions.

Each of the 19 existing purged folds retrains HFLGB and XGBoost on the joint
feature matrix.  Validation first gates the candidate's own HFLGB/XGB blend,
then gates the resulting joint model against the date-matched champion model.
Only the frozen winner is used on that fold's test segment.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from qlib_ifind_beta.model_ensemble import (
    blend_scores, correlation_metrics, gate_passes, predict_feature_matrix,
)
from scripts.diagnose_index_stage_tail_association import (
    MLRUNS_ROOT, PROVIDER_ROOT, READ_ROOT, _stage_frame, _stock_features,
)
from scripts.validate_factor_challengers import _backtest
from scripts.validate_risk_overlay_purged import _test_score
from qlib_ifind_beta.config import ROLLING_ENSEMBLE_WEIGHT

RESULT_PATH = ROOT / "data" / "index_stage_joint_model_validation.json"
SOURCE_PATH = READ_ROOT / "data" / "xgb_purged_rolling_gate_ab.json"
FRAME_CACHE = ROOT / "data" / "index_stage_joint_model_frame.pkl"
PRED_CACHE = ROOT / "data" / "index_stage_joint_predictions"

VARIANTS = {
    "rebound_joint": (
        "stage_rebound", "rebound_x_path_slope", "rebound_x_shallow_drawdown",
    ),
    "overheat_joint": (
        "stage_overheat", "overheat_x_relative_strength", "overheat_x_idio_vol",
    ),
    "pullback_joint": (
        "stage_pullback", "pullback_x_joint_confirmation",
    ),
    "rebound_pullback_joint": (
        "stage_rebound", "rebound_x_path_slope", "rebound_x_shallow_drawdown",
        "stage_pullback", "pullback_x_joint_confirmation",
    ),
}


def _daily_rank(values: pd.Series) -> pd.Series:
    return values.groupby(level="datetime").rank(method="average", pct=True)


def build_interaction_features(features: pd.DataFrame) -> pd.DataFrame:
    """Explicit T-safe interactions; zero means the regime is inactive."""
    stage = features["online_stage"]
    rebound = (stage == "rebound_start").astype(float)
    overheat = stage.isin(["uptrend", "boiling"]).astype(float)
    pullback = (stage == "pullback_continuation").astype(float)
    relative = features["relative_return_3"].where(
        stage == "uptrend", features["relative_return_5"]
    )
    pullback_parts = pd.concat([
        _daily_rank(features["relative_return_5"]),
        _daily_rank(features["accel_5m"]),
        _daily_rank(features["minute_path_max_drawdown"]),
    ], axis=1)
    return pd.DataFrame({
        "stage_rebound": rebound,
        "rebound_x_path_slope": rebound * _daily_rank(features["minute_path_slope"]),
        "rebound_x_shallow_drawdown": rebound * _daily_rank(
            features["minute_path_max_drawdown"]
        ),
        "stage_overheat": overheat,
        "overheat_x_relative_strength": overheat * _daily_rank(relative),
        "overheat_x_idio_vol": overheat * _daily_rank(features["basket_idio_vol"]),
        "stage_pullback": pullback,
        "pullback_x_joint_confirmation": pullback * pullback_parts.prod(
            axis=1, min_count=3
        ).pow(1 / 3),
    }, index=features.index).fillna(0.0).astype("float32")


def _load_base_frame(start: str, end: str) -> pd.DataFrame:
    from qlib.utils import init_instance_by_config

    config = {
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "MinuteEnhancedHandler",
                "module_path": "qlib_ifind_beta.minute_enhanced_handler",
                "kwargs": {
                    "instruments": "highbeta883926",
                    "start_time": start, "end_time": end,
                    "fit_start_time": start, "fit_end_time": end,
                    "label": ["Ref($close, -1) / $price_941 - 1"],
                },
            },
            "segments": {"all": [start, end]},
        },
    }
    dataset = init_instance_by_config(config)
    return dataset.prepare(
        "all", col_set=["feature", "label"], data_key="learn"
    ).dropna()


def load_joint_frame(start: str, end: str) -> pd.DataFrame:
    if FRAME_CACHE.exists():
        frame = pd.read_pickle(FRAME_CACHE)
        if (frame.index.get_level_values("datetime").min() <= pd.Timestamp(start)
                and frame.index.get_level_values("datetime").max() >= pd.Timestamp(end)):
            return frame
    base = _load_base_frame(start, end)
    dummy = pd.Series(0.0, index=base.index, name="score")
    stages = _stage_frame(base.index)
    explanatory = _stock_features(base.index, stages, dummy)
    interactions = build_interaction_features(explanatory).reindex(base.index).fillna(0.0)
    for name in interactions:
        base[("feature", name)] = interactions[name]
    FRAME_CACHE.parent.mkdir(parents=True, exist_ok=True)
    base.to_pickle(FRAME_CACHE)
    return base


def _dataset(frame: pd.DataFrame, segments: dict[str, list[str]]):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    return DatasetH(handler=DataHandlerLP.from_df(frame), segments=segments)


def _new_models():
    from qlib.contrib.model.highfreq_gdbt_model import HFLGBModel
    from qlib.contrib.model.xgboost import XGBModel

    hf = HFLGBModel(
        loss="binary", learning_rate=.05, max_depth=6, num_leaves=64,
        num_threads=20, lambda_l1=5.0, lambda_l2=10.0,
    )
    xgb = XGBModel(
        objective="reg:squarederror", eval_metric="rmse", eta=.05,
        max_depth=6, alpha=5.0, **{"lambda": 10.0},
        nthread=20, verbosity=0,
    )
    return hf, xgb


def _series(obj) -> pd.Series:
    return obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj


def _metrics(score: pd.Series, label: pd.Series,
             start: str, end: str) -> dict[str, float]:
    result = correlation_metrics(score, label)
    result.update(_backtest(score.to_frame("score"), start, end))
    return result


def full_gate(baseline: dict, candidate: dict) -> bool:
    """Production-strength validation gate, including both drawdowns."""
    if not gate_passes(baseline, candidate):
        return False
    return bool(
        candidate["absolute_annualized_return"] >= baseline["absolute_annualized_return"]
        and candidate["absolute_max_drawdown"] >= baseline["absolute_max_drawdown"] - 1e-6
        and candidate["excess_max_drawdown"] >= baseline["excess_max_drawdown"] - 1e-6
    )


def _train_or_load(variant: str, fold_no: int, frame: pd.DataFrame,
                   segments: dict[str, list[str]]) -> dict:
    PRED_CACHE.mkdir(parents=True, exist_ok=True)
    path = PRED_CACHE / f"{variant}_fold{fold_no:02d}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    keep = [column for column in frame["feature"].columns
            if column not in {name for fields in VARIANTS.values() for name in fields}]
    keep += list(VARIANTS[variant])
    candidate_frame = pd.concat({
        "feature": frame["feature"].loc[:, keep],
        "label": frame["label"],
    }, axis=1)
    dataset = _dataset(candidate_frame, {
        key: segments[key] for key in ("train", "valid", "test")
    })
    hf, xgb = _new_models()
    hf.fit(dataset, verbose_eval=0)
    xgb.fit(dataset, verbose_eval=False)
    valid = dataset.prepare("valid", col_set=["feature", "label"], data_key="infer").dropna()
    test = dataset.prepare("test", col_set=["feature", "label"], data_key="infer").dropna()
    result = {
        "hf_valid": predict_feature_matrix(hf, valid["feature"]),
        "xgb_valid": predict_feature_matrix(xgb, valid["feature"]),
        "valid_label": valid["label"].iloc[:, 0],
        "hf_test": predict_feature_matrix(hf, test["feature"]),
        "xgb_test": predict_feature_matrix(xgb, test["feature"]),
        "test_label": test["label"].iloc[:, 0],
        "feature_count": len(keep),
    }
    pd.to_pickle(result, path)
    return result


def main() -> None:
    import qlib
    from qlib.config import C

    C["exp_manager"]["kwargs"]["uri"] = "file:" + str(MLRUNS_ROOT)
    qlib.init(provider_uri=str(PROVIDER_ROOT), region="cn", exp_manager=C["exp_manager"])
    source = json.loads(SOURCE_PATH.read_text())
    rows = source["segments"]
    start = min(row["segments"]["train"][0] for row in rows)
    end = max(row["segments"]["test"][1] for row in rows)
    frame = load_joint_frame(start, end)
    print(f"joint frame {frame.shape}, period={start}..{end}", flush=True)
    selected = os.environ.get("INDEX_STAGE_VARIANTS", "").strip()
    selected_variants = (
        [name.strip() for name in selected.split(",") if name.strip()]
        if selected else list(VARIANTS)
    )
    unknown = set(selected_variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    result_path = Path(os.environ.get("INDEX_STAGE_RESULT_PATH", str(RESULT_PATH)))
    output = {"design": {
        "folds": len(rows), "baseline_features": 18,
        "variants": {name: list(fields) for name, fields in VARIANTS.items()},
        "model_pair": ["HFLGB", "XGBoost"], "ensemble_weight": ROLLING_ENSEMBLE_WEIGHT,
        "validation_gate": ["IC", "RankIC", "absolute_return", "absolute_drawdown",
                            "excess_return", "excess_drawdown"],
    }, "variants": {}}
    baseline_valid_cache = {}
    for variant in selected_variants:
        baseline_tests, fixed_tests, adaptive_tests, labels, fold_rows = [], [], [], [], []
        for fold_no, row in enumerate(rows, 1):
            segments = row["segments"]
            trained = _train_or_load(variant, fold_no, frame, segments)
            baseline_test, baseline_label = _test_score(row)
            from scripts.validate_risk_overlay_purged import _first_validation
            if fold_no not in baseline_valid_cache:
                baseline_valid_cache[fold_no] = _first_validation(row)
            baseline_valid, baseline_valid_label = baseline_valid_cache[fold_no]
            hf_valid = trained["hf_valid"]
            xgb_valid = trained["xgb_valid"]
            valid_label = trained["valid_label"]
            common_valid = hf_valid.index.intersection(valid_label.index)
            hf_valid, valid_label = hf_valid.loc[common_valid], valid_label.loc[common_valid]
            xgb_valid = xgb_valid.reindex(common_valid)
            candidate_hf_metrics = _metrics(
                hf_valid, valid_label, *segments["valid"]
            )
            candidate_blend = blend_scores(
                hf_valid, xgb_valid, ROLLING_ENSEMBLE_WEIGHT
            )
            candidate_blend_label = valid_label.reindex(candidate_blend.index)
            candidate_blend_metrics = _metrics(
                candidate_blend, candidate_blend_label, *segments["valid"]
            )
            candidate_weight = (
                ROLLING_ENSEMBLE_WEIGHT
                if full_gate(candidate_hf_metrics, candidate_blend_metrics) else 0.0
            )
            candidate_valid = blend_scores(hf_valid, xgb_valid, candidate_weight)
            candidate_valid_metrics = (
                candidate_blend_metrics if candidate_weight else candidate_hf_metrics
            )
            baseline_valid_metrics = _metrics(
                baseline_valid, baseline_valid_label, *segments["valid"]
            )
            feature_gate = full_gate(baseline_valid_metrics, candidate_valid_metrics)
            candidate_test = blend_scores(
                trained["hf_test"], trained["xgb_test"], candidate_weight
            )
            candidate_test_label = trained["test_label"].reindex(candidate_test.index)
            baseline_tests.append(baseline_test)
            fixed_tests.append(candidate_test)
            adaptive_tests.append(candidate_test if feature_gate else baseline_test)
            labels.append(baseline_label)
            fold_rows.append({
                "fold": fold_no, "segments": segments,
                "feature_count": trained["feature_count"],
                "candidate_ensemble_weight": candidate_weight,
                "feature_gate_passed": feature_gate,
                "validation_baseline": baseline_valid_metrics,
                "validation_candidate": candidate_valid_metrics,
            })
            output["variants"][variant] = {"folds": fold_rows}
            result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
            print(f"{variant} {fold_no:02d}/{len(rows)} "
                  f"xgb={candidate_weight:.2f} feature_gate={feature_gate}", flush=True)
        baseline = pd.concat(baseline_tests).sort_index()
        fixed = pd.concat(fixed_tests).sort_index()
        adaptive = pd.concat(adaptive_tests).sort_index()
        label = pd.concat(labels).sort_index().reindex(baseline.index)
        test_start = str(baseline.index.get_level_values("datetime").min().date())
        test_end = str(baseline.index.get_level_values("datetime").max().date())
        summary = {
            "baseline": _metrics(baseline, label, test_start, test_end),
            "fixed_joint": _metrics(fixed, label.reindex(fixed.index), test_start, test_end),
            "validation_gated_joint": _metrics(
                adaptive, label.reindex(adaptive.index), test_start, test_end
            ),
            "feature_gate_pass_count": int(sum(x["feature_gate_passed"] for x in fold_rows)),
        }
        output["variants"][variant] = {"folds": fold_rows, "summary": summary}
        result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
        print(variant, json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
