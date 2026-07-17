"""滚动重训：HFLGB 主模型 + 验证期门控 XGBoost 候选。

用 `task_train` + `OnlineToolR` 发布覆盖未来最多 20 个交易日的冻结模型段。每日入口
检查目标日是否已被
online artifact 覆盖，仅在新段首日训练。每个任务训练两个同特征模型；仅前一验证段的
IC、RankIC、精确 TD0 年化超额
都不低于 HFLGB 时，下一测试段冻结 25% XGBoost 权重，否则权重为 0。门控 artifact
写入完成后才将 HFLGB recorder 标记 online，推理失败时安全退回 HFLGB。

滚动窗口设计（与 §60 严格一致）：
  - train 窗口固定 90 交易日，valid 固定 20 交易日，测试前 embargo 1 日
  - step=20：test 段冻结 20 个交易日；每日入口不会在段内换模型
  - 示例：D0 train[2025-10-16,2026-03-02]/valid[2026-03-03,2026-03-30]/
           embargo[2026-03-31]/test[2026-04-01]

Run:
  # 目标交易日盘前触发（复用 P1 物化后的 day.bin）
  conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
      scripts/retrain.py
  # 指定 test 起始日（默认 = 上海时区今天，且必须已在交易日历中）
  conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
      scripts/retrain.py --test-start 2026-07-02
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 必须在 import qlib / mlflow 前设置（同 qrun/run.py）
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import copy

import pandas as pd

from qlib_ifind_beta.config import (
    CHAMPION_LABEL_EXPR, OVERLAY_ROOT, ROLLING_EXPERIMENT,
    ROLLING_EMBARGO_DAYS, ROLLING_GATE_ARTIFACT, ROLLING_STEP,
    ROLLING_ENSEMBLE_WEIGHT,
    ROLLING_TEST_DAYS, ROLLING_TRAIN_DAYS, ROLLING_VALID_DAYS,
    ROLLING_XGB_EXPERIMENT,
    UNIVERSE_MARKET,
)
from qlib_ifind_beta.model_ensemble import (
    blend_scores, correlation_metrics, make_gate_metadata, predict_feature_matrix,
)


def _build_task_template() -> dict:
    """Build the frozen 90/20/embargo1/test20 task template."""
    return {
        "model": {
            "class": "HFLGBModel",
            "module_path": "qlib.contrib.model.highfreq_gdbt_model",
            "kwargs": {
                "loss": "binary",
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 64,
                "num_threads": 20,
                "lambda_l1": 5.0,
                "lambda_l2": 10.0,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "MinuteEnhancedHandler",
                    "module_path": "qlib_ifind_beta.minute_enhanced_handler",
                    "kwargs": {
                        "instruments": UNIVERSE_MARKET,
                        "start_time": "2025-10-16",
                        "end_time": "2026-04-29",
                        "fit_start_time": "2025-10-16",
                        "fit_end_time": "2026-03-02",
                        "label": [CHAMPION_LABEL_EXPR],
                    },
                },
                "segments": {
                    "train": ["2025-10-16", "2026-03-02"],
                    "valid": ["2026-03-03", "2026-03-30"],
                    "test": ["2026-04-01", "2026-04-29"],
                },
            },
        },
        # Future test features do not exist at retrain time. Online inference
        # builds its own T-day DatasetH/matrix, so training only persists model/task.
        "record": [],
    }


def _xgb_task(hflgb_task: dict) -> dict:
    """Clone a generated rolling task and replace only its model definition."""
    task = copy.deepcopy(hflgb_task)
    task["model"] = {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
        "kwargs": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": 0.05,
            "max_depth": 6,
            "alpha": 5.0,
            "lambda": 10.0,
            "nthread": 20,
        },
    }
    return task


def _task_for_test_date(test_date: str | pd.Timestamp,
                        calendar: list[pd.Timestamp]) -> dict:
    """Build an exact 90d/20d/1d-embargo/<=20d-test task."""
    target = pd.Timestamp(test_date)
    try:
        pos = calendar.index(target)
    except ValueError as exc:
        raise ValueError(f"test date {target.date()} is not in the Qlib calendar") from exc
    required = ROLLING_TRAIN_DAYS + ROLLING_VALID_DAYS + ROLLING_EMBARGO_DAYS
    if pos < required:
        raise ValueError(f"calendar has fewer than {required} prior sessions for {target.date()}")
    valid_end = pos - ROLLING_EMBARGO_DAYS
    valid_start = valid_end - ROLLING_VALID_DAYS
    train_start = valid_start - ROLLING_TRAIN_DAYS
    train = calendar[train_start:valid_start]
    valid = calendar[valid_start:valid_end]
    task = _build_task_template()
    fmt = lambda value: pd.Timestamp(value).strftime("%Y-%m-%d")
    segments = task["dataset"]["kwargs"]["segments"]
    segments["train"] = [fmt(train[0]), fmt(train[-1])]
    segments["valid"] = [fmt(valid[0]), fmt(valid[-1])]
    test_end_pos = min(pos + ROLLING_TEST_DAYS - 1, len(calendar) - 1)
    segments["test"] = [fmt(target), fmt(calendar[test_end_pos])]
    handler = task["dataset"]["kwargs"]["handler"]["kwargs"]
    handler["start_time"] = fmt(train[0])
    handler["end_time"] = fmt(calendar[test_end_pos])
    handler["fit_start_time"] = fmt(train[0])
    handler["fit_end_time"] = fmt(train[-1])
    return task


def _has_validated_window(task: dict, calendar: list[pd.Timestamp]) -> bool:
    """Detect legacy 2y/60d online recorders and prevent carrying them forward."""
    try:
        segments = task["dataset"]["kwargs"]["segments"]
        positions = {date.strftime("%Y-%m-%d"): idx for idx, date in enumerate(calendar)}
        index_of = lambda value: positions[pd.Timestamp(value).strftime("%Y-%m-%d")]
        train = index_of(segments["train"][1]) - index_of(segments["train"][0]) + 1
        valid = index_of(segments["valid"][1]) - index_of(segments["valid"][0]) + 1
        test = index_of(segments["test"][1]) - index_of(segments["test"][0]) + 1
        train_to_valid = index_of(segments["valid"][0]) - index_of(segments["train"][1])
        valid_to_test = index_of(segments["test"][0]) - index_of(segments["valid"][1])
        return (train == ROLLING_TRAIN_DAYS
                and valid == ROLLING_VALID_DAYS
                and 1 <= test <= ROLLING_TEST_DAYS
                and train_to_valid == 1
                and valid_to_test == ROLLING_EMBARGO_DAYS + 1)
    except (KeyError, TypeError, ValueError):
        return False


def _dates_to_train(calendar: list[pd.Timestamp], target: pd.Timestamp,
                    latest_task: dict | None, max_test: pd.Timestamp | None
                    ) -> tuple[list[pd.Timestamp], str]:
    """Select incremental dates, or one direct migration/bootstrap target."""
    compatible = latest_task is not None and _has_validated_window(latest_task, calendar)
    if compatible and max_test >= target:
        return [], "最新"
    if compatible:
        return [target], "新冻结段"
    return [target], "首次" if latest_task is None else "旧窗口迁移"


def _recorder_id(recorder) -> str:
    return getattr(recorder, "recorder_id", None) or getattr(recorder, "id", None)


def _validation_predictions(hflgb_recorder, xgb_recorder, task: dict):
    """Predict the fully-known validation segment with both trained models."""
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config

    dataset = init_instance_by_config(task["dataset"])
    frame = dataset.prepare(
        "valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I
    ).dropna()
    features = frame["feature"]
    label = frame["label"].iloc[:, 0]
    hflgb = predict_feature_matrix(hflgb_recorder.load_object("params.pkl"), features)
    xgb = predict_feature_matrix(xgb_recorder.load_object("params.pkl"), features)
    return hflgb, xgb, label


def _gate_metrics(score: pd.Series, label: pd.Series,
                  start: str, end: str) -> dict[str, float]:
    """Calculate the same three promotion metrics used in §60 research."""
    from scripts.validate_factor_challengers import _backtest

    result = correlation_metrics(score, label)
    result.update(_backtest(score.to_frame("score"), start, end))
    return result


def _train_pair(task: dict, task_train):
    """Train a model pair and persist fail-closed gate metadata."""
    hflgb_recorder = task_train(task, experiment_name=ROLLING_EXPERIMENT)
    empty_metrics = {
        "IC": float("nan"), "RankIC": float("nan"),
        "excess_annualized_return": float("nan"),
    }
    baseline, candidate = dict(empty_metrics), dict(empty_metrics)
    xgb_id = ""
    try:
        xgb_recorder = task_train(_xgb_task(task), experiment_name=ROLLING_XGB_EXPERIMENT)
        xgb_id = _recorder_id(xgb_recorder)
        hflgb_valid, xgb_valid, label = _validation_predictions(
            hflgb_recorder, xgb_recorder, task
        )
        candidate_score = blend_scores(hflgb_valid, xgb_valid, ROLLING_ENSEMBLE_WEIGHT)
        valid_start, valid_end = task["dataset"]["kwargs"]["segments"]["valid"]
        baseline = _gate_metrics(hflgb_valid, label, valid_start, valid_end)
        candidate = _gate_metrics(candidate_score, label, valid_start, valid_end)
    except Exception as exc:
        # HFLGB is still valid. Persist the failure so online inference never
        # mistakes an incomplete pair for an enabled ensemble.
        candidate["error"] = f"{type(exc).__name__}: {exc}"

    segments = task["dataset"]["kwargs"]["segments"]
    metadata = make_gate_metadata(
        baseline, candidate, xgb_id, segments["test"], segments["valid"]
    )
    hflgb_recorder.save_objects(**{ROLLING_GATE_ARTIFACT: metadata})
    return hflgb_recorder, metadata


def _list_latest_online(tool, exp_name: str):
    """返回 (latest_recorders, max_test_end) —— 最新 online recorder 及其 test 末日。"""
    online = tool.online_models(exp_name=exp_name)
    if not online:
        return [], None
    max_test = max(
        rec.load_object("task")["dataset"]["kwargs"]["segments"]["test"][1]
        for rec in online
    )
    latest = [r for r in online
              if r.load_object("task")["dataset"]["kwargs"]["segments"]["test"][1] == max_test]
    return latest, pd.Timestamp(max_test)


def run(test_start: str | None = None) -> dict:
    """Publish a new frozen block only when the target is not covered.

    Args:
        test_start: test 段起始日。默认取上海时区今天；日历未包含今天时拒绝发布。

    Returns:
        dict: {n_new_tasks, new_recorder_ids, experiment}
    """
    import qlib
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")

    from qlib.model.trainer import task_train
    from qlib.workflow.online.utils import OnlineToolR
    from qlib.data import D

    tool = OnlineToolR(ROLLING_EXPERIMENT)

    # 1. Read the authoritative trading calendar and generate exact-count
    # windows.  Date arithmetic cannot guarantee 90/20 trading sessions.
    calendar = [pd.Timestamp(date) for date in D.calendar(freq="day")]
    target = (pd.Timestamp(test_start) if test_start is not None else
              pd.Timestamp.now(tz="Asia/Shanghai").normalize().tz_localize(None))
    if target not in calendar:
        raise ValueError(
            f"test_start={target.date()} 不在 Qlib 交易日历；请先刷新日历，"
            "或用 --test-start 显式执行历史验证（不会覆盖今天）"
        )

    # 2. Generate tasks. Legacy 2y/60d online models are not extended; migrate
    # directly to a fresh 90d/20d task for the requested target date.
    latest_records, max_test = _list_latest_online(tool, ROLLING_EXPERIMENT)
    latest_task = latest_records[0].load_object("task") if latest_records else None
    test_dates, mode = _dates_to_train(calendar, target, latest_task, max_test)
    if not test_dates:
        print(f"▶ [retrain] 无新任务（max_test={max_test.date()}, test_start={target.date()}）")
        return {"n_new_tasks": 0, "new_recorder_ids": [], "experiment": ROLLING_EXPERIMENT}
    tasks = [_task_for_test_date(date, calendar) for date in test_dates]
    print(f"▶ [retrain] {mode}：生成 {len(tasks)} 个 90d/20d/embargo1/test20 模型对，"
          f"test_start={target.date()}")

    # 3. 训练模型对；gate artifact 落盘后才发布 recorder
    new_ids = []
    for i, task in enumerate(tasks):
        segs = task["dataset"]["kwargs"]["segments"]
        print(f"  [{i+1}/{len(tasks)}] train={segs['train']} valid={segs['valid']} test={segs['test']}")
        recorder, gate = _train_pair(task, task_train)
        rid = _recorder_id(recorder)
        new_ids.append(rid)
        # 发布点：在线侧不会看到尚未写完 gate artifact 的 recorder。
        tool.reset_online_tag(recorder, exp_name=ROLLING_EXPERIMENT)
        print(f"         → recorder_id={rid} (online), gate_weight={gate['weight']:.2f}, "
              f"xgb_recorder_id={gate['xgb_recorder_id'] or 'FAILED'}")

    print(f"✅ retrain 完成：{len(new_ids)} 个模型对完成并发布 online")
    return {"n_new_tasks": len(new_ids), "new_recorder_ids": new_ids,
            "experiment": ROLLING_EXPERIMENT}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="无泄漏 HFLGB/XGBoost 滚动门控重训")
    p.add_argument("--test-start", default=None,
                   help="test 段起始日 (YYYY-MM-DD)，默认 = 上海时区今天")
    a = p.parse_args()
    result = run(a.test_start)
