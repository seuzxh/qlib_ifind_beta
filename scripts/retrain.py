"""P1 实战对接 — 模型每日滚动重训入口（qlib 原生 RollingGen）。

用 qlib workflow.task.gen.RollingGen（step=1, ROLL_SD 滑动窗口）+ task_train +
OnlineToolR 实现每日重训。每个交易日生成一个新任务（train/valid/test 同步前移 1
天），训练后把新 recorder 标记为 online（旧模型自动 offline）。inference
use_online=True 时从最新 online recorder 加载模型。

滚动窗口设计（ROLL_SD）：
  - train 窗口固定 2 年（~488 交易日），valid 固定 3 个月（~60 交易日）
  - step=1：每日生成新任务，test 段 = 未来 1 天
  - 示例：D0 train[2024-01-01,2025-12-31]/valid[2026-01-01,2026-03-31]/test[2026-04-01]
           D1 train[2024-01-02,2026-01-01]/valid[2026-01-02,2026-04-01]/test[2026-04-02]

Run:
  # 每日盘后触发（复用 P1 物化后的 day.bin）
  conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
      scripts/retrain.py
  # 指定 test 截止日（默认 = day.txt 最后一个交易日）
  conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
      scripts/retrain.py --test-end 2026-07-02
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 必须在 import qlib / mlflow 前设置（同 qrun/run.py）
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import pandas as pd

from qlib_ifind_beta.config import (
    CHAMPION_DATA_START, CHAMPION_FIT_END, CHAMPION_FIT_START,
    CHAMPION_LABEL_EXPR, OVERLAY_ROOT, ROLLING_EXPERIMENT,
    ROLLING_RTYPE, ROLLING_STEP, UNIVERSE_MARKET,
)


def _build_task_template() -> dict:
    """构建 champion FROZEN task_template（handler/model/record 全冻结）。

    RollingGen 会基于此 template 的 segments 生成滚动任务。
    """
    return {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "learning_rate": 0.05,
                "max_depth": 6,
                "num_leaves": 64,
                "num_threads": 20,
                "lambda_l1": 5.0,
                "lambda_l2": 10.0,
                "num_boost_round": 200,
                "early_stopping_rounds": 20,
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
                        "start_time": CHAMPION_DATA_START,
                        "end_time": CHAMPION_FIT_END,
                        "fit_start_time": CHAMPION_FIT_START,
                        "fit_end_time": CHAMPION_FIT_END,
                        "label": [CHAMPION_LABEL_EXPR],
                    },
                },
                "segments": {
                    "train": [CHAMPION_FIT_START, CHAMPION_FIT_END],
                    "valid": ["2026-01-01", "2026-03-31"],
                    "test": ["2026-04-01", "2026-04-01"],
                },
            },
        },
        "record": [
            {"class": "SignalRecord", "module_path": "qlib.workflow.record_temp"},
            {"class": "SigAnaRecord", "module_path": "qlib.workflow.record_temp"},
        ],
    }


def _coerce_limit_threshold(task: dict) -> None:
    """PortAnaRecord 的 limit_threshold(list) → tuple（同 qrun/run.py 本机坑绕过）。

    滚动重训不带 PortAnaRecord（无需回测），但保留此防御：若未来加 PortAnaRecord，
    list→tuple 修正自动生效。
    """
    for rec in task.get("record", []):
        if rec.get("class") != "PortAnaRecord":
            continue
        exk = (((rec.get("kwargs", {}).get("config") or {})
                .get("backtest") or {}).get("exchange_kwargs") or {})
        lt = exk.get("limit_threshold")
        if isinstance(lt, list):
            exk["limit_threshold"] = tuple(lt)


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


def run(test_end: str | None = None) -> dict:
    """执行每日滚动重训。

    Args:
        test_end: test 段截止日（默认 = day.txt 最后一个交易日）。

    Returns:
        dict: {n_new_tasks, new_recorder_ids, experiment}
    """
    import qlib
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")

    from qlib.workflow.task.gen import RollingGen
    from qlib.model.trainer import task_train
    from qlib.workflow.online.utils import OnlineToolR

    tool = OnlineToolR(ROLLING_EXPERIMENT)

    # 1. 构建 RollingGen（step=1, ROLL_SD 滑动窗口）
    rg = RollingGen(step=ROLLING_STEP, rtype=ROLLING_RTYPE)

    # 2. 生成滚动任务
    template = _build_task_template()
    latest_records, max_test = _list_latest_online(tool, ROLLING_EXPERIMENT)

    if not latest_records:
        # 首次重训：用 template 生成初始任务
        tasks = rg.generate(template)
        print(f"▶ [retrain] 首次重训：RollingGen 生成 {len(tasks)} 个初始任务")
    else:
        # 后续重训：基于最新 online recorder 生成后续任务
        tasks = []
        if test_end is None:
            from qlib.data import D
            test_end = pd.Timestamp(D.calendar(freq="day")[-1])
        else:
            test_end = pd.Timestamp(test_end)
        for rec in latest_records:
            task = rec.load_object("task")
            tasks.extend(rg.gen_following_tasks(task, test_end))
        if not tasks:
            print(f"▶ [retrain] 无新任务（max_test={max_test}, test_end={test_end}, "
                  f"step={ROLLING_STEP}，间隔不足）")
            return {"n_new_tasks": 0, "new_recorder_ids": [], "experiment": ROLLING_EXPERIMENT}
        print(f"▶ [retrain] 后续重训：从 max_test={max_test} 生成 {len(tasks)} 个新任务 "
              f"(test_end={test_end})")

    # 3. 训练每个任务
    new_ids = []
    for i, task in enumerate(tasks):
        _coerce_limit_threshold(task)
        segs = task["dataset"]["kwargs"]["segments"]
        print(f"  [{i+1}/{len(tasks)}] train={segs['train']} valid={segs['valid']} test={segs['test']}")
        recorder = task_train(task, experiment_name=ROLLING_EXPERIMENT)
        rid = getattr(recorder, "recorder_id", None) or getattr(recorder, "id", None)
        new_ids.append(rid)
        # 标记最新 recorder 为 online（旧模型自动 offline）
        tool.reset_online_tag(recorder, exp_name=ROLLING_EXPERIMENT)
        print(f"         → recorder_id={rid} (online)")

    print(f"✅ retrain 完成：{len(new_ids)} 个新模型训练并标记 online")
    return {"n_new_tasks": len(new_ids), "new_recorder_ids": new_ids,
            "experiment": ROLLING_EXPERIMENT}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="每日滚动重训（qlib RollingGen）")
    p.add_argument("--test-end", default=None,
                   help="test 段截止日 (YYYY-MM-DD)，默认 = day.txt 最后一个交易日")
    a = p.parse_args()
    result = run(a.test_end)
