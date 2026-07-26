"""Purged 20-session rolling ensemble retrain tests.

覆盖：
  - task_template 结构 = champion（18 因子 + label + HFLGBModel 超参）
  - RollingGen(step=20, ROLL_SD) compatibility
  - config 常量与 RollingGen.ROLL_SD 值对齐
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture(scope="module")
def qlib_init():
    import qlib
    qlib.init(provider_uri="data/qlib_root", region="cn")
    yield


# ---------------------------------------------------------------------------
# B4-1: task_template 结构 = champion FROZEN
# ---------------------------------------------------------------------------

def test_task_template_uses_validated_rolling_window():
    """task_template freezes the §60 90d/20d/1d model semantics."""
    from scripts.retrain import _build_task_template
    from qlib_ifind_beta.config import (
        CHAMPION_LABEL_EXPR, UNIVERSE_MARKET,
    )

    t = _build_task_template()

    # model = HFLGBModel champion 超参
    model = t["model"]
    assert model["class"] == "HFLGBModel"
    assert model["kwargs"]["loss"] == "binary"
    assert model["kwargs"]["learning_rate"] == 0.05
    assert model["kwargs"]["lambda_l1"] == 5.0
    assert model["kwargs"]["lambda_l2"] == 10.0

    # handler = MinuteEnhancedHandler(18) champion 口径
    hk = t["dataset"]["kwargs"]["handler"]["kwargs"]
    assert hk["instruments"] == UNIVERSE_MARKET
    assert hk["start_time"] == "2025-10-16"
    assert hk["fit_start_time"] == "2025-10-16"
    assert hk["fit_end_time"] == "2026-03-02"
    assert hk["label"] == [CHAMPION_LABEL_EXPR]

    # 90 train + 20 valid + 1 test sessions at the initial anchor.
    segs = t["dataset"]["kwargs"]["segments"]
    assert segs["train"] == ["2025-10-16", "2026-03-02"]
    assert segs["valid"] == ["2026-03-03", "2026-03-30"]
    assert segs["test"] == ["2026-04-01", "2026-04-29"]

    # Future test features are unavailable at training time; inference is separate.
    assert t["record"] == []


def test_xgb_task_changes_only_model_definition():
    """XGBoost pair must retain the exact handler, segments, label and records."""
    from scripts.retrain import _build_task_template, _xgb_task

    baseline = _build_task_template()
    candidate = _xgb_task(baseline)
    assert candidate["model"]["class"] == "XGBModel"
    assert candidate["model"]["kwargs"]["objective"] == "reg:squarederror"
    assert candidate["dataset"] == baseline["dataset"]
    assert candidate["record"] == baseline["record"]
    assert baseline["model"]["class"] == "HFLGBModel"  # clone did not mutate source


class _FakeRecorder:
    def __init__(self, recorder_id):
        self.recorder_id = recorder_id
        self.saved = {}

    def save_objects(self, **kwargs):
        self.saved.update(kwargs)


def test_train_pair_persists_enabled_gate(monkeypatch):
    import pandas as pd
    from qlib_ifind_beta.config import ROLLING_GATE_ARTIFACT
    from scripts import retrain

    hf, xgb = _FakeRecorder("hf-id"), _FakeRecorder("xgb-id")
    recorders = iter([hf, xgb])
    monkeypatch.setattr(retrain, "_validation_predictions", lambda *args: (
        pd.Series([1.0, 2.0], index=["A", "B"]),
        pd.Series([2.0, 1.0], index=["A", "B"]),
        pd.Series([.1, .2], index=["A", "B"]),
    ))
    metric_values = iter([
        {"IC": .10, "RankIC": .20, "excess_annualized_return": .30},
        {"IC": .11, "RankIC": .21, "excess_annualized_return": .31},
    ])
    monkeypatch.setattr(retrain, "_gate_metrics", lambda *args: next(metric_values))
    task = {"dataset": {"kwargs": {"segments": {
        "valid": ["2026-06-01", "2026-07-01"],
        "test": ["2026-07-02", "2026-07-02"],
    }}}}
    recorder, gate = retrain._train_pair(task, lambda *args, **kwargs: next(recorders))
    assert recorder is hf
    assert gate["weight"] == .25
    assert gate["xgb_recorder_id"] == "xgb-id"
    assert hf.saved[ROLLING_GATE_ARTIFACT] == gate


def test_train_pair_xgb_failure_publishes_disabled_gate():
    from qlib_ifind_beta.config import ROLLING_GATE_ARTIFACT
    from scripts import retrain

    hf = _FakeRecorder("hf-id")
    calls = 0

    def task_train(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return hf
        raise RuntimeError("xgb unavailable")

    task = {"dataset": {"kwargs": {"segments": {
        "valid": ["2026-06-01", "2026-07-01"],
        "test": ["2026-07-02", "2026-07-02"],
    }}}}
    _, gate = retrain._train_pair(task, task_train)
    assert gate["weight"] == 0
    assert gate["gate_passed"] is False
    assert "xgb unavailable" in gate["validation_candidate"]["error"]
    assert hf.saved[ROLLING_GATE_ARTIFACT] == gate


# ---------------------------------------------------------------------------
# B4-2: config 常量与 RollingGen 值对齐
# ---------------------------------------------------------------------------

def test_rolling_config_aligns_qlib():
    """config ROLLING_RTYPE/STEP 与 qlib RollingGen 常量对齐。"""
    from qlib.workflow.task.gen import RollingGen
    from qlib_ifind_beta.config import ROLLING_RTYPE, ROLLING_STEP

    assert ROLLING_RTYPE == RollingGen.ROLL_SD   # "sliding"
    assert ROLLING_STEP == 20                     # frozen 20-session test block


# ---------------------------------------------------------------------------
# B4-3: RollingGen 滚动 segments 正确性（无需 qlib.init，纯日期计算）
# ---------------------------------------------------------------------------

def test_rolling_gen_step20_slide(qlib_init):
    """RollingGen(step=20, ROLL_SD) preserves the frozen test block.

    This remains a compatibility check; production builds exact-count windows directly.
    """
    from qlib.workflow.task.gen import RollingGen
    from scripts.retrain import _build_task_template
    import pandas as pd

    rg = RollingGen(step=20, rtype=RollingGen.ROLL_SD)
    template = _build_task_template()

    # generate 产出初始任务列表
    tasks = rg.generate(template)
    assert len(tasks) >= 1

    # First task retains a multi-session frozen test block.
    first_segs = tasks[0]["dataset"]["kwargs"]["segments"]
    test_start = first_segs["test"][0]
    test_end = first_segs["test"][1]
    assert pd.Timestamp(test_end) > pd.Timestamp(test_start)

    # gen_following_tasks：从首个任务往后生成后续任务
    followings = list(rg.gen_following_tasks(tasks[0], pd.Timestamp("2026-06-30")))
    assert len(followings) > 0
    # 后续任务的 test 段 start 应晚于首个任务
    next_test_start = followings[0]["dataset"]["kwargs"]["segments"]["test"][0]
    assert pd.Timestamp(next_test_start) > pd.Timestamp(test_start), (
        f"滚动后 test_start {next_test_start} 应 > 初始 {test_start}")


def test_task_for_test_date_has_exact_trading_session_counts(qlib_init):
    import pandas as pd
    from qlib.data import D
    from qlib_ifind_beta.config import ROLLING_TRAIN_DAYS, ROLLING_VALID_DAYS
    from scripts.retrain import _has_validated_window, _task_for_test_date

    calendar = [pd.Timestamp(date) for date in D.calendar(freq="day")]
    task = _task_for_test_date("2026-07-02", calendar)
    segments = task["dataset"]["kwargs"]["segments"]
    assert segments["train"] == ["2026-01-13", "2026-06-01"]
    assert segments["valid"] == ["2026-06-02", "2026-06-30"]
    # The source calendar is updated over time; assert the frozen block size
    # rather than yesterday's final available date.
    start = calendar.index(pd.Timestamp(segments["test"][0]))
    end = calendar.index(pd.Timestamp(segments["test"][1]))
    assert segments["test"][0] == "2026-07-02"
    assert end - start + 1 == min(20, len(calendar) - start)
    assert _has_validated_window(task, calendar)
    assert ROLLING_TRAIN_DAYS == 90
    assert ROLLING_VALID_DAYS == 20


def test_dates_to_train_migrates_legacy_once_and_then_increments(qlib_init):
    import pandas as pd
    from qlib.data import D
    from scripts.retrain import _build_task_template, _dates_to_train, _task_for_test_date

    calendar = [pd.Timestamp(date) for date in D.calendar(freq="day")]
    target = pd.Timestamp("2026-07-02")
    legacy = _build_task_template()
    legacy["dataset"]["kwargs"]["segments"]["train"] = ["2024-01-01", "2026-03-03"]
    dates, mode = _dates_to_train(calendar, target, legacy, pd.Timestamp("2026-04-01"))
    assert dates == [target]
    assert mode == "旧窗口迁移"

    latest = _task_for_test_date("2026-06-03", calendar)
    dates, mode = _dates_to_train(
        calendar, target, latest, pd.Timestamp("2026-07-01")
    )
    assert [date.strftime("%Y-%m-%d") for date in dates] == ["2026-07-02"]
    assert mode == "新冻结段"

    dates, mode = _dates_to_train(calendar, target,
                                  _task_for_test_date(target, calendar), target)
    assert dates == []
    assert mode == "最新"
