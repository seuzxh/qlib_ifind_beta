"""每日滚动重训测试（item 3）。

覆盖：
  - task_template 结构 = champion FROZEN（18 因子 + label + LGBModel 超参）
  - RollingGen(step=1, ROLL_SD) 滚动 segments 正确性（train/valid/test 同步前移）
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

def test_task_template_champion_frozen():
    """task_template 的 handler/model/label 与 champion workflow 逐字一致。"""
    from scripts.retrain import _build_task_template
    from qlib_ifind_beta.config import (
        CHAMPION_DATA_START, CHAMPION_FIT_END, CHAMPION_FIT_START,
        CHAMPION_LABEL_EXPR, UNIVERSE_MARKET,
    )

    t = _build_task_template()

    # model = LGBModel champion 超参
    model = t["model"]
    assert model["class"] == "LGBModel"
    assert model["kwargs"]["learning_rate"] == 0.05
    assert model["kwargs"]["lambda_l1"] == 5.0
    assert model["kwargs"]["lambda_l2"] == 10.0
    assert model["kwargs"]["num_boost_round"] == 200

    # handler = MinuteEnhancedHandler(18) champion 口径
    hk = t["dataset"]["kwargs"]["handler"]["kwargs"]
    assert hk["instruments"] == UNIVERSE_MARKET
    assert hk["start_time"] == CHAMPION_DATA_START
    assert hk["fit_start_time"] == CHAMPION_FIT_START
    assert hk["fit_end_time"] == CHAMPION_FIT_END
    assert hk["label"] == [CHAMPION_LABEL_EXPR]

    # segments = champion 初始锚
    segs = t["dataset"]["kwargs"]["segments"]
    assert segs["train"] == [CHAMPION_FIT_START, CHAMPION_FIT_END]
    assert segs["valid"] == ["2026-01-01", "2026-03-31"]
    assert segs["test"] == ["2026-04-01", "2026-04-01"]

    # record = SignalRecord + SigAnaRecord（无 PortAnaRecord，滚动重训不做回测）
    rec_classes = [r["class"] for r in t["record"]]
    assert "SignalRecord" in rec_classes
    assert "SigAnaRecord" in rec_classes
    assert "PortAnaRecord" not in rec_classes


# ---------------------------------------------------------------------------
# B4-2: config 常量与 RollingGen 值对齐
# ---------------------------------------------------------------------------

def test_rolling_config_aligns_qlib():
    """config ROLLING_RTYPE/STEP 与 qlib RollingGen 常量对齐。"""
    from qlib.workflow.task.gen import RollingGen
    from qlib_ifind_beta.config import ROLLING_RTYPE, ROLLING_STEP

    assert ROLLING_RTYPE == RollingGen.ROLL_SD   # "sliding"
    assert ROLLING_STEP == 1                      # 每日滚动


# ---------------------------------------------------------------------------
# B4-3: RollingGen 滚动 segments 正确性（无需 qlib.init，纯日期计算）
# ---------------------------------------------------------------------------

def test_rolling_gen_step1_slide(qlib_init):
    """RollingGen(step=1, ROLL_SD) 从 champion 初始锚生成正确的滚动 segments。

    step=1 滑动窗口：train/valid/test 同步前移 1 个交易日。
    """
    from qlib.workflow.task.gen import RollingGen
    from scripts.retrain import _build_task_template

    rg = RollingGen(step=1, rtype=RollingGen.ROLL_SD)
    template = _build_task_template()

    # generate 产出初始任务列表（test 段 = step=1 天的切片）
    tasks = rg.generate(template)
    assert len(tasks) >= 1

    # 首个任务 test 段 = 1 天（step=1）
    first_segs = tasks[0]["dataset"]["kwargs"]["segments"]
    test_start = first_segs["test"][0]
    test_end = first_segs["test"][1]
    assert test_start == test_end, f"step=1 → test 段应 1 天，got {test_start}→{test_end}"

    # gen_following_tasks：从首个任务往后生成后续任务
    import pandas as pd
    followings = list(rg.gen_following_tasks(tasks[0], pd.Timestamp("2026-04-05")))
    assert len(followings) > 0
    # 后续任务的 test 段 start 应晚于首个任务
    next_test_start = followings[0]["dataset"]["kwargs"]["segments"]["test"][0]
    assert pd.Timestamp(next_test_start) > pd.Timestamp(test_start), (
        f"滚动后 test_start {next_test_start} 应 > 初始 {test_start}")


# ---------------------------------------------------------------------------
# B4-4: _coerce_limit_threshold 防御（无 PortAnaRecord 时 no-op）
# ---------------------------------------------------------------------------

def test_coerce_limit_threshold_noop_without_portana():
    """无 PortAnaRecord 时 _coerce_limit_threshold 是 no-op。"""
    from scripts.retrain import _coerce_limit_threshold

    task = {
        "record": [
            {"class": "SignalRecord", "module_path": "qlib.workflow.record_temp"},
        ]
    }
    _coerce_limit_threshold(task)   # 不应抛异常
    assert task["record"][0]["class"] == "SignalRecord"
