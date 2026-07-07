"""MinuteEnhancedTailHandler — champion(18) + 2 个最强正交 T-1 尾盘因子（goal 因子优化 Step C2）。

验证 wiring（结构校验，无需 qlib.init / 无数据抓取，经 __new__）：
  - 20 因子 get_feature_config = champion 18（ENHANCED_FIELDS）+ tail_vol_ratio_t1 + tail_accel_t1
  - TAIL_PICK 2 个因子确为 MINUTE_FACTOR_TAIL_FIELDS 子集（物化 bin 存在 + 已 shift-1 测试覆盖）
  - L1 shared DropnaProcessor(feature) 从 HighBetaAlpha158 继承（前视护栏不变）

尾盘因子的数值正确性（5 因子公式 + shift-1 + 散列到 day-cal）由
tests/test_materialize_minute.py 的 test_tail_factors_crosscheck / test_tail_window_kbar_count /
test_materialize_writes_25_bins 覆盖（16/16 PASS）；本测仅验 handler 选择与继承。
"""
from qlib_ifind_beta.config import (
    MINUTE_FACTOR_EXTRA_FIELDS,
    MINUTE_FACTOR_FIELDS,
    MINUTE_FACTOR_TAIL_FIELDS,
)
from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler
from qlib_ifind_beta.minute_enhanced_tail_handler import MinuteEnhancedTailHandler


def test_feature_config_20():
    """20 因子 = champion 18（ENHANCED_FIELDS）+ 2 tail。fields = $name，与 14 分钟因子零 Ref 模式统一。"""
    h = MinuteEnhancedTailHandler.__new__(MinuteEnhancedTailHandler)
    fields, names = h.get_feature_config()
    expected_champion = list(MINUTE_FACTOR_FIELDS) + list(MINUTE_FACTOR_EXTRA_FIELDS)
    expected_names = expected_champion + list(MinuteEnhancedTailHandler.TAIL_PICK)
    assert len(names) == 20 == len(MINUTE_FACTOR_FIELDS) + len(MINUTE_FACTOR_EXTRA_FIELDS) + 2
    assert names == expected_names
    assert fields == [f"${n}" for n in expected_names]
    # champion 18 前缀逐字保留（继承 ENHANCED_FIELDS，可复现性）
    assert names[:18] == expected_champion


def test_tail_pick_is_strongest_two():
    """TAIL_PICK = (tail_vol_ratio_t1, tail_accel_t1)，确为 IC 校准最强 2（#3/#7），且属物化尾盘族。"""
    assert MinuteEnhancedTailHandler.TAIL_PICK == ("tail_vol_ratio_t1", "tail_accel_t1")
    for n in MinuteEnhancedTailHandler.TAIL_PICK:
        assert n in MINUTE_FACTOR_TAIL_FIELDS, f"{n} not a materialized tail factor"
    # 仅取 2 个（非全 5）：避免与 tail_close_pos/mom/last5 相关徒增过拟合（Step B 教训）
    assert len(MinuteEnhancedTailHandler.TAIL_PICK) == 2


def test_l1_guard_inherited():
    """L1 shared DropnaProcessor(feature) 从 HighBetaAlpha158 继承（前视护栏不变）。"""
    procs = MinuteEnhancedTailHandler._DEFAULT_SHARED_PROCESSORS
    assert any(
        isinstance(p, dict)
        and p.get("class") == "DropnaProcessor"
        and p.get("kwargs", {}).get("fields_group") == "feature"
        for p in procs
    ), f"shared_processors must include DropnaProcessor(feature), got {procs}"


def test_inherits_champion_init():
    """唯一变量是 feature 18→20：tail handler 未 override __init__，归一化口径（RobustZScoreNorm
    时间序列，= champion，非 CSRank 横截面）靠继承冻结。infer/learn processors 由 Alpha158 __init__
    kwarg 注入，非类属性，故验 MRO 而非类属性相等。"""
    assert issubclass(MinuteEnhancedTailHandler, MinuteEnhancedHandler)
    # __init__ 未被本类 override（继承自 MinuteEnhancedHandler）→ processor 注入路径与 champion 同源
    assert "__init__" not in MinuteEnhancedTailHandler.__dict__, (
        "tail handler must not override __init__ (only variable = feature 18→20)"
    )
