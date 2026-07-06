"""MinuteOnlyHandler — 实验性 handler，仅 14 分钟因子。

验证「纯短周期分钟因子够不够预测 9:41 label」（与 v2/v3 对比）。get_feature_config 丢弃
HighBetaAlpha158 的 71 日频因子，只返回 14 个 $minute_field；L1 shared
DropnaProcessor(feature) 护栏从父类继承。

get_feature_config ignores instance state，可经 __new__ 无 qlib.init / 无数据抓取单测。
"""
from qlib_ifind_beta.config import MINUTE_FACTOR_FIELDS
from qlib_ifind_beta.minute_only_handler import MinuteOnlyHandler


def test_only_minute_fields():
    h = MinuteOnlyHandler.__new__(MinuteOnlyHandler)
    fields, names = h.get_feature_config()
    assert len(names) == len(MINUTE_FACTOR_FIELDS) == 14
    assert fields == [f"${n}" for n in MINUTE_FACTOR_FIELDS]
    assert names == list(MINUTE_FACTOR_FIELDS)


def test_no_daily_fields():
    """不应含任何 Alpha158 日频因子（无 Ref(...) lag 包装）。"""
    h = MinuteOnlyHandler.__new__(MinuteOnlyHandler)
    fields, names = h.get_feature_config()
    assert all(not f.startswith("Ref(") for f in fields), "不应有 lag 包装的日频因子"


def test_l1_guard_inherited():
    """L1 shared DropnaProcessor(feature) 从 HighBetaAlpha158 继承。"""
    procs = MinuteOnlyHandler._DEFAULT_SHARED_PROCESSORS
    assert any(
        isinstance(p, dict)
        and p.get("class") == "DropnaProcessor"
        and p.get("kwargs", {}).get("fields_group") == "feature"
        for p in procs
    ), f"shared_processors must include DropnaProcessor(feature), got {procs}"
