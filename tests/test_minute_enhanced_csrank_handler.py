"""MinuteEnhancedCSRankHandler — 特征组横截面 Rank 归一化（CSRankNorm 替换 RobustZScoreNorm）。

验证 wiring（结构校验，无需 qlib.init / 无数据抓取，经 __new__）：
  - 18 因子 get_feature_config 不变（与 MinuteEnhancedHandler 同）
  - infer_processors 实含 CSRankNorm(feature)、不含 RobustZScoreNorm
  - L1 shared DropnaProcessor(feature) 仍从 HighBetaAlpha158 继承

CSRankNorm 的数值正确性（每日 rank / mean≈0 / 值域≈[-1.73,+1.73]）是 qlib 内置保证
（processor.py:326），非本 handler 职责，故不测；本测仅验注入与继承。
"""
from qlib_ifind_beta.config import MINUTE_FACTOR_EXTRA_FIELDS, MINUTE_FACTOR_FIELDS
from qlib_ifind_beta.minute_enhanced_csrank_handler import MinuteEnhancedCSRankHandler
from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler


def test_feature_config_unchanged_18():
    """18 因子 get_feature_config 与父类 MinuteEnhancedHandler 逐字一致。"""
    h = MinuteEnhancedCSRankHandler.__new__(MinuteEnhancedCSRankHandler)
    fields, names = h.get_feature_config()
    expected_names = list(MINUTE_FACTOR_FIELDS) + list(MINUTE_FACTOR_EXTRA_FIELDS)
    assert len(names) == 18 == len(MINUTE_FACTOR_FIELDS) + len(MINUTE_FACTOR_EXTRA_FIELDS)
    assert names == expected_names
    assert fields == [f"${n}" for n in expected_names]
    # 与父类完全一致
    hp = MinuteEnhancedHandler.__new__(MinuteEnhancedHandler)
    assert fields, names == hp.get_feature_config()


def test_infer_processors_has_csranknorm():
    """infer_processors 必含 CSRankNorm(feature)。"""
    procs = MinuteEnhancedCSRankHandler._CS_INFER_PROCESSORS
    assert any(
        isinstance(p, dict)
        and p.get("class") == "CSRankNorm"
        and p.get("kwargs", {}).get("fields_group") == "feature"
        for p in procs
    ), f"infer_processors must include CSRankNorm(feature), got {procs}"


def test_infer_processors_no_robust_zscore():
    """infer_processors 不应再含 RobustZScoreNorm（已被 CSRankNorm 替换）。"""
    procs = MinuteEnhancedCSRankHandler._CS_INFER_PROCESSORS
    assert all(p.get("class") != "RobustZScoreNorm" for p in procs), (
        f"infer_processors must NOT include RobustZScoreNorm (replaced by CSRankNorm), got {procs}"
    )


def test_l1_guard_inherited():
    """L1 shared DropnaProcessor(feature) 从 HighBetaAlpha158 继承（前视护栏不变）。"""
    procs = MinuteEnhancedCSRankHandler._DEFAULT_SHARED_PROCESSORS
    assert any(
        isinstance(p, dict)
        and p.get("class") == "DropnaProcessor"
        and p.get("kwargs", {}).get("fields_group") == "feature"
        for p in procs
    ), f"shared_processors must include DropnaProcessor(feature), got {procs}"
