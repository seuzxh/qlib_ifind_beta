"""HighBetaAlpha158 appends 14 $minute-fields to Alpha158's native 158 (A5 unit part).

get_feature_config ignores instance state (Alpha158 delegates to Alpha158DL), so we
can test it via __new__ without qlib.init / data fetch. The full fetch shape (172
columns in the real df) is verified by the smoke run (Task 8).
"""
from qlib_ifind_beta.config import MINUTE_FACTOR_FIELDS
from qlib_ifind_beta.highbeta_handler import HighBetaAlpha158


def test_feature_count_is_158_plus_14():
    h = HighBetaAlpha158.__new__(HighBetaAlpha158)
    fields, names = h.get_feature_config()
    assert len(names) == 158 + 14
    assert len(fields) == len(names)


def test_minute_fields_present():
    h = HighBetaAlpha158.__new__(HighBetaAlpha158)
    fields, names = h.get_feature_config()
    for mf in MINUTE_FACTOR_FIELDS:
        assert f"${mf}" in names, mf
        assert f"${mf}" in fields, mf


def test_daily158_fields_are_lagged_to_t_minus_1():
    """v2: 158 Alpha158 日频 field 必须全包 Ref(...,1) → 压到 T-1，避免 9:41 撮合前视。"""
    h = HighBetaAlpha158.__new__(HighBetaAlpha158)
    fields, names = h.get_feature_config()
    daily_fields = fields[:158]
    assert all(f.startswith("Ref(") and f.endswith(", 1)") for f in daily_fields), (
        "all 158 Alpha158 fields must be wrapped in Ref(..., 1) to lag to T-1; "
        f"offenders: {[f for f in daily_fields if not (f.startswith('Ref(') and f.endswith(', 1)'))][:3]}"
    )
