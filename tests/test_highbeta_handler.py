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
