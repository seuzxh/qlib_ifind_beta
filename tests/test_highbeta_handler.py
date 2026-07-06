"""HighBetaAlpha158 = Alpha158(rolling windows [5,10], 全 lag T-1) + 14 $minute-fields.

v3 (2026-07-06)：rolling windows 砍到 [5,10]，剔除 20/30/60 共 87 慢因子 → 9 kbar +
4 price + 29×2 rolling = 71 日频 + 14 分钟 = 85 因子（v2 是 158+14=172）。

get_feature_config ignores instance state (Alpha158 delegates to Alpha158DL), so we
can test it via __new__ without qlib.init / data fetch. The full fetch shape (85
columns in the real df) is verified by the smoke run (Task 8).

L1 前视护栏测试：shared DropnaProcessor(feature) 配置正确、is_for_infer 安全、且只 drop
feature NaN 行（保留 label NaN 的推理样本），index 结构不变。
"""
import pandas as pd

from qlib.data.dataset import processor as qlib_processor

from qlib_ifind_beta.config import MINUTE_FACTOR_FIELDS
from qlib_ifind_beta.highbeta_handler import HighBetaAlpha158


def test_feature_count_is_71_daily_plus_14_minute():
    """v3: 71 日频（9 kbar + 4 price + 29×2 rolling，windows=[5,10]）+ 14 分钟 = 85。"""
    h = HighBetaAlpha158.__new__(HighBetaAlpha158)
    fields, names = h.get_feature_config()
    assert len(names) == 71 + 14
    assert len(fields) == len(names)


def test_no_slow_rolling_factors():
    """v3: rolling windows=[5,10] → 不应出现 window 20/30/60 的慢因子（日频字段名尾数 ≤10）。"""
    import re

    h = HighBetaAlpha158.__new__(HighBetaAlpha158)
    fields, names = h.get_feature_config()
    minute_set = set(MINUTE_FACTOR_FIELDS)
    slow = []
    for n in names:
        bare = n.lstrip("$")
        if bare in minute_set:
            continue
        m = re.search(r"(\d+)$", bare)
        if m and int(m.group(1)) > 10:
            slow.append(n)
    assert not slow, f"v3 不应有 window>10 慢因子，发现: {slow[:5]}"


def test_minute_fields_present():
    h = HighBetaAlpha158.__new__(HighBetaAlpha158)
    fields, names = h.get_feature_config()
    for mf in MINUTE_FACTOR_FIELDS:
        assert f"${mf}" in names, mf
        assert f"${mf}" in fields, mf


def test_daily_fields_are_lagged_to_t_minus_1():
    """v2: 所有日频 field（非分钟）必须包 Ref(...,1) → 压到 T-1，避免 9:41 撮合前视。
    v3: 日频数 158→71，但 lag 契约不变。"""
    h = HighBetaAlpha158.__new__(HighBetaAlpha158)
    fields, names = h.get_feature_config()
    daily_fields = [f for f, n in zip(fields, names) if n.lstrip("$") not in set(MINUTE_FACTOR_FIELDS)]
    assert all(f.startswith("Ref(") and f.endswith(", 1)") for f in daily_fields), (
        "all daily fields must be wrapped in Ref(..., 1) to lag to T-1; "
        f"offenders: {[f for f in daily_fields if not (f.startswith('Ref(') and f.endswith(', 1)'))][:3]}"
    )


# ---- L1 前视护栏（shared DropnaProcessor(feature)）----

def test_default_shared_processors_has_dropna_feature():
    """L1: 默认 shared_processors 必须含 DropnaProcessor(feature) — 防 p941 NaN 前视。"""
    procs = HighBetaAlpha158._DEFAULT_SHARED_PROCESSORS
    assert any(
        isinstance(p, dict)
        and p.get("class") == "DropnaProcessor"
        and p.get("kwargs", {}).get("fields_group") == "feature"
        for p in procs
    ), f"shared_processors must include DropnaProcessor(feature), got {procs}"


def test_default_shared_processors_infer_safe():
    """L1: shared processor 必须 is_for_infer=True，否则 DataHandlerLP._run_proc_l 在
    shared 段（check_for_infer=True）抛 TypeError。readonly=True 保证不污染原 df。"""
    from qlib.utils import init_instance_by_config

    for cfg in HighBetaAlpha158._DEFAULT_SHARED_PROCESSORS:
        proc = init_instance_by_config(cfg, qlib_processor)
        assert proc.is_for_infer(), f"{cfg} not infer-safe → cannot sit in shared_processors"
        assert proc.readonly(), f"{cfg} not readonly → shared would mutate raw data"


def test_dropna_feature_drops_only_feature_nan_rows():
    """L1 行为：DropnaProcessor(feature) 只 drop feature 列含 NaN 的行；label NaN 的推理
    样本必须保留（推理不需要 label）；列 MultiIndex 与 (instrument,datetime) 结构不变。"""
    cols = pd.MultiIndex.from_tuples(
        [("feature", "f1"), ("feature", "f2"), ("label", "LABEL0")],
        names=["col_set", "field"],
    )
    idx = pd.MultiIndex.from_tuples(
        [
            ("SH001", "2026-07-01"),  # feature NaN → 模拟 p941 NaN 票，应被 drop
            ("SH002", "2026-07-01"),  # label NaN、feature OK → 推理样本，应保留
            ("SH003", "2026-07-01"),  # 全 OK → 保留
        ],
        names=["instrument", "datetime"],
    )
    df = pd.DataFrame(
        [[None, 1.0, 0.05], [2.0, 3.0, None], [4.0, 5.0, 0.06]],
        index=idx,
        columns=cols,
        dtype="float64",
    )
    out = qlib_processor.DropnaProcessor(fields_group="feature")(df)
    assert list(out.index.get_level_values("instrument")) == ["SH002", "SH003"]
    assert out.columns.equals(cols), "column MultiIndex structure must be unchanged"
    assert out.index.names == ["instrument", "datetime"], "row index structure must be unchanged"
