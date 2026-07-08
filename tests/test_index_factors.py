"""日频情绪 + 上证指数共振 12 因子表达式正确性 + 前视 gate 测试（2026-07-07）。

核心断言（正确性即前视测试）：因子[T] == 用 ≤T-1 数据手算的值。Ref(expr,1) 把 expr
shift 1，T 行取 expr[T-1]——若因子偷看了 T 日数据，手算（只用 ≤T-1）就对不上。

qlib ops 语义（ops.py:781-999/1467-1520 实测）：
  - Rolling: `.rolling(N, min_periods=1).<func>()`，窗口含当前点 [T-N+1..T]
  - Ref(expr,N): expr.shift(N)；Ref(...,1) = expr[T-1]
  - Cov/Var pandas 默认 ddof=1，但 beta=Cov/Var 的 ddof 约掉
    → 12 因子全部 ddof 鲁棒

corr_20 已移除（qlib Corr._load_internal np.isclose 广播 bug，个股621≠指数624 时崩；
详见 index_daily_handler.py docstring）。SR/IR 保留（beta_20 用）。

手算（pandas）：先按 qlib 同语义算 raw expr（rolling 含当前点），再 .shift(1) 对应 Ref1。
对比尾部 WARMUP_CUTOFF 之后（窗口已满，避开 min_periods 前期差异）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

OVERLAY = "/home/zxh/projects/3.qlib_ifind_beta/data/qlib_root"
STOCK = "sh600519"
INDEX_CODE = "SH000001"
START, END = "2023-10-01", "2024-03-31"   # 预留 warmup（20 日窗口到 2023-11 满）
WARMUP_CUTOFF = "2024-02-01"              # 只对比此日后（所有窗口已满）

# --- expression 清单（name → qlib expr）---
IV = "ChangeInstrument('SH000001', $close)"
IVL = "ChangeInstrument('SH000001', $low)"
IVH = "ChangeInstrument('SH000001', $high)"
IVV = "ChangeInstrument('SH000001', $volume)"
SR = "$close/Ref($close,1)-1"
IR = f"{IV}/Ref({IV},1)-1"

EXPRS = {
    # 任务2 日频情绪 7 个
    "bias_5": "Ref(($close-Mean($close,5))/Mean($close,5),1)",
    "bias_20": "Ref(($close-Mean($close,20))/Mean($close,20),1)",
    "vol_ratio_20": "Ref($volume/Mean($volume,20),1)",
    "run_up_5": "Ref($close/Ref($close,5)-1,1)",
    "rsv_9": "Ref(($close-Min($low,9))/(Max($high,9)-Min($low,9)),1)",
    "dist_to_limit": "Ref($change/$limit_up,1)",
    "accel_mom": "Ref(($close/Ref($close,3)-1)-(Ref($close,3)/Ref($close,6)-1),1)",
    # 任务3 指数共振/情绪 5 个（corr_20 已移除）
    "idx_bias_20": f"Ref(({IV}-Mean({IV},20))/Mean({IV},20),1)",
    "idx_run_5": f"Ref({IV}/Ref({IV},5)-1,1)",
    "idx_rsv_9": f"Ref(({IV}-Min({IVL},9))/(Max({IVH},9)-Min({IVL},9)),1)",
    "idx_vol_ratio_20": f"Ref({IVV}/Mean({IVV},20),1)",
    "beta_20": f"Ref(Cov({SR}, {IR}, 20)/Var({IR}, 20),1)",
}


@pytest.fixture(scope="module")
def Q():
    import qlib
    qlib.init(provider_uri=OVERLAY)
    from qlib.data import D
    return D


@pytest.fixture(scope="module")
def raw(Q):
    """sh600519 + sh000001 原始字段（datetime index，dropna 对齐）。"""
    stk = Q.features([STOCK], ["$close", "$high", "$low", "$volume", "$change", "$limit_up"],
                     start_time=START, end_time=END).droplevel(0)
    idx = Q.features([INDEX_CODE], ["$close", "$high", "$low", "$volume"],
                     start_time=START, end_time=END).droplevel(0)
    df = pd.DataFrame({
        "close": stk["$close"], "high": stk["$high"], "low": stk["$low"],
        "volume": stk["$volume"], "change": stk["$change"], "limit_up": stk["$limit_up"],
        "idx_close": idx["$close"], "idx_high": idx["$high"],
        "idx_low": idx["$low"], "idx_volume": idx["$volume"],
    }).dropna()
    return df


@pytest.fixture(scope="module")
def factors(Q):
    """一次查 13 个 expression（qlib MemCache 复用 ChangeInstrument load）。"""
    df = Q.features([STOCK], list(EXPRS.values()), start_time=START, end_time=END).droplevel(0)
    return {name: df[col] for name, col in zip(EXPRS.keys(), df.columns)}


def _assert_close(got: pd.Series, exp: pd.Series, name: str):
    a = pd.concat([got.rename("g"), exp.rename("e")], axis=1).dropna()
    a = a[a.index >= WARMUP_CUTOFF]
    assert len(a) > 0, f"{name}: 无可对比行（warmup 不足？）"
    assert np.allclose(a["g"], a["e"], rtol=1e-6, equal_nan=True), (
        f"{name} mismatch（前视或语义错）:\n{a.head(8)}"
    )


# ---- ChangeInstrument 对齐 ----
def test_changeinst_index_close_aligns_with_direct_index_query(Q, raw):
    """ChangeInstrument('SH000001',$close) 对个股查询 == 直接查 sh000001 的 close（datetime 对齐）。"""
    got = Q.features([STOCK], ["ChangeInstrument('SH000001', $close)"],
                     start_time=START, end_time=END).droplevel(0).iloc[:, 0]
    a = pd.concat([got.rename("g"), raw["idx_close"].rename("e")], axis=1).dropna()
    assert np.allclose(a["g"], a["e"], rtol=1e-6), "ChangeInstrument 未返回上证综指真实值"


# ---- 日频情绪族 7 个 ----
def test_bias_5(factors, raw):
    m = raw.close.rolling(5, min_periods=1).mean()
    _assert_close(factors["bias_5"], ((raw.close - m) / m).shift(1), "bias_5")


def test_bias_20(factors, raw):
    m = raw.close.rolling(20, min_periods=1).mean()
    _assert_close(factors["bias_20"], ((raw.close - m) / m).shift(1), "bias_20")


def test_vol_ratio_20(factors, raw):
    m = raw.volume.rolling(20, min_periods=1).mean()
    _assert_close(factors["vol_ratio_20"], (raw.volume / m).shift(1), "vol_ratio_20")


def test_run_up_5(factors, raw):
    _assert_close(factors["run_up_5"], (raw.close / raw.close.shift(5) - 1).shift(1), "run_up_5")


def test_rsv_9(factors, raw):
    mn = raw.low.rolling(9, min_periods=1).min()
    mx = raw.high.rolling(9, min_periods=1).max()
    _assert_close(factors["rsv_9"], ((raw.close - mn) / (mx - mn)).shift(1), "rsv_9")


def test_dist_to_limit(factors, raw):
    _assert_close(factors["dist_to_limit"], (raw.change / raw.limit_up).shift(1), "dist_to_limit")


def test_accel_mom(factors, raw):
    raw_a = (raw.close / raw.close.shift(3) - 1) - (raw.close.shift(3) / raw.close.shift(6) - 1)
    _assert_close(factors["accel_mom"], raw_a.shift(1), "accel_mom")


# ---- 指数共振/情绪族 6 个 ----
def test_idx_bias_20(factors, raw):
    m = raw.idx_close.rolling(20, min_periods=1).mean()
    _assert_close(factors["idx_bias_20"], ((raw.idx_close - m) / m).shift(1), "idx_bias_20")


def test_idx_run_5(factors, raw):
    _assert_close(factors["idx_run_5"],
                  (raw.idx_close / raw.idx_close.shift(5) - 1).shift(1), "idx_run_5")


def test_idx_rsv_9(factors, raw):
    mn = raw.idx_low.rolling(9, min_periods=1).min()
    mx = raw.idx_high.rolling(9, min_periods=1).max()
    _assert_close(factors["idx_rsv_9"], ((raw.idx_close - mn) / (mx - mn)).shift(1), "idx_rsv_9")


def test_idx_vol_ratio_20(factors, raw):
    m = raw.idx_volume.rolling(20, min_periods=1).mean()
    _assert_close(factors["idx_vol_ratio_20"], (raw.idx_volume / m).shift(1), "idx_vol_ratio_20")


def test_beta_20(factors, raw):
    stock_ret = raw.close / raw.close.shift(1) - 1
    idx_ret = raw.idx_close / raw.idx_close.shift(1) - 1
    cov = stock_ret.rolling(20, min_periods=1).cov(idx_ret)
    var = idx_ret.rolling(20, min_periods=1).var()
    _assert_close(factors["beta_20"], (cov / var).shift(1), "beta_20")


def test_corr_20_removed():
    """corr_20 因 qlib Corr np.isclose 广播 bug 已从 handler 移除（见 index_daily_handler.py docstring）。
    此测试固化该决策：handler 不含 corr_20，避免后续误加回。"""
    from qlib_ifind_beta.index_daily_handler import IndexDailyHandler

    h = IndexDailyHandler.__new__(IndexDailyHandler)
    _, names = h.get_feature_config()
    assert "corr_20" not in names, "corr_20 已因 qlib Corr 广播 bug 移除，不应重新加回"


# ---- IndexDailyHandler 契约（get_feature_config 结构，无需 qlib.init）----
def test_index_daily_handler_config():
    """get_feature_config 返回 (fields, names)：len==12、name 与 expression 对齐、全 Ref1 lag。"""
    from qlib_ifind_beta.index_daily_handler import IndexDailyHandler

    h = IndexDailyHandler.__new__(IndexDailyHandler)   # 绕过 Alpha158.__init__（不需数据）
    fields, names = h.get_feature_config()

    assert len(fields) == 12 and len(names) == 12
    # 关键因子名在列（抽样日频 + 指数）
    for must in ("bias_5", "bias_20", "vol_ratio_20", "dist_to_limit", "accel_mom",
                 "idx_bias_20", "beta_20"):
        assert must in names, f"缺因子 {must}"
    # 13 个 expression 全 Ref(...,1) lag1（前视对齐硬约束）
    for f in fields:
        assert "Ref(" in f and f.rstrip().endswith(",1)"), f"非 Ref1 lag 表达式: {f}"
    # name ↔ expression 顺序对齐（FACTOR_FIELDS 同序）
    assert names == [n for n, _ in IndexDailyHandler.FACTOR_FIELDS]


# ---- 前视 gate（截断不变性）----
@pytest.mark.parametrize("name", ["bias_20", "rsv_9", "beta_20", "idx_bias_20"])
def test_no_lookahead_truncation_invariance(Q, name):
    """Ref1 前视 gate：用 [start,T1] 算的因子值 == [start,T2](T2>T1) 算的在 ≤T1 行。
    若因子偷看未来，加入 T1..T2 数据会改变 ≤T1 的值。"""
    expr = EXPRS[name]
    f_short = Q.features([STOCK], [expr], start_time="2024-01-01", end_time="2024-02-15").droplevel(0).iloc[:, 0]
    f_long = Q.features([STOCK], [expr], start_time="2024-01-01", end_time="2024-03-31").droplevel(0).iloc[:, 0]
    common = f_short.index.intersection(f_long.index)
    assert np.allclose(f_short.loc[common], f_long.loc[common], rtol=1e-6, equal_nan=True), (
        f"{name}: 前视！加入未来数据后 ≤{f_short.index[-1]} 的因子值改变"
    )


# ============================================================================
# §29 上证指数 T 日开盘共振因子（goal 2026-07-08「分钟尺度 regime」实做）
# 3 idx 因子（idx_open_ret_10/idx_open_mom_5m/idx_open_accel_5m）= SH000001 1min 早盘
# slot 1-10 算的大势开盘态势，broadcast 同值到每只股 overlay bin（Plan A）。详见
# materialize_minute._load_index_opening_factors + MinuteResonanceHandler + config.INDEX_OPENING_FIELDS。
# ============================================================================

# broadcast 同值验证用的两只测试股（主板 + 创业板，不同 si_dc 对齐路径）
_BROADCAST_STOCKS = ("sh600519", "sz300750")


def _idx_morning_rows():
    """min-cal 早盘 calendar rows（slot FIRST_FEATURE_SLOT..BUY_SLOT = 1..11）。"""
    from qlib_ifind_beta.materialize_minute import _load_min_calendar
    from qlib_ifind_beta.config import FIRST_FEATURE_SLOT, BUY_SLOT
    _, min_slots = _load_min_calendar()
    return np.where((min_slots >= FIRST_FEATURE_SLOT) & (min_slots <= BUY_SLOT))[0]


def test_idx_opening_formula_matches_handcalc():
    """公式正确性（= 无前视证明）：_load_index_opening_factors cache 值 == 手算。

    手算只用 SH000001 T 日 9:30-9:40（slot 1-10）close/open，与 cache 函数逐位一致 →
    idx 因子只用 T 日早盘数据，无未来信息。公式：idx_open_ret_10=c[9]/o[0]-1，
    idx_open_mom_5m=c[9]/c[4]-1，idx_open_accel_5m=(c[9]/o[5]-1)-(c[3]/o[0]-1)。
    """
    from qlib_ifind_beta.materialize_minute import (
        _load_index_opening_factors, _MORNING_WINDOW,
    )
    from qlib_ifind_beta.binio import read_bin
    from qlib_ifind_beta.config import INDEX_OPENING_SRC, FEATURES_1MIN_SRC
    idx_dir = Path(FEATURES_1MIN_SRC) / INDEX_OPENING_SRC.lower()
    si_c, close_i = read_bin(idx_dir / "close.1min.bin")
    si_o, open_i = read_bin(idx_dir / "open.1min.bin")

    morning_rows = _idx_morning_rows()
    n_min_days = morning_rows.size // _MORNING_WINDOW

    def m2d(arr, si):
        flat = np.full(morning_rows.size, np.nan, dtype=np.float64)
        valid = (morning_rows >= si) & (morning_rows < si + arr.size)
        flat[valid] = arr[morning_rows[valid] - si].astype(np.float64)
        return flat.reshape(n_min_days, _MORNING_WINDOW)

    c, o = m2d(close_i, si_c), m2d(open_i, si_o)
    with np.errstate(invalid="ignore", divide="ignore"):
        exp = {
            "idx_open_ret_10": c[:, 9] / o[:, 0] - 1.0,
            "idx_open_mom_5m": c[:, 9] / c[:, 4] - 1.0,
            "idx_open_accel_5m": (c[:, 9] / o[:, 5] - 1.0) - (c[:, 3] / o[:, 0] - 1.0),
        }

    fac = _load_index_opening_factors()
    assert fac is not None, "SH000001 1min 源缺失/错位，共振因子不可物化"
    for name, e in exp.items():
        got = fac[name]
        m = np.isfinite(got) & np.isfinite(e)
        assert m.sum() > 0, f"{name}: 无可对比 finite 行"
        assert np.allclose(got[m], e[m], rtol=1e-6), f"{name} 公式与 cache 不符（前视/scatter 错）"


def test_idx_opening_cache_shape_and_coverage():
    """cache 形状 + SH000001 1min 覆盖 smoke：3 因子等长 = n_min_days，finite 数 ~数百天。

    probe-verified（2026-07-08）：SH000001 1min bin start_index 偏晚，仅覆盖最后 ~604 个
    min-cal 日，cache finite ~576（test W1/W2 + valid 三窗 0 缺失，仅 train 早期 ~29 天 6%
    缺失 → DropnaProcessor drop，§L1 护栏）。finite 容忍 (500, 700)：数据源大幅变化时告警。
    """
    from qlib_ifind_beta.materialize_minute import (
        _load_index_opening_factors, _MORNING_WINDOW,
    )
    from qlib_ifind_beta.config import INDEX_OPENING_FIELDS
    fac = _load_index_opening_factors()
    assert fac is not None
    lens = {fac[n].size for n in INDEX_OPENING_FIELDS}
    assert len(lens) == 1, f"3 idx 因子长度不一致: {lens}"
    assert lens.pop() == _idx_morning_rows().size // _MORNING_WINDOW
    finite = int(np.isfinite(fac["idx_open_ret_10"]).sum())
    assert 500 < finite < 700, f"SH000001 1min 覆盖 {finite} 天偏离 probe ~576（数据源变化？）"


def test_minute_resonance_handler_config():
    """MinuteResonanceHandler 契约：21 因子（18 champion + 3 idx），idx 表达式 = $field 无 Ref。

    idx 因子物化为 T 日当天 bin（9:30-9:40 早盘数据），handler 用 $idx_open_ret_10 直接
    消费（无 Ref shift）→ T 日 9:40 ≪ 9:41 决策，无前视。与 §28 日频 Ref1 lag 模式对照。
    """
    from qlib_ifind_beta.minute_resonance_handler import MinuteResonanceHandler
    from qlib_ifind_beta.config import INDEX_OPENING_FIELDS
    h = MinuteResonanceHandler.__new__(MinuteResonanceHandler)
    fields, names = h.get_feature_config()
    assert len(fields) == 21 and len(names) == 21
    for must in INDEX_OPENING_FIELDS:
        assert must in names, f"缺 idx 因子 {must}"
    for idx_name in INDEX_OPENING_FIELDS:
        f = fields[names.index(idx_name)]
        assert f == f"${idx_name}" and "Ref(" not in f, f"{idx_name} 应为 $field 直接消费（无 lag）: {f}"


@pytest.fixture(scope="module")
def _materialized_broadcast():
    """物化 2 只测试股（side-effect fixture，触发 overlay bin 写入）。scope=module 只物化一次。"""
    from qlib_ifind_beta.materialize_minute import materialize_minute_instrument
    for code in _BROADCAST_STOCKS:
        assert materialize_minute_instrument(code), f"{code} 物化失败（1min 源缺失？）"


def test_idx_broadcast_same_value_across_stocks(Q, _materialized_broadcast):
    """broadcast 同值（Plan A 核心契约）：两只不同股同日 idx 因子值逐位相等。

    idx 因子是大势（SH000001）开盘态势，与个股无关 → 全池同日同值。主板 sh600519 +
    创业板 sz300750（不同 si_dc 对齐路径）在 test W1 窗同日 idx_open_ret_10/mom_5m/accel_5m
    必须相等（finite 行），证明 broadcast scatter 正确。
    """
    from qlib_ifind_beta.config import INDEX_OPENING_FIELDS
    start, end = "2026-04-01", "2026-07-02"   # test W1（probe-verified 0 缺失）
    fields = [f"${n}" for n in INDEX_OPENING_FIELDS]
    a = Q.features([_BROADCAST_STOCKS[0]], fields, start_time=start, end_time=end).droplevel(0)
    b = Q.features([_BROADCAST_STOCKS[1]], fields, start_time=start, end_time=end).droplevel(0)
    common = a.index.intersection(b.index)
    assert len(common) > 50, f"test W1 共同交易日仅 {len(common)}（异常）"
    for idx_name in INDEX_OPENING_FIELDS:
        va, vb = a.loc[common, f"${idx_name}"].values, b.loc[common, f"${idx_name}"].values
        m = np.isfinite(va) & np.isfinite(vb)
        assert m.sum() > 0, f"{idx_name}: test W1 无 finite 行"
        assert np.allclose(va[m], vb[m], rtol=1e-6), (
            f"{idx_name} broadcast 同值失败：{_BROADCAST_STOCKS} 同日不等")


def test_idx_test_window_zero_missing(Q, _materialized_broadcast):
    """test W1 0 缺失（probe 结论固化）：SH000001 1min 在 test W1(2026Q2) 0 缺失 →
    idx 因子 finite 率 == 1.0（回测公平前提，idx 因子不污染 test 窗预测）。"""
    from qlib_ifind_beta.config import INDEX_OPENING_FIELDS
    df = Q.features([_BROADCAST_STOCKS[0]], [f"${INDEX_OPENING_FIELDS[0]}"],
                    start_time="2026-04-01", end_time="2026-07-02").droplevel(0)
    finite_rate = float(np.isfinite(df.iloc[:, 0].values).mean())
    assert finite_rate == 1.0, f"test W1 idx finite 率 {finite_rate:.3f} < 1.0（probe 0 缺失结论被破坏）"
