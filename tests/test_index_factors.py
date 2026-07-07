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
