"""factor_zoo 离线单元测试（无网络、无数据依赖）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlib_ifind_beta.factor_zoo.parse import classify, iter_calls, max_window, ops_used


class TestParse:
    def test_iter_calls_simple(self):
        calls = list(iter_calls("Mean($close, 5)/$close - 1"))
        assert ("Mean", ["$close", "5"]) in calls

    def test_iter_calls_nested(self):
        calls = list(iter_calls("-1*Corr(Rank($open,1), Rank($volume,1), 10)"))
        # 顶层只产出 Corr；嵌套 Rank 由 ops_used 的递归覆盖
        assert ("Corr", ["Rank($open,1)", "Rank($volume,1)", "10"]) in calls
        assert len(calls) == 1

    def test_max_window_nested(self):
        assert max_window("Mean(Ref($close, 5), 20)/$close") == 20
        assert max_window("RSI($close, 14)") == 14
        assert max_window("($close-$open)/$open") == 0

    def test_max_window_float_not_counted(self):
        # 分位数 0.8 是浮点，不算窗口
        assert max_window("Quantile($close, 20, 0.8)") == 20

    def test_ops_used(self):
        ops = ops_used("-1*Corr(Rank(Delta(Log($volume+1),2)), Rank(($close-$open)/$open), 6)")
        assert {"Corr", "Rank", "Delta", "Log"} <= ops

    def test_classify(self):
        assert classify(5, True) == "b1"
        assert classify(8, True) == "b1"
        assert classify(60, True) == "b2_only"
        assert classify(200, True) == "too_long"
        assert classify(5, False) == "unusable"


class TestOpsCompat:
    def test_shims_built_and_picklable(self):
        import pickle
        from qlib_ifind_beta.factor_zoo.ops_compat import SHIMS
        assert len(SHIMS) >= 60
        # pickle 按模块+名反查（joblib 并行需要）
        blob = pickle.dumps(SHIMS["SMA"])
        restored = pickle.loads(blob)
        assert restored is SHIMS["SMA"]

    def test_shim_window_union(self):
        from qlib_ifind_beta.factor_zoo.ops_compat import SHIMS
        from qlib.data.ops import Feature
        op = SHIMS["ATR"](Feature("close"), Feature("high"), Feature("low"), 14)
        left, right = op.get_extended_window_size()
        assert left >= 13 and right == 0


class TestScreenLib:
    @pytest.fixture()
    def panel(self):
        rng = np.random.default_rng(7)
        days = pd.date_range("2024-01-01", periods=60)
        insts = [f"S{i:03d}" for i in range(40)]
        idx = pd.MultiIndex.from_product([days, insts], names=["datetime", "instrument"])
        label = pd.Series(rng.normal(0, 0.02, len(idx)), index=idx)
        # 因子 = label + 噪声 → IC 显著为正
        factor = label + rng.normal(0, 0.05, len(idx))
        factor.name, label.name = "F", "LABEL"
        return pd.Series(factor), label

    def test_daily_rank_ic_positive(self, panel):
        from qlib_ifind_beta.factor_zoo.screen_lib import daily_rank_ic
        factor, label = panel
        ics = daily_rank_ic(factor, label, min_ns=30)
        assert len(ics) == 60
        assert ics.mean() > 0.3  # 构造强信号

    def test_daily_rank_ic_date_intersection(self, panel):
        from qlib_ifind_beta.factor_zoo.screen_lib import daily_rank_ic
        factor, label = panel
        label2 = label.iloc[: len(label) // 2]  # 只有一半日期
        ics = daily_rank_ic(factor, label2, min_ns=30)
        assert len(ics) == 30  # 只算交集，不 KeyError

    def test_summarize_fields(self, panel):
        from qlib_ifind_beta.factor_zoo.screen_lib import summarize
        factor, label = panel
        champs = pd.DataFrame({"c1": np.random.default_rng(1).normal(size=len(factor))},
                              index=factor.index)
        m = summarize(factor, label, champs, min_days=30)
        assert m["n_days"] == 60
        assert m["ic_mean"] > 0.3
        assert m["coverage"] == pytest.approx(1.0)
        assert np.isfinite(m["resid_ic"])
