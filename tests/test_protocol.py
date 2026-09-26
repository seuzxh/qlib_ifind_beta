"""experiment/protocol.py 离线测试（不训练、不触数据）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlib_ifind_beta.experiment.protocol import (
    OPEN_COST, CLOSE_COST, FrozenResult, make_handler, replay_topn, safe_name,
)


class TestSafeName:
    def test_lowercase_single_underscore(self):
        assert safe_name("b1__alpha158__MAX5") == "fz_b1_alpha158_max5"

    def test_idempotent(self):
        once = safe_name("day__jq110__JQ110_VOL_005")
        assert safe_name(once) == once


class TestMakeHandler:
    def test_factory_adds_and_drops(self):
        H = make_handler(["fz_x", "fz_y"], ["close_pos_1m"])
        _, names = H.get_feature_config(None)  # 不依赖实例化（data loader 会要数据）
        assert "fz_x" in names and "fz_y" in names
        assert "close_pos_1m" not in names
        assert len(names) == 18 - 1 + 2


class TestReplay:
    def test_topn_equal_weight_with_costs(self):
        days = pd.date_range("2024-01-01", periods=3)
        idx = pd.MultiIndex.from_product(
            [days, [f"S{i}" for i in range(20)]], names=["datetime", "instrument"])
        rng = np.random.default_rng(3)
        pred = pd.Series(rng.normal(size=len(idx)), index=idx)
        label = pd.Series(rng.normal(0, 0.02, len(idx)), index=idx)
        r = replay_topn(pred, label, topn=10)
        assert len(r) == 3
        # 每日应恰为 top10 等权均值 - 双边成本
        d0 = days[0]
        top = pred.xs(d0).sort_values(ascending=False).head(10).index
        assert r[d0] == pytest.approx(label.xs(d0)[top].mean() - OPEN_COST - CLOSE_COST)


class TestFrozenResult:
    def test_replay_stats(self):
        r = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2024-01-01", periods=3))
        res = FrozenResult(tag="t", ic=0.05, rank_ic=0.06, pred=pd.Series(dtype=float), replay=r)
        s = res.replay_stats()
        assert "累计" in s and "回撤" in s
