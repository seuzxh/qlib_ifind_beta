"""TopkDropoutStrategyTD0 — 命门守卫：generate_trade_decision 与父类唯一差异 = shift=1→0。"""
import inspect

import pytest

from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib_ifind_beta.td0_strategy import TopkDropoutStrategyTD0


def test_td0_is_subclass_of_topkdropout():
    assert issubclass(TopkDropoutStrategyTD0, TopkDropoutStrategy)


def test_td0_overrides_generate_trade_decision():
    # 命门：必须 override（否则继承父类的 shift=1 → T+1 成交）
    assert TopkDropoutStrategyTD0.generate_trade_decision is not TopkDropoutStrategy.generate_trade_decision


def test_td0_uses_shift_zero():
    """命门：pred 窗口用 shift=0（T 日成交），不是 shift=1（T+1 成交）。"""
    src = inspect.getsource(TopkDropoutStrategyTD0.generate_trade_decision)
    assert "get_step_time(trade_step, shift=0)" in src
    assert "get_step_time(trade_step, shift=1)" not in src


def test_td0_only_diff_is_shift_zero():
    """守卫：子类 generate_trade_decision 与父类逐行对比，唯一差异是 shift=1→0。
    任何意外差异（手抖改了别的行、加了注释/docstring）都会被抓到。"""
    def norm(meth):
        return [ln.strip() for ln in inspect.getsource(meth).splitlines() if ln.strip()]
    p, c = norm(TopkDropoutStrategy.generate_trade_decision), norm(TopkDropoutStrategyTD0.generate_trade_decision)
    diffs = [(a, b) for a, b in zip(p, c) if a != b]
    assert len(diffs) == 1, f"expected exactly 1 differing line, got {len(diffs)}: {diffs}"
    a, b = diffs[0]
    assert "shift=1" in a and "shift=0" in b, f"diff must be shift=1→0: parent={a!r} child={b!r}"
