"""Position sizing tests (§55).

Tests:
1. compute_benchmark_position: basic functionality, clipping, floor
2. compute_benchmark_position: no look-ahead (first window-1 = 1.0)
3. compute_benchmark_position: edge cases (empty, single day, all NaN)
4. compute_position_for_day: single-day computation matches series
5. compute_nav with position_scale: backward compat (None = same as before)
6. compute_nav with position_scale: correctly scales returns
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlib_ifind_beta.position_sizing import (
    compute_benchmark_position,
    compute_position_for_day,
    DEFAULT_WINDOW, DEFAULT_THRESHOLD, DEFAULT_FLOOR,
)


def _make_bench(n=50, seed=42, upward=False):
    """Generate synthetic benchmark returns."""
    rng = np.random.RandomState(seed)
    base = 0.001 if upward else 0.0
    return pd.Series(rng.randn(n) * 0.01 + base)


class TestComputeBenchmarkPosition:
    def test_basic_shape(self):
        rets = _make_bench(50)
        pos = compute_benchmark_position(rets)
        assert len(pos) == 50
        assert pos.dtype == float

    def test_values_in_range(self):
        rets = _make_bench(50)
        pos = compute_benchmark_position(rets)
        assert pos.min() >= DEFAULT_FLOOR - 1e-10
        assert pos.max() <= 1.0 + 1e-10

    def test_first_window_days_full_position(self):
        """No look-ahead: first window days lack a complete T-1 window."""
        rets = _make_bench(50)
        pos = compute_benchmark_position(rets, window=20)
        assert (pos.iloc[:20] == 1.0).all()

    def test_target_day_return_does_not_affect_batch_position(self):
        rets = _make_bench(50)
        target = rets.index[40]
        pos1 = compute_benchmark_position(rets)
        changed = rets.copy()
        changed.loc[target] = 0.50
        pos2 = compute_benchmark_position(changed)
        assert pos1.loc[target] == pytest.approx(pos2.loc[target])

    def test_floor_respected(self):
        """Even in severe downtrend, position >= floor."""
        rets = pd.Series([-0.02] * 50)  # continuous decline
        pos = compute_benchmark_position(rets, floor=0.3)
        assert pos.min() >= 0.3 - 1e-10

    def test_upward_trend_full_position(self):
        """Strong upward trend → position = 1.0."""
        rets = pd.Series([0.005] * 50)  # consistent positive
        pos = compute_benchmark_position(rets, window=20, threshold=0.02)
        # After 20 days of +0.5% daily = ~10% momentum >> 2% threshold
        assert (pos.iloc[20:] == 1.0).all()

    def test_invalid_threshold(self):
        rets = _make_bench(10)
        with pytest.raises(ValueError):
            compute_benchmark_position(rets, threshold=0)

    def test_invalid_floor(self):
        rets = _make_bench(10)
        with pytest.raises(ValueError):
            compute_benchmark_position(rets, floor=-0.1)
        with pytest.raises(ValueError):
            compute_benchmark_position(rets, floor=1.5)

    def test_empty_series(self):
        pos = compute_benchmark_position(pd.Series(dtype=float))
        assert len(pos) == 0

    def test_short_series(self):
        """Less than window days → all 1.0."""
        rets = _make_bench(10)
        pos = compute_benchmark_position(rets, window=20)
        assert (pos == 1.0).all()

    def test_custom_params(self):
        rets = _make_bench(50)
        pos = compute_benchmark_position(rets, window=10, threshold=0.01, floor=0.5)
        assert len(pos) == 50
        assert pos.min() >= 0.5 - 1e-10
        assert pos.max() <= 1.0 + 1e-10


class TestComputePositionForDay:
    def test_matches_series(self):
        """Single-day computation should match the series value."""
        rets = _make_bench(50)
        series = compute_benchmark_position(rets, window=20, threshold=0.02, floor=0.3)
        target = rets.index[40]
        single = compute_position_for_day(rets, target, window=20, threshold=0.02, floor=0.3)
        assert abs(single - series[target]) < 1e-6

    def test_insufficient_history(self):
        """Less than window days before target → 1.0."""
        rets = _make_bench(10)
        target = rets.index[5]
        pos = compute_position_for_day(rets, target, window=20)
        assert pos == 1.0

    def test_no_lookahead(self):
        """Target day's return should not affect position."""
        rets = _make_bench(50)
        target = rets.index[40]
        # Modify the target day's return — should not change position
        rets2 = rets.copy()
        rets2[target] = 0.05  # huge positive return
        pos1 = compute_position_for_day(rets, target)
        pos2 = compute_position_for_day(rets2, target)
        assert abs(pos1 - pos2) < 1e-10


class TestComputeNavWithPosition:
    def _make_settle(self):
        """Create a simple settle DataFrame."""
        return pd.DataFrame({
            "signal_date": ["2025-01-02", "2025-01-02"],
            "code": ["SH600001", "SH600002"],
            "buy_price": [10.0, 20.0],
            "sell_date": ["2025-01-03", "2025-01-03"],
            "sell_price": [10.5, 21.0],
            "blocked": [False, False],
        })

    def test_backward_compat_none(self, tmp_path):
        """position_scale=None → same as before (满仓)."""
        from qlib_ifind_beta.live.track import compute_nav
        settle = self._make_settle()
        out = tmp_path / "nav.csv"
        daily = compute_nav(settle, out, position_scale=None)
        assert "position" in daily.columns
        assert (daily["position"] == 1.0).all()

    def test_position_scale_applied(self, tmp_path):
        """position_scale=0.5 → daily returns halved."""
        from qlib_ifind_beta.live.track import compute_nav
        settle = self._make_settle()
        out = tmp_path / "nav.csv"
        daily_full = compute_nav(settle, out, position_scale=None)
        out2 = tmp_path / "nav_half.csv"
        pos = pd.Series([0.5], index=["2025-01-03"])
        daily_half = compute_nav(settle, out2, position_scale=pos)
        # Net return should be roughly halved (not exact due to cost structure)
        ratio = daily_half["daily_ret_net"].iloc[0] / daily_full["daily_ret_net"].iloc[0]
        assert abs(ratio - 0.5) < 0.05  # approximately halved
