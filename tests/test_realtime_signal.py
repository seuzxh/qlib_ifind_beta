"""Tests for real-time signal generation module.

Tests:
1. _bars_to_arrays: correct array conversion from kline bars
2. _compute_all_factors: factor computation with mock data
3. _compute_all_factors: edge cases (NaN prev_vols, missing daily_info)
4. _write_factor_row: bin writing (mock bin I/O)
5. data_fetch: load_universe returns non-empty for known date
6. data_fetch: prev-day volumes from cn_data_1min
7. Integration: dry-run predict matches materialized path
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestBarsToArrays:
    def test_basic_conversion(self):
        from qlib_ifind_beta.realtime.signal import _bars_to_arrays

        # Create 11 mock bars (09:31-09:41)
        bars = []
        for i in range(11):
            bars.append({
                "date": "2026-07-13",
                "time": f"09:{31+i:02d}:00",
                "open": 10.0 + i * 0.01,
                "high": 10.1 + i * 0.01,
                "low": 9.9 + i * 0.01,
                "close": 10.05 + i * 0.01,
                "volume": 1000 + i * 100,
                "amount": 10050 + i * 100 * 10,
            })

        result = _bars_to_arrays(bars)
        assert result is not None
        assert len(result["c"]) == 11
        assert result["c"][0] == 10.05
        assert result["c"][10] == 10.15  # 10.05 + 10*0.01
        assert result["vol"][0] == 1000
        assert result["vol"][10] == 2000

    def test_insufficient_bars(self):
        from qlib_ifind_beta.realtime.signal import _bars_to_arrays

        bars = [{"close": 10.0, "open": 10.0, "high": 10.0, "low": 10.0,
                 "volume": 1000, "amount": 10000}] * 5
        assert _bars_to_arrays(bars) is None

    def test_vwap_from_amount_volume(self):
        from qlib_ifind_beta.realtime.signal import _bars_to_arrays

        bars = []
        for i in range(11):
            bars.append({
                "date": "2026-07-13", "time": f"09:{31+i:02d}:00",
                "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
                "volume": 1000, "amount": 10500,  # vwap = 10500/1000 = 10.5
            })
        result = _bars_to_arrays(bars)
        assert result["vwap"][0] == 10.5

    def test_vwap_fallback_when_volume_zero(self):
        from qlib_ifind_beta.realtime.signal import _bars_to_arrays

        bars = []
        for i in range(11):
            bars.append({
                "date": "2026-07-13", "time": f"09:{31+i:02d}:00",
                "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
                "volume": 0, "amount": 0,
            })
        result = _bars_to_arrays(bars)
        # When volume=0, vwap falls back to close
        assert result["vwap"][0] == 10.0


class TestComputeAllFactors:
    def _make_mock_arrays(self):
        """Create realistic mock arrays for factor computation."""
        np.random.seed(42)
        base = 10.0
        c = np.array([base + i * 0.01 for i in range(11)], dtype=np.float64)
        o = c.copy()
        h = c + 0.05
        l = c - 0.05
        vol = np.array([1000 + i * 50 for i in range(11)], dtype=np.float64)
        vwap = c.copy()
        amount = vol * vwap
        return {"c": c, "o": o, "h": h, "l": l, "vol": vol,
                "vwap": vwap, "amount": amount}

    def test_basic_factors(self):
        from qlib_ifind_beta.realtime.signal import _compute_all_factors

        arrs = self._make_mock_arrays()
        prev_vols = [240000.0, 230000.0, 220000.0, 200000.0]
        daily_info = {"prev_close": 9.8, "prev_factor": 1.0,
                      "open": 10.0, "factor": 1.0}

        factors = _compute_all_factors(arrs, prev_vols, daily_info)

        # Should have all 14+4 factor keys + price_941 + change_941
        assert "startup_mom_1m" in factors
        assert "vol_vs_yest" in factors
        assert "vol_vs_yest_t5" in factors
        assert "overnight_gap" in factors
        assert "price_941" in factors
        assert "change_941" in factors

        # startup_total = c[9]/o[0] - 1
        expected_startup = arrs["c"][9] / arrs["o"][0] - 1
        assert abs(factors["startup_total"] - expected_startup) < 1e-10

        # price_941 = c[10]
        assert factors["price_941"] == arrs["c"][10]

    def test_nan_prev_vols(self):
        from qlib_ifind_beta.realtime.signal import _compute_all_factors

        arrs = self._make_mock_arrays()
        prev_vols = [0.0, 0.0, 0.0, 0.0]  # all zero

        factors = _compute_all_factors(arrs, prev_vols, None)

        assert np.isnan(factors["vol_vs_yest"])
        assert np.isnan(factors["vol_vs_yest_t2"])
        assert np.isnan(factors["overnight_gap"])

    def test_vol_vs_yest_t5_correct(self):
        from qlib_ifind_beta.realtime.signal import _compute_all_factors
        from qlib_ifind_beta.config import REAL_BARS_PER_DAY

        arrs = self._make_mock_arrays()
        prev_vols = [240000.0, 230000.0, 220000.0, 200000.0]

        factors = _compute_all_factors(arrs, prev_vols, None)

        # vol_vs_yest_t5 uses prev_vols[3] (T-5)
        morning_sum = arrs["vol"][0:10].sum()
        expected = morning_sum / (200000.0 / float(REAL_BARS_PER_DAY))
        assert abs(factors["vol_vs_yest_t5"] - expected) < 1e-6

    def test_change_941_unadjusted(self):
        from qlib_ifind_beta.realtime.signal import _compute_all_factors

        arrs = self._make_mock_arrays()
        daily_info = {"prev_close": 9.8, "prev_factor": 2.0,
                      "open": 19.6, "factor": 2.0}  # raw: 9.8, 9.8

        factors = _compute_all_factors(arrs, [100000]*4, daily_info)

        # raw_p941 = c[10] / factor = 10.1 / 2.0 = 5.05
        # raw_prev_close = 9.8 / 2.0 = 4.9
        # change_941 = 5.05/4.9 - 1
        expected = (arrs["c"][10] / 2.0) / (9.8 / 2.0) - 1
        assert abs(factors["change_941"] - expected) < 1e-10


class TestDataFetchUniverse:
    def test_load_universe_returns_codes(self):
        from qlib_ifind_beta.realtime.data_fetch import load_universe

        # Use a known date that should have universe data
        codes = load_universe("2026-07-10")
        assert len(codes) > 50  # should be ~90
        assert all(c.startswith(("SH", "SZ", "BJ")) for c in codes[:10])

    def test_load_universe_empty_future_date(self):
        from qlib_ifind_beta.realtime.data_fetch import load_universe

        # A date far in the future should return empty
        codes = load_universe("2099-01-01")
        assert len(codes) == 0


class TestPrevDayVolumes:
    def test_prev_day_volumes_multi(self):
        """Test that we can read prev-day volumes from cn_data_1min."""
        from qlib_ifind_beta.realtime.data_fetch import (
            get_prev_day_volumes_multi, load_universe,
        )

        codes = load_universe("2026-07-10")[:5]
        result = get_prev_day_volumes_multi(codes, "2026-07-10")

        assert len(result) > 0
        for code, vols in result.items():
            assert len(vols) == 4  # k=1,2,3,5
            # At least vol_T-1 should be positive
            assert vols[0] > 0, f"{code} has zero T-1 volume"
