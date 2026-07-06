"""Materialization correctness (spec A1) — cross-check materialized day.bin against
an independent raw read of the 1min source for SH600519."""
from pathlib import Path

import numpy as np
import pytest

from qlib_ifind_beta import materialize_minute as mm
from qlib_ifind_beta.binio import read_bin
from qlib_ifind_beta.config import (
    BUY_SLOT, FEATURES_1MIN_SRC, FEATURES_DST, FEATURES_SRC, FIRST_FEATURE_SLOT,
    MINUTE_DEAL_PRICE_FIELD, MINUTE_FACTOR_FIELDS, SLOTS_PER_DAY,
)
from qlib_ifind_beta.minute_factors import compute_day_factors

# Morning window width (slots FIRST_FEATURE_SLOT..BUY_SLOT inclusive = 11).
_MORNING_WINDOW = BUY_SLOT - FIRST_FEATURE_SLOT + 1


def _stock_arrays(code="SH600519"):
    """Read 1min + daily arrays; scatter 1min onto the calendar morning grid.

    Mirrors materialize_minute's calendar-grid mapping (slots 1-11 →
    (n_min_days, 11)) so hand-computed expected values are correct. The dataset's
    slot 0 (09:30) is universally NaN pool-wide (probe 2026-07-06), so the window
    starts at slot 1 (first real bar). A naive reshape(242) would be off-by-one
    everywhere. See spec §数据基础 / §物化架构.
    """
    si_m, c1m = read_bin(Path(FEATURES_1MIN_SRC) / code.lower() / "close.1min.bin")
    _, v1m = read_bin(Path(FEATURES_1MIN_SRC) / code.lower() / "volume.1min.bin")
    si_dc, close_d = read_bin(Path(FEATURES_SRC) / code.lower() / "close.day.bin")

    _, min_slots = mm._load_min_calendar()
    morning_rows = np.where(
        (min_slots >= FIRST_FEATURE_SLOT) & (min_slots <= BUY_SLOT)
    )[0]
    n_min_days = morning_rows.size // _MORNING_WINDOW

    def to_morning2d(arr):
        flat = np.full(morning_rows.size, np.nan, dtype=np.float64)
        valid = (morning_rows >= si_m) & (morning_rows < si_m + arr.size)
        flat[valid] = arr[morning_rows[valid] - si_m].astype(np.float64)
        return flat.reshape(n_min_days, _MORNING_WINDOW)

    return si_m, to_morning2d(c1m), to_morning2d(v1m), si_dc, close_d, n_min_days


def _min_day_in_day_cal(k, morning_rows, min_dates, date_to_row):
    """Return day-calendar row for min-cal day k, or -1 if its date isn't in day-cal.

    Guards against the extra 2026-07-03 min-cal day that has no day-cal counterpart
    (the IMPL handles this via clip+oob mask; tests that map min-dates to day-rows
    need the same guard to avoid IndexError on date_to_row).
    """
    first_row = morning_rows[k * _MORNING_WINDOW]   # slot-FIRST_FEATURE_SLOT row
    date_ord = min_dates[first_row]
    if date_ord < 0 or date_ord >= date_to_row.size:
        return -1
    return int(date_to_row[date_ord])


def _full_day_minute_vol(code="SH600519"):
    """Replicate impl's per-min-day full-day minute volume sum (NaN-safe) for cross-check."""
    si_m, v1m = read_bin(Path(FEATURES_1MIN_SRC) / code.lower() / "volume.1min.bin")
    min_dates, min_slots = mm._load_min_calendar()
    morning_rows = np.where(
        (min_slots >= FIRST_FEATURE_SLOT) & (min_slots <= BUY_SLOT)
    )[0]
    n_min_days = morning_rows.size // _MORNING_WINDOW

    bin_rows = si_m + np.arange(v1m.size, dtype=np.int64)
    in_range = (bin_rows >= 0) & (bin_rows < min_slots.size)
    day_idx_all = np.where(in_range, bin_rows, 0) // SLOTS_PER_DAY
    in_range &= day_idx_all < n_min_days
    src_vals = v1m[in_range].astype(np.float64)
    src_vals = np.where(np.isfinite(src_vals), src_vals, 0.0)
    fdv = np.zeros(n_min_days, dtype=np.float64)
    np.add.at(fdv, day_idx_all[in_range], src_vals)
    return fdv


def test_materialize_returns_true_for_liquid_stock():
    assert mm.materialize_minute_instrument("SH600519") is True


def test_materialize_writes_15_bins():
    mm.materialize_minute_instrument("SH600519")
    d = Path(FEATURES_DST) / "sh600519"
    from qlib_ifind_beta.config import MINUTE_FACTOR_FIELDS, MINUTE_DEAL_PRICE_FIELD
    for name in MINUTE_FACTOR_FIELDS:
        assert (d / f"{name}.day.bin").exists(), name
    assert (d / f"{MINUTE_DEAL_PRICE_FIELD}.day.bin").exists()


def test_day_bin_aligned_to_daily_close():
    """materialized bin start_index == stock's daily close.bin start_index."""
    si_dc, _ = read_bin(Path(FEATURES_SRC) / "sh600519" / "close.day.bin")
    si_out, _ = read_bin(Path(FEATURES_DST) / "sh600519" / "price_941.day.bin")
    assert si_out == si_dc


def test_startup_mom_and_price941_crosscheck():
    """A1: materialized startup_mom_1m / price_941 == raw 1min hand-compute."""
    mm.materialize_minute_instrument("SH600519")
    si_out, startup = read_bin(Path(FEATURES_DST) / "sh600519" / "startup_mom_1m.day.bin")
    _, p941 = read_bin(Path(FEATURES_DST) / "sh600519" / "price_941.day.bin")

    si_m, c2d, v2d, si_dc, close_d, n_days_m = _stock_arrays()
    min_dates, min_slots = mm._load_min_calendar()
    _, date_to_row = mm._load_day_calendar_lookup()
    morning_rows = np.where(
        (min_slots >= FIRST_FEATURE_SLOT) & (min_slots <= BUY_SLOT)
    )[0]

    # find a recent min-cal day with valid feature bars (indices 8,9) and buy (10),
    # AND whose date is in the day calendar (skip extra min-cal days like 2026-07-03).
    km = None
    for k in range(n_days_m - 1, -1, -1):
        if _min_day_in_day_cal(k, morning_rows, min_dates, date_to_row) < 0:
            continue   # min-cal day not in day-cal (extra 2026-07-03)
        if np.all(np.isfinite(c2d[k, [8, 9, 10]])):
            km = k
            break
    assert km is not None, "no valid 1min day for SH600519"

    day_row = _min_day_in_day_cal(km, morning_rows, min_dates, date_to_row)
    out_row = day_row - si_out
    assert 0 <= out_row < startup.size

    # startup_mom_1m = c[index9] / c[index8] - 1; price_941 = c[index10] (slot 11)
    expected_startup = c2d[km, 9] / c2d[km, 8] - 1.0
    expected_p941 = c2d[km, 10]
    assert abs(startup[out_row] - expected_startup) < 1e-4
    assert abs(p941[out_row] - expected_p941) < 1e-2


def test_vol_vs_yest_crosscheck():
    """A1: vol_vs_yest (minute-only both sides) == raw hand-compute."""
    mm.materialize_minute_instrument("SH600519")
    si_out, vol_vs_yest = read_bin(Path(FEATURES_DST) / "sh600519" / "vol_vs_yest.day.bin")

    si_m, c2d, v2d, si_dc, close_d, n_days_m = _stock_arrays()
    min_dates, min_slots = mm._load_min_calendar()
    _, date_to_row = mm._load_day_calendar_lookup()
    morning_rows = np.where(
        (min_slots >= FIRST_FEATURE_SLOT) & (min_slots <= BUY_SLOT)
    )[0]

    # full-day minute volume per min-cal day (replicates impl logic).
    full_day_vol = _full_day_minute_vol()

    km = None
    for k in range(n_days_m - 1, -1, -1):
        if _min_day_in_day_cal(k, morning_rows, min_dates, date_to_row) < 0:
            continue   # extra min-cal day (e.g. 2026-07-03)
        if k == 0:
            continue   # first min-cal day → prev_day NaN
        if np.all(np.isfinite(v2d[k, 0:10])):
            km = k
            break
    assert km is not None

    day_row = _min_day_in_day_cal(km, morning_rows, min_dates, date_to_row)
    out_row = day_row - si_out
    # numerator = sum of slots 1-10 vol = v2d[km, 0:10].sum()
    # denominator = prev min-cal day's full-day vol / 240
    expected = v2d[km, 0:10].sum() / (full_day_vol[km - 1] / 240.0)
    assert abs(vol_vs_yest[out_row] - expected) < 1e-3


def test_vectorized_matches_perrow_all_fields_real_data():
    """Strongest guard: for every SH600519 trading day whose 11 morning bars are
    all non-NaN across all 5 fields, ALL 15 materialized outputs match the per-row
    oracle (compute_day_factors) within tolerance.

    Independent reference: re-read cn_data_1min bins for all 5 fields, scatter to
    the morning grid ourselves, derive prev-day full minute volume from the raw
    volume bin, and feed per-day arrays to compute_day_factors. Then read back the
    15 day.bins and compare field-by-field. Catches any drift between the
    vectorized formulas and the per-row oracle on real data.
    """
    code = "SH600519"
    lcode = code.lower()
    src_dir = Path(FEATURES_1MIN_SRC) / lcode

    # --- independent raw read of all 5 1min fields (own start_index check) ---
    raw = {}
    si_m_ref = None
    for f in ("close", "open", "high", "low", "volume"):
        si, a = read_bin(src_dir / f"{f}.1min.bin")
        assert a.size > 0 and si is not None, f"{f}.1min.bin missing/empty"
        raw[f] = a
        if si_m_ref is None:
            si_m_ref = si
        else:
            assert si == si_m_ref, f"{f} start_index {si} != {si_m_ref}"

    min_dates, min_slots = mm._load_min_calendar()
    _, date_to_row = mm._load_day_calendar_lookup()
    morning_rows = np.where(
        (min_slots >= FIRST_FEATURE_SLOT) & (min_slots <= BUY_SLOT)
    )[0]
    n_min_days = morning_rows.size // _MORNING_WINDOW

    def scatter_morning(arr):
        flat = np.full(morning_rows.size, np.nan, dtype=np.float64)
        valid = (morning_rows >= si_m_ref) & (morning_rows < si_m_ref + arr.size)
        flat[valid] = arr[morning_rows[valid] - si_m_ref].astype(np.float64)
        return flat.reshape(n_min_days, _MORNING_WINDOW)

    c2d = scatter_morning(raw["close"]); o2d = scatter_morning(raw["open"])
    h2d = scatter_morning(raw["high"]);   l2d = scatter_morning(raw["low"])
    v2d = scatter_morning(raw["volume"])

    # prev-day FULL-DAY minute volume, built here from the raw volume bin as an
    # independent reference (mirrors impl logic: NaN→0, scatter-sum by //SLOTS_PER_DAY).
    v1m = raw["volume"]
    bin_rows = si_m_ref + np.arange(v1m.size, dtype=np.int64)
    in_rng = (bin_rows >= 0) & (bin_rows < min_slots.size)
    day_idx_all = np.where(in_rng, bin_rows, 0) // SLOTS_PER_DAY
    in_rng &= day_idx_all < n_min_days
    vals = v1m[in_rng].astype(np.float64)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    full_day_vol = np.zeros(n_min_days, dtype=np.float64)
    np.add.at(full_day_vol, day_idx_all[in_rng], vals)

    # --- materialize, then read back all 15 day.bins (start_index == daily close) ---
    assert mm.materialize_minute_instrument(code) is True
    si_dc_ref, _ = read_bin(Path(FEATURES_SRC) / lcode / "close.day.bin")
    dst_dir = Path(FEATURES_DST) / lcode
    mat = {}
    for name in MINUTE_FACTOR_FIELDS:
        si_o, vals = read_bin(dst_dir / f"{name}.day.bin")
        assert si_o == si_dc_ref, f"{name} start_index {si_o} != daily {si_dc_ref}"
        mat[name] = vals
    si_op, p941 = read_bin(dst_dir / f"{MINUTE_DEAL_PRICE_FIELD}.day.bin")
    assert si_op == si_dc_ref
    mat[MINUTE_DEAL_PRICE_FIELD] = p941

    all_fields = list(MINUTE_FACTOR_FIELDS) + [MINUTE_DEAL_PRICE_FIELD]

    # --- collect usable days: full non-NaN morning (5 fields × 11 slots), date in
    #     day-cal, and prev-day full vol > 0 (so vol_vs_yest is finite on both sides) ---
    usable = []
    for k in range(n_min_days):
        if _min_day_in_day_cal(k, morning_rows, min_dates, date_to_row) < 0:
            continue   # min-cal day with no day-cal counterpart (e.g. extra 2026-07-03)
        if k == 0 or not np.isfinite(full_day_vol[k - 1]) or full_day_vol[k - 1] <= 0:
            continue   # vol_vs_yest undefined → oracle NaN; skip to keep comparison clean
        if (np.all(np.isfinite(c2d[k])) and np.all(np.isfinite(o2d[k]))
                and np.all(np.isfinite(h2d[k])) and np.all(np.isfinite(l2d[k]))
                and np.all(np.isfinite(v2d[k]))):
            usable.append(k)
    if len(usable) < 6:
        pytest.skip(f"only {len(usable)} usable days for {code} (need >=6)")

    # --- compare every usable day, every field (NaN-aware + tolerance) ---
    for k in usable:
        day_row = _min_day_in_day_cal(k, morning_rows, min_dates, date_to_row)
        out_row = day_row - si_dc_ref
        assert 0 <= out_row < mat[MINUTE_DEAL_PRICE_FIELD].size
        oracle = compute_day_factors(
            c2d[k], o2d[k], h2d[k], l2d[k], v2d[k],
            prev_day_minute_vol=float(full_day_vol[k - 1]),
        )
        for name in all_fields:
            tol = 1e-2 if name == MINUTE_DEAL_PRICE_FIELD else 1e-4
            m_val = float(mat[name][out_row])
            o_val = float(oracle[name])
            if np.isnan(o_val):
                assert np.isnan(m_val), (
                    f"{name} day k={k}: oracle NaN but materialized={m_val}"
                )
            else:
                assert np.isfinite(m_val), (
                    f"{name} day k={k}: oracle={o_val} but materialized not finite"
                )
                assert abs(m_val - o_val) < tol, (
                    f"{name} day k={k} out_row={out_row}: "
                    f"mat={m_val} oracle={o_val} diff={abs(m_val - o_val)} tol={tol}"
                )


def test_missing_1min_bin_returns_false(tmp_path, monkeypatch):
    """A code whose cn_data_1min dir/bin is absent → materialize returns False,
    no exception (caller treats as 'no minute data'; qlib reads NaN)."""
    # Point FEATURES_1MIN_SRC at an empty tmp dir so SH600519's 1min bins are not
    # found → _read_1min_fields returns (None, None) → graceful False.
    monkeypatch.setattr(mm, "FEATURES_1MIN_SRC", str(tmp_path / "no_such_1min_dir"))
    assert mm.materialize_minute_instrument("SH600519") is False


def test_missing_daily_close_bin_returns_false(tmp_path, monkeypatch):
    """1min data present but daily close.day.bin empty → materialize returns
    False, no exception (alignment reference unreadable)."""
    # Keep real FEATURES_1MIN_SRC (so _read_1min_fields succeeds for SH600519),
    # but redirect FEATURES_SRC to an overlay whose sh600519/close.day.bin is a
    # 0-byte file → read_bin returns (None, empty) → graceful False.
    fake_src = tmp_path / "fake_daily"
    (fake_src / "sh600519").mkdir(parents=True)
    (fake_src / "sh600519" / "close.day.bin").touch()   # 0-byte file
    monkeypatch.setattr(mm, "FEATURES_SRC", str(fake_src))
    assert mm.materialize_minute_instrument("SH600519") is False
