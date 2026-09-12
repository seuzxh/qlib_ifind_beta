"""Materialization correctness (spec A1) — cross-check materialized day.bin against
an independent raw read of the 1min source for SH600519."""
from pathlib import Path

import numpy as np
import pytest

from qlib_ifind_beta import materialize_minute as mm
from qlib_ifind_beta.binio import read_bin
from qlib_ifind_beta.config import (
    BUY_SLOT, FEATURES_1MIN_SRC, FEATURES_DST, FEATURES_SRC, FIRST_FEATURE_SLOT,
    MINUTE_DEAL_PRICE_FIELD, MINUTE_FACTOR_EXTRA_FIELDS, MINUTE_FACTOR_FIELDS,
    MINUTE_FACTOR_PATH_FIELDS,
    SLOTS_PER_DAY,
)
from qlib_ifind_beta.minute_factors import compute_day_factors

# Morning window width (slots FIRST_FEATURE_SLOT..BUY_SLOT inclusive = 11).
_MORNING_WINDOW = BUY_SLOT - FIRST_FEATURE_SLOT + 1


def _stock_arrays(code="SH600519"):
    """Read 1min + daily arrays; scatter 1min onto the calendar morning grid.

    Mirrors materialize_minute's calendar-grid mapping (slots 1-11 →
    (n_min_days, 11)) so hand-computed expected values are correct. The dataset's
    slot 0 = 09:31 (first real bar since 2026-07-11 rebuild, 240 slots/day).
    A naive reshape(240) would be off-by-one
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


def test_materialize_writes_required_bins():
    """All champion execution and feature bins are present."""
    mm.materialize_minute_instrument("SH600519")
    d = Path(FEATURES_DST) / "sh600519"
    from qlib_ifind_beta.config import (MINUTE_CHANGE_941_FIELD,
        MINUTE_CLOSE_0941_FIELD, MINUTE_CLOSE_1500_FIELD)
    expected = (
        list(MINUTE_FACTOR_FIELDS)               # 14 baseline
        + list(MINUTE_FACTOR_EXTRA_FIELDS)       # 4 enhanced extras (incl. overnight_gap)
        + [MINUTE_DEAL_PRICE_FIELD, MINUTE_CHANGE_941_FIELD,
           MINUTE_CLOSE_0941_FIELD, MINUTE_CLOSE_1500_FIELD]   # label v2 两腿
    )
    assert len(expected) == 22
    for name in expected:
        assert (d / f"{name}.day.bin").exists(), name


def test_day_bin_aligned_to_daily_close():
    """materialized bin start_index == stock's daily close.bin start_index."""
    si_dc, _ = read_bin(Path(FEATURES_SRC) / "sh600519" / "close.day.bin")
    si_out, _ = read_bin(Path(FEATURES_DST) / "sh600519" / "price_941.day.bin")
    assert si_out == si_dc


def test_opening_path_candidates_match_raw_ten_bars():
    """Shadow path bins retain volatility/range/drawdown information."""
    code = "SH600519"
    assert mm.materialize_minute_instrument(code)
    src = Path(FEATURES_1MIN_SRC) / code.lower()
    raw = {}
    start = None
    for field in ("close", "open", "high", "low"):
        si, values = read_bin(src / f"{field}.1min.bin")
        start = si if start is None else start
        assert si == start
        raw[field] = values
    min_dates, min_slots = mm._load_min_calendar()
    _, date_to_row = mm._load_day_calendar_lookup()
    rows = np.where((min_slots >= FIRST_FEATURE_SLOT) & (min_slots <= BUY_SLOT))[0]
    n_days = len(rows) // _MORNING_WINDOW

    def morning(values):
        output = np.full(len(rows), np.nan)
        valid = (rows >= start) & (rows < start + len(values))
        output[valid] = values[rows[valid] - start]
        return output.reshape(n_days, _MORNING_WINDOW)

    close, opening = morning(raw["close"]), morning(raw["open"])
    high, low = morning(raw["high"]), morning(raw["low"])
    si_day, _ = read_bin(Path(FEATURES_SRC) / code.lower() / "close.day.bin")
    candidates = {}
    for field in MINUTE_FACTOR_PATH_FIELDS:
        si, values = read_bin(Path(FEATURES_DST) / code.lower() / f"{field}.day.bin")
        assert si == si_day
        candidates[field] = values
    for k in range(n_days):
        day_row = _min_day_in_day_cal(k, rows, min_dates, date_to_row)
        out_row = day_row - si_day
        if day_row < 0 or not 0 <= out_row < len(candidates[MINUTE_FACTOR_PATH_FIELDS[0]]):
            continue
        if not all(np.isfinite(array[k, :10]).all() for array in (close, opening, high, low)):
            continue
        ret = close[k, :10] / opening[k, :10] - 1
        nav = close[k, :10] / opening[k, 0]
        expected = {
            "minute_return_vol": np.std(ret),
            "minute_range_mean": np.mean(high[k, :10] / low[k, :10] - 1),
            "minute_path_max_drawdown": np.min(nav / np.maximum.accumulate(nav) - 1),
        }
        for field, value in expected.items():
            assert candidates[field][out_row] == pytest.approx(value, abs=1e-6)
        break
    else:
        pytest.skip("no complete ten-bar day")


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
        day_row = _min_day_in_day_cal(k, morning_rows, min_dates, date_to_row)
        if not (0 <= day_row - si_out < startup.size):
            continue   # minute source can be newer than the daily output bin
        if np.all(np.isfinite(c2d[k, [8, 9, 10]])):
            km = k
            break
    assert km is not None, "no valid 1min day for SH600519"

    day_row = _min_day_in_day_cal(km, morning_rows, min_dates, date_to_row)
    out_row = day_row - si_out
    assert 0 <= out_row < startup.size

    # startup_mom_1m = c[index9] / c[index8] - 1; price_941 = c[index10] (slot 10)
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
        day_row = _min_day_in_day_cal(k, morning_rows, min_dates, date_to_row)
        if not (0 <= day_row - si_out < vol_vs_yest.size):
            continue
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
        day_row = _min_day_in_day_cal(k, morning_rows, min_dates, date_to_row)
        if not (0 <= day_row - si_dc_ref < mat[MINUTE_DEAL_PRICE_FIELD].size):
            continue   # minute source can lead the daily source intraday
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


def test_change_941_matches_hand_formula():
    """A1b: change_941[T] = (price_941[T]/factor[T]) / (close[T-1]/factor[T-1]) - 1，首日 NaN。"""
    mm.materialize_minute_instrument("SH600519")
    from qlib_ifind_beta.config import FEATURES_SRC, FEATURES_DST, MINUTE_CHANGE_941_FIELD
    _, close = read_bin(Path(FEATURES_SRC) / "sh600519" / "close.day.bin")
    _, factor = read_bin(Path(FEATURES_SRC) / "sh600519" / "factor.day.bin")
    _, p941 = read_bin(Path(FEATURES_DST) / "sh600519" / "price_941.day.bin")
    _, ch941 = read_bin(Path(FEATURES_DST) / "sh600519" / f"{MINUTE_CHANGE_941_FIELD}.day.bin")
    raw_p = p941.astype(np.float64) / factor.astype(np.float64)
    raw_c = close.astype(np.float64) / factor.astype(np.float64)
    prev = np.full_like(raw_c, np.nan)
    prev[1:] = raw_c[:-1]
    expected = raw_p / prev - 1.0
    m = np.isfinite(expected) & np.isfinite(ch941)
    assert np.nanmax(np.abs(expected[m] - ch941[m])) < 1e-5
    assert np.isnan(ch941[0])   # 首日无昨收 → NaN


# ---------------------------------------------------------------------------
# enhanced champion 的 4 extra 因子（2026-07-07）：vol_vs_yest_t2/t3/t5 + overnight_gap。
# 详见 backtest-log §22（MinuteEnhancedHandler，14 + 4 = 18 因子）。
# ---------------------------------------------------------------------------


def test_minute_window_kbar_count():
    """CLAUDE.md：分钟因子测试必须验证 K 线数量。开盘窗 = 恰好 10 根特征 K（slots
    0-9 = 09:31-09:40）+ 1 根买入 K（slot 10 = 09:41）；1min 日历每个交易日贡献
    恰好 11 行 morning rows。"""
    _, min_slots = mm._load_min_calendar()
    morning_rows = np.where(
        (min_slots >= FIRST_FEATURE_SLOT) & (min_slots <= BUY_SLOT)
    )[0]
    # 每个交易日恰好 11 行（整除），无溢出/缺失。
    assert morning_rows.size % _MORNING_WINDOW == 0
    # 抽查第 0 个交易日的 11 行：slots 必须是 1..11 连续。
    first_day_slots = min_slots[morning_rows[:_MORNING_WINDOW]]
    assert list(first_day_slots) == list(range(FIRST_FEATURE_SLOT, BUY_SLOT + 1))
    # 特征 K = 10 根（slots 1-10），买入 K = slot 11。
    assert (BUY_SLOT - FIRST_FEATURE_SLOT) == 10   # 10 feature bars
    assert _MORNING_WINDOW == 11                   # 10 feature + 1 buy


@pytest.mark.parametrize("shift,fname", [
    (2, "vol_vs_yest_t2"),
    (3, "vol_vs_yest_t3"),
    (5, "vol_vs_yest_t5"),
])
def test_vol_vs_yest_family_shift_crosscheck(shift, fname):
    """vol_vs_yest_t{2,3,5}[T] = morning_vol_sum[T] / (full_day_vol[T-shift] / 240)。

    三层断言：
    1. shift 正确：bin[T] 用的分母是 T-shift 日的全天量（与 vol_vs_yest 的 shift-1
       同形，仅 shift 改 2/3/5）。
    2. 与 vol_vs_yest 在同一日取值不同（证明确实是不同 shift，非复制粘贴 bug）。
    3. NaN-safe：min-cal 首 shift 日（k<shift）fac 为 NaN。
    """
    mm.materialize_minute_instrument("SH600519")
    si_out, fac_bin = read_bin(Path(FEATURES_DST) / "sh600519" / f"{fname}.day.bin")
    _, vvy = read_bin(Path(FEATURES_DST) / "sh600519" / "vol_vs_yest.day.bin")

    si_m, c2d, v2d, si_dc, close_d, n_days_m = _stock_arrays()
    min_dates, min_slots = mm._load_min_calendar()
    _, date_to_row = mm._load_day_calendar_lookup()
    morning_rows = np.where(
        (min_slots >= FIRST_FEATURE_SLOT) & (min_slots <= BUY_SLOT)
    )[0]
    full_day_vol = _full_day_minute_vol()

    # 找一个近期、k>=shift、晨窗量非 NaN 的 min-cal 日。
    km = None
    for k in range(n_days_m - 1, -1, -1):
        if _min_day_in_day_cal(k, morning_rows, min_dates, date_to_row) < 0:
            continue   # 额外 min-cal 日（如 2026-07-03）
        day_row = _min_day_in_day_cal(k, morning_rows, min_dates, date_to_row)
        if not (0 <= day_row - si_out < fac_bin.size):
            continue   # 分钟源可能领先日频输出
        if k < shift:
            continue   # fac[fname] 在 k<shift 处为 NaN
        if np.all(np.isfinite(v2d[k, 0:10])):
            km = k
            break
    assert km is not None, f"no valid day for {fname}"

    day_row = _min_day_in_day_cal(km, morning_rows, min_dates, date_to_row)
    out_row = day_row - si_out
    expected = v2d[km, 0:10].sum() / (full_day_vol[km - shift] / 240.0)
    assert abs(fac_bin[out_row] - expected) < 1e-3, (
        f"{fname} shift={shift}: bin={fac_bin[out_row]} expected={expected}")

    # 与 vol_vs_yest（shift-1）同一日必须不同（分母用不同日 → 不同值），证明非复制 bug。
    assert abs(fac_bin[out_row] - vvy[out_row]) > 1e-6, (
        f"{fname} 与 vol_vs_yest 在 out_row={out_row} 完全相同 → 疑似 shift 未生效")


def test_overnight_gap_crosscheck():
    """overnight_gap[T] = (open[T]/factor[T]) / (close[T-1]/factor[T-1]) - 1。

    口径：不复权（分子分母同除 factor 抵消后复权 → 反映真实开盘跳空；除权日跳空
    不被复权抹平）。与 change_941 同源（都是 raw 价 pct）。
    断言：start_index == daily close；长度 == close（day-cal 对齐）；首日 NaN；
    全 finite 日与手算公式吻合。
    """
    from qlib_ifind_beta.config import MINUTE_CHANGE_941_FIELD  # noqa: F401 (keep import style parity)
    mm.materialize_minute_instrument("SH600519")
    si_close, close_d = read_bin(Path(FEATURES_SRC) / "sh600519" / "close.day.bin")
    _, open_d = read_bin(Path(FEATURES_SRC) / "sh600519" / "open.day.bin")
    _, factor_d = read_bin(Path(FEATURES_SRC) / "sh600519" / "factor.day.bin")
    si_out, og = read_bin(Path(FEATURES_DST) / "sh600519" / "overnight_gap.day.bin")

    assert si_out == si_close                 # day-cal 对齐
    assert og.size == close_d.size            # 长度 == 日频 close（每日一个值）
    assert np.isnan(og[0])                    # 首日无昨收 → NaN

    raw_open = open_d.astype(np.float64) / factor_d.astype(np.float64)
    raw_close = close_d.astype(np.float64) / factor_d.astype(np.float64)
    prev = np.full_like(raw_close, np.nan)
    prev[1:] = raw_close[:-1]
    expected = raw_open / prev - 1.0
    m = np.isfinite(expected) & np.isfinite(og)
    assert m.sum() > 100, f"only {m.sum()} finite days for overnight_gap cross-check"
    assert np.nanmax(np.abs(expected[m] - og[m])) < 1e-5, (
        f"overnight_gap max abs diff = {np.nanmax(np.abs(expected[m] - og[m]))}")

    # 不复权口径自洽：除权日（factor 跳变）overnight_gap ≠ 后复权口径 ($open/Ref($close,1)-1)。
    # 后复权口径会把除权缺口抹平；这里验证两者在 factor 变化日确有差异（口径正确的副作用）。
    adj_open = open_d.astype(np.float64)      # 后复权 open（qlib bin 原值）
    adj_prev_close = np.full_like(close_d.astype(np.float64), np.nan)
    adj_prev_close[1:] = close_d[:-1].astype(np.float64)   # 后复权 prev close
    adj_gap = adj_open / adj_prev_close - 1.0
    factor_changed = np.abs(np.diff(factor_d.astype(np.float64))) > 1e-9
    factor_changed = np.concatenate(([False], factor_changed))   # 对齐到 T
    diff_mask = factor_changed & np.isfinite(og) & np.isfinite(adj_gap)
    # 至少有一个除权日，且不复权与后复权口径在该日确实不同（证明口径选择有实际后果）。
    if diff_mask.sum() > 0:
        assert np.nanmax(np.abs(og[diff_mask] - adj_gap[diff_mask])) > 1e-4, (
            "overnight_gap 与后复权口径在除权日无差异 → 口径选择未生效")
