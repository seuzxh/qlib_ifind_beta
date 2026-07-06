"""Materialize 14 minute factors + $price_941 as day.bin into the overlay.

Reads cn_data_1min 1min bins (close/open/high/low/volume) per stock, slices each
trading day's morning window (slots 1-11 = 09:31-09:41: 10 feature bars slots 1-10
+ 1 buy bar slot 11), computes the 14 factors + 09:41 close, and writes them as
day.bin aligned to qlib_data's day
calendar (same start_index and length as the stock's daily close.bin, so
$price_941[T] row-aligns with $close[T]).

vol_vs_yest denominator = previous min-cal trading day's TOTAL minute volume / REAL_BARS_PER_DAY
(= 240 real bars/day; computed from cn_data_1min itself — minute volume on BOTH sides). The
minute-vs-daily volume unit mismatch (per-stock ratio 1.0-192.7, ≈ cumulative
adjustment factor: 科创板≈1, 茅台 5.84, 平安银行 192.7) ruled out the prior
qlib_data daily-volume denominator.

Vectorized per stock: scatter the 1min bin onto the global (day, slot) calendar
grid by absolute start_index, take morning slots 1-11 (09:31-09:41) →
(n_min_days, 11), compute factors column-wise, scatter into the day-aligned
output. Calendar-grid alignment is robust to the dataset's UNIVERSAL slot-0 NaN
(every day's 09:30 bar is NaN pool-wide, 0/604 non-NaN — probe 2026-07-06) and
to stock-local holes; missing cells become NaN. A naive reshape(242) is
off-by-one. See spec §物化架构.

Run: see scripts/materialize_minute.py (full universe) or call
materialize_minute_instrument(code) directly.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from .binio import read_bin, write_bin
from .config import (
    BUY_SLOT, DAY_CAL, FEATURES_1MIN_SRC, FEATURES_DST, FEATURES_SRC, FREQ,
    FIRST_FEATURE_SLOT, MIN_CAL, MINUTE_DEAL_PRICE_FIELD, MINUTE_FACTOR_FIELDS,
    REAL_BARS_PER_DAY, SLOTS_PER_DAY,
)

_1MIN_FIELDS = ("close", "open", "high", "low", "volume")
# Morning window: slots FIRST_FEATURE_SLOT..BUY_SLOT inclusive = 11 calendar rows.
_MORNING_WINDOW = BUY_SLOT - FIRST_FEATURE_SLOT + 1   # 11
_cal_cache: dict = {}


def _load_min_calendar():
    """Return (min_dates_ord, min_slots) as int arrays over the 1min calendar.

    min_dates_ord[i] = date.toordinal() of 1min row i; min_slots[i] = slot-within-day.
    Cached (1.55M rows, ~once per process).
    """
    if "min" not in _cal_cache:
        dts = []
        with open(MIN_CAL) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    dts.append(datetime.strptime(line, "%Y-%m-%d %H:%M:%S"))
        dates = np.empty(len(dts), dtype=np.int64)
        slots = np.empty(len(dts), dtype=np.int16)
        cur_date = None
        slot = 0
        for i, dt in enumerate(dts):
            if dt.date() != cur_date:
                cur_date = dt.date()
                slot = 0
            dates[i] = dt.toordinal()
            slots[i] = slot
            slot += 1
        _cal_cache["min"] = (dates, slots)
    return _cal_cache["min"]


def _load_day_calendar_lookup():
    """Return (n_days, date_to_row) where date_to_row[ordinal] = day-calendar row.

    date_to_row is a numpy int32 array indexed by date.toordinal(); -1 if the date
    is not a trading day. Sized to the max ordinal in day.txt.
    """
    if "day" not in _cal_cache:
        with open(DAY_CAL) as fp:
            rows = [line.strip() for line in fp if line.strip()]
        ordinals = [
            datetime.strptime(s, "%Y-%m-%d").toordinal() for s in rows
        ]
        max_ord = max(ordinals)
        lookup = np.full(max_ord + 1, -1, dtype=np.int32)
        for i, ord_ in enumerate(ordinals):
            lookup[ord_] = i
        _cal_cache["day"] = (len(rows), lookup)
    return _cal_cache["day"]


def _read_1min_fields(code: str):
    """Return (start_index, dict field→float32 array) or (None, None) if missing/misaligned."""
    d = Path(FEATURES_1MIN_SRC) / code.lower()
    arrs = {}
    si = None
    for f in _1MIN_FIELDS:
        p = d / f"{f}.1min.bin"
        if not p.exists():
            return None, None
        s, a = read_bin(p)
        if a.size == 0 or s is None:
            return None, None
        arrs[f] = a
        if si is None:
            si = s
        elif s != si:
            return None, None   # fields disagree on start_index
    # Length-consistency guard: a truncated bin (e.g. partial write) would scatter
    # with mismatched sizes → silent field misalignment. Treat as "no minute data".
    if len({a.size for a in arrs.values()}) > 1:
        return None, None
    return si, arrs


def materialize_minute_instrument(code: str) -> bool:
    """Read 1min + daily bins for `code`, write 14 factor day.bins + price_941.day.bin.

    Returns True on success; False if the 1min source is missing/misaligned, or the
    daily close bin is missing/misaligned (caller treats as "no minute data for
    this stock" — qlib reads NaN).
    """
    si_m, m = _read_1min_fields(code)
    if si_m is None:
        return False

    # daily close.bin = alignment reference (output start_index + length). The
    # previous implementation also read daily volume.bin for vol_vs_yest, but
    # that crossed minute↔daily units — now both sides use cn_data_1min minute
    # volume, so the daily-volume read is dropped.
    ddir = Path(FEATURES_SRC) / code.lower()
    si_dc, close_d = read_bin(ddir / f"close.{FREQ}.bin")
    if close_d.size == 0 or si_dc is None:
        return False

    # Calendar-grid alignment: scatter each 1min bin onto the global (day, slot)
    # grid by absolute calendar row, then take morning slots FIRST_FEATURE_SLOT
    # (1) .. BUY_SLOT (11) → 09:31-09:41. Robust to the dataset's UNIVERSAL
    # slot-0 NaN (every day's 09:30 bar is NaN pool-wide, probe 2026-07-06) and
    # to stock-local holes; missing cells become NaN. A naive reshape(242) is
    # off-by-one. See spec §物化架构.
    min_dates, min_slots = _load_min_calendar()
    _, date_to_row = _load_day_calendar_lookup()

    morning_rows = np.where(
        (min_slots >= FIRST_FEATURE_SLOT) & (min_slots <= BUY_SLOT)
    )[0]   # 11 rows/day (slots 1-11), ordered
    assert morning_rows.size % _MORNING_WINDOW == 0, (
        f"1min calendar malformed: morning_rows={morning_rows.size} not divisible "
        f"by {_MORNING_WINDOW} (slots/day={SLOTS_PER_DAY})"
    )
    n_min_days = morning_rows.size // _MORNING_WINDOW   # total trading days in 1min cal

    def morning2d(field):
        """Stock's morning bars on the global grid → (n_min_days, _MORNING_WINDOW).

        Column j corresponds to calendar slot FIRST_FEATURE_SLOT+j (1..11 =
        09:31-09:41). NaN where the stock has no bar at that calendar row
        (universal slot-0 NaN doesn't enter — window starts at slot 1 — but
        suspensions and pre-listing still produce NaN).
        """
        arr = m[field]
        flat = np.full(morning_rows.size, np.nan, dtype=np.float64)
        valid = (morning_rows >= si_m) & (morning_rows < si_m + arr.size)
        flat[valid] = arr[morning_rows[valid] - si_m].astype(np.float64)
        return flat.reshape(n_min_days, _MORNING_WINDOW)

    c, o, h, l, v = (morning2d("close"), morning2d("open"), morning2d("high"),
                     morning2d("low"), morning2d("volume"))

    # vol_vs_yest denominator (minute-only both sides): per-day FULL-DAY minute
    # volume from cn_data_1min, summed across all real (non-NaN) slots that day.
    # Computed from the scattered grid (uses the WHOLE volume bin, not just the
    # morning window) → per-min-day total. Then shift by 1 min-cal trading day
    # (first day → NaN). NaN-safe: NaN placeholder slots contribute 0 to the sum.
    n_total_min_rows = min_slots.size
    vol_bin = m["volume"]
    bin_rows = si_m + np.arange(vol_bin.size, dtype=np.int64)
    in_range = (bin_rows >= 0) & (bin_rows < n_total_min_rows)
    day_idx_all = np.where(in_range, bin_rows, 0) // SLOTS_PER_DAY
    in_range &= day_idx_all < n_min_days
    src_vals = vol_bin[in_range].astype(np.float64)
    src_vals = np.where(np.isfinite(src_vals), src_vals, 0.0)
    full_day_vol = np.zeros(n_min_days, dtype=np.float64)
    np.add.at(full_day_vol, day_idx_all[in_range], src_vals)
    prev_day_full_vol = np.full(n_min_days, np.nan, dtype=np.float64)
    if n_min_days > 1:
        prev_day_full_vol[1:] = full_day_vol[:-1]

    # per-min-day date → day-calendar row → output-relative index.
    # morning_rows[::_MORNING_WINDOW] = each day's slot-FIRST_FEATURE_SLOT (slot 1)
    # cal row (date is constant in-day). clip ordinal before fancy-index
    # (defensive: a stray 1min date outside the day calendar — e.g. the extra
    # 2026-07-03 min-cal day not in day-cal — must not crash the whole stock;
    # mark it invalid instead).
    first_rows = morning_rows[::_MORNING_WINDOW]
    day_dates = min_dates[first_rows]
    max_ord = date_to_row.size - 1
    safe = np.clip(day_dates, 0, max_ord)
    oob = (day_dates < 0) | (day_dates > max_ord)
    day_rows = np.where(oob, -1, date_to_row[safe])   # -1 where not a trading day
    rel = day_rows - si_dc

    # vectorized factors over n_min_days; features use columns 0-9 (slots 1-10),
    # buy price uses column 10 (slot 11). Mirror compute_day_factors exactly.
    fac = {}
    n = c.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        # A. startup momentum
        fac["startup_mom_1m"] = c[:, 9] / c[:, 8] - 1.0
        fac["startup_mom_3m"] = c[:, 9] / c[:, 6] - 1.0
        fac["startup_mom_5m"] = c[:, 9] / c[:, 4] - 1.0
        fac["startup_total"] = c[:, 9] / o[:, 0] - 1.0
        # B. acceleration
        fac["accel_1m"] = (c[:, 9] / o[:, 9] - 1.0) - (c[:, 0] / o[:, 0] - 1.0)
        fac["accel_3m"] = (c[:, 9] / o[:, 7] - 1.0) - (c[:, 1] / o[:, 0] - 1.0)
        fac["accel_5m"] = (c[:, 9] / o[:, 5] - 1.0) - (c[:, 3] / o[:, 0] - 1.0)
        # C. close position in window
        fac["close_pos_1m"] = (c[:, 9] - l[:, 9]) / (h[:, 9] - l[:, 9])
        h3, l3 = h[:, 7:10].max(axis=1), l[:, 7:10].min(axis=1)
        fac["close_pos_3m"] = (c[:, 9] - l3) / (h3 - l3)
        h5, l5 = h[:, 5:10].max(axis=1), l[:, 5:10].min(axis=1)
        fac["close_pos_5m"] = (c[:, 9] - l5) / (h5 - l5)
        # D. volume ratio (NaN where front-seg mean ≤ 0 — match compute_day_factors guard)
        d1 = v[:, 0:9].mean(axis=1)
        fac["vol_ratio_1m"] = np.full(n, np.nan)
        np.divide(v[:, 9], d1, out=fac["vol_ratio_1m"], where=(d1 > 0))
        d3 = v[:, 0:2].mean(axis=1)
        fac["vol_ratio_3m"] = np.full(n, np.nan)
        np.divide(v[:, 7:10].mean(axis=1), d3, out=fac["vol_ratio_3m"], where=(d3 > 0))
        d5 = v[:, 0:4].mean(axis=1)
        fac["vol_ratio_5m"] = np.full(n, np.nan)
        np.divide(v[:, 5:10].mean(axis=1), d5, out=fac["vol_ratio_5m"], where=(d5 > 0))
        # E. cross-day volume (minute-only): prev min-cal day's full-day vol / REAL_BARS_PER_DAY
        prev_safe = np.where(prev_day_full_vol > 0, prev_day_full_vol, np.nan)
        fac["vol_vs_yest"] = v[:, 0:10].sum(axis=1) / (prev_safe / float(REAL_BARS_PER_DAY))
    price_941 = c[:, 10]

    # scatter into day-aligned output (length = daily close.bin length)
    n_out = close_d.size
    out = {name: np.full(n_out, np.nan, dtype=np.float32) for name in MINUTE_FACTOR_FIELDS}
    out_p941 = np.full(n_out, np.nan, dtype=np.float32)
    valid = (rel >= 0) & (rel < n_out) & (day_rows >= 0)
    rel_v = rel[valid]
    for name in MINUTE_FACTOR_FIELDS:
        out[name][rel_v] = fac[name][valid].astype(np.float32)
    out_p941[rel_v] = price_941[valid].astype(np.float32)

    dst_dir = Path(FEATURES_DST) / code.lower()
    for name in MINUTE_FACTOR_FIELDS:
        write_bin(dst_dir / f"{name}.{FREQ}.bin", si_dc, out[name])
    write_bin(dst_dir / f"{MINUTE_DEAL_PRICE_FIELD}.{FREQ}.bin", si_dc, out_p941)
    return True
