"""First-principles: pin the cn_data_1min 1min.bin format before materialization.

Guards four invariants the materialize layer depends on:
  1. start_index is a float32 calendar-row index (same FileFeatureStorage format as
     day.bin). If it were uint32-reinterpreted, the value would be ~1e9 and fail
     the `< len(calendar)` check.
  2. A stock's first stored 1min bar is day-aligned and lands at market open
     (9:31) — the dataset has 240 real bars/day (09:31~15:00, no NaN placeholders
     since 2026-07-11 rebuild), so the materialize layer aligns by absolute
     calendar row, NOT by reshape(240).
  3. Calendar slot 10 == 9:41 (the buy-price bar), anchored on the calendar grid.
  4. 240 slots per trading day.

History: cn_data_1min was rebuilt on 2026-07-11 from 242 slots (with slot 0 = 09:30
NaN and slot 121 = 13:00 NaN placeholders) to 240 pure real bars. Tests updated to
match the new format.
"""
import functools
from datetime import datetime, time
from pathlib import Path

from qlib_ifind_beta.binio import read_bin
from qlib_ifind_beta.config import FEATURES_1MIN_SRC, MIN_CAL, SLOTS_PER_DAY


@functools.lru_cache(maxsize=1)
def _load_1min_cal():
    dts = []
    with open(MIN_CAL) as fp:
        for line in fp:
            line = line.strip()
            if line:
                dts.append(datetime.strptime(line, "%Y-%m-%d %H:%M:%S"))
    return dts


def test_start_index_is_float32_calendar_row():
    """binio.read_bin (float32) yields a start_index inside the 1min calendar."""
    dts = _load_1min_cal()
    p = Path(FEATURES_1MIN_SRC) / "sh600519" / "close.1min.bin"
    assert p.exists(), f"missing test fixture {p}"
    si, arr = read_bin(p)
    assert si is not None and si >= 0
    assert si < len(dts), f"start_index {si} >= calendar len {len(dts)} (uint32 misread?)"
    assert arr.size + si <= len(dts), "bin overflows 1min calendar"


def test_first_bar_day_aligned_near_open():
    """Stock's first stored 1min bar is at market open (slot 0 = 09:31).

    The dataset has 240 real bars/day (09:31~15:00, no NaN placeholders since
    2026-07-11 rebuild). The first stored bar of every bin is at slot 0 (09:31)
    of its first trading day.
    """
    dts = _load_1min_cal()
    p = Path(FEATURES_1MIN_SRC) / "sh600519" / "close.1min.bin"
    si, arr = read_bin(p)
    first = dts[si]
    assert first.time() == time(9, 31), f"first bar not 09:31: {first}"
    # consecutive 1-min spacing within the first day
    assert (dts[si + 1] - dts[si]).total_seconds() == 60


def test_slot10_is_941():
    """Calendar slot 10 (buy-price bar) lands on 9:41.

    With 240 slots/day (09:31~15:00), slot 0 = 09:31, slot 10 = 09:41.
    """
    dts = _load_1min_cal()
    p = Path(FEATURES_1MIN_SRC) / "sh600519" / "close.1min.bin"
    si, _ = read_bin(p)
    slot_of_si = si % SLOTS_PER_DAY
    day_slot0_row = si - slot_of_si
    slot10_time = dts[day_slot0_row + 10].time()
    assert slot10_time == time(9, 41), (
        f"calendar slot 10 = {slot10_time}, expected 9:41. "
        "If this fails, the spec slot-map is wrong — do NOT silently change; "
        "re-probe cn_data_1min and update spec + BUY_SLOT."
    )


def test_slots_per_day_is_240():
    """240 rows per trading day in the 1min calendar (reshape precondition)."""
    dts = _load_1min_cal()
    # count rows in the first full trading day
    first_date = dts[0].date()
    count = sum(1 for d in dts if d.date() == first_date)
    assert count == SLOTS_PER_DAY, f"first day has {count} slots, expected {SLOTS_PER_DAY}"
