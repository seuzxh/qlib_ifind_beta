"""First-principles: pin the cn_data_1min 1min.bin format before materialization.

Guards four invariants the materialize layer depends on:
  1. start_index is a float32 calendar-row index (same FileFeatureStorage format as
     day.bin). If it were uint32-reinterpreted, the value would be ~1e9 and fail
     the `< len(calendar)` check.
  2. A stock's first stored 1min bar is day-aligned and lands at/near market open
     (9:30 or 9:31) — the dataset has a universal head-hole (every bin starts at
     9:31 of 2024-01-02), so the materialize layer aligns by absolute calendar
     row, NOT by reshape(242).
  3. Calendar slot 11 == 9:41 (the buy-price bar), anchored on the calendar grid —
     robust to the head-hole (the naive bin-offset slot 11 would land on 9:42).
  4. 242 slots per trading day.
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
    """Stock's first stored 1min bar is at/near market open (slot 0 or 1).

    The dataset has a universal head-hole: every stock's bin starts at calendar
    slot 1 (09:31) of 2024-01-02 — the 09:30 bar is missing pool-wide (see spec
    §数据基础). So the first stored bar is 09:31, not 09:30. The materialize layer
    must NOT assume the bin starts at slot 0; it aligns by absolute calendar row
    (calendar-grid alignment, Task 4). Both 09:30 and 09:31 are accepted here.
    """
    dts = _load_1min_cal()
    p = Path(FEATURES_1MIN_SRC) / "sh600519" / "close.1min.bin"
    si, arr = read_bin(p)
    first = dts[si]
    assert first.time() in (time(9, 30), time(9, 31)), f"first bar not open: {first}"
    # consecutive 1-min spacing within the first day
    assert (dts[si + 1] - dts[si]).total_seconds() == 60


def test_slot11_is_941():
    """Calendar slot 11 (buy-price bar) lands on 9:41 — spec §数据基础 slot map.

    Anchored on the CALENDAR, not the bin offset: derive the stock's in-day slot
    from si (si % 242), back up to that day's slot-0 row, then check slot 0+11.
    Robust to the universal head-hole (bin starts at slot 1, so the naive
    dts[si+11] would land on 9:42 — that's the bug the calendar-grid fix repairs).
    """
    dts = _load_1min_cal()
    p = Path(FEATURES_1MIN_SRC) / "sh600519" / "close.1min.bin"
    si, _ = read_bin(p)
    slot_of_si = si % SLOTS_PER_DAY              # 1 (dataset head-hole)
    day_slot0_row = si - slot_of_si              # calendar row of that day's 09:30
    slot11_time = dts[day_slot0_row + 11].time()
    assert slot11_time == time(9, 41), (
        f"calendar slot 11 = {slot11_time}, expected 9:41. "
        "If this fails, the spec slot-map is wrong — do NOT silently change; "
        "re-probe cn_data_1min and update spec + PRICE_941_SLOT."
    )


def test_slots_per_day_is_242():
    """242 rows per trading day in the 1min calendar (reshape precondition)."""
    dts = _load_1min_cal()
    # count rows in the first full trading day
    first_date = dts[0].date()
    count = sum(1 for d in dts if d.date() == first_date)
    assert count == SLOTS_PER_DAY, f"first day has {count} slots, expected {SLOTS_PER_DAY}"
