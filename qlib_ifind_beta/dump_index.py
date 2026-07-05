"""Dump the SH883926 benchmark index (883926.TI) from iFinD into qlib day-bins.

qlib_data has no 883xxx index bins, so we fetch 883926.TI via the iFinD
``history_data`` API and write the 7 base fields (open/high/low/close/volume/
vwap + factor=1.0) under ``features/sh883926/`` aligned to the qlib day calendar.

**Direct dump, no reconstruction.** Per the project decision (2026-07-05):
883926 is a published 同花顺 index; iFinD returns its values as-is, and we dump
them faithfully — no pct_chg rebuild, no return capping. Whatever the source
returns is what lands in the bins.

⚠️ **Source field note (read-only finding, 2026-07-05)**: iFinD's ``close`` for
883xxx indices drifts ~2800x over 4y then snaps back, while ``vwap`` stays in a
normal 4~88 band — close and vwap are decoupled by ~4 orders of magnitude. This
is a field-level property of the source; how the benchmark consumes it (close vs
vwap vs alternative) is a separate, user-decided config matter. This module
just dumps what the source gives.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .binio import write_bin
from .config import DAY_CAL, FEATURES_DST, FREQ, INDEX_CODE_IFIND, INDEX_CODE_QLIB
from . import ifind

# The 7 base fields written for the index. factor is constant 1.0 (no ex-rights
# for an index); the other 6 come straight from the iFinD source.
INDEX_OHLCVW_FIELDS = ("open", "high", "low", "close", "volume", "vwap")


def load_day_calendar(path: Path = DAY_CAL) -> list[str]:
    """Load ``calendars/day.txt`` → list of 'YYYY-MM-DD' (one per trading day)."""
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def _map_to_calendar(dates: list[str], values: np.ndarray, calendar: list[str]
                     ) -> tuple[int | None, np.ndarray | None]:
    """Map (date, value) pairs onto qlib calendar → (start_index, values from si)."""
    idx_map = {d: i for i, d in enumerate(calendar)}
    present = [(idx_map[d], float(v)) for d, v in zip(dates, values)
               if d in idx_map and pd.notna(v)]
    if not present:
        return None, None
    present.sort()
    start, end = present[0][0], present[-1][0]
    arr = np.full(end - start + 1, np.nan, dtype=np.float32)
    for i, v in present:
        arr[i - start] = v
    return start, arr


def dump_index(start: str, end: str, code_qlib: str = INDEX_CODE_QLIB,
               code_ifind: str = INDEX_CODE_IFIND,
               dst_features: Path = FEATURES_DST, freq: str = FREQ) -> pd.DataFrame:
    """Fetch 883926.TI history and dump the 7 base fields verbatim to day-bins.

    Writes under ``features/<code_qlib.lower()>/``: open/high/low/close/volume/vwap
    (from source) + factor (constant 1.0). All aligned to the qlib day calendar.
    Returns the raw fetched DataFrame.
    """
    df = ifind.fetch_history_data(code_ifind, start, end)
    if df.empty:
        raise RuntimeError(f"iFinD returned no rows for {code_ifind} {start}..{end}")
    df = df.sort_values("date").reset_index(drop=True)

    missing = [f for f in INDEX_OHLCVW_FIELDS if f not in df.columns]
    if missing:
        raise RuntimeError(f"history_data missing fields {missing} for {code_ifind}")

    calendar = load_day_calendar()
    dst_dir = Path(dst_features) / code_qlib.lower()
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 6 OHLCVW fields straight from source
    for fld in INDEX_OHLCVW_FIELDS:
        vals = pd.to_numeric(df[fld], errors="coerce").to_numpy(dtype=np.float64)
        si, arr = _map_to_calendar(df["date"].tolist(), vals, calendar)
        if si is None:
            raise RuntimeError(f"no calendar-aligned values for {code_ifind} {fld}")
        write_bin(dst_dir / f"{fld}.{freq}.bin", si, arr)

    # factor = 1.0 over the same span as close
    si_c, close_arr = _map_to_calendar(
        df["date"].tolist(),
        pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64),
        calendar,
    )
    write_bin(dst_dir / f"factor.{freq}.bin", si_c,
              np.full(close_arr.size, 1.0, dtype=np.float32))

    print(f"✓ {code_qlib}: dumped open/high/low/close/volume/vwap + factor(=1.0) "
          f"({len(close_arr)} days, si={si_c})")
    return df
