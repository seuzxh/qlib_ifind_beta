"""Low-level read/write for qlib ``.day.bin`` feature files.

Binary format (pyqlib ``FileFeatureStorage``, source-confirmed at
``qlib/data/storage/file_storage.py``):

* dtype: little-endian float32 (``<f4``)
* first 4 bytes: ``start_index`` — the calendar row where this instrument's data
  begins (stored as float32; exact for indices < 2**24, our ~6419-day calendar
  is well within range).
* subsequent 4-byte values: one per calendar day, aligned from ``start_index``.

qlib 0.9.7 ships no ``DumpBinAll`` helper, so we read/write the raw layout
directly. Round-trips byte-for-byte with qlib's own ``write``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DTYPE = np.dtype("<f4")


def read_bin(path) -> tuple[int | None, np.ndarray]:
    """Return ``(start_index, values)``.

    ``values`` is a float32 array of length = number of stored days, aligned to
    the calendar starting at ``start_index``. ``(None, empty)`` if the file is
    empty/missing-handled-by-caller.
    """
    raw = np.fromfile(path, dtype=DTYPE)
    if raw.size == 0:
        return None, np.array([], dtype=DTYPE)
    return int(raw[0]), raw[1:]


def write_bin(path, start_index: int, values) -> None:
    """Write ``values`` (aligned to calendar from ``start_index``). Overwrites."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=np.float32)
    arr = np.empty(values.size + 1, dtype=DTYPE)
    arr[0] = start_index
    arr[1:] = values
    arr.tofile(path)


def bin_length(path) -> int:
    """Number of data points stored (``filesize // 4 - 1``); 0 if absent."""
    p = Path(path)
    if not p.exists():
        return 0
    return p.stat().st_size // DTYPE.itemsize - 1
