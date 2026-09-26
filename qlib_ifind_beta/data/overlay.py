"""Build the minimal symlink-farm overlay over readonly ``qlib_data``.

qlib_data is read-only; we cannot drop derived bins into it. Instead we expose a
fresh provider_uri (``data/qlib_root/``) that symlinks the readonly calendars /
instruments / per-stock base bins and adds our own derived bins + market files
alongside them.

Layout under ``OVERLAY_ROOT``::

    calendars/              -> symlink to qlib_data/calendars (whole dir)
    instruments/             real dir; all.txt -> symlink; highbeta883926.txt real
    features/<code>/         real dir per stock; 7 base bins symlinked from qlib_data
                             + 3 derived bins (change/limit_up/limit_down) real
    features/sh883926/       real dir; 7 base bins dumped from iFinD (no symlink)
"""
from __future__ import annotations

import os
from pathlib import Path

from .config import (
    BASE_FIELDS,
    CALENDAR_DST,
    FEATURES_DST,
    FEATURES_SRC,
    FREQ,
    INSTRUMENTS_DST,
    QLIB_DATA,
)


def _symlink(src: Path, dst: Path) -> None:
    """Idempotent symlink: replace any existing node at dst."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)


def link_calendars() -> None:
    """Symlink the whole readonly calendars dir (day/1min/5min)."""
    _symlink(QLIB_DATA / "calendars", CALENDAR_DST)


def link_instruments(market_files: dict[str, str] | None = None) -> None:
    """Set up instruments dir: symlink all.txt from qlib_data + write our market files.

    ``market_files`` maps filename → TSV text content (e.g. ``highbeta883926.txt``).
    """
    INSTRUMENTS_DST.mkdir(parents=True, exist_ok=True)
    src = QLIB_DATA / "instruments" / "all.txt"
    if src.exists():
        _symlink(src, INSTRUMENTS_DST / "all.txt")
    for name, text in (market_files or {}).items():
        (INSTRUMENTS_DST / name).write_text(text)


def write_market_file(market: str, records) -> Path:
    """Write ``instruments/<market>.txt`` as TSV (code\\tstart\\tend), no header."""
    INSTRUMENTS_DST.mkdir(parents=True, exist_ok=True)
    path = INSTRUMENTS_DST / f"{market.lower()}.txt"
    lines = [f"{c}\t{s}\t{e}" for (c, s, e) in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


def link_stock(code: str, freq: str = FREQ) -> Path:
    """Create overlay feature dir for ``code`` with 7 base bins symlinked from qlib_data.

    Returns the overlay dir (caller then drops derived bins into it).
    """
    c = code.lower()
    src_dir = FEATURES_SRC / c
    dst_dir = FEATURES_DST / c
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in BASE_FIELDS:
        src = src_dir / f"{f}.{freq}.bin"
        if src.exists():
            _symlink(src, dst_dir / f"{f}.{freq}.bin")
    return dst_dir


def ensure_stock_dir(code: str) -> Path:
    """Create (empty) overlay feature dir for a non-symlinked instrument (e.g. SH883926)."""
    dst_dir = FEATURES_DST / code.lower()
    dst_dir.mkdir(parents=True, exist_ok=True)
    return dst_dir


def link_benchmark(code: str) -> Path:
    """Symlink a benchmark's whole feature dir from qlib_data (e.g. SH000300).

    Benchmarks (CSI300/500/1000 etc.) live in qlib_data with clean 26y bins; we
    reuse them read-only. Different from ``link_stock`` (per-file) because we want
    the entire dir verbatim.
    """
    src = FEATURES_SRC / code.lower()
    dst = FEATURES_DST / code.lower()
    if not src.exists():
        raise FileNotFoundError(f"benchmark {code} not in qlib_data: {src}")
    _symlink(src, dst)
    return dst
