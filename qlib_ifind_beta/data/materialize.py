"""Materialize derived feature bins: ``$change``, ``$limit_up``, ``$limit_down``.

Why these exist — qlib's Exchange reads them via ``D.features`` to enforce A-share
price-limit (涨跌停) blocking:

* ``$change`` — unadjusted-price daily pct change. Exchange compares it against
  the limit threshold to decide buy/sell blocks (``$change >= $limit_up`` ⇒ 封涨停
  ⇒ 禁买; ``$change <= $limit_down`` ⇒ 封跌停 ⇒ 禁卖). Computed on **raw** price
  (``close/factor``) so ex-dividend days (where ``factor`` jumps) don't misfire.
* ``$limit_up`` / ``$limit_down`` — per-board constant threshold. A-share limit
  tiers differ by board (主板 ±10%, 创业板/科创板 ±20%, 北交所 ±30%), so a single
  global float can't model it; we materialize one constant per instrument.

The ``limit_threshold=["$change >= $limit_up", "$change <= $limit_down"]`` form is
qlib's native ``LT_TP_EXP`` (``exchange.py:46/258-292``) — no Exchange subclassing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .binio import read_bin, write_bin
from .config import FEATURES_SRC, FEATURES_DST, FREQ

# --- A-share board limit tiers ------------------------------------------------
# qlib convention: "just below" the nominal limit (0.095 not 0.10) so the float
# edge at the exact limit price doesn't leak (matches LT_FLT's ``.ge()``).
INDEX_LIMIT = (1.0, -1.0)  # indices are not traded -> never trigger


def board_limit(code: str) -> tuple[float, float]:
    """Return ``(limit_up, limit_down)`` by instrument code prefix.

    ST/*ST (±5% → 0.045) is NOT distinguished here — the 7-field data carries no
    ST flag. Upgrade path: pull iFinD ST status and turn these into per-day bins.
    """
    c = code.upper()
    # indices / non-tradeable benchmarks
    if c.startswith(("SH000", "SH88", "SZ399", "SH999")):
        return INDEX_LIMIT
    # 北交所 ±30%
    if c.startswith("BJ"):
        return 0.295, -0.295
    # 创业板 / 科创板 ±20%
    if c.startswith(("SZ300", "SZ301", "SH688", "SH689")):
        return 0.195, -0.195
    # 沪深主板 ±10% (default; B股 SH900/SZ200 also ±10%)
    return 0.095, -0.095


def compute_change(close: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Unadjusted-price daily pct change. Day 0 is NaN (no prior close)."""
    raw = close / factor  # restore unadjusted price (qlib $close is post-adjust)
    out = np.empty(raw.shape, dtype=np.float32)
    out[0] = np.nan
    out[1:] = raw[1:] / raw[:-1] - 1.0
    return out


def materialize_instrument(
    code: str,
    src_features: Path = FEATURES_SRC,
    dst_features: Path = FEATURES_DST,
    freq: str = FREQ,
) -> bool:
    """Read close/factor for ``code`` from src, write change/limit_up/limit_down to dst.

    Returns True on success, False if source close/factor bins are missing or
    misaligned (caller decides whether to warn).
    """
    src_dir = Path(src_features) / code.lower()
    close_path = src_dir / f"close.{freq}.bin"
    factor_path = src_dir / f"factor.{freq}.bin"
    if not close_path.exists() or not factor_path.exists():
        return False

    si_c, close = read_bin(close_path)
    si_f, factor = read_bin(factor_path)
    if close.size == 0 or si_c is None or si_c != si_f:
        return False

    dst_dir = Path(dst_features) / code.lower()
    write_bin(dst_dir / f"change.{freq}.bin", si_c, compute_change(close, factor))
    lu, ld = board_limit(code)
    n = close.size
    write_bin(dst_dir / f"limit_up.{freq}.bin", si_c, np.full(n, lu, dtype=np.float32))
    write_bin(dst_dir / f"limit_down.{freq}.bin", si_c, np.full(n, ld, dtype=np.float32))
    return True
