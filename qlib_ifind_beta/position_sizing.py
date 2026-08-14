"""Position sizing: benchmark-trend continuous position (§55 C1).

position = clip(benchmark_20d_momentum / threshold, floor, 1.0)

原理：基准指数 20 日动量为正 → 满仓；动量转负 → 按比例减仓；最低保留 floor 比例。
与 §48 G1/G2 的 binary on/off 不同，本方案是连续仓位，避免"全仓↔空仓"的断崖切换。

2026-07-16 审计结论：旧 §55 批量回测使用了当日收盘后才能知道的 benchmark
return 来决定同日 09:41 仓位，旧 Calmar 9.56 及参数鲁棒性结论全部作废。修复后，
361 日快速代理中满仓基线（绝对累计 +166.7%、几何超额 +118.1%、标准 Calmar
2.85）优于 C1（绝对累计 +87.6%、几何超额 +53.4%、标准 Calmar 2.43）。
因此该模块保留为研究接口，但当前默认不启用 C1 overlay。

无 look-ahead：position[T] 只用 T-1 及之前的 benchmark 收益计算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 20
DEFAULT_THRESHOLD = 0.02   # 2% per 20 days → full position when benchmark trending up
DEFAULT_FLOOR = 0.3        # never go below 30% invested


def compute_benchmark_position(
    bench_returns: pd.Series,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    floor: float = DEFAULT_FLOOR,
) -> pd.Series:
    """Compute position scale [floor, 1.0] from benchmark trailing momentum.

    Parameters
    ----------
    bench_returns : pd.Series
        Daily benchmark returns (e.g. SH000300 pct_change), indexed by date.
    window : int
        Trailing window for momentum calculation (trading days).
    threshold : float
        Momentum level at which position = 1.0 (full investment).
        Momentum > threshold → position = 1.0.
    floor : float
        Minimum position (never fully exit).

    Returns
    -------
    pd.Series
        Position scale in [floor, 1.0], same index as bench_returns.
        First `window-1` entries are 1.0 (insufficient history → full position).
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}")

    # The position for row T is decided at 09:41 on T.  The benchmark's T-day
    # close return is not known yet, therefore the rolling window must end at
    # T-1.  Keeping the shift here (rather than at callers) makes the batch API
    # consistent with compute_position_for_day and prevents accidental leakage.
    known_returns = bench_returns.shift(1)
    mom = known_returns.rolling(window).apply(lambda x: np.prod(1 + x) - 1, raw=True)

    position = (mom / threshold).clip(lower=floor, upper=1.0)
    # No look-ahead: insufficient T-1 history → full position.
    position.iloc[:window] = 1.0
    position = position.fillna(1.0)

    return position


def compute_position_for_day(
    bench_returns: pd.Series,
    target_date,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
    floor: float = DEFAULT_FLOOR,
) -> float:
    """Compute position for a single target date (for live/inference use).

    Uses only bench_returns up to target_date (exclusive — target_date's return
    is not yet known at decision time).

    Parameters
    ----------
    bench_returns : pd.Series
        Full benchmark return series (will be sliced to [:target_date]).
    target_date : date-like
        The date to compute position for.
    window, threshold, floor : see compute_benchmark_position.

    Returns
    -------
    float
        Position scale in [floor, 1.0].
    """
    # Use only data strictly before target_date (no look-ahead)
    past = bench_returns.loc[:target_date].iloc[:-1]
    if len(past) < window:
        return 1.0  # insufficient history → full position

    mom = np.prod(1 + past.iloc[-window:].values) - 1
    return float(np.clip(mom / threshold, floor, 1.0))
