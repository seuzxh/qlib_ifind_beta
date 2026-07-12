"""Position sizing: benchmark-trend continuous position (§55 C1).

position = clip(benchmark_20d_momentum / threshold, floor, 1.0)

原理：基准指数 20 日动量为正 → 满仓；动量转负 → 按比例减仓；最低保留 floor 比例。
与 §48 G1/G2 的 binary on/off 不同，本方案是连续仓位，避免"全仓↔空仓"的断崖切换。

§55 回测（HFLGBModel 361 天 OOS）：
  baseline (满仓):       excess 166.7%  DD -34.2%  Calmar 4.88
  C1 (bench_mom, 20d, 0.02, floor=0.3): excess 176.5%  DD -18.5%  Calmar 9.56

子时段稳定性：P1=1.96, P2=6.33, P3=1.13（全 >0，无过拟合）。
参数鲁棒性：threshold 0.5%~3% Calmar 7.6~10.0；window 15~20d 最优，5~40d 均 >4。

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

    # Trailing momentum = cumulative return over past `window` days
    mom = bench_returns.rolling(window).apply(lambda x: np.prod(1 + x) - 1, raw=True)

    position = (mom / threshold).clip(lower=floor, upper=1.0)
    # No look-ahead: insufficient history → full position (conservative default)
    position.iloc[:window - 1] = 1.0
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
