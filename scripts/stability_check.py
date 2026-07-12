"""Sub-period stability check for position sizing candidates."""
from __future__ import annotations
import os, sys
from pathlib import Path
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scripts.compare_position_sizing import (
    _load_data, _bench_returns, _baseline_returns, _pool_mean_returns, _daily_ic,
    scheme_A1_vol_window, scheme_A3_excess_vol, scheme_B2_alpha_signal,
    scheme_B3_ic_zscore, scheme_C1_bench_momentum, scheme_C3_combined,
)


def metrics_sub(series):
    cum = np.cumprod(1 + series.values)
    max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
    ann_ret = (cum[-1] - 1) * 100
    calmar = ann_ret / abs(max_dd * 100) if max_dd < 0 else float("inf")
    ir = series.mean() / series.std() * np.sqrt(250) if series.std() > 0 else 0
    return ann_ret, ir, max_dd * 100, calmar


def main():
    pred, label, dates = _load_data()
    bench_ret = _bench_returns(dates)
    s_net = _baseline_returns(pred, label, dates)
    pool_mean = _pool_mean_returns(pred, label, dates)
    ic_series = _daily_ic(pred, label, dates)

    candidates = {"baseline": s_net}
    candidates["A1_w5"] = scheme_A1_vol_window(s_net, bench_ret, windows=(5,))["A1_w5"]
    candidates["A3"] = scheme_A3_excess_vol(s_net, bench_ret)["A3"]
    candidates["B2"] = scheme_B2_alpha_signal(s_net, bench_ret, pool_mean)["B2"]
    candidates["B3"] = scheme_B3_ic_zscore(s_net, bench_ret, ic_series)["B3"]
    candidates["C1"] = scheme_C1_bench_momentum(s_net, bench_ret)["C1"]
    candidates["C3"] = scheme_C3_combined(s_net, bench_ret)["C3"]

    n = len(s_net)
    p1_end = s_net.index[n // 3]
    p2_end = s_net.index[2 * n // 3]

    print("=== Sub-period stability check ===")
    print(f"P1: {s_net.index[0].date()} → {p1_end.date()}")
    print(f"P2: {p1_end.date()} → {p2_end.date()}")
    print(f"P3: {p2_end.date()} → {s_net.index[-1].date()}")
    print()
    print(f"{'Scheme':<12} {'P1 Calmar':>10} {'P2 Calmar':>10} {'P3 Calmar':>10} {'Min':>8} {'Stable':>7}")
    print("-" * 62)

    for name, series in candidates.items():
        p1 = metrics_sub(series.loc[:p1_end])
        p2 = metrics_sub(series.loc[p1_end:p2_end])
        p3 = metrics_sub(series.loc[p2_end:])
        min_c = min(p1[3], p2[3], p3[3])
        stable = "✓" if min_c > 0 else "✗"
        print(f"{name:<12} {p1[3]:>10.2f} {p2[3]:>10.2f} {p3[3]:>10.2f} {min_c:>8.2f} {stable:>7}")

    # Position distribution
    print("\n=== Position distribution ===")
    for name in ["A3", "B3", "C1", "C3"]:
        series = candidates[name]
        pos = (series / s_net).clip(lower=0, upper=1.5)
        print(f"{name}: mean={pos.mean():.3f} min={pos.min():.3f} <0.5: {(pos<0.5).mean()*100:.0f}% "
              f"=1: {(pos>=0.99).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
