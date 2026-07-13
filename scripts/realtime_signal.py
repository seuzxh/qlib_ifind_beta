"""Real-time signal generation CLI entry point.

Usage:
  # At 9:41 AM each trading day:
  conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
      scripts/realtime_signal.py --date 2026-07-13

  # Dry-run (use last available day's data from cn_data_1min, no kline fetch):
  conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
      scripts/realtime_signal.py --date 2026-07-10 --dry-run

Output:
  - Console: top10 signal list with scores + prices
  - CSV: data/realtime_signals/YYYY-MM-DD.csv
"""
from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Real-time signal generation")
    parser.add_argument("--date", required=True, help="Target date YYYY-MM-DD")
    parser.add_argument("--topk", type=int, default=10, help="Top K stocks")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Parallel fetch workers")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use cn_data_1min data instead of real-time fetch "
                             "(for testing on non-trading days)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.dry_run:
        _dry_run(args.date, args.topk)
    else:
        _realtime_run(args.date, args.topk, args.max_workers)


def _realtime_run(date: str, topk: int, max_workers: int):
    """Real-time mode: fetch from kline-fetcher."""
    from qlib_ifind_beta.realtime.signal import generate_realtime_signal

    result = generate_realtime_signal(date, topk=topk, max_workers=max_workers)

    _print_result(result, date)
    _save_csv(result, date)


def _dry_run(date: str, topk: int):
    """Dry-run mode: use existing cn_data_1min data (no kline fetch).

    This directly calls predict_day after ensuring factors are materialized
    from the already-synced cn_data_1min bins.
    """
    logging.info(f"▶ Dry-run mode for {date}: using existing cn_data_1min data")

    # In dry-run, we use the existing materialization pipeline
    # (factors already in overlay bins from materialize_minute)
    from qlib_ifind_beta.live.inference import predict_day

    result = predict_day(date, topk=topk)

    _print_result(result, date)
    _save_csv(result, date)


def _print_result(result: dict, date: str):
    """Print signal to console."""
    print("\n" + "=" * 72)
    print(f"  🎯 实时信号 {date}")
    print("=" * 72)

    topk = result.get("topk", [])
    if not topk:
        print("  ⚠️ 无信号（候选不足或数据缺失）")
        return

    print(f"\n  Top {len(topk)} 标的池（已剔除封涨停）:\n")
    print(f"  {'#':<3} {'代码':<12} {'Score':>10} {'9:41价':>10} {'涨跌幅':>8} {'涨停':>6} {'跌停':>6}")
    print(f"  {'-'*62}")

    for i, s in enumerate(topk):
        code = s["code"]
        score = s.get("score", 0)
        price = s.get("price_941", 0)
        change = s.get("change_941", 0)
        limit_up = s.get("limit_up", 0)
        limit_down = s.get("limit_down", 0)
        print(f"  {i+1:<3} {code:<12} {score:>10.4f} {price:>10.2f} "
              f"{change*100:>7.2f}% {limit_up*100:>5.1f}% {limit_down*100:>5.1f}%")

    n_candidates = result.get("n_candidates", 0)
    print(f"\n  候选总数: {n_candidates}，入选: {len(topk)}")
    print("=" * 72)


def _save_csv(result: dict, date: str):
    """Save signal to CSV."""
    import pandas as pd

    out_dir = Path("data/realtime_signals")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}.csv"

    rows = []
    for i, s in enumerate(result.get("topk", [])):
        rows.append({
            "rank": i + 1,
            "code": s["code"],
            "score": s.get("score", 0),
            "price_941": s.get("price_941", 0),
            "change_941": s.get("change_941", 0),
            "limit_up": s.get("limit_up", 0),
            "limit_down": s.get("limit_down", 0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n  💾 已保存: {out_path}")


if __name__ == "__main__":
    main()
