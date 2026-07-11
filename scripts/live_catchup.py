"""模拟盘跟踪 — 批量补跑入口。

遍历 [start, end] 区间内的每个交易日，逐日调用 live_forward.run(date)：
  [1] universe 增量 → [2] 物化 → [3] 推理 → [4] 结算 → [5] NAV

首日 skip_universe=False（需要拉 iFinD 快照），后续 skip_universe=True（已刷新到位）。
materialize 始终跑（池内增量，幂等）。

Run:
  # 补跑 07-03 ~ 07-10（6 个交易日）
  conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
      scripts/live_catchup.py --start 2026-07-03 --end 2026-07-10
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse


def run(start: str, end: str) -> None:
    # 复用 live_forward 的 run()，避免逻辑重复
    from qlib.data import D
    import qlib
    from qlib_ifind_beta.config import OVERLAY_ROOT
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")

    from scripts.live_forward import run as run_day

    cal = D.calendar(start_time=start, end_time=end, freq="day")
    dates = [d.strftime("%Y-%m-%d") for d in cal]
    print(f"▶ live_catchup: {len(dates)} 个交易日 [{dates[0]} ~ {dates[-1]}]")

    for i, date in enumerate(dates):
        # 首日跑 universe 增量（拉 iFinD 快照），后续跳过（已刷新）
        skip_universe = i > 0
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(dates)}] {date}  (skip_universe={skip_universe})")
        print(f"{'='*60}")
        run_day(date, skip_universe=skip_universe, skip_materialize=False)

    print(f"\n✅ live_catchup 完成：{len(dates)} 个交易日")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="模拟盘批量补跑")
    p.add_argument("--start", required=True, help="起始交易日 (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="截止交易日 (YYYY-MM-DD)")
    a = p.parse_args()
    run(a.start, a.end)
