"""Pre-market universe snapshot updater (cron entry, universe-only).

Fetches missing 883926 constituent snapshots up to ``--end`` (incremental,
cache-resumable) and rewrites ``instruments/highbeta883926.txt`` from the full
cache.  Deliberately does NOT re-materialize factors or touch anything else —
materialization stays a manual ``build_overlay`` step by project decision
(2026-09-11).

The qlib day calendar only gains day T after the 15:30 market-data cron, so an
08:30 run's calendar walk stops at T-1.  Since p03473 serves the T-day snapshot
pre-market (verified 2026-07-10), this script additionally fetches ``--end``
itself when it is a weekday and not yet cached (guarded by the exact-100 row
check so holidays/weekends are skipped silently).

Run:  conda run -n qlib_ifind_beta python -m scripts.update_universe [--end DATE]
Cron: scripts/cron_update_universe.sh (08:30, Mon-Fri)
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qlib_ifind_beta import universe
from qlib_ifind_beta.config import INDEX_CODE_IFIND


def _cache_dates() -> set[str]:
    import pandas as pd
    if not universe.SNAPSHOT_CACHE.exists():
        return set()
    df = pd.read_csv(universe.SNAPSHOT_CACHE)
    return set(df["date"].astype(str).str[:10])


def _fetch_single_day(date: str) -> bool:
    """Fetch one day's snapshot directly; append to cache only on the exact-100 gate."""
    df = universe.fetch_constituents(INDEX_CODE_IFIND, iv_date=date.replace("-", ""))
    if len(df) != 100:
        print(f"  · {date}: {len(df)} rows (≠100, holiday/stale?) — skip")
        return False
    df = df.assign(date=date)[["date", "code_ifind", "code_qlib", "name"]]
    import pandas as pd
    cached = pd.read_csv(universe.SNAPSHOT_CACHE) if universe.SNAPSHOT_CACHE.exists() else pd.DataFrame()
    universe._flush_cache(universe.SNAPSHOT_CACHE, cached, [df])
    print(f"  · {date}: fetched 100 rows pre-market → cache")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="盘前 universe 快照增量更新（仅名单）")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=dt.date.today().isoformat())
    args = parser.parse_args()

    end = args.end
    before = _cache_dates()
    # T-day snapshot is calendar-invisible at 08:30 — fetch --end explicitly.
    if dt.date.fromisoformat(end).weekday() < 5 and end not in before:
        _fetch_single_day(end)

    universe.dump_universe(args.start, end)

    after = _cache_dates()
    print(f"✓ universe cache: {len(before)} → {len(after)} days "
          f"({min(after) if after else '-'} → {max(after) if after else '-'})")
    if end in after or (dt.date.fromisoformat(end).weekday() >= 5):
        return 0
    print(f"⚠ end date {end} still missing from cache (fetch failed?)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
