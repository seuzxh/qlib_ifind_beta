"""One-shot orchestrator: build the qlib_data overlay end-to-end.

Idempotent — safe to re-run. Steps:
  1. link_calendars()           — symlink readonly calendars/ dir
  2. link_instruments()         — symlink instruments/all.txt
  3. dump_universe(start,end)   — iFinD p03473 daily snapshots → T-1 time-varying
                                   instruments/highbeta883926.txt (+ historical code superset)
  4. per code: link_stock()     — 7 base bins symlinked from qlib_data
            + materialize_instrument() — change/limit_up/limit_down derived
            (codes = every stock ever in 883926 over [start,end]; delisted/no-bin
             codes fall to `miss` — link_stock leaves an empty dir, qlib reads NaN)
  5. link_stock(BENCHMARK) + materialize_instrument(BENCHMARK) — SH000300 reuse qlib_data + derive

Run:  conda run -n qlib_ifind_beta python -m scripts.build_overlay
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow `python -m scripts.build_overlay` from project root + `python scripts/build_overlay.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qlib_ifind_beta import overlay, universe, materialize, materialize_minute
from qlib_ifind_beta.config import BENCHMARK, INDEX_FACTOR_SOURCES, OVERLAY_ROOT, UNIVERSE_MARKET

# window covering train/valid/test (2024-01-01 → 2026-07-02)
DUMP_START, DUMP_END = "2024-01-01", "2026-07-02"


def build(start: str = DUMP_START, end: str = DUMP_END) -> dict:
    summary: dict = {}
    print(f"▶ overlay root: {OVERLAY_ROOT}")

    # 1-2. calendars + base instruments
    overlay.link_calendars()
    overlay.link_instruments()
    print("✓ calendars + instruments/all.txt symlinked")

    # 3. universe (iFinD p03473, time-varying T-1 pool over [start, end])
    _, codes = universe.dump_universe(start, end)
    summary["n_constituents"] = len(codes)
    print(f"✓ universe: {len(codes)} historical codes (T-1 time-varying) "
          f"→ instruments/{UNIVERSE_MARKET}.txt")

    # 4. per-stock overlay + derived bins (daily change/limit + minute factors)
    ok, miss = [], []
    min_ok, min_miss = 0, 0
    for i, code in enumerate(codes, 1):
        overlay.link_stock(code)               # 7 base bins (symlink)
        if materialize.materialize_instrument(code):  # change/limit_up/limit_down
            ok.append(code)
        else:
            miss.append(code)
        if materialize_minute.materialize_minute_instrument(code):  # 14 minute factors + price_941
            min_ok += 1
        else:
            min_miss += 1
        if i % 25 == 0:
            print(f"  …{i}/{len(codes)} stock dirs done")
    summary["materialized_ok"] = len(ok)
    summary["materialized_missing"] = miss
    summary["minute_ok"] = min_ok
    summary["minute_missing"] = min_miss
    print(f"✓ derived bins: {len(ok)} ok" + (f", {len(miss)} missing: {miss}" if miss else ""))
    print(f"✓ minute factors: {min_ok} ok, {min_miss} missing (delisted/no-1min)")

    # 5. benchmark SH000300 — reuse qlib_data's clean 26y bins (symlink) + derive change
    overlay.link_stock(BENCHMARK)
    if not materialize.materialize_instrument(BENCHMARK):
        raise RuntimeError(f"failed to materialize $change for benchmark {BENCHMARK}")
    summary["benchmark"] = BENCHMARK
    print(f"✓ benchmark {BENCHMARK} linked + derived → features/{BENCHMARK.lower()}/")

    # 6. index factor sources (e.g. SH000001) — link 7 base bins for ChangeInstrument refs.
    # NOT benchmark; indices don't trade → no change/limit materialization needed.
    for idx in INDEX_FACTOR_SOURCES:
        overlay.link_stock(idx)
        print(f"✓ index source {idx} linked → features/{idx.lower()}/")
    summary["index_sources"] = list(INDEX_FACTOR_SOURCES)

    print("\n✅ overlay build complete")
    print(f"   provider_uri = {OVERLAY_ROOT}")
    print(f"   universe = {len(codes)} historical codes (time-varying); benchmark = {BENCHMARK}")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build qlib_data overlay for 883926 MVP")
    p.add_argument("--start", default=DUMP_START)
    p.add_argument("--end", default=DUMP_END)
    a = p.parse_args()
    build(a.start, a.end)
