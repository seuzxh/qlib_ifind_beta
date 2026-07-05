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

from qlib_ifind_beta import overlay, universe, materialize
from qlib_ifind_beta.config import BENCHMARK, OVERLAY_ROOT, UNIVERSE_MARKET

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

    # 4. per-stock overlay + derived bins
    ok, miss = [], []
    for i, code in enumerate(codes, 1):
        overlay.link_stock(code)               # 7 base bins (symlink)
        if materialize.materialize_instrument(code):  # change/limit_up/limit_down
            ok.append(code)
        else:
            miss.append(code)
        if i % 25 == 0:
            print(f"  …{i}/{len(codes)} stock dirs done")
    summary["materialized_ok"] = len(ok)
    summary["materialized_missing"] = miss
    print(f"✓ derived bins: {len(ok)} ok" + (f", {len(miss)} missing: {miss}" if miss else ""))

    # 5. benchmark SH000300 — reuse qlib_data's clean 26y bins (symlink) + derive change
    overlay.link_stock(BENCHMARK)
    if not materialize.materialize_instrument(BENCHMARK):
        raise RuntimeError(f"failed to materialize $change for benchmark {BENCHMARK}")
    summary["benchmark"] = BENCHMARK
    print(f"✓ benchmark {BENCHMARK} linked + derived → features/{BENCHMARK.lower()}/")

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
