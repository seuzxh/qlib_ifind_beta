"""Re-materialize minute factors (38 day.bins/stock) without redoing the full overlay.

Writes 14 baseline minute factors + 4 enhanced extras (vol_vs_yest_t2/t3/t5 +
overnight_gap) + 5 tail (§25) + 5 T-1 opening + 5 T-2 opening (§26) +
price_941 + change_941 + 3 上证指数开盘共振（§29 idx_open_ret_10 /
idx_open_mom_5m / idx_open_accel_5m，broadcast 同值 per stock） per stock. Use
after tweaking qlib_ifind_beta/materialize_minute.py formulas — avoids
re-pulling the iFinD universe (unlike scripts/build_overlay.py). Reads the
existing instruments/highbeta883926.txt + ensures each stock's overlay dir
exists, then calls materialize_minute_instrument per code.

Run: conda run -n qlib_ifind_beta python scripts/materialize_minute.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qlib_ifind_beta import materialize_minute, overlay
from qlib_ifind_beta.config import INSTRUMENTS_DST, UNIVERSE_MARKET


def _load_codes():
    p = INSTRUMENTS_DST / f"{UNIVERSE_MARKET}.txt"
    codes = []
    with open(p) as fp:
        for line in fp:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                codes.append(parts[0])
    return sorted(set(codes))


def main():
    codes = _load_codes()
    print(f"▶ materializing minute factors for {len(codes)} codes")
    ok, miss = [], []
    for i, code in enumerate(codes, 1):
        overlay.link_stock(code)                       # ensure feature dir + 7 base bins
        if materialize_minute.materialize_minute_instrument(code):
            ok.append(code)
        else:
            miss.append(code)
        if i % 100 == 0:
            print(f"  …{i}/{len(codes)}  (ok={len(ok)}, miss={len(miss)})")
    print(f"✓ minute factors: {len(ok)} ok, {len(miss)} missing")
    if miss:
        print(f"  missing sample: {miss[:12]}")
        print(f"  (expected ~76 — delisted/suspended stocks with no 1min/daily bins)")


if __name__ == "__main__":
    main()
