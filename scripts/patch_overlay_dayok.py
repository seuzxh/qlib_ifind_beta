"""overlay 数据一致性补丁（一次性，2026-09-25）— PortAna 修复的数据侧。

① limit_up/limit_down 对齐日线 change 跨度（同空可比）；
② 生成过滤 universe highbeta883926_dayok（剔除 99 只缺日线 bin 股票）。
Run: conda run -n qlib_ifind_beta python scripts/patch_overlay_dayok.py
"""
"""limit bin 对齐日线 change 跨度（而非分钟 change_941 跨度），保证
Ge/Le 两侧在回测窗口内标签一致（同空可比）。"""
import sys
sys.path.insert(0, ".")
import numpy as np
from qlib_ifind_beta.config import OVERLAY_ROOT
from qlib_ifind_beta.materialize import board_limit
from qlib_ifind_beta import binio

feat = OVERLAY_ROOT / "features"
fixed = 0
for d in sorted(feat.iterdir()):
    chg = d / "change.day.bin"
    lu = d / "limit_up.day.bin"
    if not chg.exists():
        continue
    si, v = binio.read_bin(chg)
    if si is None:
        continue
    up, dn = board_limit(d.name.upper())
    # 无论 limit bin 现状如何，统一重写为 change 同跨度（幂等）
    if lu.exists():
        si2, v2 = binio.read_bin(lu)
        if si2 == si and v2.size == v.size:
            continue
    binio.write_bin(lu, si, np.full(v.size, up, dtype=np.float32))
    binio.write_bin(d / "limit_down.day.bin", si, np.full(v.size, dn, dtype=np.float32))
    fixed += 1
print(f"limit bin 对齐日线 change 跨度: 重写 {fixed} 只")
# SH603071 诊断
si, v = binio.read_bin(feat / "sh603071" / "change.day.bin")
cal = [x.strip() for x in (OVERLAY_ROOT / "calendars/day.txt").read_text().splitlines() if x.strip()]
if si is not None and v.size:
    print("SH603071 change 跨度:", cal[si], "→", cal[si + v.size - 1], "| 回测窗 2026-04→07 是否覆盖:",
          si <= cal.index("2026-04-01") < si + v.size)


# --- ② 过滤 universe ---
sys.path.insert(0, ".")
from pathlib import Path
from qlib_ifind_beta.config import OVERLAY_ROOT

feat = OVERLAY_ROOT / "features"
rows = []
for ln in (OVERLAY_ROOT / "instruments" / "highbeta883926.txt").read_text().splitlines():
    p = ln.split("\t")
    if len(p) >= 3:
        rows.append(p)

def day_ok(code: str) -> bool:
    d = feat / code.lower()
    return (d / "close.day.bin").exists() and (d / "change.day.bin").exists() and (d / "factor.day.bin").exists()

bad = {p[0] for p in rows if not day_ok(p[0])}
print(f"universe 股票 {len({p[0] for p in rows})}，缺日线源 bin: {len(bad)} → {sorted(bad)[:12]}")
kept = [p for p in rows if p[0] not in bad]
out = OVERLAY_ROOT / "instruments" / "highbeta883926_dayok.txt"
out.write_text("\n".join("\t".join(p) for p in kept) + "\n")
print(f"写过滤 universe → {out.name}（保留 {len({p[0] for p in kept})} 股 / {len(kept)} 行）")
