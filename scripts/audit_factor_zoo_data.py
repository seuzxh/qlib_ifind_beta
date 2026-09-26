"""Phase 0.4 数据审计 — 多日族（B2）的可行性前提。

审计项：
  ① 截面时点：T 日成分 08:30 盘前 cron 已得（update_universe），09:41 前可用 ——
     静态事实，打印核对 cron 文件存在性。
  ② T-k 完整 240bar 覆盖率：对每个交易日 T，T 日成分中在 T-k（k=1/2/3/5）
     拥有完整 240 根非 NaN 分钟 bar 的比例（活跃=成交量>0 另计）。
  ③ 1min 字段结构：抽样核对 7 字段 bin 齐备（含 factor.1min.bin）。
  ④ 复权口径：分钟域一律 1min 原值（名义口径），与 label v2 / overnight_gap
     决策一致 —— 静态声明，配合 2026-09-11 gap A/B 结论。

产物：reports/factor_zoo_data_audit.json
Run: conda run -n qlib_ifind_beta python scripts/audit_factor_zoo_data.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from qlib_ifind_beta.config import CN_DATA_1MIN, OVERLAY_ROOT

OUT = ROOT / "reports" / "factor_zoo_data_audit.json"
SLOTS = 240


def read_bin_f32(path: Path) -> tuple[int | None, np.ndarray]:
    if not path.exists():
        return None, np.array([])
    raw = np.fromfile(path, dtype="<f4")
    if raw.size < 2:
        return None, np.array([])
    return int(raw[0]), raw[1:]


def load_universe() -> dict[str, list[str]]:
    """day(YYYY-MM-DD) → 当日成分列表。"""
    uni: dict[str, list[str]] = {}
    for ln in (OVERLAY_ROOT / "instruments" / "highbeta883926.txt").read_text().splitlines():
        parts = ln.split("\t")
        if len(parts) >= 3:
            uni.setdefault(parts[1], []).append(parts[0])  # 以首见 start 日期聚合
    return uni


def main() -> None:
    # ① 截面时点（静态）
    cron = ROOT / "scripts" / "cron_update_universe.sh"
    timing_ok = cron.exists() and "08:30" in cron.read_text()
    print(f"① 截面时点: universe 08:30 盘前 cron = {timing_ok}")

    day_cal = [d.strip() for d in (OVERLAY_ROOT / "calendars" / "day.txt").read_text().splitlines() if d.strip()]
    uni_rows = []
    for ln in (OVERLAY_ROOT / "instruments" / "highbeta883926.txt").read_text().splitlines():
        p = ln.split("\t")
        if len(p) >= 3:
            uni_rows.append((p[0], p[1], p[2]))
    print(f"   universe 行数 {len(uni_rows)}，日历 {day_cal[0]}→{day_cal[-1]} 共 {len(day_cal)} 日")

    # 每股逐日完整性：day_idx → 0=缺/NaN 1=全bar非NaN 2=全bar非NaN且有成交
    # 注意 1min bin 的 start_index 是【分钟日历】索引：day = start_index // 240。
    day_pos = {d: i for i, d in enumerate(day_cal)}
    min_cal = [t.strip() for t in (CN_DATA_1MIN / "calendars" / "1min.txt").read_text().splitlines() if t.strip()]
    min_days = [min_cal[i * SLOTS][:10] for i in range(len(min_cal) // SLOTS)]
    min_day_to_cal = {}
    for i, d in enumerate(min_days):
        if d in day_pos:
            min_day_to_cal[i] = day_pos[d]
    feat_dir = CN_DATA_1MIN / "features"
    stocks = sorted({s for s, _, _ in uni_rows})
    day_start = day_cal.index("2024-01-02")
    n_days = len(day_cal)
    comp = np.zeros((len(stocks), n_days), dtype=np.int8)
    sidx = {s: i for i, s in enumerate(stocks)}
    n_missing_vol = 0
    for s in stocks:
        si_min, vol = read_bin_f32(feat_dir / s.lower() / "volume.1min.bin")
        if si_min is None:
            n_missing_vol += 1
            continue
        first_min_day = si_min // SLOTS
        if si_min % SLOTS:  # 非整日起始：从次日起算，保守
            first_min_day += 1
        n_full_days = min(len(vol) // SLOTS, len(min_days) - first_min_day)
        if n_full_days <= 0:
            continue
        off = first_min_day * SLOTS - si_min
        v = vol[off: off + n_full_days * SLOTS].reshape(n_full_days, SLOTS)
        ok_nan = ~np.isnan(v).any(axis=1)
        ok_amt = ok_nan & (v.sum(axis=1) > 0)
        for j in range(n_full_days):
            cal_i = min_day_to_cal.get(first_min_day + j)
            if cal_i is not None:
                comp[sidx[s], cal_i] = 2 if ok_amt[j] else (1 if ok_nan[j] else 0)
    print(f"   volume.1min.bin 缺失股票数: {n_missing_vol}/{len(stocks)}"
          f"；分钟日历 {min_days[0]}→{min_days[-1]} 共 {len(min_days)} 日")

    # ② T-k 覆盖率（行级区间展开：start ≤ d ≤ end 均为当日成分）
    day_pos = {d: i for i, d in enumerate(day_cal)}
    uni_by_day: dict[int, list[int]] = {}
    for s, a, b in uni_rows:
        if a not in day_pos:
            continue
        i0, i1 = day_pos[a], day_pos.get(b, len(day_cal) - 1)
        si_ = sidx[s]
        for di in range(i0, min(i1, n_days - 1) + 1):
            uni_by_day.setdefault(di, []).append(si_)
    report = {"timing_ok": timing_ok, "stocks": len(stocks), "coverage": {}}
    for k in (1, 2, 3, 5):
        fracs, fracs_amt = [], []
        for di in range(max(day_start, k), n_days):
            members = uni_by_day.get(di)
            if not members:
                continue
            prev = comp[members, di - k]
            fracs.append(float((prev >= 1).mean()))
            fracs_amt.append(float((prev == 2).mean()))
        report["coverage"][f"T-{k}"] = {
            "完整240bar_均值": round(float(np.mean(fracs)), 4),
            "完整且有成交_均值": round(float(np.mean(fracs_amt)), 4),
            "最低单日_完整": round(float(np.min(fracs)), 4),
            "P5_完整": round(float(np.percentile(fracs, 5)), 4),
        }
        print(f"② T-{k}: 完整240bar 均值 {np.mean(fracs):.2%}"
              f"（P5 {np.percentile(fracs, 5):.2%}，最低 {np.min(fracs):.2%}）；"
              f"完整且有成交均值 {np.mean(fracs_amt):.2%}")

    # ③ 字段结构抽样
    import random
    random.seed(0)
    sample = random.sample(stocks, min(50, len(stocks)))
    fields = ["open", "high", "low", "close", "volume", "vwap", "factor"]
    miss = {f: sum(not (feat_dir / s.lower() / f"{f}.1min.bin").exists() for s in sample) for f in fields}
    print(f"③ 抽样50股字段缺失: {miss}")
    report["field_missing_sample50"] = miss

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"saved → {OUT}")


if __name__ == "__main__":
    main()
