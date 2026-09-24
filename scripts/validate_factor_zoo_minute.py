"""Phase 1B 分钟域移植 — B1（当日 09:40 采样）/ B2（T-k 日 15:00 采样 + shift k）。

求值通道：qlib 引擎 + freq="1min"（provider=cn_data_1min）+ vendored 算子 shim，
与路线 A 完全同一套表达式语义。连续分钟序列上：
- B1 = b1 候选（窗口≤8）在 T 日 09:40 bar 的取值（嵌套链>10 会自然 NaN）；
- B2 = b1+b2 候选（窗口≤120）在 T-k 日 15:00 bar 的取值，shift k 个交易日
  （k=1 全量；t2/t3/t5 仅对 shortlist 追加）。
universe 成员过滤：按 overlay universe 行级 span 后置过滤。

用法：
  bench: 30 股 × 前 2 块表达式，测速定分块
  feat : 全量（股票 150/块 × 表达式 40/块），产出 samples pkl
Run:
  conda run -n qlib_ifind_beta python scripts/validate_factor_zoo_minute.py bench|feat
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import qlib

from qlib_ifind_beta.config import CN_DATA_1MIN, OVERLAY_ROOT
from qlib_ifind_beta.factor_zoo import register_zoo_ops

OUTDIR = ROOT / "reports" / "factor_zoo"
START, END = "2024-01-02", "2026-09-24"
STOCK_CHUNK = 150
FIELD_CHUNK = 40
SLOT_B1 = " 09:40:00"
SLOT_EOD = " 15:00:00"


def load_universe_rows() -> list[tuple[str, str, str]]:
    rows = []
    for ln in (OVERLAY_ROOT / "instruments" / "highbeta883926.txt").read_text().splitlines():
        p = ln.split("\t")
        if len(p) >= 3:
            rows.append((p[0], p[1], p[2]))
    return rows


def member_pairs() -> pd.MultiIndex:
    """(datetime[Timestamp], instrument) 成员对（行级 span 展开；与采样帧同类型同顺序）。"""
    rows = load_universe_rows()
    cal = [d.strip() for d in (OVERLAY_ROOT / "calendars" / "day.txt").read_text().splitlines() if d.strip()]
    cal_set = {d: i for i, d in enumerate(cal)}
    pairs = set()
    for s, a, b in rows:
        if a not in cal_set:
            continue
        i0 = cal_set[a]
        i1 = cal_set.get(b, len(cal) - 1)
        for di in range(i0, min(i1, len(cal) - 1) + 1):
            pairs.add((pd.Timestamp(cal[di]), s))
    return pd.MultiIndex.from_tuples(sorted(pairs), names=["datetime", "instrument"])


def eval_chunk(D, stocks: list[str], rows: list[tuple[str, str, str]]):
    """一块 (stocks × exprs) 的 09:40 / 15:00 采样帧（向量化布尔掩码，无字符串）。"""
    fields = [expr for _, _, expr in rows]
    names = [f"{lib}__{name}" for lib, name, _ in rows]
    df = D.features(stocks, fields, start_time=START, end_time=END, freq="1min")
    df.columns = names
    ts = df.index.get_level_values(-1)                 # datetime64 级
    hhmm = (ts.hour * 100 + ts.minute).to_numpy()      # 向量化时刻
    dates_arr = ts.normalize().to_numpy()
    inst_arr = (df.index.get_level_values(0) if df.index.nlevels == 2
                else df.index.get_level_values(1)).to_numpy()

    def rekey(mask: np.ndarray) -> pd.DataFrame:
        sub = df.loc[mask]
        sub.index = pd.MultiIndex.from_arrays(
            [dates_arr[mask], inst_arr[mask]], names=["datetime", "instrument"])
        return sub[~sub.index.duplicated(keep="last")]

    return rekey(hhmm == 940), rekey(hhmm == 1500)


def run(mode: str) -> None:
    qlib.init(provider_uri=str(CN_DATA_1MIN), region="cn")
    register_zoo_ops()
    from qlib.data import D

    parse_rows = json.loads((OUTDIR / "factor_zoo_parse.json").read_text())
    cands = [r for r in parse_rows if r["class"] in ("b1", "b2_only")]
    rows = [(r["lib"], r["name"], r["expr"]) for r in cands]
    print(f"分钟域候选 {len(rows)}（b1={sum(1 for r in parse_rows if r['class']=='b1')}）")

    stocks = sorted({s for s, _, _ in load_universe_rows()})
    if mode == "bench":
        stocks = stocks[:30]
        rows = rows[: FIELD_CHUNK * 2]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    b1_parts, eod_parts = [], []
    t0 = time.time()
    n_schunk = int(np.ceil(len(stocks) / STOCK_CHUNK))
    n_fchunk = int(np.ceil(len(rows) / FIELD_CHUNK))
    for fi in range(n_fchunk):
        frows = rows[fi * FIELD_CHUNK: (fi + 1) * FIELD_CHUNK]
        for si in range(n_schunk):
            sgrp = stocks[si * STOCK_CHUNK: (si + 1) * STOCK_CHUNK]
            try:
                b1f, eodf = eval_chunk(D, sgrp, frows)
            except Exception as e:  # noqa: BLE001 - 降级逐表达式
                print(f"  f{fi}s{si} 块失败({type(e).__name__})，逐表达式重试: {str(e)[:80]}")
                b1f = eodf = None
                for k, row in enumerate(frows):
                    try:
                        b, e = eval_chunk(D, sgrp, [row])
                        b1f = b if b1f is None else b1f.join(b, how="outer")
                        eodf = e if eodf is None else eodf.join(e, how="outer")
                    except Exception as e2:  # noqa: BLE001
                        print(f"    弃用 {row[0]}__{row[1]}: {type(e2).__name__}")
            if b1f is not None and b1f.shape[1]:
                b1_parts.append(b1f)
            if eodf is not None and eodf.shape[1]:
                eod_parts.append(eodf)
        el = time.time() - t0
        done = (fi + 1) * FIELD_CHUNK
        print(f"  表达式 {min(done,len(rows))}/{len(rows)}，累计 {el:.0f}s"
              f"（外推全量 {el/max(done,1)*len(rows)/60:.1f} min）", flush=True)
        if mode == "bench":
            break

    b1_all = pd.concat(b1_parts).sort_index() if b1_parts else pd.DataFrame()
    eod_all = pd.concat(eod_parts).sort_index() if eod_parts else pd.DataFrame()
    members = member_pairs()
    b1_all = b1_all.reindex(members.intersection(b1_all.index))
    eod_all = eod_all.reindex(members.intersection(eod_all.index))
    suffix = "_bench" if mode == "bench" else ""
    b1_all.to_pickle(OUTDIR / f"minute_b1{suffix}.pkl")
    eod_all.to_pickle(OUTDIR / f"minute_eod{suffix}.pkl")
    print(f"B1 {b1_all.shape} / EOD {eod_all.shape} → minute_*{suffix}.pkl"
          f"；总用时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "bench")
