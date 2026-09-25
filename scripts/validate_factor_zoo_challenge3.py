"""Phase 2-replace 终审 — purged 分段滚动 A/B（BASE vs C1 vs C5）。

继承仓库 purged 19 段纪律：train 90 交易日 / embargo 1 日 / test 20 日，
滑动覆盖 2025-01 → 2026-07；三方同段对照，报告逐段 IC 与全期重放。
（正式 purged rolling 管线 rolling_validate.py 的同口径轻量版；PortAnaRecord
因本机死锁继续用 SignalRecord + 手工重放。）

Run: conda run -n qlib_ifind_beta python scripts/validate_factor_zoo_challenge3.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import qlib
from ruamel.yaml import YAML

import scripts.validate_factor_zoo_augment as A
from qlib_ifind_beta.config import OVERLAY_ROOT

OUTDIR = ROOT / "reports" / "factor_zoo"
B1 = "b1__alpha158__"
C1_ADD = [B1 + "MAX5", B1 + "MIN5", B1 + "QTLD5"]
C1_DROP = ["close_pos_1m", "close_pos_3m", "close_pos_5m"]
C5_ADD = [B1 + "STD5"] + C1_ADD + [B1 + "HIGH0"]
C5_DROP = C1_DROP + ["accel_1m", "accel_3m", "accel_5m"]
TRAIN_D, EMBARGO_D, TEST_D = 90, 1, 20
SEG_START, SEG_END = "2025-01-02", "2026-07-02"


def build_segments() -> list[dict]:
    cal = [d.strip() for d in (OVERLAY_ROOT / "calendars" / "day.txt").read_text().splitlines() if d.strip()]
    cal = pd.to_datetime(pd.Series(cal))
    cal = cal[(cal >= SEG_START) & (cal <= SEG_END)].reset_index(drop=True)
    segs, i = [], 0
    while i + TRAIN_D + EMBARGO_D + TEST_D <= len(cal):
        a, b, c, d = (cal.iloc[i], cal.iloc[i + TRAIN_D - 1],
                      cal.iloc[i + TRAIN_D + EMBARGO_D],
                      cal.iloc[i + TRAIN_D + EMBARGO_D + TEST_D - 1])
        segs.append({"train": [str(a.date()), str(b.date())],
                     "valid": [str(b.date()), str(b.date())],  # 触发用占位，无早停
                     "test": [str(c.date()), str(d.date())]})
        i += TEST_D
    return segs


def run_seg(seg: dict, handler: str, module: str, drop: list[str], add: list[str]):
    from qlib.model.trainer import task_train
    A.FZ_FIELDS.clear(), A.DROP_FIELDS.clear()
    A.DROP_FIELDS.extend(drop)
    A.FZ_FIELDS.extend(add)
    cfg = YAML(typ="safe").load(open(ROOT / "qrun" / "workflow_minute_enhanced_tk10_nd8.yaml"))
    task = cfg["task"]
    h = task["dataset"]["kwargs"]["handler"]
    h["class"], h["module_path"] = handler, module
    task["dataset"]["kwargs"]["segments"] = seg
    task["record"] = [r for r in task["record"] if r.get("class") != "PortAnaRecord"]
    rec = task_train(task, experiment_name="factor_zoo_challenge3")
    m = rec.list_metrics()
    pred = rec.load_object("pred.pkl")
    pred = (pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred)
    return float(m.get("IC", np.nan)), pred.reorder_levels(
        ["datetime", "instrument"]).sort_index()


def main() -> None:
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    meta = pd.read_pickle(OUTDIR / "dayf_meta.pkl")
    meta = meta.reorder_levels(["datetime", "instrument"]).sort_index()
    label = meta["LABEL"]

    fz_c1 = [A.materialize(s) for s in C1_ADD]
    fz_c5 = [A.materialize(s) for s in C5_ADD]
    sides = [("BASE", "MinuteEnhancedHandler", "qlib_ifind_beta.minute_enhanced_handler", [], []),
             ("C1", "MinuteEnhancedFZHandler", "scripts.validate_factor_zoo_augment", C1_DROP, fz_c1),
             ("C5", "MinuteEnhancedFZHandler", "scripts.validate_factor_zoo_augment", C5_DROP, fz_c5)]

    segs = build_segments()
    print(f"purged 分段：{len(segs)} 段（train{TRAIN_D}/embargo{EMBARGO_D}/test{TEST_D}，"
          f"{SEG_START}→{SEG_END}）", flush=True)
    rows = []
    daily = {n: {} for n, *_ in sides}
    for k, seg in enumerate(segs):
        line = f"seg{k:02d} test {seg['test'][0]}→{seg['test'][1]}:"
        for name, handler, module, drop, add in sides:
            ic, pred = run_seg(seg, handler, module, drop, add)
            r = A.replay_topn(pred, label)
            for d, v in r.items():
                daily[name].setdefault(d, []).append(v)
            line += f"  {name} IC={ic:+.4f} 重放{(1+r).prod()-1:+.1%}"
            rows.append({"seg": k, "side": name, "test": seg["test"][0], "ic": ic,
                         "seg_ret": float((1 + r).prod() - 1)})
        print(line, flush=True)
        pd.DataFrame(rows).to_csv(OUTDIR / "challenge3_segments.csv", index=False)

    print("\n=== purged 滚动汇总 ===")
    for name, *_ in sides:
        ics = [r["ic"] for r in rows if r["side"] == name]
        rets = [r["seg_ret"] for r in rows if r["side"] == name]
        base_rets = [r["seg_ret"] for r in rows if r["side"] == "BASE"]
        win = float(np.mean([a > b for a, b in zip(rets, base_rets)]))
        # 全期拼接日收益（段内逐日，段间拼接）
        merged = {}
        for d, vs in daily[name].items():
            merged[d] = float(np.mean(vs))  # 相邻段不重叠（test 步进=窗口长）
        s = pd.Series(merged).sort_index()
        cum = float((1 + s).prod() - 1)
        dd = float(((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min())
        print(f"  {name}: 段均值IC={np.mean(ics):+.4f} 段胜率(vs BASE)={win:.0%} "
              f"拼接累计={cum:+.1%} 拼接回撤={dd:.1%}", flush=True)


if __name__ == "__main__":
    main()
