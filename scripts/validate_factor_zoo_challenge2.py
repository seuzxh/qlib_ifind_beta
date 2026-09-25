"""Phase 2-replace 终审复核 — C1/C5 挑战者的确定性与第二窗口 A/B。

① 确定性：同配置重跑 C1（窗口1），IC 应逐位复现（HFLGB 固定种子）。
② 第二窗口（W2）：所有方用同一替代切分 train 2024-01→2025-06 / valid
   2025-07→08 / test 2025-09→12（与 W1 的 2026-04→07 完全不相交），
   BASE/C1/C5 三方 A/B——继承仓库 W1/W2 双窗纪律。

Run: conda run -n qlib_ifind_beta python scripts/validate_factor_zoo_challenge2.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import qlib
from ruamel.yaml import YAML

import scripts.validate_factor_zoo_augment as A

OUTDIR = ROOT / "reports" / "factor_zoo"
B1 = "b1__alpha158__"
C1_ADD = [B1 + "MAX5", B1 + "MIN5", B1 + "QTLD5"]
C1_DROP = ["close_pos_1m", "close_pos_3m", "close_pos_5m"]
C5_ADD = [B1 + "STD5"] + C1_ADD + [B1 + "HIGH0"]
C5_DROP = C1_DROP + ["accel_1m", "accel_3m", "accel_5m"]
W2_SEG = {"train": ["2024-01-01", "2025-06-30"],
          "valid": ["2025-07-01", "2025-08-31"],
          "test": ["2025-09-01", "2025-12-31"]}


def run_task_seg(handler_class: str, module_path: str, seg: dict | None):
    from qlib.model.trainer import task_train
    cfg = YAML(typ="safe").load(open(ROOT / "qrun" / "workflow_minute_enhanced_tk10_nd8.yaml"))
    task = cfg["task"]
    h = task["dataset"]["kwargs"]["handler"]
    h["class"] = handler_class
    h["module_path"] = module_path
    if seg:
        task["dataset"]["kwargs"]["segments"] = seg
    task["record"] = [r for r in task["record"] if r.get("class") != "PortAnaRecord"]
    return task_train(task, experiment_name="factor_zoo_challenge2")


def run_and_report(tag: str, handler: str, module: str, seg: dict | None,
                   label: pd.Series, drop: list[str], add: list[str]) -> pd.Series:
    A.FZ_FIELDS.clear(), A.DROP_FIELDS.clear()
    A.DROP_FIELDS.extend(drop)
    A.FZ_FIELDS.extend(add)
    rec = run_task_seg(handler, module, seg)
    m = rec.list_metrics()
    pred = rec.load_object("pred.pkl")
    pred = (pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred)
    pred = pred.reorder_levels(["datetime", "instrument"]).sort_index()
    r = A.replay_topn(pred, label)
    print(f"  {tag}: IC={m.get('IC'):.4f} RankIC={m.get('Rank IC'):.4f} | {A.replay_stats(r)}",
          flush=True)
    return r


def main() -> None:
    qlib.init(provider_uri=str(A.OVERLAY_ROOT), region="cn")
    meta = pd.read_pickle(OUTDIR / "dayf_meta.pkl")
    meta = meta.reorder_levels(["datetime", "instrument"]).sort_index()
    label = meta["LABEL"]

    fz = {"c1": [n for n in (A.materialize(s) for s in C1_ADD)],
          "c5": [n for n in (A.materialize(s) for s in C5_ADD)]}

    print("▶ ① 确定性重跑（窗口1，应复现 IC=0.0951/0.0943）", flush=True)
    run_and_report("C1@W1-dup", "MinuteEnhancedFZHandler",
                   "scripts.validate_factor_zoo_augment", None, label, C1_DROP, fz["c1"])

    print("▶ ② 第二窗口 A/B（test 2025-09→12，三方同切分）", flush=True)
    rb = run_and_report("BASE@W2", "MinuteEnhancedHandler",
                        "qlib_ifind_beta.minute_enhanced_handler", W2_SEG, label, [], [])
    r1 = run_and_report("C1@W2", "MinuteEnhancedFZHandler",
                        "scripts.validate_factor_zoo_augment", W2_SEG, label, C1_DROP, fz["c1"])
    r5 = run_and_report("C5@W2", "MinuteEnhancedFZHandler",
                        "scripts.validate_factor_zoo_augment", W2_SEG, label, C5_DROP, fz["c5"])
    for name, r in (("C1", r1), ("C5", r5)):
        both = pd.DataFrame({"b": rb, "c": r}).dropna()
        h = len(both) // 2
        print(f"  {name}@W2 Δ日均 vs BASE: 全窗 {both.c.sub(both.b).mean():+.4%} | "
              f"前半 {both.c.iloc[:h].sub(both.b.iloc[:h]).mean():+.4%} | "
              f"后半 {both.c.iloc[h:].sub(both.b.iloc[h:]).mean():+.4%}", flush=True)


if __name__ == "__main__":
    main()
