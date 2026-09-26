"""Phase 2-replace 终审-II — TD0 完整 PortAna 回测（BASE vs C1 vs C5）。

PortAna 死锁三层修复后的首次正式回测：
  ① joblib_backend=threading（绕开 loky 池重入死锁）；
  ② handler/exchange 双双使用过滤 universe `highbeta883926_dayok`
     （剔除 99 只仅分钟源、缺日线 bin 的股票，成员行仅减 0.9%，三方一致）；
  ③ exchange_kwargs 显式 codes（默认 "all" 会载入全市场）。
完整协议：TopkDropoutStrategyTD0 topk10/n_drop8/hold_thresh1/
forbid_all_trade_at_limit，deal_price [$price_941,$close]，open 0.0005 /
close 0.0015 / min_cost 5，涨跌停 Ge($change_941,$limit_up)/Le($change,$limit_down)。

Run: conda run -n qlib_ifind_beta python scripts/validate_factor_zoo_challenge4.py
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

import examples.factor_zoo.validate_factor_zoo_augment as A
from qlib_ifind_beta.config import OVERLAY_ROOT

B1 = "b1__alpha158__"
C1_ADD = [B1 + "MAX5", B1 + "MIN5", B1 + "QTLD5"]
C1_DROP = ["close_pos_1m", "close_pos_3m", "close_pos_5m"]
C5_ADD = [B1 + "STD5"] + C1_ADD + [B1 + "HIGH0"]
C5_DROP = C1_DROP + ["accel_1m", "accel_3m", "accel_5m"]


def run_portana(handler: str, module: str, drop: list[str], add: list[str],
                w2: bool = False):
    from qlib.model.trainer import task_train
    A.FZ_FIELDS.clear(), A.DROP_FIELDS.clear()
    A.DROP_FIELDS.extend(drop)
    A.FZ_FIELDS.extend(add)
    cfg = YAML(typ="safe").load(open(ROOT / "examples" / "champion" / "workflow_minute_enhanced_tk10_nd8.yaml"))
    task = cfg["task"]
    h = task["dataset"]["kwargs"]["handler"]
    h["class"], h["module_path"] = handler, module
    h["kwargs"]["instruments"] = "highbeta883926_dayok"
    if w2:  # 第二窗口：切分与回测窗同步切换（与 challenge2 的 W2 切分一致）
        task["dataset"]["kwargs"]["segments"] = {
            "train": ["2024-01-01", "2025-06-30"],
            "valid": ["2025-07-01", "2025-08-31"],
            "test": ["2025-09-01", "2025-12-31"]}
    for rec in task.get("record", []):
        if rec.get("class") == "PortAnaRecord":
            bk = (rec.get("kwargs", {}).get("config") or {}).get("backtest") or {}
            exk = bk.get("exchange_kwargs") or {}
            if isinstance(exk.get("limit_threshold"), list):
                exk["limit_threshold"] = tuple(exk["limit_threshold"])
            exk["codes"] = "highbeta883926_dayok"
            if w2:
                bk["start_time"], bk["end_time"] = "2025-09-01", "2025-12-31"
    return task_train(task, experiment_name="factor_zoo_challenge4")


def port_metrics(recorder) -> dict:
    rep = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    s, b = rep["return"].fillna(0), rep["bench"].fillna(0)
    cum = float((1 + s).prod() - 1)
    ann = float((1 + cum) ** (252 / len(s)) - 1)
    dd = float(((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min())
    exc = float((1 + (s - b)).prod() - 1)
    ir = float((s - b).mean() / (s - b).std() * 252 ** 0.5)
    return {"IC": None, "累计": cum, "年化": ann, "超额": exc, "IR": ir,
            "回撤": dd, "Calmar": ann / abs(dd) if dd else float("inf"),
            "天数": len(s)}


def main() -> None:
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib.config import C
    C["joblib_backend"] = "threading"   # 死锁修复①
    w2 = len(sys.argv) > 1 and sys.argv[1] == "w2"

    fz_c1 = [A.materialize(s) for s in C1_ADD]
    fz_c5 = [A.materialize(s) for s in C5_ADD]
    sides = [
        ("BASE", "MinuteEnhancedHandler", "qlib_ifind_beta.factor.minute_enhanced_handler", [], []),
        ("C1", "MinuteEnhancedFZHandler", "examples.factor_zoo.validate_factor_zoo_augment", C1_DROP, fz_c1),
        ("C5", "MinuteEnhancedFZHandler", "examples.factor_zoo.validate_factor_zoo_augment", C5_DROP, fz_c5),
    ]
    win = "W2 test 2025-09→12" if w2 else "W1 test 2026-04→07"
    print(f"▶ TD0 完整 PortAna（{win}，过滤 universe 三方一致）", flush=True)
    rows = []
    for tag, handler, module, drop, add in sides:
        rec = run_portana(handler, module, drop, add, w2=w2)
        m = rec.list_metrics()
        pm = port_metrics(rec)
        pm["IC"] = float(m.get("IC", np.nan))
        pm["RankIC"] = float(m.get("Rank IC", np.nan))
        pm["tag"] = tag
        rows.append(pm)
        print(f"  {tag}: IC={pm['IC']:.4f} 累计={pm['累计']:+.1%} 超额={pm['超额']:+.1%} "
              f"IR={pm['IR']:.2f} 回撤={pm['回撤']:.1%} Calmar={pm['Calmar']:.2f}", flush=True)
        df = pd.DataFrame(rows)
        out = ROOT / "reports" / "factor_zoo" / ("challenge4_td0_w2.csv" if w2 else "challenge4_td0.csv")
        df.to_csv(out, index=False)
    print(f"\nsaved → reports/factor_zoo/{'challenge4_td0_w2.csv' if w2 else 'challenge4_td0.csv'}", flush=True)


if __name__ == "__main__":
    main()
