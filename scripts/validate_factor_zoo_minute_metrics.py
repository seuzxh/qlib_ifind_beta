"""Phase 1D — B1/B2 分钟域特征的双轨指标汇总。

输入：minute_b1.pkl（09:40 采样）、minute_eod.pkl（15:00 采样）、dayf_meta.pkl。
产出：reports/factor_zoo/minute_b1_metrics.csv、minute_b2_t1_metrics.csv
（t2/t3/t5 变体对 top 幸存者另跑：--lags t2,t3,t5 --names a,b）。

Run:
  conda run -n qlib_ifind_beta python scripts/validate_factor_zoo_minute_metrics.py
  conda run -n qlib_ifind_beta python scripts/validate_factor_zoo_minute_metrics.py \
      --lags t2 --names alpha101__ALPHA001
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from qlib_ifind_beta.config import OVERLAY_ROOT
from qlib_ifind_beta.factor_zoo.screen_lib import summarize

OUTDIR = ROOT / "reports" / "factor_zoo"
LAG_N = {"t1": 1, "t2": 2, "t3": 3, "t5": 5}


def shift_by_day(df: pd.DataFrame, k: int, day_cal: list[str]) -> pd.DataFrame:
    """每股按交易日历 shift k（T 日特征 = T-k 日采样值）。"""
    parts = []
    for s, g in df.groupby(level="instrument"):
        gg = g.droplevel("instrument").reindex(day_cal)
        parts.append(gg.shift(k))
    out = pd.concat(parts).rename_axis(["instrument", "datetime"]).swaplevel()
    return out[~out.index.duplicated(keep="last")]


def main(lags: list[str], names: list[str] | None) -> None:
    meta: pd.DataFrame = pd.read_pickle(OUTDIR / "dayf_meta.pkl")
    meta = meta.reorder_levels(["datetime", "instrument"]).sort_index()
    label, champs = meta["LABEL"], meta.drop(columns=["LABEL"])
    day_cal = sorted(meta.index.get_level_values("datetime").unique())

    # B1：09:40 采样直接可用（index 已过滤到成员）
    b1: pd.DataFrame = pd.read_pickle(OUTDIR / "minute_b1.pkl")
    b1 = b1.reindex(label.index)
    recs = []
    for col in b1.columns:
        recs.append({"name": f"b1__{col}", **summarize(b1[col], label, champs)})
        if len(recs) % 50 == 0:
            print(f"  b1 {len(recs)}/{b1.shape[1]}", flush=True)
    pd.DataFrame(recs).to_csv(OUTDIR / "minute_b1_metrics.csv", index=False)

    eod: pd.DataFrame = pd.read_pickle(OUTDIR / "minute_eod.pkl")
    for lag in lags:
        k = LAG_N[lag]
        cols = eod.columns if names is None else [c for c in eod.columns if c in names]
        df = shift_by_day(eod[cols], k, day_cal).reindex(label.index)
        recs = []
        for col in df.columns:
            recs.append({"name": f"b2_{lag}__{col}", **summarize(df[col], label, champs)})
            if len(recs) % 100 == 0:
                print(f"  {lag} {len(recs)}/{len(cols)}", flush=True)
        out = OUTDIR / ("minute_b2_t1_metrics.csv" if lag == "t1" else f"minute_b2_{lag}_metrics.csv")
        pd.DataFrame(recs).to_csv(out, index=False)
        print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lags", default="t1")
    ap.add_argument("--names", default=None, help="逗号分隔，仅对这些列算多日变体")
    a = ap.parse_args()
    main(a.lags.split(","), a.names.split(",") if a.names else None)
