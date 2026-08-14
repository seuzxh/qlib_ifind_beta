"""Blend champion18 tail selection with pruned15's stronger cross-sectional IC."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.validate_factor_challengers import _backtest, _daily_corr
from qlib_ifind_beta.config import OVERLAY_ROOT


RUNS = {
    "W2_2025Q2": {
        "champion": "beaa3a394ae04b04818e92fb6d663d51",
        "pruned": "4ff0bb6a72ef4256a9574bd63ea18edb",
        "start": "2025-04-01", "end": "2025-07-02",
    },
    "W1_2026Q2": {
        "champion": "e0ca66e6b67745ccb4ef244119609e7c",
        "pruned": "5bc794f50ef8498d9948258db808dba7",
        "start": "2026-04-01", "end": "2026-07-02",
    },
    "W3_2025Q4": {
        "champion": "f4d6572fa4c44e789da3c2f96b8f3717",
        "pruned": None,
        "start": "2025-10-01", "end": "2025-12-31",
    },
}
PRUNED_WEIGHTS = (0.25, 0.50, 0.75)


def _series(obj):
    return obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj


def _cs_rank(score: pd.Series) -> pd.Series:
    return score.groupby(level="datetime").rank(method="average", pct=True)


def main():
    import qlib
    from qlib.config import C
    from qlib.workflow import R

    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(ROOT / "mlruns")
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn", exp_manager=exp_manager)

    results = {}
    for window, cfg in RUNS.items():
        if cfg["pruned"] is None:
            continue
        rc = R.get_recorder(recorder_id=cfg["champion"], experiment_name="factor_challenger_ab")
        rp = R.get_recorder(recorder_id=cfg["pruned"], experiment_name="factor_challenger_ab")
        champion = _series(rc.load_object("pred.pkl"))
        pruned = _series(rp.load_object("pred.pkl"))
        label = _series(rc.load_object("label.pkl"))
        common = champion.index.intersection(pruned.index).intersection(label.index)
        champion, pruned, label = champion.loc[common], pruned.loc[common], label.loc[common]
        rc_rank, rp_rank = _cs_rank(champion), _cs_rank(pruned)

        results[window] = {}
        for weight in PRUNED_WEIGHTS:
            score = (1.0 - weight) * rc_rank + weight * rp_rank
            pearson = _daily_corr(score, label, "pearson")
            rankic = _daily_corr(score, label, "spearman")
            pred = score.to_frame("score")
            metrics = {
                "pruned_weight": weight,
                "IC": float(pearson.mean()),
                "ICIR": float(pearson.mean() / pearson.std()),
                "RankIC": float(rankic.mean()),
                "RankICIR": float(rankic.mean() / rankic.std()),
                "RankIC_positive_rate": float((rankic > 0).mean()),
            }
            metrics.update(_backtest(pred, cfg["start"], cfg["end"]))
            results[window][str(weight)] = metrics
            print(window, json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

    out = ROOT / "data" / "prediction_blend_ab.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
