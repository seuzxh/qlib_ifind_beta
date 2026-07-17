"""Four-quarter screen for chase and member-basket tracking penalties.

This stage uses a fast daily Top10 rebuild to eliminate unstable parameters.
Survivors must still pass the purged exact-TD0 validation; this script cannot
promote anything by itself.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from qlib_ifind_beta.config import CHAMPION_LABEL_EXPR, OVERLAY_ROOT, UNIVERSE_MARKET
from qlib_ifind_beta.model_ensemble import correlation_metrics
from qlib_ifind_beta.risk_overlay import (
    apply_top_pool_penalty, compute_chase_risk, compute_tracking_risk,
)
from scripts.diagnose_minute_basket_resonance import build_candidates
from scripts.validate_prediction_blend import _series

RESULT_PATH = ROOT / "data" / "risk_overlay_quarter_screen.json"
CHAMPION_RECORDERS = {
    "W4_2024Q4": "cef0056252f64ac2ae81092683d7ea63",
    "W2_2025Q2": "beaa3a394ae04b04818e92fb6d663d51",
    "W3_2025Q4": "f4d6572fa4c44e789da3c2f96b8f3717",
    "W1_2026Q2": "e0ca66e6b67745ccb4ef244119609e7c",
}
GRID = tuple((q, weight) for q in (.75, .85, .90) for weight in (.05, .10, .15))


def _datetime_instrument(obj: pd.Series | pd.DataFrame):
    if obj.index.names == ["instrument", "datetime"]:
        return obj.reorder_levels(["datetime", "instrument"]).sort_index()
    return obj.sort_index()


def load_chase_features(index: pd.MultiIndex) -> pd.DataFrame:
    """Load only fields available by T 09:40 and derive settled 5-day state."""
    from qlib.data import D

    start = index.get_level_values("datetime").min()
    end = index.get_level_values("datetime").max()
    expressions = [f"Ref($close,{lag})" for lag in range(1, 7)] + [
        "$overnight_gap", "$startup_total", "$accel_5m",
    ]
    raw = D.features(
        D.instruments(UNIVERSE_MARKET), expressions,
        start_time=start, end_time=end, freq="day", disk_cache=0,
    )
    raw = _datetime_instrument(raw).reindex(index)
    closes = raw.iloc[:, :6]
    result = pd.DataFrame(index=index)
    result["mom_5d"] = closes.iloc[:, 0] / closes.iloc[:, 5] - 1.0
    daily_positive = [closes.iloc[:, i] > closes.iloc[:, i + 1] for i in range(5)]
    result["positive_days_5"] = sum(item.astype(float) for item in daily_positive) / 5.0
    result[["overnight_gap", "startup_total", "accel_5m"]] = raw.iloc[:, 6:].to_numpy()
    return result


def fast_metrics(score: pd.Series, label: pd.Series) -> dict[str, float]:
    """Signal-level proxy relative to the daily equal-weight member basket."""
    common = score.dropna().index.intersection(label.dropna().index)
    score, label = score.loc[common], label.loc[common]
    corr = correlation_metrics(score, label)
    rank = score.groupby(level="datetime").rank(method="first", ascending=False)
    selected = label.loc[rank <= 10].groupby(level="datetime").mean()
    basket = label.groupby(level="datetime").mean().reindex(selected.index)
    # Full-rebuild proxy deliberately charges a conservative round-trip cost.
    portfolio_net = selected - 0.002
    active = portfolio_net - basket
    nav = (1 + active).cumprod()
    drawdown = nav / nav.cummax() - 1
    return {
        **corr,
        "basket_active_annualized_proxy": float(active.mean() * 250),
        "basket_active_ir_proxy": float(active.mean() / active.std() * np.sqrt(250)),
        "basket_active_max_drawdown_proxy": float(drawdown.min()),
        "portfolio_p05": float(portfolio_net.quantile(.05)),
        "portfolio_loss5_rate": float((portfolio_net < -.05).mean()),
        "days": int(len(active)),
    }


def main() -> None:
    import qlib
    from qlib.config import C
    from qlib.workflow import R

    C["exp_manager"]["kwargs"]["uri"] = "file:" + str(ROOT / "mlruns")
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn", exp_manager=C["exp_manager"])
    output = {}
    for window, recorder_id in CHAMPION_RECORDERS.items():
        recorder = R.get_recorder(
            recorder_id=recorder_id, experiment_name="factor_challenger_ab"
        )
        score = _datetime_instrument(_series(recorder.load_object("pred.pkl")))
        label = _datetime_instrument(_series(recorder.load_object("label.pkl")))
        common = score.dropna().index.intersection(label.dropna().index)
        score, label = score.loc[common], label.loc[common]
        chase = compute_chase_risk(load_chase_features(common))
        paths = build_candidates(common)
        tracking = compute_tracking_risk(paths)
        output[window] = {"baseline": fast_metrics(score, label), "chase": {}, "tracking": {}}
        for threshold, weight in GRID:
            key = f"q{threshold:.2f}_w{weight:.2f}"
            output[window]["chase"][key] = fast_metrics(
                apply_top_pool_penalty(score, chase, threshold, weight), label
            )
            output[window]["tracking"][key] = fast_metrics(
                apply_top_pool_penalty(score, tracking, threshold, weight), label
            )
        RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
        print(f"{window}: complete", flush=True)

    for family in ("chase", "tracking"):
        print(f"\n{family}")
        keys = next(iter(output.values()))[family]
        for key in keys:
            excess_delta = np.mean([
                output[w][family][key]["basket_active_annualized_proxy"]
                - output[w]["baseline"]["basket_active_annualized_proxy"] for w in output
            ])
            dd_delta = np.mean([
                output[w][family][key]["basket_active_max_drawdown_proxy"]
                - output[w]["baseline"]["basket_active_max_drawdown_proxy"] for w in output
            ])
            wins = sum(
                output[w][family][key]["basket_active_annualized_proxy"]
                >= output[w]["baseline"]["basket_active_annualized_proxy"] for w in output
            )
            print(f"{key}: excess_delta={excess_delta:+.2%} dd_delta={dd_delta:+.2%} wins={wins}/4")
    print(f"Saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
