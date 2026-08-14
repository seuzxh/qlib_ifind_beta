"""Purged five-day validation gate for chase and basket-tracking overlays."""
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

from qlib_ifind_beta.config import OVERLAY_ROOT, UNIVERSE_MARKET
from qlib_ifind_beta.model_ensemble import blend_scores, correlation_metrics
from qlib_ifind_beta.risk_overlay import apply_top_pool_penalty
from scripts.diagnose_minute_basket_resonance import build_candidates
from scripts.validate_factor_challengers import _backtest
from scripts.validate_prediction_blend import _series
from scripts.validate_risk_overlay_quarters import fast_metrics
from scripts.validate_xgb_purged_rolling_gate import _validation_scores

SOURCE_PATH = ROOT / "data" / "xgb_purged_rolling_gate_ab.json"
RESULT_PATH = ROOT / "data" / "risk_overlay_purged_ab.json"
HF_EXPERIMENT = "rolling_90d_purged_hflgb"
XGB_EXPERIMENT = "rolling_90d_purged_xgb"
BLOCK_DAYS = 5
HISTORY_DAYS = 20
CANDIDATES = {
    "baseline": None,
    "track_q075_w005": (.75, .05),
    "track_q090_w010": (.90, .10),
}


def _datetime_instrument(series: pd.Series) -> pd.Series:
    if series.index.names == ["instrument", "datetime"]:
        return series.reorder_levels(["datetime", "instrument"]).sort_index()
    return series.sort_index()


def _recorder(recorder_id: str, experiment: str):
    from qlib.workflow import R
    return R.get_recorder(recorder_id=recorder_id, experiment_name=experiment)


def _test_score(row: dict) -> tuple[pd.Series, pd.Series]:
    hf = _recorder(row["hflgb_recorder_id"], HF_EXPERIMENT)
    xgb = _recorder(row["xgb_recorder_id"], XGB_EXPERIMENT)
    hf_score = _datetime_instrument(_series(hf.load_object("pred.pkl")))
    xgb_score = _datetime_instrument(_series(xgb.load_object("pred.pkl")))
    label = _datetime_instrument(_series(hf.load_object("label.pkl")))
    return blend_scores(hf_score, xgb_score, float(row["chosen_weight"])), label


def _first_validation(row: dict) -> tuple[pd.Series, pd.Series]:
    hf = _recorder(row["hflgb_recorder_id"], HF_EXPERIMENT)
    xgb = _recorder(row["xgb_recorder_id"], XGB_EXPERIMENT)
    task = hf.load_object("task")
    hf_score, xgb_score, label = _validation_scores(hf, xgb, task)
    return blend_scores(hf_score, xgb_score, float(row["chosen_weight"])), label


def basket_close_returns(start: str, end: str) -> pd.Series:
    """Costless equal-weight return of each date's actual member basket."""
    from qlib.data import D
    raw = D.features(
        D.instruments(UNIVERSE_MARKET), ["$close", "Ref($close,1)"],
        start_time=start, end_time=end, freq="day", disk_cache=0,
    )
    values = raw.iloc[:, 0] / raw.iloc[:, 1] - 1.0
    return values.groupby(level="datetime").mean().sort_index()


def _selected_instruments(score: pd.Series) -> pd.Series:
    rank = score.groupby(level="datetime").rank(method="first", ascending=False)
    return pd.Series(rank <= 10, index=score.index)


def _changed_days(base: pd.Series, candidate: pd.Series) -> int:
    base_selected = _selected_instruments(base)
    cand_selected = _selected_instruments(candidate)
    dates = base.index.get_level_values("datetime").unique()
    changed = 0
    for date in dates:
        a = set(base_selected.xs(date)[lambda x: x].index)
        b = set(cand_selected.xs(date)[lambda x: x].index)
        changed += a != b
    return changed


def _apply(name: str, score: pd.Series, tracking: pd.Series) -> pd.Series:
    params = CANDIDATES[name]
    if params is None:
        return score.copy()
    return apply_top_pool_penalty(score, tracking, params[0], params[1])


def _choose(history_score: pd.Series, history_label: pd.Series,
            history_tracking: pd.Series) -> tuple[str, dict]:
    baseline = fast_metrics(history_score, history_label)
    chosen, best_return = "baseline", baseline["basket_active_annualized_proxy"]
    metrics = {"baseline": baseline}
    for name in CANDIDATES:
        if name == "baseline":
            continue
        candidate_score = _apply(name, history_score, history_tracking)
        result = fast_metrics(candidate_score, history_label)
        result["changed_days"] = _changed_days(history_score, candidate_score)
        metrics[name] = result
        passes = (
            result["changed_days"] > 0
            and result["IC"] >= baseline["IC"] - .001
            and result["RankIC"] >= baseline["RankIC"] - .001
            and result["basket_active_annualized_proxy"] >= best_return
            and result["basket_active_max_drawdown_proxy"]
                >= baseline["basket_active_max_drawdown_proxy"]
        )
        if passes:
            chosen = name
            best_return = result["basket_active_annualized_proxy"]
    return chosen, metrics


def _summary(score: pd.Series, label: pd.Series,
             basket: pd.Series, start: str, end: str) -> dict:
    result = correlation_metrics(score, label)
    result.update(_backtest(score.to_frame("score"), start, end, active_benchmark=basket))
    result.update({f"proxy_{key}": value for key, value in fast_metrics(score, label).items()})
    return result


def _benchmark_summary(returns: pd.Series) -> dict:
    """Make the unusually strong daily-rebalanced basket auditable."""
    returns = returns.dropna()
    nav = (1.0 + returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    return {
        "days": int(len(returns)),
        "mean_daily_return": float(returns.mean()),
        "arithmetic_annualized_return": float(returns.mean() * 250),
        "compound_annualized_return": float(nav.iloc[-1] ** (250 / len(nav)) - 1),
        "total_return": float(nav.iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
    }


def main() -> None:
    import qlib
    from qlib.config import C
    from qlib.data import D

    C["exp_manager"]["kwargs"]["uri"] = "file:" + str(ROOT / "mlruns")
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn", exp_manager=C["exp_manager"])
    source = json.loads(SOURCE_PATH.read_text())
    test_scores, labels = [], []
    for row in source["segments"]:
        score, label = _test_score(row)
        test_scores.append(score)
        labels.append(label)
    test_score = pd.concat(test_scores).sort_index()
    label = pd.concat(labels).sort_index().reindex(test_score.index)
    first_valid_score, first_valid_label = _first_validation(source["segments"][0])
    history_score = pd.concat([first_valid_score, test_score]).sort_index()
    history_score = history_score[~history_score.index.duplicated(keep="last")]
    history_label = pd.concat([first_valid_label, label]).sort_index().reindex(history_score.index)

    print(f"building basket risk for {len(history_score)} rows", flush=True)
    paths = build_candidates(history_score.index)
    from qlib_ifind_beta.risk_overlay import compute_tracking_risk
    tracking = compute_tracking_risk(paths).reindex(history_score.index)

    calendar = [pd.Timestamp(date) for date in D.calendar(freq="day")]
    test_dates = sorted(test_score.index.get_level_values("datetime").unique())
    adjusted_blocks, block_rows = [], []
    for offset in range(0, len(test_dates), BLOCK_DAYS):
        block_dates = test_dates[offset:offset + BLOCK_DAYS]
        start = block_dates[0]
        pos = calendar.index(pd.Timestamp(start))
        settled_cutoff = calendar[pos - 2]
        available = sorted(
            date for date in history_score.index.get_level_values("datetime").unique()
            if date <= settled_cutoff
        )[-HISTORY_DAYS:]
        history_mask = history_score.index.get_level_values("datetime").isin(available)
        chosen, validation = _choose(
            history_score.loc[history_mask], history_label.loc[history_mask],
            tracking.loc[history_mask],
        )
        block_mask = test_score.index.get_level_values("datetime").isin(block_dates)
        block_score = test_score.loc[block_mask]
        # ``tracking`` also contains the prepended validation history, so a
        # positional boolean mask from ``test_score`` cannot be applied to it.
        # Align by the full (datetime, instrument) key instead.
        block_tracking = tracking.reindex(block_score.index)
        adjusted_blocks.append(_apply(chosen, block_score, block_tracking))
        block_rows.append({
            "test": [str(block_dates[0].date()), str(block_dates[-1].date())],
            "settled_validation": [str(available[0].date()), str(available[-1].date())],
            "chosen": chosen,
            "validation": validation,
        })
        print(f"{block_dates[0].date()}..{block_dates[-1].date()} -> {chosen}", flush=True)

    adjusted = pd.concat(adjusted_blocks).sort_index().reindex(test_score.index)
    start, end = str(test_dates[0].date()), str(test_dates[-1].date())
    basket = basket_close_returns(start, end)
    summary = {
        "baseline": _summary(test_score, label, basket, start, end),
        "validation_gated": _summary(adjusted, label, basket, start, end),
        "member_basket_benchmark": _benchmark_summary(basket),
        "changed_days": _changed_days(test_score, adjusted),
        "chosen_counts": pd.Series([row["chosen"] for row in block_rows]).value_counts().to_dict(),
    }
    output = {"block_days": BLOCK_DAYS, "history_days": HISTORY_DAYS,
              "blocks": block_rows, "summary": summary}
    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
