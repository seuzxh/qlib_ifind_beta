"""Two-window clean A/B for champion and the close-position ablation.

Each run keeps HFLGBModel, label, segments, strategy and exchange fixed.  The
only variable is the handler's feature list.  Promotion metrics come from the
actual TopkDropoutStrategyTD0 Qlib backtest, not the fast daily-rebuild proxy.
"""
from __future__ import annotations

import json
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from qlib_ifind_beta.config import CHAMPION_LABEL_EXPR, OVERLAY_ROOT, UNIVERSE_MARKET


HANDLERS = {
    "champion18": ("MinuteEnhancedHandler", "qlib_ifind_beta.minute_enhanced_handler"),
    "pruned15": ("MinutePrunedHandler", "qlib_ifind_beta.minute_pruned_handler"),
    "turnover19": ("MinuteTurnoverHandler", "qlib_ifind_beta.minute_turnover_handler"),
    "tailrank18": ("MinuteEnhancedHandler", "qlib_ifind_beta.minute_enhanced_handler"),
    "tailbinary18": ("MinuteEnhancedHandler", "qlib_ifind_beta.minute_enhanced_handler"),
    "regression18": ("MinuteEnhancedHandler", "qlib_ifind_beta.minute_enhanced_handler"),
    "xgb18": ("MinuteEnhancedHandler", "qlib_ifind_beta.minute_enhanced_handler"),
    "path21": ("MinutePathHandler", "qlib_ifind_beta.minute_path_handler"),
}

WINDOWS = {
    "W4_2024Q4": {
        "train": ["2024-01-01", "2024-06-30"],
        "valid": ["2024-07-01", "2024-09-30"],
        "test": ["2024-10-01", "2024-12-31"],
    },
    "W2_2025Q2": {
        "train": ["2024-01-01", "2024-12-31"],
        "valid": ["2025-01-01", "2025-03-31"],
        "test": ["2025-04-01", "2025-07-02"],
    },
    "W1_2026Q2": {
        "train": ["2024-01-01", "2025-12-31"],
        "valid": ["2026-01-01", "2026-03-31"],
        "test": ["2026-04-01", "2026-07-02"],
    },
    # Held-out robustness window: added only after the turnover-confirmation
    # hypothesis and its rerank weight grid were fixed on W2/W1.
    "W3_2025Q4": {
        "train": ["2024-01-01", "2025-06-30"],
        "valid": ["2025-07-01", "2025-09-30"],
        "test": ["2025-10-01", "2025-12-31"],
    },
}


def _task(handler_name: str, module_path: str, segs: dict, variant: str) -> dict:
    if variant == "xgb18":
        model = {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
            "kwargs": {
                "objective": "reg:squarederror", "eval_metric": "rmse",
                "eta": 0.05, "max_depth": 6, "alpha": 5.0,
                "lambda": 10.0, "nthread": 20, "verbosity": 0,
            },
        }
    elif variant == "regression18":
        model = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse", "learning_rate": 0.05,
                "max_depth": 6, "num_leaves": 64, "num_threads": 20,
                "lambda_l1": 5.0, "lambda_l2": 10.0,
                "num_boost_round": 200, "early_stopping_rounds": 20,
            },
        }
    elif variant in {"tailrank18", "tailbinary18"}:
        model = {
            "class": "TailRankLGBModel" if variant == "tailrank18" else "TailBinaryLGBModel",
            "module_path": "qlib_ifind_beta.tail_rank_model",
            "kwargs": {
                "learning_rate": 0.05, "max_depth": 6, "num_leaves": 64,
                "num_threads": 20, "lambda_l1": 5.0, "lambda_l2": 10.0,
            },
        }
    else:
        model = {
            "class": "HFLGBModel",
            "module_path": "qlib.contrib.model.highfreq_gdbt_model",
            "kwargs": {
                "loss": "binary", "learning_rate": 0.05,
                "max_depth": 6, "num_leaves": 64, "num_threads": 20,
                "lambda_l1": 5.0, "lambda_l2": 10.0,
            },
        }
    return {
        "model": model,
        "dataset": {
            "class": "DatasetH", "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": handler_name, "module_path": module_path,
                    "kwargs": {
                        "instruments": UNIVERSE_MARKET,
                        "start_time": segs["train"][0],
                        "end_time": segs["test"][1],
                        "fit_start_time": segs["train"][0],
                        "fit_end_time": segs["train"][1],
                        "label": [CHAMPION_LABEL_EXPR],
                    },
                },
                "segments": segs,
            },
        },
        "record": [
            {"class": "SignalRecord", "module_path": "qlib.workflow.record_temp"},
            {"class": "SigAnaRecord", "module_path": "qlib.workflow.record_temp"},
        ],
    }


def _daily_corr(pred: pd.Series, label: pd.Series, method: str) -> np.ndarray:
    values = []
    dates = pred.index.get_level_values("datetime").unique()
    for date in dates:
        p = pred.xs(date, level="datetime").dropna()
        y = label.xs(date, level="datetime").dropna()
        common = p.index.intersection(y.index)
        if len(common) >= 20:
            value = p.reindex(common).corr(y.reindex(common), method=method)
            if pd.notna(value):
                values.append(value)
    return np.asarray(values)


def _backtest(pred, start: str, end: str,
              active_benchmark: pd.Series | None = None):
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    from qlib.config import C
    from qlib.data import D
    from unittest.mock import patch

    cal = D.calendar(start_time=start, end_time=end, freq="day")
    bt_end = pd.Timestamp(cal[-2])  # final prediction needs T+1 close; do not open it
    signal_index = pred.index.to_frame(index=False)
    first_signal = signal_index.groupby("instrument")["datetime"].min()
    # A code cannot enter the portfolio before its first signal.  From then on
    # keep quotes through bt_end so a limit-down-blocked position can be carried
    # for arbitrarily many days without loading every code for the full window.
    exchange_codes = {
        code: [(pd.Timestamp(first_date), bt_end)]
        for code, first_date in first_signal.items()
    }
    # Qlib expression comparisons in Exchange's dataset query are extremely
    # slow on this 2,500-code rotating universe.  Load the raw fields once and
    # compute the exact same boolean limits vectorially, then inject them via
    # Exchange.extra_quote (an official Exchange extension point).
    raw_quote = D.features(
        exchange_codes,
        ["$price_941", "$close", "$change", "$factor", "$volume",
         "$change_941", "$limit_up", "$limit_down"],
        start_time=start, end_time=bt_end, freq="day", disk_cache=0,
    )
    suspended = raw_quote["$close"].isna()
    extra_quote = raw_quote[["$price_941", "$close", "$factor", "$volume"]].copy()
    extra_quote["limit_buy"] = (raw_quote["$change_941"] >= raw_quote["$limit_up"]) | suspended
    extra_quote["limit_sell"] = (raw_quote["$change"] <= raw_quote["$limit_down"]) | suspended
    strategy = {
        "class": "TopkDropoutStrategyTD0",
        "module_path": "qlib_ifind_beta.td0_strategy",
        "kwargs": {
            "signal": pred, "topk": 10, "n_drop": 8, "hold_thresh": 1,
            "forbid_all_trade_at_limit": True,
        },
    }
    raw_features = D.features
    def no_dataset_cache(*args, **kwargs):
        # Exchange hard-codes disk_cache=True. Interrupted exploratory runs can
        # leave a cache build lock; direct provider reads are deterministic and
        # fast enough for a 62-day promotion gate.
        kwargs["disk_cache"] = 0
        return raw_features(*args, **kwargs)

    old_limit = C.limit_threshold
    C.limit_threshold = None
    try:
        with patch.object(D, "features", side_effect=no_dataset_cache):
            report, _ = backtest_daily(
                start_time=start, end_time=bt_end, strategy=strategy,
                account=100_000_000, benchmark="SH000300",
                exchange_kwargs={
                    "freq": "day", "codes": ["SH000300"],
                    "extra_quote": extra_quote,
                    "limit_threshold": None,
                    "deal_price": ["$price_941", "$close"],
                    "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5,
                },
            )
    finally:
        C.limit_threshold = old_limit
    abs_net = report["return"] - report["cost"]
    if active_benchmark is None:
        benchmark_return = report["bench"]
    else:
        benchmark_return = active_benchmark.copy()
        benchmark_return.index = pd.to_datetime(benchmark_return.index)
        benchmark_return = benchmark_return.reindex(report.index)
        if benchmark_return.isna().any():
            missing = benchmark_return[benchmark_return.isna()].index.strftime("%Y-%m-%d").tolist()
            raise ValueError(f"active benchmark missing report dates: {missing[:5]}")
    active_net = report["return"] - benchmark_return - report["cost"]
    abs_risk = risk_analysis(abs_net, freq="day")["risk"]
    active_risk = risk_analysis(active_net, freq="day")["risk"]
    return {
        "absolute_annualized_return": float(abs_risk["annualized_return"]),
        "absolute_max_drawdown": float(abs_risk["max_drawdown"]),
        "excess_annualized_return": float(active_risk["annualized_return"]),
        "excess_information_ratio": float(active_risk["information_ratio"]),
        "excess_max_drawdown": float(active_risk["max_drawdown"]),
    }


def main(window_filter: str | None = None, variant_filter: str | None = None,
         skip_backtest: bool = False):
    import qlib
    from qlib.config import C
    from qlib.model.trainer import task_train

    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(ROOT / "mlruns")
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn", exp_manager=exp_manager)

    out = ROOT / "data" / "factor_challenger_ab.json"
    if (window_filter or variant_filter) and out.exists():
        results = json.loads(out.read_text())
    else:
        results = {}
    for window_name, segs in WINDOWS.items():
        if window_filter and window_name != window_filter:
            continue
        results.setdefault(window_name, {})
        for variant, (handler, module) in HANDLERS.items():
            if variant_filter and variant != variant_filter:
                continue
            print(f"\n=== {window_name} / {variant} ===", flush=True)
            rec = task_train(_task(handler, module, segs, variant),
                             experiment_name="factor_challenger_ab")
            pred = rec.load_object("pred.pkl")
            label = rec.load_object("label.pkl")
            p = pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred
            y = label.iloc[:, 0] if isinstance(label, pd.DataFrame) else label
            pearson = _daily_corr(p, y, "pearson")
            rankic = _daily_corr(p, y, "spearman")
            metrics = {
                "recorder_id": rec.id,
                "IC": float(pearson.mean()),
                "ICIR": float(pearson.mean() / pearson.std()),
                "RankIC": float(rankic.mean()),
                "RankICIR": float(rankic.mean() / rankic.std()),
                "RankIC_positive_rate": float((rankic > 0).mean()),
            }
            if not skip_backtest:
                metrics.update(_backtest(pred, segs["test"][0], segs["test"][1]))
            results[window_name][variant] = metrics
            print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", choices=WINDOWS)
    parser.add_argument("--variant", choices=HANDLERS)
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()
    main(args.window, args.variant, args.skip_backtest)
