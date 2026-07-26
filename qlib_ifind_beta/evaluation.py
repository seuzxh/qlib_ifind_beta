"""Production evaluation helpers shared by retraining and model governance."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd


def backtest_td0(
    pred,
    start: str,
    end: str,
    active_benchmark: pd.Series | None = None,
    sell_at_941: bool = False,
) -> dict[str, float]:
    """Run the exact Top10/n_drop=8 TD0 promotion backtest."""
    from qlib.config import C
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    from qlib.data import D

    calendar = D.calendar(start_time=start, end_time=end, freq="day")
    bt_end = pd.Timestamp(calendar[-2])
    signal_index = pred.index.to_frame(index=False)
    first_signal = signal_index.groupby("instrument")["datetime"].min()
    exchange_codes = {
        code: [(pd.Timestamp(first_date), bt_end)]
        for code, first_date in first_signal.items()
    }
    raw_quote = D.features(
        exchange_codes,
        [
            "$price_941",
            "$close",
            "$change",
            "$factor",
            "$volume",
            "$change_941",
            "$limit_up",
            "$limit_down",
        ],
        start_time=start,
        end_time=bt_end,
        freq="day",
        disk_cache=0,
    )
    suspended = raw_quote["$close"].isna()
    extra_quote = raw_quote[["$price_941", "$close", "$factor", "$volume"]].copy()
    extra_quote["limit_buy"] = (
        raw_quote["$change_941"] >= raw_quote["$limit_up"]
    ) | suspended
    sell_change = raw_quote["$change_941"] if sell_at_941 else raw_quote["$change"]
    extra_quote["limit_sell"] = (
        sell_change <= raw_quote["$limit_down"]
    ) | suspended
    strategy = {
        "class": "TopkDropoutStrategyTD0",
        "module_path": "qlib_ifind_beta.td0_strategy",
        "kwargs": {
            "signal": pred,
            "topk": 10,
            "n_drop": 8,
            "hold_thresh": 1,
            "forbid_all_trade_at_limit": True,
        },
    }

    raw_features = D.features

    def no_dataset_cache(*args, **kwargs):
        kwargs["disk_cache"] = 0
        return raw_features(*args, **kwargs)

    old_limit = C.limit_threshold
    C.limit_threshold = None
    try:
        with patch.object(D, "features", side_effect=no_dataset_cache):
            report, _ = backtest_daily(
                start_time=start,
                end_time=bt_end,
                strategy=strategy,
                account=100_000_000,
                benchmark="SH000300",
                exchange_kwargs={
                    "freq": "day",
                    "codes": ["SH000300"],
                    "extra_quote": extra_quote,
                    "limit_threshold": None,
                    "deal_price": [
                        "$price_941",
                        "$price_941" if sell_at_941 else "$close",
                    ],
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
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
            missing = (
                benchmark_return[benchmark_return.isna()]
                .index.strftime("%Y-%m-%d")
                .tolist()
            )
            raise ValueError(
                f"active benchmark missing report dates: {missing[:5]}"
            )
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
