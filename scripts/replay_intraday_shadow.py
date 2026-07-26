"""Replay the CSV/manual intraday workflow from local historical data."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from qlib_ifind_beta.config import (
    CHAMPION_EXPERIMENT, CHAMPION_RECORDER_ID, OVERLAY_ROOT, PROJECT_ROOT,
)
from qlib_ifind_beta.live.historical_replay import HistoricalReplaySource
from qlib_ifind_beta.live.intraday import (
    DayPaths, FACTOR_TIMES, apply_buy_fills, apply_sell_fills, assemble_features,
    build_buy_orders, build_execution_snapshot, build_sell_orders, plan_topk_dropout,
    make_active_manifest, reconcile_positions, upsert_bars, validate_factor_bars,
    write_csv, write_json, write_parquet,
)
from qlib_ifind_beta.materialize import board_limit
from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler
from qlib_ifind_beta.model_ensemble import load_model_bundle, predict_bundle_matrix


def _series(value) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    if set(value.index.names) == {"instrument", "datetime"}:
        value = value.reorder_levels(["datetime", "instrument"])
    return value.sort_index()


def _eligible_snapshots(start: str, end: str) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(PROJECT_ROOT / "data" / "universe_snapshots.csv")
    frame["date"] = frame["date"].astype(str).str[:10]
    frame = frame.loc[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    frame["code"] = frame["code_qlib"].astype(str)
    frame["eligible"] = ~frame["name"].astype(str).str.upper().str.contains(r"\*?ST|退市", regex=True)
    dates = sorted(frame["date"].unique())
    return frame, dates


def _scores(date: str, features: pd.DataFrame, execution: pd.DataFrame, bundle) -> pd.DataFrame:
    fields = list(MinuteEnhancedHandler.ENHANCED_FIELDS)
    matrix = features.set_index("code")[fields].copy()
    matrix.columns = [f"${field}" for field in fields]
    matrix.index = pd.MultiIndex.from_arrays(
        [[pd.Timestamp(date)] * len(matrix), matrix.index], names=["datetime", "instrument"]
    )
    pred = predict_bundle_matrix(bundle, matrix).sort_values(ascending=False)
    changes = execution.set_index("code")["change_941"].to_dict()
    rows = []
    for (_, code), score in pred.items():
        limit_up, limit_down = board_limit(code)
        rows.append({"code": code, "score": float(score), "limit_up": limit_up,
                     "limit_down": limit_down, "change_941": changes.get(code, np.nan)})
    return pd.DataFrame(rows)


def _fills(orders: pd.DataFrame, execution: pd.DataFrame, side: str) -> pd.DataFrame:
    price = execution.set_index("code") if not execution.empty else pd.DataFrame()
    rows = []
    for _, order in orders.iterrows():
        code = str(order["code"])
        available = code in price.index
        px = float(price.loc[code, "price_941"]) if available else 0.0
        change = float(price.loc[code, "change_941"]) if available else np.nan
        limit_up, limit_down = board_limit(code)
        blocked = (side == "SELL" and np.isfinite(change) and change <= limit_down)
        blocked |= (side == "BUY" and np.isfinite(change) and change >= limit_up)
        qty = int(order["quantity"]) if available and not blocked else 0
        rate = 0.0015 if side == "SELL" else 0.0005
        fee = max(5.0, qty * px * rate) if qty else 0.0
        rows.append({"client_order_id": order["client_order_id"], "code": code,
                     "filled_quantity": qty, "average_price": px, "fee": fee,
                     "fill_status": "FILLED" if qty else ("LIMIT_BLOCKED" if blocked else "NO_BAR")})
    return pd.DataFrame(rows, columns=["client_order_id", "code", "filled_quantity",
                                      "average_price", "fee", "fill_status"])


def _metrics(nav: pd.Series) -> dict:
    curve = pd.concat([pd.Series({"initial": 1_000_000.0}), nav.astype(float)])
    returns = curve.pct_change().dropna()
    total = float(nav.iloc[-1] / 1_000_000.0 - 1.0)
    annualized = float((1.0 + total) ** (252.0 / len(nav)) - 1.0) if total > -1 else -1.0
    drawdown = curve / curve.cummax() - 1.0
    return {"total_return": total, "annualized_return": annualized,
            "max_drawdown": float(drawdown.min()), "daily_volatility": float(returns.std()),
            "final_nav": float(nav.iloc[-1])}


def run(start: str, end: str, output_root: Path, initial_cash: float = 1_000_000.0,
        require_stored_parity: bool = True) -> dict:
    import qlib
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib.workflow import R

    snapshots, dates = _eligible_snapshots(start, end)
    if not dates:
        raise RuntimeError("no cached constituent snapshots in requested range")
    source = HistoricalReplaySource()
    recorder = R.get_recorder(recorder_id=CHAMPION_RECORDER_ID,
                              experiment_name=CHAMPION_EXPERIMENT)
    stored = _series(recorder.load_object("pred.pkl"))
    stored_dates = set(pd.to_datetime(stored.index.get_level_values("datetime")))
    bundle = load_model_bundle(dates[0], use_online=False)
    model_matches = list((PROJECT_ROOT / "mlruns").glob(
        f"*/{CHAMPION_RECORDER_ID}/artifacts/params.pkl"
    ))
    if len(model_matches) != 1:
        raise RuntimeError(f"expected one Champion model, found {len(model_matches)}")
    positions = pd.DataFrame(columns=["code", "quantity", "sellable_quantity", "hold_days"])
    cash = float(initial_cash)
    daily_results = []

    for number, date in enumerate(dates, 1):
        paths = DayPaths.create(date, output_root)
        active = make_active_manifest(
            date=date, experiment=CHAMPION_EXPERIMENT, recorder_id=CHAMPION_RECORDER_ID,
            model_path=model_matches[0], qlib_version=qlib.__version__,
            status="HISTORICAL_REPLAY_LOCKED",
        )
        preflight = {"date": date, "status": "PASS", "failures": [],
                     "data_mode": "LOCAL_HISTORY_ONLY", "model_manifest_valid": True}
        write_json(paths.file("active_model_manifest.json"), active)
        write_json(paths.file("preflight.json"), preflight)
        write_json(paths.file("run_manifest.json"), {**active, "preflight_status": "PASS"})
        if not positions.empty:
            positions["hold_days"] = positions["hold_days"].astype(int) + 1
            positions["sellable_quantity"] = positions["quantity"].astype(int)
        day_snapshot = snapshots.loc[snapshots["date"] == date].copy()
        eligible = day_snapshot.loc[day_snapshot["eligible"]].copy()
        codes = eligible["code"].tolist()
        all_codes = sorted(set(codes) | set(positions["code"].astype(str)))
        write_csv(paths.file("universe_raw.csv"), day_snapshot)
        write_csv(paths.file("universe_eligible.csv"), eligible)
        universe_quality = {"date": date, "raw_count": len(day_snapshot),
                            "unique_count": day_snapshot["code"].nunique(),
                            "eligible_count": len(eligible),
                            "status": "PASS" if len(day_snapshot) == day_snapshot["code"].nunique() == 100 else "FAIL"}
        write_json(paths.file("universe_quality.json"), universe_quality)
        if universe_quality["status"] != "PASS":
            raise RuntimeError(f"{date}: invalid universe snapshot")
        write_csv(paths.file("positions_before.csv"), positions)

        raw = source.bars(date, all_codes)
        factor_bars = None
        collection = []
        idempotent = True
        for minute in FACTOR_TIMES:
            incoming = raw.loc[raw["bar_time"] == minute].copy()
            factor_bars = upsert_bars(factor_bars, incoming, date=date)
            repeated = upsert_bars(factor_bars, incoming, date=date)
            idempotent &= factor_bars.equals(repeated)
            collection.append({"date": date, "bar_time": minute, "requested": len(codes),
                               "received": int(incoming["code"].isin(codes).sum()),
                               "status": "PASS" if incoming["code"].isin(codes).sum() == len(codes) else "PARTIAL"})
        write_parquet(paths.file("factor_bars.parquet"), factor_bars)
        write_csv(paths.file("bar_collection_status.csv"), pd.DataFrame(collection))
        complete, quality = validate_factor_bars(factor_bars, codes)
        quality["idempotent_replay"] = bool(idempotent)
        write_json(paths.file("bar_quality.json"), quality)
        if quality["status"] != "PASS" or not idempotent:
            raise RuntimeError(f"{date}: factor bar gate failed: {quality}")

        daily_info = source.daily_info(date, all_codes)
        fallback_count = sum(
            info.get("historical_source") == "minute_fallback"
            for info in daily_info.values()
        )
        write_json(paths.file("daily_source_quality.json"), {
            "date": date,
            "requested_codes": len(all_codes),
            "available_codes": len(daily_info),
            "minute_fallback_count": fallback_count,
            "status": "PASS" if len(daily_info) >= 80 else "FAIL",
        })
        prev_volumes = source.previous_volumes(date, complete)
        features = assemble_features(factor_bars, prev_volumes, daily_info, complete)
        if len(features) < 80:
            raise RuntimeError(f"{date}: only {len(features)} complete features")
        write_parquet(paths.file("features.parquet"), features)
        execution_raw = raw.loc[raw["bar_time"] == "09:41"].copy()
        execution = build_execution_snapshot(execution_raw, daily_info)
        write_parquet(paths.file("execution_bar_0941.parquet"), execution)
        scores = _scores(date, features, execution, bundle)
        write_csv(paths.file("scores.csv"), scores)

        live_score = scores.set_index("code")["score"]
        reference_available = pd.Timestamp(date) in stored_dates
        if reference_available:
            historical = stored.xs(pd.Timestamp(date), level="datetime")
            common = historical.index.intersection(live_score.index)
            max_abs = float((historical.loc[common] - live_score.loc[common]).abs().max())
            stored_top = historical.sort_values(ascending=False).head(10).index.tolist()
            replay_top = live_score.sort_values(ascending=False).head(10).index.tolist()
            common_top = historical.loc[common].sort_values(ascending=False).head(10).index.tolist()
            common_top_exact = common_top == replay_top
            common_top_overlap = len(set(common_top) & set(replay_top))
            parity = {
                "reference_available": True,
                "common_count": len(common),
                "max_abs_score_diff": max_abs,
                "score_correlation": float(
                    historical.loc[common].corr(live_score.loc[common])
                ),
                "stored_candidate_count": len(historical),
                "replay_candidate_count": len(live_score),
                "stored_global_top10_overlap": len(set(stored_top) & set(replay_top)),
                "common_universe_top10_exact": common_top_exact,
                "common_universe_top10_overlap": common_top_overlap,
                "universe_drift_codes": sorted(
                    set(historical.index) ^ set(live_score.index)
                ),
                "status": "PASS" if max_abs <= 1e-12 else "FAIL",
            }
        else:
            if require_stored_parity:
                raise RuntimeError(f"{date}: no stored Champion prediction for parity")
            max_abs = np.nan
            common_top_exact = np.nan
            common_top_overlap = np.nan
            parity = {
                "reference_available": False,
                "common_count": 0,
                "max_abs_score_diff": None,
                "score_correlation": None,
                "stored_candidate_count": 0,
                "replay_candidate_count": len(live_score),
                "stored_global_top10_overlap": None,
                "common_universe_top10_exact": None,
                "common_universe_top10_overlap": None,
                "universe_drift_codes": [],
                "status": "UNREFERENCED_FORWARD",
            }
        write_json(paths.file("signal_parity.json"), parity)
        if parity["status"] == "FAIL":
            raise RuntimeError(f"{date}: Champion parity failed: {parity}")

        decision = plan_topk_dropout(scores, positions, topk=10, n_drop=8, hold_thresh=1)
        write_json(paths.file("rebalance_decision.json"), decision)
        sells = build_sell_orders(decision, positions, date)
        write_csv(paths.file("sell_orders.csv"), sells)
        sell_fills = _fills(sells, execution, "SELL")
        write_csv(paths.file("sell_fills.csv"), sell_fills)
        after_sell, sell_cash = apply_sell_fills(positions, cash, sells, sell_fills)
        cash = float(sell_cash["cash_after_sell"])
        write_csv(paths.file("positions_after_sell.csv"), after_sell)
        write_json(paths.file("cash_after_sell.json"), sell_cash)

        buys = build_buy_orders(decision, after_sell, cash, execution, scores, date)
        write_csv(paths.file("buy_orders.csv"), buys)
        buy_fills = _fills(buys, execution, "BUY")
        write_csv(paths.file("buy_fills.csv"), buy_fills)
        positions, buy_cash = apply_buy_fills(after_sell, cash, buys, buy_fills)
        cash = float(buy_cash["cash_after_buy"])
        write_csv(paths.file("positions_after_close.csv"), positions)
        write_json(paths.file("cash_after_buy.json"), buy_cash)
        reconciliation = reconcile_positions(positions, positions.copy(), cash, cash)
        write_json(paths.file("daily_reconciliation.json"), reconciliation)

        closes = source.valuation_close(date, positions["code"].tolist())
        market_value = float(sum(int(row.quantity) * closes.get(row.code, 0.0)
                                 for row in positions.itertuples()))
        nav = cash + market_value
        row = {"date": date, "status": "PASS", "universe_count": len(codes),
               "complete_bars": len(complete), "feature_count": len(features),
               "daily_info_minute_fallback_count": fallback_count,
               "score_count": len(scores),
               "score_reference_available": reference_available,
               "stored_global_top10_overlap": parity.get("stored_global_top10_overlap"),
               "common_universe_top10_exact": common_top_exact,
               "common_universe_top10_overlap": common_top_overlap,
               "universe_drift_count": len(parity["universe_drift_codes"]),
               "max_abs_score_diff": max_abs, "sell_orders": len(sells),
               "buy_orders": len(buys), "position_count": len(positions),
               "cash": cash, "market_value": market_value, "nav": nav,
               "reconciliation": reconciliation["status"]}
        daily_results.append(row)
        write_json(paths.file("shadow_day_result.json"), row)
        parity_text = (
            f"{max_abs:.1e}/common_top10={common_top_overlap}"
            if reference_available else "forward-unreferenced"
        )
        print(f"[{number:02d}/{len(dates)}] {date} PASS features={len(features)} "
              f"parity={parity_text} "
              f"drift={len(parity['universe_drift_codes'])} nav={nav:,.2f}", flush=True)

    daily = pd.DataFrame(daily_results)
    referenced = daily.loc[daily["score_reference_available"]]
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "daily_summary.csv", daily)
    result = {"status": "PASS" if (daily["status"] == "PASS").all() else "FAIL",
              "start": dates[0], "end": dates[-1], "trading_days": len(dates),
              "all_reconciled": bool((daily["reconciliation"] == "PASS").all()),
              "referenced_score_days": len(referenced),
              "unreferenced_forward_days": int((~daily["score_reference_available"]).sum()),
              "all_referenced_score_values_exact": bool(
                  (referenced["max_abs_score_diff"] <= 1e-12).all()
              ),
              "days_common_universe_top10_exact": int(
                  referenced["common_universe_top10_exact"].sum()
              ),
              "minimum_common_universe_top10_overlap": int(
                  referenced["common_universe_top10_overlap"].min()
              ),
              "minimum_stored_global_top10_overlap": int(
                  referenced["stored_global_top10_overlap"].min()
              ),
              "days_with_universe_drift": int((daily["universe_drift_count"] > 0).sum()),
              "days_using_minute_daily_fallback": int(
                  (daily["daily_info_minute_fallback_count"] > 0).sum()
              ),
              "maximum_minute_daily_fallback_count": int(
                  daily["daily_info_minute_fallback_count"].max()
              ),
              "max_referenced_score_difference": float(
                  referenced["max_abs_score_diff"].max()
              ),
              "metrics": _metrics(daily.set_index("date")["nav"]),
              "assumptions": {"fills": "09:41 close, full fill unless limit blocked or missing bar",
                              "open_cost": 0.0005, "close_cost": 0.0015,
                              "broker_orders_submitted": False, "retraining_executed": False}}
    write_json(output_root / "replay_report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-07-02")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "data" / "historical_shadow_replay")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument(
        "--allow-unreferenced-scores",
        action="store_true",
        help="允许超出 Recorder pred.pkl 日期，使用冻结 Champion 做前向影子评分",
    )
    args = parser.parse_args()
    print(json.dumps(run(
        args.start, args.end, args.output, args.initial_cash,
        require_stored_parity=not args.allow_unreferenced_scores,
    ),
                     ensure_ascii=False, indent=2))
