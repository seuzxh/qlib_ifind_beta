"""First-stage CSV/manual intraday production orchestrator.

Each subcommand consumes and writes explicit files under
data/production_signals/YYYY-MM-DD.  It never submits a broker order.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from qlib_ifind_beta.config import (
    CHAMPION_EXPERIMENT, CHAMPION_RECORDER_ID, CHAMPION_TOPK, DAY_CAL, PROJECT_ROOT,
)
from qlib_ifind_beta.live.intraday import (
    DayPaths, FACTOR_TIMES, apply_sell_fills, assemble_features,
    build_buy_orders, build_execution_snapshot, build_sell_orders,
    make_active_manifest, plan_topk_dropout, reconcile_positions,
    upsert_bars, validate_factor_bars, write_csv, write_json, write_parquet,
)


def _model_path() -> Path:
    matches = list((PROJECT_ROOT / "mlruns").glob(
        f"*/{CHAMPION_RECORDER_ID}/artifacts/params.pkl"
    ))
    if len(matches) != 1:
        raise RuntimeError(f"expected one Champion params.pkl, found {len(matches)}")
    return matches[0]


def preflight(date: str, base: Path | None = None) -> dict:
    import qlib
    paths = DayPaths.create(date, base)
    calendar = {line.strip() for line in Path(DAY_CAL).read_text().splitlines() if line.strip()}
    failures = []
    if date not in calendar:
        failures.append("not_in_qlib_calendar")
    model = _model_path()
    active = make_active_manifest(
        date=date, experiment=CHAMPION_EXPERIMENT, recorder_id=CHAMPION_RECORDER_ID,
        model_path=model, qlib_version=qlib.__version__,
    )
    write_json(paths.file("active_model_manifest.json"), active)
    result = {"date": date, "status": "PASS" if not failures else "FAIL",
              "failures": failures, "model_manifest_valid": True,
              "generated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    write_json(paths.file("preflight.json"), result)
    write_json(paths.file("run_manifest.json"), {**active, "preflight_status": result["status"]})
    return result


def universe_snapshot(date: str, base: Path | None = None) -> dict:
    from qlib_ifind_beta import universe
    paths = DayPaths.create(date, base)
    raw = universe.fetch_constituents(iv_date=date.replace("-", ""))
    raw = raw.rename(columns={"code_qlib": "code"})
    raw["eligible"] = ~raw["name"].astype(str).str.upper().str.contains(r"\*?ST|退市", regex=True)
    raw["exclude_reason"] = raw["eligible"].map({True: "", False: "ST_OR_DELISTING"})
    raw["source"] = "ifind_p03473"
    write_csv(paths.file("universe_raw.csv"), raw)
    write_csv(paths.file("universe_eligible.csv"), raw.loc[raw["eligible"]].copy())
    quality = {"date": date, "raw_count": len(raw), "unique_count": raw["code"].nunique(),
               "eligible_count": int(raw["eligible"].sum())}
    quality["status"] = "PASS" if quality["raw_count"] == quality["unique_count"] == 100 else "FAIL"
    write_json(paths.file("universe_quality.json"), quality)
    if quality["status"] != "PASS":
        raise RuntimeError(f"invalid 883926 snapshot: {quality}")
    # Update Qlib's authoritative time-varying instrument file from the cached history.
    universe.dump_universe("2024-01-01", date)
    return quality


def collect_factor_minute(date: str, minute: str, base: Path | None = None,
                          max_workers: int = 10) -> dict:
    if minute not in FACTOR_TIMES:
        raise ValueError(f"minute must be one of {FACTOR_TIMES}")
    from qlib_ifind_beta.realtime.data_fetch import fetch_bars_parallel
    paths = DayPaths.create(date, base)
    eligible = pd.read_csv(paths.file("universe_eligible.csv"))
    codes = eligible["code"].astype(str).tolist()
    fetched = fetch_bars_parallel(codes, count=1, max_workers=max_workers)
    rows = []
    now = datetime.now().astimezone().isoformat(timespec="milliseconds")
    for code, values in fetched.items():
        if not values:
            continue
        bar = values[-1]
        bar_time = str(bar.get("time", ""))[0:5]
        bar_date = str(bar.get("date", date))[0:10]
        if bar_time != minute or bar_date != date:
            continue
        rows.append({"date": date, "code": code, "bar_time": minute,
                     **{k: bar.get(k) for k in ("open", "high", "low", "close", "volume", "amount")},
                     "fetch_time": now, "source": "kline"})
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        raise RuntimeError(f"no closed {minute} bars returned")
    path = paths.file("factor_bars.parquet")
    existing = pd.read_parquet(path) if path.exists() else None
    merged = upsert_bars(existing, incoming, date=date)
    write_parquet(path, merged)
    status = pd.DataFrame([{"date": date, "bar_time": minute,
                            "requested": len(codes), "received": len(incoming),
                            "status": "PASS" if len(incoming) == len(codes) else "PARTIAL"}])
    status_path = paths.file("bar_collection_status.csv")
    old = pd.read_csv(status_path) if status_path.exists() else None
    status_all = pd.concat([old, status], ignore_index=True) if old is not None else status
    status_all = status_all.drop_duplicates(["date", "bar_time"], keep="last")
    write_csv(status_path, status_all)
    return status.iloc[0].to_dict()


def score_and_plan(date: str, positions_file: Path, base: Path | None = None,
                   factor_confirmed: bool = False) -> dict:
    from qlib_ifind_beta.realtime.data_fetch import get_daily_close_factor, get_prev_day_volumes_multi
    from qlib_ifind_beta.realtime.signal import _predict_in_memory
    paths = DayPaths.create(date, base)
    universe = pd.read_csv(paths.file("universe_eligible.csv"))
    codes = universe["code"].astype(str).tolist()
    bars = pd.read_parquet(paths.file("factor_bars.parquet"))
    complete, quality = validate_factor_bars(bars, codes)
    write_json(paths.file("bar_quality.json"), quality)
    if quality["status"] != "PASS":
        raise RuntimeError("factor-bar quality gate failed")
    prev = get_prev_day_volumes_multi(complete, date)
    daily = get_daily_close_factor(complete, date)
    for code, info in daily.items():
        if factor_confirmed:
            info["factor"] = info.get("prev_factor")
            info["factor_confirmed"] = True
    features = assemble_features(bars, prev, daily, complete)
    if len(features) < 80:
        raise RuntimeError(f"valid feature coverage too low: {len(features)}")
    write_parquet(paths.file("features.parquet"), features)
    feature_rows = features.set_index("code").to_dict("index")
    pred = _predict_in_memory(date, feature_rows, CHAMPION_TOPK,
                              change_941_map={}, use_online=False)
    scores = pd.DataFrame(pred["candidates"])
    write_csv(paths.file("scores.csv"), scores)
    positions = pd.read_csv(positions_file)
    write_csv(paths.file("positions_before.csv"), positions)
    decision = plan_topk_dropout(scores, positions, topk=10, n_drop=8, hold_thresh=1)
    write_json(paths.file("rebalance_decision.json"), decision)
    sells = build_sell_orders(decision, positions, date)
    write_csv(paths.file("sell_orders.csv"), sells)
    return {"candidate_count": len(scores), **decision}


def build_buys(date: str, sell_fills_file: Path, cash_before: float,
               base: Path | None = None, max_workers: int = 10,
               factor_confirmed: bool = False) -> dict:
    from qlib_ifind_beta.realtime.data_fetch import fetch_bars_parallel, get_daily_close_factor
    paths = DayPaths.create(date, base)
    positions = pd.read_csv(paths.file("positions_before.csv"))
    sells = pd.read_csv(paths.file("sell_orders.csv"))
    fills = pd.read_csv(sell_fills_file)
    after_sell, cash = apply_sell_fills(positions, cash_before, sells, fills)
    write_csv(paths.file("sell_fills.csv"), fills)
    write_csv(paths.file("positions_after_sell.csv"), after_sell)
    write_json(paths.file("cash_after_sell.json"), cash)
    scores = pd.read_csv(paths.file("scores.csv"))
    codes = scores["code"].astype(str).tolist()
    fetched = fetch_bars_parallel(codes, count=1, max_workers=max_workers)
    rows = []
    for code, values in fetched.items():
        if values:
            bar = values[-1]
            if str(bar.get("time", ""))[0:5] == "09:41":
                rows.append({"code": code, "bar_time": "09:41", **bar})
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("no closed 09:41 execution bars returned")
    daily = get_daily_close_factor(codes, date)
    if factor_confirmed:
        for info in daily.values():
            info["factor"] = info.get("prev_factor")
            info["factor_confirmed"] = True
    execution = build_execution_snapshot(raw, daily)
    write_parquet(paths.file("execution_bar_0941.parquet"), execution)
    decision = json.loads(paths.file("rebalance_decision.json").read_text())
    buys = build_buy_orders(decision, after_sell, cash["cash_after_sell"],
                            execution, scores, date)
    write_csv(paths.file("buy_orders.csv"), buys)
    return {"cash_after_sell": cash["cash_after_sell"], "buy_order_count": len(buys)}


def reconcile(date: str, expected_positions: Path, broker_positions: Path,
              expected_cash: float, broker_cash: float, base: Path | None = None) -> dict:
    paths = DayPaths.create(date, base)
    expected, broker = pd.read_csv(expected_positions), pd.read_csv(broker_positions)
    result = reconcile_positions(expected, broker, expected_cash, broker_cash)
    write_csv(paths.file("positions_after_close.csv"), broker)
    write_json(paths.file("daily_reconciliation.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="883926 CSV/manual intraday production")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "universe"):
        cmd = sub.add_parser(name); cmd.add_argument("--date", required=True)
    cmd = sub.add_parser("collect-factor-bars"); cmd.add_argument("--date", required=True)
    cmd.add_argument("--minute", required=True); cmd.add_argument("--max-workers", type=int, default=10)
    cmd = sub.add_parser("score-and-plan-sells"); cmd.add_argument("--date", required=True)
    cmd.add_argument("--positions", type=Path, required=True); cmd.add_argument("--factor-confirmed", action="store_true")
    cmd = sub.add_parser("build-buy-orders"); cmd.add_argument("--date", required=True)
    cmd.add_argument("--sell-fills", type=Path, required=True); cmd.add_argument("--cash-before", type=float, required=True)
    cmd.add_argument("--factor-confirmed", action="store_true")
    cmd = sub.add_parser("reconcile"); cmd.add_argument("--date", required=True)
    cmd.add_argument("--expected-positions", type=Path, required=True); cmd.add_argument("--broker-positions", type=Path, required=True)
    cmd.add_argument("--expected-cash", type=float, required=True); cmd.add_argument("--broker-cash", type=float, required=True)
    cmd = sub.add_parser("retrain"); cmd.add_argument("--asof", required=True)
    cmd = sub.add_parser("promote-model"); cmd.add_argument("--manifest", type=Path, required=True)
    cmd.add_argument("--approved-by", required=True)
    args = parser.parse_args()
    if args.command == "preflight": result = preflight(args.date)
    elif args.command == "universe": result = universe_snapshot(args.date)
    elif args.command == "collect-factor-bars": result = collect_factor_minute(args.date, args.minute, max_workers=args.max_workers)
    elif args.command == "score-and-plan-sells": result = score_and_plan(args.date, args.positions, factor_confirmed=args.factor_confirmed)
    elif args.command == "build-buy-orders": result = build_buys(
        args.date, args.sell_fills, args.cash_before, factor_confirmed=args.factor_confirmed
    )
    elif args.command == "reconcile": result = reconcile(args.date, args.expected_positions, args.broker_positions, args.expected_cash, args.broker_cash)
    elif args.command == "retrain":
        from scripts.retrain import run; result = run(args.asof)
    else:
        from scripts.retrain import promote_candidate; result = promote_candidate(args.manifest, args.approved_by)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
