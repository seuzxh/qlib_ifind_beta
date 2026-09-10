import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qlib_ifind_beta.live.intraday import (
    FACTOR_TIMES, SCHEME_B_STATUS, apply_buy_fills, apply_sell_fills, assemble_features,
    build_buy_orders, build_execution_snapshot, build_sell_orders,
    plan_topk_dropout, reconcile_positions, upsert_bars, validate_factor_bars,
)


def _bars(codes=("A",), times=FACTOR_TIMES):
    rows = []
    for code in codes:
        for i, tm in enumerate(times):
            rows.append({"date": "2026-07-20", "code": code, "bar_time": tm,
                         "open": 10 + i * .01, "high": 10.2 + i * .01,
                         "low": 9.9 + i * .01, "close": 10.1 + i * .01,
                         "volume": 1000 + i, "fetch_time": f"{tm}:02"})
    return pd.DataFrame(rows)


def test_bar_upsert_is_idempotent_and_quality_requires_exact_window():
    bars = _bars(("A", "B"))
    twice = upsert_bars(bars, bars, date="2026-07-20")
    assert len(twice) == 20
    complete, quality = validate_factor_bars(twice, ["A", "B"], min_candidates=2)
    assert complete == ["A", "B"] and quality["status"] == "PASS"
    incomplete = twice.loc[~((twice.code == "B") & (twice.bar_time == "09:36"))]
    complete, quality = validate_factor_bars(incomplete, ["A", "B"], min_candidates=2)
    assert complete == ["A"] and quality["missing"]["B"] == ["09:36"]


def test_assemble_features_has_frozen_order_and_never_needs_0941():
    from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler
    bars = _bars()
    daily = {"A": {"prev_close": 9.9, "prev_factor": 1.0,
                    "factor": 1.0, "factor_confirmed": True}}
    frame = assemble_features(bars, {"A": [240000] * 4}, daily, ["A"])
    assert frame.columns.tolist() == ["code", *MinuteEnhancedHandler.ENHANCED_FIELDS]
    assert len(frame) == 1
    assert "price_941" not in frame.columns and "change_941" not in frame.columns


def test_unknown_today_factor_fails_closed_per_stock():
    bars = _bars()
    daily = {"A": {"prev_close": 9.9, "prev_factor": 1.0,
                    "factor": None, "factor_confirmed": False}}
    assert assemble_features(bars, {"A": [240000] * 4}, daily, ["A"]).empty


def test_stateful_topk_dropout_and_sell_orders():
    scores = pd.DataFrame({"code": list("ABCDEFGH"),
                           "score": [.9, .8, .7, .6, .5, .4, .3, .2]})
    positions = pd.DataFrame({"code": ["A", "G", "X"], "quantity": [100, 200, 300],
                              "sellable_quantity": [100, 200, 300], "hold_days": [2, 2, 2]})
    decision = plan_topk_dropout(scores, positions, topk=3, n_drop=2, hold_thresh=1)
    assert decision["keep"] == ["A"]
    assert set(decision["sell"]) == {"G", "X"}
    assert decision["buy_ranked"] == ["B", "C"]
    orders = build_sell_orders(decision, positions, "2026-07-20")
    assert set(orders.code) == {"G", "X"}
    assert set(orders.execution_policy) == {SCHEME_B_STATUS}


def test_real_sell_fills_drive_cash_and_remaining_position():
    positions = pd.DataFrame({"code": ["A"], "quantity": [1000], "sellable_quantity": [1000]})
    orders = pd.DataFrame({"client_order_id": ["S1"], "code": ["A"], "quantity": [1000]})
    fills = pd.DataFrame({"client_order_id": ["S1"], "code": ["A"],
                          "filled_quantity": [400], "average_price": [10.0], "fee": [5.0]})
    remaining, cash = apply_sell_fills(positions, 100.0, orders, fills)
    assert remaining.iloc[0].quantity == 600
    assert cash["cash_after_sell"] == pytest.approx(4095.0)


def test_real_buy_fills_create_t1_locked_position():
    positions = pd.DataFrame(columns=["code", "quantity", "sellable_quantity", "hold_days"])
    orders = pd.DataFrame({"client_order_id": ["B1"], "code": ["A"], "quantity": [100]})
    fills = pd.DataFrame({"client_order_id": ["B1"], "code": ["A"],
                          "filled_quantity": [100], "average_price": [10.0], "fee": [5.0]})
    after, cash = apply_buy_fills(positions, 2000.0, orders, fills)
    assert after.iloc[0].quantity == 100
    assert after.iloc[0].sellable_quantity == 0
    assert after.iloc[0].hold_days == 0
    assert cash["cash_after_buy"] == pytest.approx(995.0)


def test_execution_limit_filter_and_buy_lot_rounding():
    raw = pd.DataFrame({"code": ["B", "C"], "bar_time": ["09:41", "09:41"],
                        "close": [11.0, 10.0]})
    info = {code: {"prev_close": 10.0, "prev_factor": 1.0, "factor": 1.0,
                   "factor_confirmed": True} for code in ["B", "C"]}
    execution = build_execution_snapshot(raw, info)
    scores = pd.DataFrame({"code": ["B", "C"], "score": [.9, .8],
                           "limit_up": [.095, .095]})
    decision = {"topk": 1, "buy_ranked": ["B", "C"]}
    orders = build_buy_orders(decision, pd.DataFrame(), 10000, execution, scores,
                              "2026-07-20")
    assert orders.code.tolist() == ["C"]  # B is +10%, blocked
    assert orders.iloc[0].quantity % 100 == 0


def test_reconciliation_is_fail_closed():
    expected = pd.DataFrame({"code": ["A"], "quantity": [100]})
    actual = pd.DataFrame({"code": ["A"], "quantity": [90]})
    result = reconcile_positions(expected, actual, 1000, 1000)
    assert result["status"] == "FAIL" and result["position_mismatches"]


def test_calendar_gate_accepts_pre_market_next_weekday():
    from qlib_ifind_beta.live.intraday import calendar_gate

    cal = ["2026-09-09", "2026-09-10"]  # Thu; T=Fri not yet synced
    assert calendar_gate("2026-09-11", cal) == "next_trading_day_pre_market"
    assert calendar_gate("2026-09-10", cal) == "in_calendar"
    # Weekend and back-dated gaps still fail closed.
    assert calendar_gate("2026-09-12", cal) == "not_in_qlib_calendar"
    assert calendar_gate("2026-09-08", cal) == "not_in_qlib_calendar"
    # A later calendar entry existing means the date was skipped, not pending.
    assert calendar_gate("2026-09-11", ["2026-09-10", "2026-09-15"]) == "not_in_qlib_calendar"


def test_candidate_sidecars_are_json_not_model_files(tmp_path, monkeypatch):
    from scripts import retrain

    class Recorder:
        recorder_id = "hf-id"

    task = {"dataset": {"kwargs": {"segments": {
        "train": ["2026-01-01", "2026-05-01"],
        "valid": ["2026-05-02", "2026-06-01"],
        "test": ["2026-06-03", "2026-06-30"],
    }}}}
    gate = {"weight": 0.25, "xgb_recorder_id": "xgb-id", "gate_passed": True,
            "validation_baseline": {"IC": .1}, "validation_candidate": {"IC": .2}}
    manifest, report = retrain._write_candidate_artifacts(Recorder(), gate, task, tmp_path)
    assert json.loads(manifest.read_text())["hflgb_artifact"] == "params.pkl"
    assert json.loads(manifest.read_text())["status"] == "CANDIDATE"
    assert json.loads(report.read_text())["promotion_recommendation"] == "REVIEW"
