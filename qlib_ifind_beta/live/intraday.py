"""Auditable intraday production state and order assembly.

Pure transformations live here; network/model adapters stay in their existing
modules.  Every public step is deterministic and safe to rerun with the same
inputs, which makes the manual CSV workflow recoverable after interruption.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date as date_type, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from qlib_ifind_beta.config import CHAMPION_TOPK, PROJECT_ROOT
from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler
from qlib_ifind_beta.minute_factors import compute_champion_factors

FACTOR_TIMES = tuple(f"09:{minute:02d}" for minute in range(31, 41))
EXECUTION_TIME = "09:41"
MIN_CANDIDATES = 80
SCHEME_B_STATUS = "SCHEME_B_UNVALIDATED"


def calendar_gate(date: str, calendar: "Iterable[str]") -> str:
    """Classify a trade date against the qlib day calendar (preflight gate).

    The day calendar advances at T-1 close sync, so a normal 08:50 pre-market
    run necessarily finds T missing. Returns one of:

    - ``in_calendar``: T already synced — post-market rerun timing.
    - ``next_trading_day_pre_market``: T is a weekday strictly after the
      calendar end and nothing later exists in the calendar — the normal
      pre-market timing. Local weekday logic cannot distinguish CN holidays,
      so a mis-entered holiday date also lands here; downstream steps then
      fail closed on missing realtime bars.
    - ``not_in_qlib_calendar``: everything else (weekends, dates before the
      calendar end that were never synced, dates with later entries present).
    """
    cal = sorted(calendar)
    if date in set(cal):
        return "in_calendar"
    if not cal:
        return "not_in_qlib_calendar"
    day = date_type.fromisoformat(date)
    if day.weekday() >= 5:
        return "not_in_qlib_calendar"
    end = date_type.fromisoformat(cal[-1])
    if day > end:
        return "next_trading_day_pre_market"
    return "not_in_qlib_calendar"


@dataclass(frozen=True)
class DayPaths:
    date: str
    root: Path

    @classmethod
    def create(cls, date: str, base: Path | None = None) -> "DayPaths":
        root = Path(base or PROJECT_ROOT / "data" / "production_signals") / date
        root.mkdir(parents=True, exist_ok=True)
        return cls(date=date, root=root)

    def file(self, name: str) -> Path:
        return self.root / name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_sha256(fields: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(fields).encode()).hexdigest()


def _atomic_replace(path: Path, writer) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    writer(tmp)
    os.replace(tmp, path)
    return path


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp, datetime, date_type)):
        return value.isoformat()
    return value


def write_json(path: Path, value: dict) -> Path:
    return _atomic_replace(
        path, lambda tmp: tmp.write_text(
            json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    )


def write_csv(path: Path, frame: pd.DataFrame) -> Path:
    return _atomic_replace(path, lambda tmp: frame.to_csv(tmp, index=False))


def write_parquet(path: Path, frame: pd.DataFrame) -> Path:
    return _atomic_replace(path, lambda tmp: frame.to_parquet(tmp, index=False))


def upsert_bars(existing: pd.DataFrame | None, incoming: pd.DataFrame,
                *, date: str, allowed_times: Iterable[str] = FACTOR_TIMES) -> pd.DataFrame:
    """Idempotently merge bars on (date, code, bar_time), latest fetch wins."""
    required = {"date", "code", "bar_time", "open", "high", "low", "close", "volume"}
    missing = required - set(incoming.columns)
    if missing:
        raise ValueError(f"bar input missing columns: {sorted(missing)}")
    incoming = incoming.copy()
    incoming["date"] = incoming["date"].astype(str)
    incoming["bar_time"] = incoming["bar_time"].astype(str).str.slice(0, 5)
    if set(incoming["date"]) != {date}:
        raise ValueError("bar input contains a different trade date")
    bad_times = set(incoming["bar_time"]) - set(allowed_times)
    if bad_times:
        raise ValueError(f"unexpected bar times: {sorted(bad_times)}")
    numeric = ["open", "high", "low", "close", "volume"]
    incoming[numeric] = incoming[numeric].apply(pd.to_numeric, errors="coerce")
    invalid = (
        incoming[numeric].isna().any(axis=1)
        | (incoming["volume"] < 0)
        | (incoming["low"] > incoming[["open", "close", "high"]].min(axis=1))
        | (incoming["high"] < incoming[["open", "close", "low"]].max(axis=1))
    )
    if invalid.any():
        raise ValueError(f"invalid OHLCV rows: {incoming.index[invalid].tolist()[:10]}")
    frame = pd.concat([existing, incoming], ignore_index=True) if existing is not None else incoming
    sort_cols = [c for c in ("date", "code", "bar_time", "fetch_time") if c in frame]
    frame = frame.sort_values(sort_cols).drop_duplicates(
        ["date", "code", "bar_time"], keep="last"
    )
    return frame.sort_values(["code", "bar_time"]).reset_index(drop=True)


def validate_factor_bars(bars: pd.DataFrame, universe_codes: Iterable[str],
                         *, min_candidates: int = MIN_CANDIDATES) -> tuple[list[str], dict]:
    expected = set(FACTOR_TIMES)
    codes = list(dict.fromkeys(universe_codes))
    complete, missing = [], {}
    for code in codes:
        found = set(bars.loc[bars["code"] == code, "bar_time"].astype(str).str.slice(0, 5))
        if found == expected:
            complete.append(code)
        else:
            missing[code] = sorted(expected - found)
    quality = {
        "universe_count": len(codes), "complete_bar_count": len(complete),
        "coverage": len(complete) / len(codes) if codes else 0.0,
        "missing": missing, "bar_window": [FACTOR_TIMES[0], FACTOR_TIMES[-1]],
        "status": "PASS" if len(complete) >= min_candidates else "FAIL",
    }
    return complete, quality


def assemble_features(bars: pd.DataFrame, prev_volumes: dict[str, list[float]],
                      daily_info: dict[str, dict], codes: Iterable[str]) -> pd.DataFrame:
    """Build the exact ordered 18-column matrix from strict ten-bar groups."""
    fields = list(MinuteEnhancedHandler.ENHANCED_FIELDS)
    rows = []
    for code in codes:
        group = bars.loc[bars["code"] == code].copy()
        group["bar_time"] = group["bar_time"].astype(str).str.slice(0, 5)
        group = group.set_index("bar_time").reindex(FACTOR_TIMES)
        if len(group) != 10 or group[["open", "high", "low", "close", "volume"]].isna().any().any():
            continue
        info = daily_info.get(code, {})
        today_factor = info.get("factor")
        prev_factor = info.get("prev_factor")
        # Fail closed for known/possible factor changes; callers may explicitly
        # supply factor_confirmed=True after checking corporate actions.
        if not info.get("factor_confirmed", today_factor is not None):
            continue
        factors = compute_champion_factors(
            group["close"].to_numpy(), group["open"].to_numpy(),
            group["high"].to_numpy(), group["low"].to_numpy(),
            group["volume"].to_numpy(), prev_volumes.get(code, []),
            prev_close=info.get("prev_close"), prev_factor=prev_factor,
            today_open=float(group["open"].iloc[0]), today_factor=today_factor,
        )
        row = {"code": code, **{field: factors.get(field, np.nan) for field in fields}}
        rows.append(row)
    frame = pd.DataFrame(rows, columns=["code", *fields])
    if frame.empty:
        return frame
    frame[fields] = frame[fields].replace([np.inf, -np.inf], np.nan)
    return frame.dropna(subset=fields).reset_index(drop=True)


def plan_topk_dropout(scores: pd.DataFrame, positions: pd.DataFrame,
                      *, topk: int = CHAMPION_TOPK, n_drop: int = 8,
                      hold_thresh: int = 1) -> dict:
    """Mirror Qlib TopkDropout member selection without estimating fills."""
    score = scores.set_index("code")["score"].astype(float).sort_values(ascending=False)
    pos = positions.copy()
    if pos.empty:
        pos = pd.DataFrame(columns=["code", "quantity", "sellable_quantity", "hold_days"])
    pos = pos.drop_duplicates("code").set_index("code")
    held = list(pos.index)
    last = list(score.reindex(held).sort_values(ascending=False, na_position="last").index)
    unheld = [code for code in score.index if code not in held]
    today = unheld[: max(0, n_drop + topk - len(last))]
    combined = pd.concat([score.reindex(last), score.reindex(today)]).sort_values(
        ascending=False, na_position="last"
    )
    bottom = set(combined.tail(n_drop).index)
    proposed_sell = [code for code in last if code in bottom]
    sell, blocked = [], []
    for code in proposed_sell:
        row = pos.loc[code]
        quantity = float(row.get("quantity", 0))
        sellable = float(row.get("sellable_quantity", quantity))
        hold_days = int(row.get("hold_days", hold_thresh))
        if sellable >= quantity > 0 and hold_days >= hold_thresh:
            sell.append(code)
        else:
            blocked.append(code)
    buy_count = max(0, min(len(today), len(sell) + topk - len(last)))
    buy = today[:buy_count]
    keep = [code for code in held if code not in sell]
    return {"keep": keep, "sell": sell, "blocked_sell": blocked,
            "buy_ranked": buy, "topk": topk, "n_drop": n_drop,
            "hold_thresh": hold_thresh}


def build_sell_orders(decision: dict, positions: pd.DataFrame, date: str) -> pd.DataFrame:
    pos = positions.set_index("code") if not positions.empty else pd.DataFrame()
    rows = []
    for idx, code in enumerate(decision["sell"], 1):
        quantity = int(pos.loc[code, "quantity"])
        rows.append({"client_order_id": f"{date.replace('-', '')}-S-{idx:03d}",
                     "code": code, "action": "SELL", "quantity": quantity,
                     "execution_policy": SCHEME_B_STATUS, "reason": "dropout_bottom"})
    return pd.DataFrame(rows, columns=[
        "client_order_id", "code", "action", "quantity", "execution_policy", "reason"
    ])


def apply_sell_fills(positions: pd.DataFrame, cash_before: float,
                     sell_orders: pd.DataFrame, fills: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply broker fills once by client order id and return authoritative cash."""
    if fills["client_order_id"].duplicated().any():
        raise ValueError("duplicate sell fill client_order_id")
    order = sell_orders.set_index("client_order_id")
    pos = positions.copy().set_index("code")
    cash = float(cash_before)
    for _, fill in fills.iterrows():
        oid = fill["client_order_id"]
        if oid not in order.index:
            raise ValueError(f"unknown sell fill {oid}")
        expected = order.loc[oid]
        if str(fill["code"]) != str(expected["code"]):
            raise ValueError(f"sell fill code mismatch for {oid}")
        qty = int(fill["filled_quantity"])
        if qty < 0 or qty > int(expected["quantity"]):
            raise ValueError(f"invalid filled quantity for {oid}")
        code = str(fill["code"])
        pos.loc[code, "quantity"] = max(0, int(pos.loc[code, "quantity"]) - qty)
        if "sellable_quantity" in pos:
            pos.loc[code, "sellable_quantity"] = max(
                0, int(pos.loc[code, "sellable_quantity"]) - qty
            )
        cash += qty * float(fill["average_price"]) - float(fill.get("fee", 0.0))
    pos = pos.loc[pos["quantity"].astype(float) > 0].reset_index()
    return pos, {"cash_after_sell": cash, "status": "PASS"}


def apply_buy_fills(positions: pd.DataFrame, cash_before: float,
                    buy_orders: pd.DataFrame, fills: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply broker buy fills once and create T+1-locked positions."""
    if fills["client_order_id"].duplicated().any():
        raise ValueError("duplicate buy fill client_order_id")
    order = buy_orders.set_index("client_order_id")
    pos = positions.copy()
    if pos.empty:
        pos = pd.DataFrame(columns=["code", "quantity", "sellable_quantity", "hold_days"])
    pos = pos.set_index("code")
    cash = float(cash_before)
    for _, fill in fills.iterrows():
        oid = fill["client_order_id"]
        if oid not in order.index:
            raise ValueError(f"unknown buy fill {oid}")
        expected = order.loc[oid]
        if str(fill["code"]) != str(expected["code"]):
            raise ValueError(f"buy fill code mismatch for {oid}")
        qty = int(fill["filled_quantity"])
        if qty < 0 or qty > int(expected["quantity"]):
            raise ValueError(f"invalid filled quantity for {oid}")
        cost = qty * float(fill["average_price"]) + float(fill.get("fee", 0.0))
        if cost > cash + 1e-8:
            raise ValueError(f"insufficient cash for {oid}")
        code = str(fill["code"])
        if code not in pos.index:
            pos.loc[code, ["quantity", "sellable_quantity", "hold_days"]] = [0, 0, 0]
        pos.loc[code, "quantity"] = int(pos.loc[code, "quantity"]) + qty
        # A-share buys are unavailable for same-day sale.
        pos.loc[code, "sellable_quantity"] = int(pos.loc[code, "sellable_quantity"])
        pos.loc[code, "hold_days"] = 0
        cash -= cost
    pos = pos.reset_index()
    for column in ("quantity", "sellable_quantity", "hold_days"):
        pos[column] = pos[column].astype(int)
    return pos.sort_values("code").reset_index(drop=True), {
        "cash_after_buy": cash, "status": "PASS"
    }


def build_execution_snapshot(bars_0941: pd.DataFrame, daily_info: dict[str, dict]) -> pd.DataFrame:
    frame = bars_0941.copy()
    frame["bar_time"] = frame["bar_time"].astype(str).str.slice(0, 5)
    if set(frame["bar_time"]) - {EXECUTION_TIME}:
        raise ValueError("execution snapshot must contain only the closed 09:41 bar")
    rows = []
    for _, bar in frame.drop_duplicates("code", keep="last").iterrows():
        code = str(bar["code"])
        info = daily_info.get(code, {})
        values = [info.get("prev_close"), info.get("prev_factor"), info.get("factor")]
        if not info.get("factor_confirmed", info.get("factor") is not None) or not all(
            value is not None and math.isfinite(float(value)) for value in values
        ):
            continue
        raw_prev = float(values[0]) / float(values[1])
        price = float(bar["close"])
        change = (price / float(values[2])) / raw_prev - 1.0
        rows.append({"code": code, "price_941": price, "change_941": change})
    return pd.DataFrame(rows)


def build_buy_orders(decision: dict, positions_after_sell: pd.DataFrame,
                     cash_after_sell: float, execution: pd.DataFrame,
                     scores: pd.DataFrame, date: str, *, open_cost: float = 0.0005,
                     min_cost: float = 5.0, safety_buffer: float = 0.002,
                     lot_size: int = 100) -> pd.DataFrame:
    held = set(positions_after_sell["code"]) if not positions_after_sell.empty else set()
    slots = max(0, int(decision["topk"]) - len(held))
    aux = execution.set_index("code")
    ranked = list(decision["buy_ranked"])
    # Continue through the full score list when a planned candidate is blocked.
    ranked += [c for c in scores.sort_values("score", ascending=False)["code"]
               if c not in ranked and c not in held]
    eligible = []
    for code in ranked:
        if code not in aux.index:
            continue
        row = aux.loc[code]
        score_row = scores.loc[scores["code"] == code]
        limit_up = float(score_row.iloc[0].get("limit_up", np.nan)) if not score_row.empty else np.nan
        if np.isfinite(limit_up) and float(row["change_941"]) >= limit_up:
            continue
        eligible.append(code)
        if len(eligible) == slots:
            break
    if not eligible:
        return pd.DataFrame(columns=["client_order_id", "rank", "code", "action", "quantity"])
    per_name = float(cash_after_sell) * (1.0 - safety_buffer) / len(eligible)
    rows = []
    for rank, code in enumerate(eligible, 1):
        price = float(aux.loc[code, "price_941"])
        budget = max(0.0, per_name - max(min_cost, per_name * open_cost))
        quantity = int(budget / price / lot_size) * lot_size
        if quantity <= 0:
            continue
        rows.append({"client_order_id": f"{date.replace('-', '')}-B-{rank:03d}",
                     "rank": rank, "code": code, "action": "BUY", "quantity": quantity,
                     "reference_price": price, "estimated_amount": quantity * price,
                     "execution_policy": SCHEME_B_STATUS})
    return pd.DataFrame(rows)


def reconcile_positions(expected: pd.DataFrame, broker: pd.DataFrame,
                        expected_cash: float, broker_cash: float,
                        *, cash_tolerance: float = 1.0) -> dict:
    exp = expected.groupby("code")["quantity"].sum().astype(int).to_dict()
    got = broker.groupby("code")["quantity"].sum().astype(int).to_dict()
    mismatches = [{"code": code, "expected": exp.get(code, 0), "actual": got.get(code, 0)}
                  for code in sorted(set(exp) | set(got)) if exp.get(code, 0) != got.get(code, 0)]
    cash_diff = float(broker_cash) - float(expected_cash)
    return {"status": "PASS" if not mismatches and abs(cash_diff) <= cash_tolerance else "FAIL",
            "position_mismatches": mismatches, "cash_difference": cash_diff,
            "actual_position_count": len(got)}


def make_active_manifest(*, date: str, experiment: str, recorder_id: str,
                         model_path: Path, qlib_version: str, status: str = "LOCKED") -> dict:
    return {"trade_date": date, "experiment": experiment, "recorder_id": recorder_id,
            "artifact": "params.pkl", "model_sha256": sha256_file(model_path),
            "feature_schema_sha256": schema_sha256(MinuteEnhancedHandler.ENHANCED_FIELDS),
            "qlib_version": qlib_version, "status": status,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
