"""P1 实战对接 — 信号记录 / T+1 结算 / NAV 累积 / 日度 IC。

纯 pandas 状态机（无 qlib 依赖，可单测）。口径见 spec §3.2/§4。

简化纸面撮合（spec §4 + §6.4 已声明偏差）：
- T 日 9:41 买 top10 equal-weight（每只 1/topk 仓位），T+1 日收盘全卖（hold 1 天）。
- 成本：open_cost / close_cost 比例近似（min_cost 需资金规模，P1 用比例近似，NAV 偏差 spec §6.4）。
- 涨跌停拦截：buy 封涨停剔出（inference.py 已做）；sell 封跌停 → blocked（sell_price=NaN，该笔 ret=0 持平）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_SIGNAL_COLS = ["date", "code", "score", "price_941", "change_941",
                "limit_up", "limit_down", "in_topk"]
_SETTLE_COLS = ["signal_date", "code", "buy_price", "sell_date", "sell_price", "blocked"]


def record_signal(result: dict, path: Path) -> None:
    """追加 predict_day 结果到 live_signals.csv（幂等：同 date 先删后写）。

    记录全部 candidates（含封涨停未买的）+ in_topk 标记：
    - 全部 candidates 保留 → 事后 IC 复算（IC 需全部 score，非仅 topk）。
    - in_topk=True → 实际买入集（已剔除封涨停的前 topk，spec §3.2 [5] NAV 口径）。
    """
    path = Path(path)
    topk_codes = {c["code"] for c in result["topk"]}
    rows = []
    for c in result["candidates"]:
        rows.append({"date": result["date"], "code": c["code"], "score": c["score"],
                     "price_941": c["price_941"], "change_941": c["change_941"],
                     "limit_up": c["limit_up"], "limit_down": c["limit_down"],
                     "in_topk": c["code"] in topk_codes})
    new = pd.DataFrame(rows, columns=_SIGNAL_COLS)
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["date"] != result["date"]]   # 幂等：覆盖同日
        pd.concat([old, new], ignore_index=True).to_csv(path, index=False)
    else:
        new.to_csv(path, index=False)


def settle_prev(prev_date: str, today: str,
                close_lookup: dict[str, float],
                change_lookup: dict[str, float],
                limit_down_lookup: dict[str, float],
                signals_path: Path, settle_path: Path) -> int:
    """回填 prev_date 信号的 sell_price=close[today]，涨跌停 sell 拦截。

    *_lookup: code → 当日值（由调用方从 day.bin 取，≤today 无前视）。
    封跌停（change <= limit_down）→ sell_price=NaN, blocked=True（该笔 ret=0 持平）。
    返回结算笔数。仅结算 topk（已 buy 拦截后的可买集）：buy_price=price_941。

    注：P1 v1 settle 全部 candidates 的可买集（封涨停的 buy_price=NaN 自动 drop）。
    """
    signals_path, settle_path = Path(signals_path), Path(settle_path)
    if not signals_path.exists():
        return 0
    sig = pd.read_csv(signals_path)
    # 仅结算实际买入集 in_topk=True（spec §3.2 [5]: NAV = equal-weight topk10）。
    # in_topk 已在 record_signal 阶段剔除封涨停（change_941 >= limit_up）。
    prev_day = sig[sig["date"] == prev_date]
    if "in_topk" in prev_day.columns:
        prev_topk = prev_day[prev_day["in_topk"].astype(bool)]
    else:
        prev_topk = prev_day   # backward-compat（无 in_topk 列 → 全结算）
    rows = []
    for _, r in prev_topk.iterrows():
        code = r["code"]
        chg = change_lookup.get(code)
        ld = limit_down_lookup.get(code)
        blocked = (chg is not None) and (ld is not None) and (chg <= ld)
        rows.append({
            "signal_date": prev_date, "code": code,
            "buy_price": r["price_941"], "sell_date": today,
            "sell_price": float("nan") if blocked else close_lookup.get(code, float("nan")),
            "blocked": blocked,
        })
    new = pd.DataFrame(rows, columns=_SETTLE_COLS)
    if settle_path.exists():
        old = pd.read_csv(settle_path)
        old = old[old["signal_date"] != prev_date]   # 幂等：覆盖同 signal_date
        pd.concat([old, new], ignore_index=True).to_csv(settle_path, index=False)
    else:
        new.to_csv(settle_path, index=False)
    return len(rows)


def compute_nav(settle: pd.DataFrame, out_path: Path,
                open_cost: float = 0.0005, close_cost: float = 0.0015,
                position_scale: pd.Series | None = None) -> pd.DataFrame:
    """equal-weight compound NAV（按 sell_date 聚合日收益）。

    每笔：ret_gross = sell/buy - 1；ret_net = sell*(1-close_cost)/(buy*(1+open_cost)) - 1。
    日收益 = 该 sell_date 下所有可结算笔（非 NaN）的等权均值（blocked/NaN 笔 dropna）。
    gross_nav / net_nav 从 1.0 compound。

    position_scale: 可选，pd.Series index=date → [0,1] 仓位比例。如提供，日收益 *= position。
    用于 §55 仓位层 overlay（基准趋势连续仓位），默认 None = 满仓（向后兼容）。
    """
    out_path = Path(out_path)
    s = settle.dropna(subset=["buy_price", "sell_price"]).copy()
    s["ret_gross"] = s["sell_price"] / s["buy_price"] - 1
    s["ret_net"] = s["sell_price"] * (1 - close_cost) / (s["buy_price"] * (1 + open_cost)) - 1
    if s.empty:
        daily = pd.DataFrame(columns=["sell_date", "daily_ret_gross", "daily_ret_net",
                                      "n_held", "gross_nav", "net_nav"])
        daily.to_csv(out_path, index=False)
        return daily
    grp = s.groupby("sell_date")
    daily = pd.DataFrame({
        "daily_ret_gross": grp["ret_gross"].mean(),
        "daily_ret_net": grp["ret_net"].mean(),
        "n_held": grp.size(),
    }).reset_index()

    # §55: apply position scale if provided (e.g. benchmark-trend continuous position)
    if position_scale is not None:
        pos = position_scale.reindex(daily["sell_date"]).fillna(1.0)
        daily["daily_ret_gross"] *= pos.values
        daily["daily_ret_net"] *= pos.values
        daily["position"] = pos.values
    else:
        daily["position"] = 1.0

    daily["gross_nav"] = (1 + daily["daily_ret_gross"]).cumprod()
    daily["net_nav"] = (1 + daily["daily_ret_net"]).cumprod()
    daily.to_csv(out_path, index=False)
    return daily


def daily_ic(signals_path: Path, label_lookup: dict[str, float], date: str) -> float | None:
    """单日 rank IC（pred score vs T+1 label 的 Spearman）。

    label_lookup: code → label（T+1 收盘后回填，含 close[T+1] 已知）。
    无前视：仅当 T+1 收盘后调用。< 5 对返回 None（边界）。
    """
    if not Path(signals_path).exists():
        return None
    sig = pd.read_csv(signals_path)
    day = sig[sig["date"] == date]
    pairs = [(r["score"], label_lookup.get(r["code"])) for _, r in day.iterrows()
             if label_lookup.get(r["code"]) is not None]
    if len(pairs) < 5:
        return None
    s = pd.DataFrame(pairs, columns=["score", "label"]).dropna()
    if len(s) < 5:
        return None
    return float(s["score"].corr(s["label"], method="spearman"))
