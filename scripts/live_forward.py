"""P1 实战对接 — 每日入口（orchestration）。

工作日 15:35 触发（cn_data_1min 15:30 同步后）。五步：
  [1] universe 增量（iFinD p03473 T 日盘前快照）—— 可 --skip-universe 跳过（无网络/离线时）
  [2] materialize 池内 day.bins（~100 codes）—— 可 --skip-materialize 跳过（dry-run 时）
  [3] predict_day(T) → record_signal
  [4] settle_prev(T-1, T) —— 用 T 日 close/change/limit_down 回填 T-1 信号
  [5] compute_nav + daily_ic

Run:
  conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
      scripts/live_forward.py --date 2026-07-02 --skip-universe --skip-materialize
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import pandas as pd

from qlib_ifind_beta.config import UNIVERSE_MARKET
from qlib_ifind_beta.live.inference import predict_day
from qlib_ifind_beta.live.materialize_live import load_pool, materialize_pool
from qlib_ifind_beta.live.track import compute_nav, record_signal, settle_prev

DATA = Path(__file__).resolve().parent.parent / "data"
SIGNALS = DATA / "live_signals.csv"
SETTLE = DATA / "live_settle.csv"
NAV = DATA / "live_nav.csv"


def _aux_lookups(date: str, fields: list[str],
                 codes: list[str] | None = None) -> dict[str, dict[str, float]]:
    """从 day.bin 读单日多字段 → {field(stripped): {code: value}}。≤date 无前视。

    D.features 返回列名带 '\$' 前缀（'\$close'），此处 strip 之 → 'close'。

    codes:
      - None → D.instruments(market)（qlib 按 date 过滤成**当日池**，~100 只）。
        用于 predict 段（只关心当日可买标的）。
      - 显式 list → 不受 universe 池过滤，直接取 bin（list 语义）。
        用于 settle 段：T-1 持仓在 T 日掉出观察池仍可查 close（base bin 全票都有），
        避免 883926 每日换手 ~90% 导致 90% 持仓 sell_price=NaN。
    """
    from qlib.data import D
    instruments = codes if codes is not None else D.instruments(market=UNIVERSE_MARKET)
    df = D.features(instruments, fields, start_time=date, end_time=date)
    if df.empty:
        return {f.replace("$", ""): {} for f in fields}
    level = "datetime" if "datetime" in df.index.names else 1
    day = df.xs(date, level=level)
    out = {}
    for col in day.columns:
        out[col.replace("$", "")] = day[col].dropna().to_dict()
    return out


def run(date: str, skip_universe: bool = False, skip_materialize: bool = False) -> None:
    print(f"▶ P1 live_forward date={date} skip_universe={skip_universe} skip_materialize={skip_materialize}")

    # [1] universe 增量（iFinD p03473 T 日盘前快照；失败用既有池继续）
    if not skip_universe:
        try:
            from qlib_ifind_beta import universe
            universe.dump_universe("2024-01-01", date)   # 全量幂等（resumable cache）
            print(f"✓ [1] universe 刷新到 {date}")
        except Exception as e:
            print(f"⚠ [1] universe 失败（用既有池继续）: {e}")
    else:
        print("⏭ [1] universe 跳过")

    # [2] materialize 池内（T 日 membership，~100 codes；幂等全量重算）
    if not skip_materialize:
        codes = load_pool(date)
        summ = materialize_pool(codes)
        print(f"✓ [2] materialize pool(n={summ['n']}): minute ok={summ['minute_ok']}/{summ['n']} "
              f"change ok={summ['change_ok']}/{summ['n']}")
    else:
        print("⏭ [2] materialize 跳过")

    # [3] predict_day(T) → 记录信号
    result = predict_day(date)
    record_signal(result, SIGNALS)
    topk_codes = [c["code"] for c in result["topk"]]
    print(f"✓ [3] predict: n_candidates={result['n_candidates']} topk={topk_codes}")

    # [4] settle_prev(T-1, T)：用 T 日 close/change/limit_down 回填 T-1 信号
    #     close 按 prev candidates code 列表查（list 不受当日池过滤，持仓掉出观察池仍可查）
    from qlib.data import D
    cal = D.calendar(start_time="2026-01-01", end_time=date, freq="day")
    if len(cal) >= 2:
        prev = pd.Timestamp(cal[-2]).strftime("%Y-%m-%d")
        prev_cands = []
        if SIGNALS.exists():
            sig_df = pd.read_csv(SIGNALS)
            prev_cands = sig_df.loc[sig_df["date"] == prev, "code"].tolist()
        aux = _aux_lookups(date, ["$close", "$change", "$limit_down"], codes=prev_cands)
        n = settle_prev(prev, date,
                        close_lookup=aux.get("close", {}),
                        change_lookup=aux.get("change", {}),
                        limit_down_lookup=aux.get("limit_down", {}),
                        signals_path=SIGNALS, settle_path=SETTLE)
        print(f"✓ [4] settle_prev({prev}→{date}): {n} 笔 (sell_price 非空 "
              f"close_lookup={len(aux.get('close', {}))})")
    else:
        print("⏭ [4] settle_prev 跳过（首日无 T-1）")

    # [5] NAV 累积
    if SETTLE.exists():
        settle_df = pd.read_csv(SETTLE)
        nav = compute_nav(settle_df, NAV)
        if not nav.empty:
            print(f"✓ [5] NAV: 末日 net_nav={nav['net_nav'].iloc[-1]:.4f} "
                  f"gross_nav={nav['gross_nav'].iloc[-1]:.4f} ({len(nav)} 日)")
        else:
            print("⏭ [5] NAV 空（无可结算笔）")
    print("✅ live_forward 完成")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="P1 纸面前向跟踪每日入口")
    p.add_argument("--date", required=True, help="推理日 T (YYYY-MM-DD)")
    p.add_argument("--skip-universe", action="store_true", help="跳过 universe 刷新（离线/dry-run）")
    p.add_argument("--skip-materialize", action="store_true", help="跳过 materialize（dry-run，bin 已存在）")
    a = p.parse_args()
    run(a.date, a.skip_universe, a.skip_materialize)
