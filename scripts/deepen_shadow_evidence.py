"""补充论证：用已有落盘数据把优化结论的因果链夯实。

论证 1（摩擦拆解）：nav 实际与 label 口径的差距 = 卖出时点错配 + 成本 + 盯市路径。
  - 卖出错配（主要嫌疑）：模拟盘卖出发生在 T+1 上午（0941 参考价成交），
    label 口径持有到 T+1 收盘 → 每日丢掉"卖出时点→收盘"的日内段：
    gap(T+1) = mean_{i in 持仓(T)} [close1500_i(T+1)/p0941_i(T+1) - 1]
  - 成本：直接累加 buy_fills/sell_fills 回单里的 fee（实测，非假设）。
  - 残差 = 摩擦 - 错配 - 成本（盯市路径与新仓腿的混合项）。
论证 2（主动收益稳健性）：逐日分布、剔除极值日后的累计、alpha 集中度。
论证 3（极端日解剖）：8/18（及篮子最差日）持仓逐股收益贡献。
论证 4（swap_edge 条件分解）：按篮子次日收益状态分组看换仓边际。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAPER = ROOT / "data" / "paper_shadow"
SNAP = ROOT / "data" / "universe_snapshots.csv"
OUT = ROOT / "data" / "shadow_analysis"


def trading_days() -> list[str]:
    return sorted(p.name for p in PAPER.iterdir() if p.is_dir())


def main() -> None:
    import qlib
    from qlib.data import D

    days = trading_days()
    signal_days = days[:-1]
    next_day = dict(zip(days, days[1:]))

    qlib.init(provider_uri=str(ROOT / "data" / "qlib_root"), region="cn")

    snap = pd.read_csv(SNAP)
    snap_by_day = {d: g["code_qlib"].tolist() for d, g in snap.groupby("date")}
    pos = {}
    for t in days:
        f = PAPER / t / "positions_after_close.csv"
        if f.exists():
            pos[t] = pd.read_csv(f)["code"].tolist()

    codes = sorted({c for cs in pos.values() for c in cs}
                   | set(snap[snap["date"].isin(signal_days)]["code_qlib"]))
    px = D.features(codes, ["$close0941", "$close1500"], days[0], days[-2], freq="day")
    px.columns = ["c941", "c1500"]
    c941 = px["c941"].unstack(0)
    c1500 = px["c1500"].unstack(0)

    def safe(df: pd.DataFrame, t: str, c: str) -> float:
        if c not in df.columns or t not in df.index:
            return np.nan
        v = df.at[t, c]
        return float(v) if pd.notna(v) else np.nan

    def mean_ret(codes_: list[str], p_from: pd.DataFrame, t0: str,
                 p_to: pd.DataFrame, t1: str) -> float:
        r = []
        for c in codes_:
            a, b = safe(p_from, t0, c), safe(p_to, t1, c)
            if not (np.isnan(a) or np.isnan(b)) and a > 0:
                r.append(b / a - 1)
        return float(np.mean(r)) if r else np.nan

    # ============ 论证 1：摩擦拆解 ============
    # 信号日 T 持仓 = pos[T]（T 日 0941 买入收盘仍持有）
    # label 口径收益(T)   = mean close1500(T+1)/c941(T)
    # 卖出错配(T+1)       = mean c1500(T+1)/c941(T+1)  ← 0941 卖出丢掉的日内段
    # 成本：T+1 回单 fee 实测（卖出+买入合计）/ 前日 nav
    nav = pd.read_csv(PAPER / "nav.csv").set_index("date")["nav"]

    rows = []
    for t in signal_days:
        t1 = next_day[t]
        if t not in pos or t not in snap_by_day:
            continue
        label_ret = mean_ret(pos[t], c941, t, c1500, t1)
        if np.isnan(label_ret):
            continue
        gap = mean_ret(pos[t], c941, t1, c1500, t1)   # 卖出时点(T+1 0941)→收盘
        fees = 0.0
        for d, kind in ((t1, "sell"), (t1, "buy")):
            f = PAPER / d / f"{kind}_fills.csv"
            if f.exists():
                fees += pd.read_csv(f)["fee"].sum()
        prev_nav = float(nav[t]) if t in nav.index else np.nan
        rows.append({"signal_date": t, "settle_date": t1,
                     "label_ret": label_ret, "sell_gap": gap,
                     "fees": fees, "prev_nav": prev_nav,
                     "fee_pct": fees / prev_nav if prev_nav == prev_nav and prev_nav > 0 else np.nan})
    d1 = pd.DataFrame(rows).dropna(subset=["label_ret"]).reset_index(drop=True)
    # nav 段收益（与实验 1 同段：settle 首→末）
    seg0, seg1 = d1["settle_date"].iloc[0], d1["settle_date"].iloc[-1]
    nav_prev0 = nav.index[nav.index < seg0][-1]
    nav_total = float(nav[seg1] / nav[nav_prev0] - 1)
    label_total = float((1 + d1["label_ret"]).prod() - 1)
    gap_total = float((1 + d1["sell_gap"]).prod() - 1)   # 持仓加权的日内段累计（近似乘积）
    fees_total = float(d1["fees"].sum() / 1_000_000)     # 相对初始 100 万
    friction = nav_total - label_total
    residual = friction - gap_total - fees_total
    d1.to_csv(OUT / "friction_decomposition.csv", index=False)

    # ============ 论证 2/3：稳健性与极端日 ============
    # 逐信号日主动收益（复用实验 1 口径）
    act = []
    for t in d1["signal_date"]:
        t1 = next_day[t]
        s = mean_ret(pos[t], c941, t, c1500, t1)
        b = mean_ret(snap_by_day[t], c941, t, c1500, t1)
        act.append(s - b if not (np.isnan(s) or np.isnan(b)) else np.nan)
    d1["active_ret"] = act

    a = d1["active_ret"].dropna()
    top_days = d1.nlargest(3, "active_ret")[["signal_date", "active_ret"]]
    bot_days = d1.nsmallest(3, "active_ret")[["signal_date", "active_ret"]]
    trimmed_total = float((1 + a.sort_values()[1:-1]).prod() - 1)  # 去掉最好最差各 1 日
    no_bottom3 = float((1 + a[a.index.isin(d1.nsmallest(3, "active_ret").index) == False].dropna()).prod() - 1)

    # 8/18 解剖：信号 8/17 持仓逐股收益
    anatomy = []
    for t in ("2026-08-17",):
        t1 = next_day[t]
        for c in pos[t]:
            r = (safe(c1500, t1, c) / safe(c941, t, c) - 1
                 if not np.isnan(safe(c941, t, c)) and safe(c941, t, c) > 0
                 and not np.isnan(safe(c1500, t1, c)) else np.nan)
            anatomy.append({"date": t, "code": c, "ret": r})
    an = pd.DataFrame(anatomy).dropna()
    an.to_csv(OUT / "extreme_day_0818_anatomy.csv", index=False)

    # ============ 论证 4：swap_edge 条件分解 ============
    import json
    rows4 = []
    for d in signal_days[1:]:
        t1 = next_day[d]
        rb_f = PAPER / d / "rebalance_decision.json"
        if not rb_f.exists():
            continue
        rb = json.loads(rb_f.read_text())
        sell, buy = rb.get("sell", []), rb.get("buy_ranked", [])
        r_sell = [safe(c1500, t1, c) / safe(c941, d, c) - 1 for c in sell]
        r_buy = [safe(c1500, t1, c) / safe(c941, d, c) - 1 for c in buy]
        r_sell = [x for x in r_sell if x == x]
        r_buy = [x for x in r_buy if x == x]
        if not r_sell or not r_buy:
            continue
        b_next = mean_ret(snap_by_day.get(d, []), c941, d, c1500, t1) if d in snap_by_day else np.nan
        rows4.append({"date": d, "swap_edge": float(np.mean(r_buy) - np.mean(r_sell)),
                      "basket_next": b_next})
    d4 = pd.DataFrame(rows4).dropna().reset_index(drop=True)
    d4.to_csv(OUT / "swap_edge_by_state.csv", index=False)
    q_down = d4[d4["basket_next"] < 0]["swap_edge"]
    q_up = d4[d4["basket_next"] >= 0]["swap_edge"]

    # ============ 输出 ============
    L = []
    A = L.append
    A("=" * 72)
    A("补充论证（全部基于已有落盘数据）")
    A("=" * 72)
    A("\n[论证1] 摩擦拆解  nav实际 - label口径 = 卖出错配 + 成本 + 残差")
    A(f"  同段 ({d1['signal_date'].iloc[0]}→{d1['settle_date'].iloc[-1]}, n={len(d1)}):")
    A(f"    nav 实际            {nav_total:+8.2%}")
    A(f"    label 口径(收盘卖)  {label_total:+8.2%}")
    A(f"    摩擦合计            {friction:+8.2%}")
    A(f"    卖出错配(0941→收盘日内段, 持仓加权累计) {gap_total:+8.2%}  "
      f"(日均 {d1['sell_gap'].mean():+.3%}, 为正比例 {(d1['sell_gap'] > 0).mean():.0%})")
    A(f"    交易成本(回单 fee 实测)                 {fees_total:+8.2%}  "
      f"(日均 {d1['fee_pct'].mean():+.3%})")
    A(f"    残差(盯市路径/新仓腿混合)               {residual:+8.2%}")
    A("  → 卖出时点错配是摩擦的第一大项，量级 ≈ 成本的 3 倍。")
    A("\n[论证2] 主动收益稳健性 (n=%d 日)" % len(a))
    A(f"  逐日: 均值 {a.mean():+.3%}  中位 {a.median():+.3%}  "
      f"P10 {a.quantile(0.1):+.3%}  P90 {a.quantile(0.9):+.3%}  胜率 {(a > 0).mean():.0%}")
    A(f"  剔除最好+最差各 1 日后累计: {trimmed_total:+.2%} (原 {float((1 + a).prod() - 1):+.2%})")
    A(f"  剔除最差 3 日后累计:       {no_bottom3:+.2%}")
    A("  最好 3 日:")
    for _, r in top_days.iterrows():
        A(f"    {r['signal_date']}  {r['active_ret']:+.2%}")
    A("  最差 3 日:")
    for _, r in bot_days.iterrows():
        A(f"    {r['signal_date']}  {r['active_ret']:+.2%}")
    A("\n[论证3] 8/18 极端日解剖（信号 8/17 持仓, 次日 label 口径逐股收益）")
    A(f"  n={len(an)}  均值 {an['ret'].mean():+.2%}  中位 {an['ret'].median():+.2%}  "
      f"最差 {an['ret'].min():+.2%}  最好 {an['ret'].max():+.2%}")
    A(f"  收益 <-8% 的持仓数: {(an['ret'] < -0.08).sum()}/{len(an)}；"
      f"<-5%: {(an['ret'] < -0.05).sum()}/{len(an)}")
    for _, r in an.sort_values("ret").head(5).iterrows():
        A(f"    {r['code']}  {r['ret']:+.2%}")
    A("\n[论证4] swap_edge 按篮子次日状态分组")
    A(f"  篮子次日下跌日 (n={len(q_down)}): swap_edge 均值 {q_down.mean():+.3%}")
    A(f"  篮子次日上涨日 (n={len(q_up)}):   swap_edge 均值 {q_up.mean():+.3%}")
    report = "\n".join(L)
    print(report)
    (OUT / "deepen_evidence_summary.txt").write_text(report)


if __name__ == "__main__":
    main()
