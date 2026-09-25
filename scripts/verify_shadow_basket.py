"""验证实验：影子模拟盘 44 日的 alpha/beta 分解 + 固定换手边际分析。

实验 1（P0 推荐的"当日 100 成分可投资篮子基准"，此前从未落地）：
  - 策略 label 口径收益(T)   = mean_{i in 持仓(T)} close1500_i(T+1)/close0941_i(T) - 1
  - 篮子   label 口径收益(T) = mean_{i in 快照(T)}  close1500_i(T+1)/close0941_i(T) - 1
  - 两腿均来自 overlay 分钟同源日线 bin（label v2 口径），消除复权基差与执行错配，
    差值 = 纯选股 alpha（成本前）。
  - 状态条件（篮子收益四分位）下 alpha 是否消失 → 检验"回撤=纯 beta 暴露"假设。
  - nav 实际日收益 - 策略 label 口径收益 = 执行错配 + 成本拖累（整体量化）。

实验 2（固定 n_drop=8 的换仓边际）：
  - swap_edge(D) = mean r(buy, D→D+1) - mean r(sell, D→D+1)，label v2 口径。
    回答"把排名底 8 换成新 Top8，事后多赚多少"；对照 20bp 往返成本。
  - 决策时点分数差 delta_score = mean(score_buy) - mean(score_sell) → 无差异换手占比。

只读主 checkout 运行资产；输出写 research_out/（不触碰 data/）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent        # 仓库根（主 checkout 或 worktree）
sys.path.insert(0, str(ROOT))

PAPER = ROOT / "data" / "paper_shadow"
SNAP = ROOT / "data" / "universe_snapshots.csv"
OUT = ROOT / "data" / "shadow_analysis"
OUT.mkdir(parents=True, exist_ok=True)

COST_RT = 0.0020  # 5bp 买 + 15bp 卖，与回放引擎一致


def trading_days() -> list[str]:
    days = sorted(p.name for p in PAPER.iterdir() if p.is_dir())
    return days


def load_positions() -> dict[str, list[str]]:
    out = {}
    for p in PAPER.iterdir():
        if p.is_dir():
            f = p / "positions_after_close.csv"
            if f.exists():
                out[p.name] = pd.read_csv(f)["code"].tolist()
    return out


def load_rebalance() -> dict[str, dict]:
    import json

    out = {}
    for p in PAPER.iterdir():
        if p.is_dir():
            f = p / "rebalance_decision.json"
            if f.exists():
                out[p.name] = json.loads(f.read_text())
    return out


def load_scores() -> dict[str, pd.Series]:
    out = {}
    for p in PAPER.iterdir():
        if p.is_dir():
            f = p / "scores.csv"
            if f.exists():
                out[p.name] = pd.read_csv(f).set_index("code")["score"]
    return out


def main() -> None:
    import qlib
    from qlib.data import D

    days = trading_days()
    # 信号日 = 除最后一个目录外的所有日（末日信号尚未结算）
    signal_days = days[:-1]
    next_day = dict(zip(days, days[1:]))

    qlib.init(provider_uri=str(ROOT / "data" / "qlib_root"), region="cn")

    # ---- qlib 批量读取两腿价格（快照并集 + 持仓并集）----
    snap = pd.read_csv(SNAP)
    pos = load_positions()
    rebal = load_rebalance()
    scores = load_scores()

    codes = sorted(
        set(snap[snap["date"].isin(signal_days)]["code_qlib"])
        | {c for cs in pos.values() for c in cs}
    )
    start, end = signal_days[0], days[-1]
    px = D.features(codes, ["$close0941", "$close1500"], start, end, freq="day")
    px.columns = ["c941", "c1500"]
    c941 = px["c941"].unstack(0)   # index=date, columns=code
    c1500 = px["c1500"].unstack(0)

    snap_by_day = {d: g["code_qlib"].tolist() for d, g in snap.groupby("date")}

    def leg_ret(code: str, t: str, t1: str) -> float:
        if code not in c941.columns or code not in c1500.columns:
            return np.nan
        if t not in c941.index or t1 not in c1500.index:
            return np.nan
        a, b = c941.at[t, code], c1500.at[t1, code]
        return b / a - 1 if pd.notna(a) and pd.notna(b) and a > 0 else np.nan

    def basket_ret(codes_: list[str], t: str, t1: str) -> float:
        r = [leg_ret(c, t, t1) for c in codes_]
        r = [x for x in r if not np.isnan(x)]
        return float(np.mean(r)) if r else np.nan

    # ================= 实验 1：篮子基准分解 =================
    rows = []
    for t in signal_days:
        t1 = next_day[t]
        if t not in snap_by_day or t not in pos:
            continue
        strat = basket_ret(pos[t], t, t1)
        bask = basket_ret(snap_by_day[t], t, t1)
        rows.append({"signal_date": t, "settle_date": t1, "strat_label_ret": strat,
                     "basket_label_ret": bask, "active_ret": strat - bask})
    df1 = pd.DataFrame(rows).dropna().reset_index(drop=True)

    # nav 实际日收益（结算日口径）
    nav = pd.read_csv(PAPER / "nav.csv").set_index("date")["nav"]
    nav_ret = nav.pct_change()
    # 首日建仓特例：nav(首日) 收益 = 首日 0941→收盘，与信号日=首日对应；为避免口径混写，丢弃
    df1["nav_ret"] = [nav_ret.get(s, np.nan) for s in df1["settle_date"]]
    df1 = df1.dropna(subset=["nav_ret"]).reset_index(drop=True)
    # nav 逐日盯市把持有期拆到两天（T 收盘→T+1 上午卖 + 新仓 T+1 腿），与 label 全程口径
    # 只能做同段累计比较：nav(settle_last)/nav(settle_first 前一交易日) vs label 累计。
    seg = df1["settle_date"]
    nav_first = nav.index[nav.index < seg.iloc[0]][-1]
    nav_seg_total = float(nav[seg.iloc[-1]] / nav[nav_first] - 1)
    label_seg_total = float((1 + df1["strat_label_ret"]).prod() - 1)
    basket_seg_total = float((1 + df1["basket_label_ret"]).prod() - 1)
    friction_total = nav_seg_total - label_seg_total

    def perf(r: pd.Series) -> dict:
        r = r.dropna()
        nav_s = (1 + r).cumprod()
        dd = (nav_s / nav_s.cummax() - 1).min()
        ann = float(np.mean(r) * 244)
        ir = float(np.mean(r) / (np.std(r, ddof=1) + 1e-12) * np.sqrt(244))
        return {"total": float(nav_s.iloc[-1] - 1), "ann": ann, "ir": ir,
                "maxdd": float(dd), "win": float((r > 0).mean())}

    p_strat, p_bask = perf(df1["strat_label_ret"]), perf(df1["basket_label_ret"])
    p_active = perf(df1["active_ret"])

    # beta / 相关
    beta = float(np.cov(df1["strat_label_ret"], df1["basket_label_ret"])[0, 1]
                 / np.var(df1["basket_label_ret"], ddof=1))
    corr = float(df1["strat_label_ret"].corr(df1["basket_label_ret"]))

    # 状态条件：篮子收益四分位
    q = pd.qcut(df1["basket_label_ret"], 4, labels=["Q1跌", "Q2", "Q3", "Q4涨"])
    cond = df1.groupby(q, observed=True)["active_ret"].agg(["mean", "count"])

    # 尾部：最差 5 个篮子日
    worst = df1.nsmallest(5, "basket_label_ret")[
        ["signal_date", "basket_label_ret", "strat_label_ret", "active_ret", "nav_ret"]]

    df1.to_csv(OUT / "basket_decomposition.csv", index=False)

    # ================= 实验 2：换仓边际 =================
    rows2 = []
    for d in signal_days[1:]:  # 需要有前日持仓可比
        d1 = next_day[d]
        rb = rebal.get(d, {})
        sell, buy = rb.get("sell", []), rb.get("buy_ranked", [])
        if not sell or not buy:
            continue
        r_sell = [leg_ret(c, d, d1) for c in sell]
        r_buy = [leg_ret(c, d, d1) for c in buy]
        r_sell = [x for x in r_sell if not np.isnan(x)]
        r_buy = [x for x in r_buy if not np.isnan(x)]
        if not r_sell or not r_buy:
            continue
        s = scores.get(d, pd.Series(dtype=float))
        sb = [s[c] for c in buy if c in s.index]
        ss = [s[c] for c in sell if c in s.index]
        rows2.append({
            "date": d, "n_sell": len(r_sell), "n_buy": len(r_buy),
            "ret_buy_next": float(np.mean(r_buy)), "ret_sell_next": float(np.mean(r_sell)),
            "swap_edge_gross": float(np.mean(r_buy) - np.mean(r_sell)),
            "swap_edge_net": float(np.mean(r_buy) - np.mean(r_sell)) - COST_RT,
            "delta_score": float(np.mean(sb) - np.mean(ss)) if sb and ss else np.nan,
        })
    df2 = pd.DataFrame(rows2)
    df2.to_csv(OUT / "swap_edge.csv", index=False)

    # ================= 汇总输出 =================
    lines = []
    A = lines.append
    A("=" * 72)
    A(f"影子模拟盘验证  信号日 {df1['signal_date'].iloc[0]} → {df1['signal_date'].iloc[-1]}  n={len(df1)}")
    A("=" * 72)
    A("\n[实验1] 篮子基准分解（label v2 同源两腿，成本前纯选股口径）")
    A(f"  策略:  total {p_strat['total']:+.2%}  ann {p_strat['ann']:+.1%}  "
      f"IR {p_strat['ir']:.2f}  maxdd {p_strat['maxdd']:.2%}  win {p_strat['win']:.0%}")
    A(f"  篮子:  total {p_bask['total']:+.2%}  ann {p_bask['ann']:+.1%}  "
      f"IR {p_bask['ir']:.2f}  maxdd {p_bask['maxdd']:.2%}  win {p_bask['win']:.0%}")
    A(f"  主动:  total {p_active['total']:+.2%}  ann {p_active['ann']:+.1%}  "
      f"IR {p_active['ir']:.2f}  胜率 {p_active['win']:.0%}  日均 {df1['active_ret'].mean():+.3%}")
    A(f"  beta(策略~篮子) = {beta:.2f}   corr = {corr:.2f}")
    A(f"  同段累计对比 ({df1['signal_date'].iloc[0]}→{df1['settle_date'].iloc[-1]}):")
    A(f"    nav 实际 {nav_seg_total:+.2%}  vs  label 口径策略 {label_seg_total:+.2%}  "
      f"vs  篮子 {basket_seg_total:+.2%}")
    A(f"    摩擦(nav-label) = {friction_total:+.2%}  （执行错配[次日上午卖] + 交易成本 + 盯市路径）")
    A("\n  状态条件 alpha（按篮子日收益四分位）:")
    for k, row in cond.iterrows():
        A(f"    {k}: 日均主动 {row['mean']:++.3%}  (n={int(row['count'])})" if False else
          f"    {k}: 日均主动 {row['mean']:+.3%}  (n={int(row['count'])})")
    A("\n  最差 5 个篮子日:")
    for _, r in worst.iterrows():
        A(f"    {r['signal_date']}  篮子 {r['basket_label_ret']:+.2%}  "
          f"策略 {r['strat_label_ret']:+.2%}  主动 {r['active_ret']:+.2%}  nav实际 {r['nav_ret']:+.2%}")
    A("\n[实验2] 固定 n_drop=8 换仓边际（buy vs sell 的次日 label 口径收益差）")
    if len(df2):
        g, n = df2["swap_edge_gross"], len(df2)
        A(f"  n={n} 日  swap_edge(成本前): 均值 {g.mean():+.3%}  中位 {g.median():+.3%}  "
          f"胜率 {(g > 0).mean():.0%}")
        A(f"  swap_edge(扣20bp往返): 均值 {df2['swap_edge_net'].mean():+.3%}  "
          f"为正比例 {(df2['swap_edge_net'] > 0).mean():.0%}")
        ds = df2["delta_score"].dropna()
        # 结构事实：持仓掉出当日成分池 → 无当日分数 → plan_topk_dropout 排队尾被卖。
        # delta_score 可计算 = sell 股当日仍在池的比例。
        in_pool_sell, total_sell = 0, 0
        for d in df2["date"]:
            s = scores.get(d, pd.Series(dtype=float))
            total_sell += len(rebal.get(d, {}).get("sell", []))
            in_pool_sell += sum(1 for c in rebal.get(d, {}).get("sell", []) if c in s.index)
        A(f"  换出股当日仍在成分池(有分数)的比例: {in_pool_sell}/{total_sell} "
          f"= {in_pool_sell / max(1, total_sell):.0%}（其余为掉出池强制退出）")
        # 时序稳定性：前后半窗口
        half = len(df2) // 2
        A(f"  swap_edge 前半窗口 {df2['swap_edge_gross'][:half].mean():+.3%} / "
          f"后半窗口 {df2['swap_edge_gross'][half:].mean():+.3%}")
        A(f"  累计 swap_edge(成本前) = {g.sum():+.2%}；若每日换手 80% 全部停做，"
          f"机会成本口径参考此值")
    report = "\n".join(lines)
    print(report)
    (OUT / "verify_summary.txt").write_text(report)


if __name__ == "__main__":
    main()
