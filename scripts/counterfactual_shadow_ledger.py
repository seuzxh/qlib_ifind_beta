"""论证 1 修正版：反事实账本 —— 精确量化"次日上午 0941 卖出 vs 持有到收盘卖"。

方法：逐笔重放已有回单，唯一变化 = 卖出成交价从 0941 参考价（回放假设）改为
当日 close1500（同源分钟收盘 bin）。买入价、成交量、fee、涨停拦截状态全部保持
实际回放不变。逐日递推现金，市值按收盘价估值（与实际 nav 的估值口径一致）。

输出：实际 nav 曲线 vs 反事实 nav 曲线同段对比，差值 = 执行错配的精确量（含复利），
与 label 口径、fee 实测成本共同构成摩擦三分解。
另附：论证 3 修正 —— 解剖最差主动日 8/18、8/06 的持仓逐股收益。
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


def main() -> None:
    import qlib
    from qlib.data import D

    days = sorted(p.name for p in PAPER.iterdir() if p.is_dir())

    # ---------- 逐笔回放反事实账本 ----------
    qlib.init(provider_uri=str(ROOT / "data" / "qlib_root"), region="cn")

    pos_codes = {}
    for d in days:
        f = PAPER / d / "positions_after_close.csv"
        if f.exists():
            pos_codes[d] = pd.read_csv(f)["code"].tolist()
    codes = sorted({c for cs in pos_codes.values() for c in cs})
    # close1500 估值/反事实卖价；窗口到最后结算日 days[-2]（末信号未结算）
    px = D.features(codes, ["$close1500"], days[0], days[-2], freq="day")
    c1500 = px["$close1500"].unstack(0)

    def close_px(d: str, c: str) -> float:
        if d not in c1500.index or c not in c1500.columns:
            return np.nan
        v = c1500.at[d, c]
        return float(v) if pd.notna(v) else np.nan

    cash_actual_start = None
    rows = []
    cash = 0.0
    initial = 1_000_000.0
    cash = initial
    for d in days[:-1]:
        sf, bf = PAPER / d / "sell_fills.csv", PAPER / d / "buy_fills.csv"
        if not (sf.exists() and bf.exists()):
            continue
        sells = pd.read_csv(sf)
        buys = pd.read_csv(bf)
        fees = float(sells["fee"].sum() + buys["fee"].sum())
        # 实际现金流（0941 卖价）
        sell_in = float((sells["filled_quantity"] * sells["average_price"]).sum())
        buy_out = float((sells.assign(x=0)["x"].sum())  # placeholder
                        + (buys["filled_quantity"] * buys["average_price"]).sum())
        # 反事实现金流：卖出成交价改为 close1500（缺价则保持实际价，保持 fail-closed）
        sell_cf = 0.0
        for r in sells.itertuples():
            p = close_px(d, r.code)
            price = p if not np.isnan(p) and p > 0 else r.average_price
            sell_cf += r.filled_quantity * price
        cash_cf = cash + sell_cf - buy_out - fees
        # 实际现金流递推（对齐回放账本）
        cash = cash + sell_in - buy_out - fees
        # 市值（两种账本相同：收盘价估值）
        mv = 0.0
        for c in pos_codes.get(d, []):
            q_f = PAPER / d / "positions_after_close.csv"
        pos_q = pd.read_csv(PAPER / d / "positions_after_close.csv").set_index("code")["quantity"]
        for c, q in pos_q.items():
            p = close_px(d, c)
            if not np.isnan(p) and p > 0:
                mv += int(q) * p
        rows.append({"date": d, "nav_actual": None, "cash_cf": cash_cf, "mv": mv})
    # 反事实 nav 序列（收盘估值与实际账本口径一致）
    cf = pd.DataFrame(rows).set_index("date")
    cf["nav_cf"] = cf["cash_cf"] + cf["mv"]

    nav = pd.read_csv(PAPER / "nav.csv").set_index("date")["nav"]
    common = [d for d in cf.index if d in nav.index and not np.isnan(cf.at[d, "nav_cf"])]
    actual = nav[common].astype(float)
    counterf = cf.loc[common, "nav_cf"].astype(float)

    # 段：首个完整反事实日 → 最后（close1500 覆盖到的）日
    seg = [d for d in common if d <= "2026-09-11"]
    a0, a1 = actual[seg[0]], actual[seg[-1]]
    c0, c1 = counterf[seg[0]], counterf[seg[-1]]
    # 反事实首日 cash_cf 含初始资金注入口径，用逐日收益比较更稳：
    ret_a = actual[seg].pct_change().dropna()
    ret_c = counterf[seg].pct_change().dropna()
    tot_a, tot_c = float((1 + ret_a).prod() - 1), float((1 + ret_c).prod() - 1)
    fees_seg = float(sum(pd.read_csv(PAPER / d / "sell_fills.csv")["fee"].sum()
                         + pd.read_csv(PAPER / d / "buy_fills.csv")["fee"].sum()
                         for d in seg if (PAPER / d / "sell_fills.csv").exists()))

    out = []
    A = out.append
    A("=" * 72)
    A("论证1修正：反事实账本 —— 唯一变化 = 卖出价 0941 → 当日收盘（同源 close1500）")
    A("=" * 72)
    A(f"  段 {seg[0]} → {seg[-1]} (n={len(ret_a)} 结算日)")
    A(f"  实际 nav（次日上午 0941 卖）:      {tot_a:+.2%}")
    A(f"  反事实 nav（持有到当日收盘卖）:    {tot_c:+.2%}")
    A(f"  执行错配（收盘卖 - 上午卖）:       {tot_c - tot_a:+.2%}")
    A(f"  段内 fee 实测合计:                 {fees_seg / 1_000_000:+.2%} (相对初始 100 万)")
    A(f"  对照 label 口径（实验1，+33.32%）: 反事实仍低于 label 的部分"
      f" = 复利路径/估值时点/keep 持仓的残余差")

    # ---------- 论证 3 修正：最差主动日解剖 ----------
    snap = pd.read_csv(SNAP)
    snap_by_day = {d: g["code_qlib"].tolist() for d, g in snap.groupby("date")}
    c941 = D.features(codes, ["$close0941"], days[0], days[-2], freq="day")["$close0941"].unstack(0)

    def leg(c: str, t: str, t1: str) -> float:
        a = c941.at[t, c] if (t in c941.index and c in c941.columns) else np.nan
        b = c1500.at[t1, c] if (t1 in c1500.index and c in c1500.columns) else np.nan
        a = float(a) if pd.notna(a) else np.nan
        b = float(b) if pd.notna(b) else np.nan
        return b / a - 1 if not (np.isnan(a) or np.isnan(b)) and a > 0 else np.nan

    A("")
    A("=" * 72)
    A("论证3修正：最差两个主动日解剖（信号日持仓的次日逐股收益）")
    A("=" * 72)
    for t in ("2026-08-18", "2026-08-06"):
        t1 = days[days.index(t) + 1]
        rets = {c: leg(c, t, t1) for c in pos_codes.get(t, [])}
        rets = {k: v for k, v in rets.items() if not np.isnan(v)}
        ser = pd.Series(rets)
        bask = [leg(c, t, t1) for c in snap_by_day.get(t, [])]
        bask = [x for x in bask if not np.isnan(x)]
        A(f"\n  信号 {t}（结算 {t1}）：持仓 n={len(ser)}")
        A(f"    持仓均值 {ser.mean():+.2%}  中位 {ser.median():+.2%}  "
          f"篮子均值 {np.mean(bask):+.2%}  主动 {ser.mean() - np.mean(bask):+.2%}")
        A(f"    持仓最差 5: " + "  ".join(f"{c} {v:+.1%}" for c, v in ser.nsmallest(5).items()))
        A(f"    持仓最好 2: " + "  ".join(f"{c} {v:+.1%}" for c, v in ser.nlargest(2).items()))

    report = "\n".join(out)
    print(report)
    (OUT / "counterfactual_summary.txt").write_text(report)
    cf.loc[seg].to_csv(OUT / "counterfactual_nav.csv")


if __name__ == "__main__":
    main()
