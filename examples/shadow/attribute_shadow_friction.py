"""摩擦完整归因：label 口径 → 实际账本之间的 -25pp 缺口逐项分解。

已确立（反事实账本 + 实测）：
  label 口径(收盘卖、每日全新 Top10、满仓、零成本) +33.32%
  反事实账本(收盘卖、真实结构、含成本)               +8.63%
  实际账本(上午卖、真实结构、含成本)                  +8.11%
  fee 实测                                           ~ +6.4pp

待归因的 ~18pp 结构缺口，逐项量化：
  G1 keep 结构差：账本持有 2 只昨日 keep（今日高分旧仓），label 口径则为
     "第 9-10 名新股"。同段对比 r(第9-10名) - r(keep) 的累计。
  G2 现金拖累：账本平均 ~4.7% 现金仓位的日收益机会成本。
  G3 复利路径/权重残差：等权 label 乘积 vs 真实资金路径的残余。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAPER = ROOT / "data" / "paper_shadow"
OUT = ROOT / "data" / "shadow_analysis"


def main() -> None:
    import qlib
    from qlib.data import D

    days = sorted(p.name for p in PAPER.iterdir() if p.is_dir())
    next_day = dict(zip(days, days[1:]))
    seg_days = [d for d in days if "2026-07-21" <= d <= "2026-09-10"]  # 信号日

    pos_codes = {}
    scores_d = {}
    import json
    rebal = {}
    for d in days:
        f = PAPER / d / "positions_after_close.csv"
        s = PAPER / d / "scores.csv"
        r = PAPER / d / "rebalance_decision.json"
        if f.exists():
            pos_codes[d] = pd.read_csv(f)["code"].tolist()
        if s.exists():
            sc = pd.read_csv(s)
            scores_d[d] = sc.set_index("code")["score"].astype(float)
        if r.exists():
            rebal[d] = json.loads(r.read_text())

    codes = sorted({c for cs in pos_codes.values() for c in cs}
                   | {c for s in scores_d.values() for c in s.index})
    qlib.init(provider_uri=str(ROOT / "data" / "qlib_root"), region="cn")
    px = D.features(codes, ["$close0941", "$close1500"], days[0], "2026-09-11", freq="day")
    px.columns = ["c941", "c1500"]
    c941 = px["c941"].unstack(0)
    c1500 = px["c1500"].unstack(0)

    def pxv(df: pd.DataFrame, t: str, c: str) -> float:
        if t not in df.index or c not in df.columns:
            return np.nan
        v = df.at[t, c]
        return float(v) if pd.notna(v) else np.nan

    # label 口径下的"第 9-10 名"：当日池按分数排序，剔除已持有后第 9、10 位候选
    rows = []
    for t in seg_days[:-1]:
        t1 = next_day[t]
        rb = rebal.get(t, {})
        keep = rb.get("keep", [])
        buy = rb.get("buy_ranked", [])
        if t not in scores_d or not keep:
            continue
        sc = scores_d[t].sort_values(ascending=False)
        held = set(keep) | set(buy)
        rest = [c for c in sc.index if c not in held]
        rank910 = rest[:2]
        if len(rank910) < 2:
            continue
        # 同段对比（T 收盘 → T+1 0941，账本 keep 股的实际持有段）
        def seg_ret(c: str) -> float:
            a, b = pxv(c1500, t, c), pxv(c941, t1, c)
            return b / a - 1 if not (np.isnan(a) or np.isnan(b)) and a > 0 else np.nan

        r_keep = [seg_ret(c) for c in keep]
        r_910 = [seg_ret(c) for c in rank910]
        r_keep = [x for x in r_keep if x == x]
        r_910 = [x for x in r_910 if x == x]
        if not r_keep or not r_910:
            continue
        rows.append({"date": t, "keep_next": float(np.mean(r_keep)),
                     "rank910_next": float(np.mean(r_910)),
                     "struct_gap": float(np.mean(r_910) - np.mean(r_keep))})
    g = pd.DataFrame(rows)

    # 现金拖累：现金占比 × 当日账本收益（机会成本）
    nav = pd.read_csv(PAPER / "nav.csv").set_index("date")
    nav_seg = nav.loc["2026-07-21":"2026-09-11"]
    ret = nav_seg["nav"].pct_change().dropna()
    cash_w = nav_seg["cash"] / nav_seg["nav"]
    cash_drag = float((cash_w.reindex(ret.index).fillna(0) * ret).sum())

    L = []
    A = L.append
    A("=" * 72)
    A("摩擦完整归因（段 7/21→9/11, 38 结算日）")
    A("=" * 72)
    A("  label 口径（收盘卖/每日全新Top10/满仓/零成本）      +33.32%")
    A("  实际账本（上午卖/真实结构/含成本）                  +8.11%")
    A("  ─────────────────────────────────────────")
    A("  摩擦合计                                            -25.21%")
    A(f"  fee 实测                                            -6.42%")
    A(f"  执行错配(上午卖 vs 收盘卖, 反事实账本)              -0.52%")
    if len(g):
        w_keep = 0.2
        keep_gap_total = float(g["struct_gap"].sum()) * w_keep
        A(f"  G1 keep 结构差 (0.2×Σ[r(第9-10名)−r(keep)], n={len(g)} 日)")
        A(f"     keep 股段收益日均 {g['keep_next'].mean():+.3%} vs 第9-10名 {g['rank910_next'].mean():+.3%}"
          f"  → 结构差日均 {g['struct_gap'].mean():+.3%}")
        A(f"     累计贡献 ≈ {keep_gap_total:+.2%}")
    A(f"  G2 现金拖累 (Σ 现金占比×日收益)                     {cash_drag:+.2%}")
    A(f"  G3 残差 (复利路径/权重/估值时点)                     "
      f"{-25.21 - (-6.42) - (-0.52) - (float(g['struct_gap'].sum()) * 0.2 if len(g) else 0) - cash_drag:+.2%}")
    A("")
    A("  注：G1/G2 为机会成本口径（对量级归因），G3 吸收复利与路径混合效应。")
    report = "\n".join(L)
    print(report)
    (OUT / "friction_attribution.txt").write_text(report)
    g.to_csv(OUT / "keep_struct_gap.csv", index=False)


if __name__ == "__main__":
    main()
