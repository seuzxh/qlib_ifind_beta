"""Phase 2-risk — 冻结 Champion 分数 × 加速状态软惩罚 的 Top10 重放对照。

不重训模型：读取冻结 recorder 的 pred.pkl，在 Top20 内做
    score' = rank_pct(pred) - λ · accel_score_t1（横截面 rank_pct）
按 score' 取 Top10，与 λ=0（=champion 原选择）对照。手工重放口径（含成本
open 0.0005 / close 0.0015，buy @0941 / sell @T+1 1500 与 label v2 同腿）——
按 excluded.json 方法论规则 2：**只用于排序与方向判断，不作终判**。

λ 只在训练半窗（前 50% 天）网格选择，测试半窗报告；并输出"已加速日 vs
非加速日"的分组差异，验证非加速日不受伤。

Run: conda run -n qlib_ifind_beta python scripts/validate_factor_zoo_overlay.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import qlib

from qlib_ifind_beta.config import CHAMPION_RECORDER_ID, OVERLAY_ROOT

OUTDIR = ROOT / "reports" / "factor_zoo"
OPEN_COST, CLOSE_COST = 0.0005, 0.0015
LAMBDA_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]


def replay(pred: pd.Series, penalty: pd.Series | None, lam: float) -> pd.Series:
    """逐日 Top20 内软惩罚选 Top10，返回逐日组合净收益。"""
    daily = {}
    for d, g in pred.groupby(level="datetime"):
        top20 = g.sort_values(ascending=False).head(20)
        if penalty is None:
            sel = top20.sort_values(ascending=False).head(10)
        else:
            p = penalty.xs(d, level="datetime").reindex(top20.index.get_level_values("instrument"))
            p.index = top20.index
            score = top20.rank(pct=True) - lam * p.rank(pct=True).fillna(0.5)
            sel = score.sort_values(ascending=False).head(10)
        # label v2 即 T 09:41 → T+1 15:00 毛收益；成本按换手近似：全额双边
        # xs 后是单层 instrument 索引，用 instrument 层对齐
        y = LABEL.xs(d, level="datetime").reindex(sel.index.get_level_values("instrument"))
        daily[d] = float(y.mean()) - OPEN_COST - CLOSE_COST
    return pd.Series(daily).sort_index()


def stats(r: pd.Series) -> dict:
    cum = float((1 + r).prod() - 1)
    ann = float((1 + cum) ** (252 / max(len(r), 1)) - 1)
    dd = float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())
    return {"累计": f"{cum:+.1%}", "年化": f"{ann:+.1%}", "最大回撤": f"{dd:.1%}",
            "Calmar": f"{ann/abs(dd):.2f}" if dd else "inf", "天数": len(r)}


LABEL: pd.Series


def main() -> None:
    global LABEL
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib.workflow import R
    rec = R.get_recorder(recorder_id=CHAMPION_RECORDER_ID,
                         experiment_name="minute_enhanced_tk10_nd8")
    pred = rec.load_object("pred.pkl")
    pred = (pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred)
    pred = pred.reorder_levels(["datetime", "instrument"]).sort_index()
    print(f"pred 天数 {pred.index.get_level_values('datetime').nunique()}")

    meta: pd.DataFrame = pd.read_pickle(OUTDIR / "dayf_meta.pkl")
    meta = meta.reorder_levels(["datetime", "instrument"]).sort_index()
    LABEL = meta["LABEL"]
    accel: pd.DataFrame = pd.read_pickle(OUTDIR / "accel_state.pkl")
    score_t1 = accel["accel_score_t1"]

    base = replay(pred, None, 0.0)
    print("\nλ=0 基线（champion 原选择，手工重放口径）:", stats(base))

    days = base.index
    split = days[len(days) // 2]
    print(f"训练半窗 ≤ {split.date()}，测试半窗 > {split.date()}")

    # 训练半窗选 λ
    best_lam, best_calmar = 0.0, -np.inf
    for lam in LAMBDA_GRID:
        r = replay(pred, score_t1, lam)
        tr = r[r.index <= split]
        s = stats(tr)
        calmar = float(s["Calmar"]) if s["Calmar"] != "inf" else 99.0
        print(f"  train λ={lam:<4}: {s}")
        if calmar > best_calmar:
            best_calmar, best_lam = calmar, lam
    print(f"选定 λ* = {best_lam}")

    r_star = replay(pred, score_t1, best_lam)
    print("\n测试半窗对照:")
    print("  λ=0  :", stats(base[base.index > split]))
    print(f"  λ={best_lam}:", stats(r_star[r_star.index > split]))
    print("全窗对照:")
    print("  λ=0  :", stats(base))
    print(f"  λ={best_lam}:", stats(r_star))

    # 已加速日 vs 非加速日分组（全窗，λ*）
    day_accel = score_t1.groupby(level="datetime").mean()
    med = day_accel.median()
    acc_days = set(day_accel[day_accel > med].index)
    for name, sel in (("加速市(状态高分日)", acc_days), ("平静日", set(day_accel.index) - acc_days)):
        rr = r_star[[d in sel for d in r_star.index]]
        bb = base[[d in sel for d in base.index]]
        print(f"  {name}: Δ日均 {rr.mean()-bb.mean():+.4%}（λ* {stats(rr)['累计']} vs 基线 {stats(bb)['累计']}）")


if __name__ == "__main__":
    main()
