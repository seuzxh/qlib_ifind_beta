"""筛选公共指标库：日度 Rank IC / 残差 IC / 双半窗一致性。

Route A（日频）与 B1/B2（分钟域采样）共用同一套口径，保证跨路线可比。
所有序列均以 (datetime, instrument) MultiIndex Series 传入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_rank_ic(factor: pd.Series, label: pd.Series, min_ns: int = 30) -> pd.Series:
    """逐日 Spearman Rank IC（截面 ≥ min_ns 只才计；日期取与 label 的交集）。"""
    out = {}
    f = factor.dropna()
    lab_days = label.index.get_level_values("datetime").unique()
    f = f[f.index.get_level_values("datetime").isin(lab_days)]
    for d, g in f.groupby(level="datetime"):
        try:
            y = label.xs(d, level="datetime", drop_level=False).reindex(g.index)
        except KeyError:  # pragma: no cover - label 无该日
            continue
        yy = y.dropna()
        if len(yy) >= min_ns:
            out[d] = g.loc[yy.index].rank().corr(yy.rank())
    return pd.Series(out)


def residual_ic(factor: pd.Series, label: pd.Series, champs: pd.DataFrame,
                min_ns: int = 50) -> float:
    """逐日对 18 champion 因子 OLS 残差化后的 Rank IC 均值。"""
    ics: list[float] = []
    lab_days = label.index.get_level_values("datetime").unique()
    champ_days = champs.index.get_level_values("datetime").unique()
    for d, g in factor.groupby(level="datetime"):
        if d not in lab_days or d not in champ_days:
            continue
        g = g.dropna()
        if len(g) < min_ns:
            continue
        X = champs.xs(d, level="datetime").reindex(g.index)
        y = label.xs(d, level="datetime").reindex(g.index)
        m = X.notna().all(axis=1) & y.notna()
        if int(m.sum()) < min_ns:
            continue
        Xm, gm, ym = X.loc[m].to_numpy(), g.loc[m].to_numpy(), y.loc[m].to_numpy()
        A = np.column_stack([np.ones(len(Xm)), Xm])
        bf, *_ = np.linalg.lstsq(A, gm, rcond=None)
        by, *_ = np.linalg.lstsq(A, ym, rcond=None)
        rf, ry = gm - A @ bf, ym - A @ by
        ics.append(pd.Series(rf).corr(pd.Series(ry), method="spearman"))
    return float(np.mean(ics)) if ics else float("nan")


def summarize(factor: pd.Series, label: pd.Series, champs: pd.DataFrame) -> dict:
    """单因子指标汇总：IC/IR/t/覆盖率/双半窗/残差 IC。"""
    ics = daily_rank_ic(factor, label)
    if len(ics) < 100:
        return {"n_days": len(ics)}
    half = len(ics) // 2
    ic1, ic2 = ics.iloc[:half], ics.iloc[half:]
    cov = float(factor.groupby(level="datetime").apply(lambda g: float(g.notna().mean())).mean())
    return {
        "ic_mean": float(ics.mean()),
        "ic_ir": float(ics.mean() / ics.std()),
        "t": float(ics.mean() / ics.std() * np.sqrt(len(ics))),
        "n_days": int(len(ics)),
        "coverage": cov,
        "half1_ic": float(ic1.mean()),
        "half2_ic": float(ic2.mean()),
        "same_sign": bool(np.sign(ic1.mean()) == np.sign(ic2.mean())),
        "resid_ic": residual_ic(factor, label, champs),
    }
