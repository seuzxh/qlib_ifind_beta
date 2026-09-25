"""Phase 2-replace — 对冻结 Champion 的全面替代挑战战役。

用户指令（2026-09-25）：不止验证加速，用更多因子全面挑战 Champion，找可
替代方案。战役假设：18+2 叠加失败源于与现役列的信息冲突，故挑战者以
【替换】与【正交化】为主：

  C1  替换 close_pos 族（3 列）→ B1 价格位置族代表 3 个
  C2  替换动量/加速度族（7 列）→ B1 去重池头部 7 个（含新王 STD5）
  C3  全 zoo 18（drop 全部现役列，纯 zoo 池 top-18）
  C4  正交化 augment（b1_STD5/b1_MAX5 对 18 因子逐日截面残差化为 2 列）
  C5  强替换（close_pos×3 + accel×3 → STD5/MAX5/MIN5/GTJA158/HIGH0/QTLD5）

协议冻结：同 HFLGB 超参/切分/label/Top10-n_drop（重放口径）；同批基线
每个进程必跑（excluded.json 规则 1）；PortAnaRecord 因本机死锁用
SignalRecord + 手工重放替代（只排序不终判）。

Run: conda run -n qlib_ifind_beta python scripts/validate_factor_zoo_challenge.py
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

import scripts.validate_factor_zoo_augment as A
from qlib_ifind_beta import binio
from qlib_ifind_beta.config import OVERLAY_ROOT
from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler

OUTDIR = ROOT / "reports" / "factor_zoo"
B1 = "b1__alpha158__"
DROP_CLOSE_POS = ["close_pos_1m", "close_pos_3m", "close_pos_5m"]
DROP_MOM = ["startup_mom_1m", "startup_mom_3m", "startup_mom_5m",
            "startup_total", "accel_1m", "accel_3m", "accel_5m"]
C5_ADD = [B1 + "STD5", B1 + "MAX5", B1 + "MIN5",
          "b1__gtja191__GTJA158", B1 + "HIGH0", B1 + "QTLD5"]


def b1_pool() -> list[str]:
    """B1 基础门槛通过者按 |残差IC| 降序、spearman>0.85 去重后的池。"""
    m = pd.read_csv(OUTDIR / "minute_b1_metrics.csv")
    ok = m[(m.n_days >= 100) & (m.ic_ir.abs() >= 0.3) & (m.ic_mean.abs() >= 0.015)
           & (m.coverage >= 0.95) & (m.same_sign == True)].copy()
    feats = pd.read_pickle(OUTDIR / "minute_b1.pkl")
    ok["col"] = ok.name.str.replace("b1__", "", regex=False)
    ok = ok[ok.col.isin(feats.columns)]
    corr = feats[ok.col.tolist()].corr(method="spearman").abs()
    order = ok.sort_values("resid_ic", key=lambda s: s.abs(), ascending=False)
    picked: list[str] = []
    for n in order.col:
        if any(abs(corr.loc[n, p]) > 0.85 for p in picked):
            continue
        picked.append(n)
    return picked


def materialize_series(f: pd.Series, fname: str) -> str:
    """Series → overlay 研究 bin（全小写字段名；augment.materialize 的 Series 版）。"""
    day_cal = [d.strip() for d in (OVERLAY_ROOT / "calendars" / "day.txt").read_text().splitlines() if d.strip()]
    cal_pos = {d: i for i, d in enumerate(day_cal)}
    for s, g in f.groupby(level="instrument"):
        out = OVERLAY_ROOT / "features" / s.lower() / f"{fname}.day.bin"
        if out.exists():
            continue
        gg = g.droplevel("instrument")
        gg.index = pd.to_datetime(gg.index).strftime("%Y-%m-%d")
        vals = np.full(len(day_cal), np.nan, dtype=np.float32)
        for d, v in gg.items():
            i = cal_pos.get(d)
            if i is not None and pd.notna(v):
                vals[i] = np.float32(v)
        out.parent.mkdir(parents=True, exist_ok=True)
        start = int(np.argmax(~np.isnan(vals))) if (~np.isnan(vals)).any() else 0
        binio.write_bin(out, start, vals[start:])
    print(f"  物化 {fname}（{f.groupby(level='instrument').ngroups} 股）", flush=True)
    return fname


def orth_residual(spec: str, fname: str, champs: pd.DataFrame) -> str:
    """zoo 特征对 18 champion 因子的逐日截面 OLS 残差 → 物化为新特征。"""
    f = A.load_feature(spec)
    f = f[f.index.isin(champs.index)]
    parts = []
    for d, g in f.groupby(level="datetime"):
        g2 = g.droplevel("datetime").dropna()
        if len(g2) < 50:
            continue
        X = champs.xs(d, level="datetime").reindex(g2.index)
        m = X.notna().all(axis=1)
        if int(m.sum()) < 50:
            continue
        Xm = X.loc[m].to_numpy()
        A_ = np.column_stack([np.ones(len(Xm)), Xm])
        beta, *_ = np.linalg.lstsq(A_, g2.loc[m].to_numpy(), rcond=None)
        r = g2.loc[m].to_numpy() - A_ @ beta
        parts.append(pd.Series(r, index=pd.MultiIndex.from_product(
            [[d], g2.index[m]], names=["datetime", "instrument"])))
    resid = pd.concat(parts) if parts else pd.Series(dtype=float)
    print(f"  正交化 {spec} → {fname}（{len(resid)} 样本）", flush=True)
    return materialize_series(resid, fname)


def run_one(tag: str, add_specs: list[str], drop_cols: list[str], label: pd.Series,
            preds: dict) -> None:
    A.FZ_FIELDS.clear(), A.DROP_FIELDS.clear()
    if drop_cols:
        A.DROP_FIELDS.extend(drop_cols)
    names = []
    for spec in add_specs:
        if spec.startswith("ORTH:"):          # 正交化特征已由 orth_residual 物化
            names.append(spec.split(":", 2)[2])
        else:
            names.append(A.materialize(spec))
    A.FZ_FIELDS.extend(names)
    print(f"\n▶ {tag}: drop={drop_cols or '无'} add={names}", flush=True)
    rec = A.run_task("MinuteEnhancedFZHandler", portana=False)
    m = rec.list_metrics()
    pred = rec.load_object("pred.pkl")
    pred = (pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred)
    pred = pred.reorder_levels(["datetime", "instrument"]).sort_index()
    preds[tag] = pred
    r = A.replay_topn(pred, label)
    preds[f"{tag}__replay"] = r
    print(f"  IC={m.get('IC'):.4f} RankIC={m.get('Rank IC'):.4f} | 重放 {A.replay_stats(r)}", flush=True)


def main() -> None:
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    meta = pd.read_pickle(OUTDIR / "dayf_meta.pkl")
    meta = meta.reorder_levels(["datetime", "instrument"]).sort_index()
    label, champs = meta["LABEL"], meta.drop(columns=["LABEL"])

    pool = b1_pool()
    print(f"B1 去重池 {len(pool)} 个；挑战战役开始", flush=True)

    preds: dict = {}
    # 同批基线
    A.FZ_FIELDS.clear(), A.DROP_FIELDS.clear()
    base_rec = A.run_task("MinuteEnhancedHandler",
                          "qlib_ifind_beta.minute_enhanced_handler", portana=False)
    bm = base_rec.list_metrics()
    bp = base_rec.load_object("pred.pkl")
    bp = (bp.iloc[:, 0] if isinstance(bp, pd.DataFrame) else bp)
    bp = bp.reorder_levels(["datetime", "instrument"]).sort_index()
    rb = A.replay_topn(bp, label)
    preds["BASE"] = bp
    preds["BASE__replay"] = rb
    print(f"▶ BASE(18) IC={bm.get('IC'):.4f} RankIC={bm.get('Rank IC'):.4f} | {A.replay_stats(rb)}", flush=True)

    # 正交化特征（C4 用）
    champ18 = meta[list(MinuteEnhancedHandler.ENHANCED_FIELDS)]
    orth_std5 = orth_residual(B1 + "STD5", "fz_orth_std5", champ18)
    orth_max5 = orth_residual("b1__alpha158__MAX5", "fz_orth_max5", champ18)

    c3_add = [f"b1__{c}" for c in pool[:16]] + ["day__alpha101__ALPHA006", "day__jq110__JQ110_VOL_005"]
    challengers = [
        ("C1_替换close_pos", [B1 + "MAX5", B1 + "MIN5", B1 + "QTLD5"], DROP_CLOSE_POS),
        ("C2_替换动量族", [B1 + "STD5", "b1__gtja191__GTJA158", B1 + "HIGH0",
                          B1 + "LOW0", B1 + "IMAX5", "b1__jq110__JQ110_TVMA_06",
                          "b1__gtja191__GTJA189"], DROP_MOM),
        ("C3_全zoo18", c3_add, list(MinuteEnhancedHandler.ENHANCED_FIELDS)),
        ("C4_正交化augment", [f"ORTH:ignore:{orth_std5}", f"ORTH:ignore:{orth_max5}"], []),
        ("C5_强替换6v6", C5_ADD, DROP_CLOSE_POS + ["accel_1m", "accel_3m", "accel_5m"]),
    ]
    for tag, add, drop in challengers:
        try:
            run_one(tag, add, drop, label, preds)
        except Exception as e:  # noqa: BLE001
            print(f"  {tag} 失败: {type(e).__name__}: {str(e)[:120]}", flush=True)

    # 汇总
    print("\n=== 挑战战役汇总（同批基线对照）===")
    print(f"BASE IC={bm.get('IC'):.4f} | {A.replay_stats(rb)}")
    rows = []
    for tag, _, _ in challengers:
        p = preds.get(tag)
        if p is None:
            continue
        r = preds[f"{tag}__replay"]
        both = pd.DataFrame({"b": rb, "c": r}).dropna()
        h = len(both) // 2
        rows.append({"tag": tag,
                     "半窗Δ日均_前": both.c.iloc[:h].sub(both.b.iloc[:h]).mean(),
                     "半窗Δ日均_后": both.c.iloc[h:].sub(both.b.iloc[h:]).mean(),
                     "累计": f"{(1 + r).prod() - 1:+.1%}"})
    print(pd.DataFrame(rows).to_string(index=False))
    pd.to_pickle({k: v for k, v in preds.items() if not k.endswith("__replay")},
                 OUTDIR / "challenge_preds.pkl")


if __name__ == "__main__":
    main()
