"""Phase 1C 加速状态变量 — 多日分钟形态构造"已经加速"代理 + 风险轨诊断。

状态变量（每股每交易日，全部只用该日 240 根分钟 bar，之后 shift 到 T 日）：
  day_ret      全天收益 close[239]/close[0]-1（1min 原值口径）
  tail30_mom   尾盘 30 分钟动量 close[239]/close[209]-1
  tail60_mom   尾盘 60 分钟动量
  max_gain     盘中最大涨幅 max(high)/prev_close-1（昨收从分钟序列前一日 close[239]）
  near_limit   近涨停分钟数（bar close/prev_close-1 ≥ 0.97×涨停幅；涨停幅按
               板块：sh688/sz300/bj→20%，其余→10%）
  vol_ratio_d1 当日总量/昨日总量
  up_bars      上涨分钟占比（全天 PSY）
  close_pos    尾盘位置 close[239] 在全天 high-low 区间位置
  path_dd      日内路径最大回撤（cummax-close)/cummax 的最大值
  amp          全天振幅 (max_high-min_low)/prev_close

加速状态（T 日 09:41 可得 = T-1 值 shift1，另做 t2/t3/t5 变体）：
  accel_score = 横截面 rank 均值[ day_ret, tail30_mom, max_gain, vol_ratio_d1 ]
  accel_top   = accel_score ≥ 0.8（"已经加速"布尔状态）

诊断（对齐 label v2 与 champion Top10）：
  A. 各分量/合成状态的 IC、双半窗
  B. 状态覆盖与持续性
  C. 条件分布：label | accel_top vs 其他（均值/中位/P10/bottom-decile/P(label<-5%)）
  D. champion Top10 中 accel_top 占比 vs 全池占比；Top10 内条件对照（overlay 前瞻）

Run: conda run -n qlib_ifind_beta python scripts/validate_factor_zoo_accel.py
产物：reports/factor_zoo/accel_state.pkl + 控制台诊断
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

from qlib_ifind_beta.config import CN_DATA_1MIN, OVERLAY_ROOT

OUTDIR = ROOT / "reports" / "factor_zoo"
SLOTS = 240
DAY_START = "2023-06-01"   # 留预热，状态从 2024-01-02 起 shift 使用


def read_bin_f32(path: Path):
    if not path.exists():
        return None, np.array([])
    raw = np.fromfile(path, dtype="<f4")
    if raw.size < 2:
        return None, np.array([])
    return int(raw[0]), raw[1:].astype(np.float64)


def limit_rate(stock: str) -> float:
    s = stock.lower()
    if s.startswith(("sh688", "sz300", "bj")):
        return 0.20
    return 0.10


def stock_states(stock: str, min_cal_days: list[str]) -> dict[str, np.ndarray] | None:
    feat = CN_DATA_1MIN / "features" / stock.lower()
    arrays = {}
    for f in ("open", "high", "low", "close", "volume"):
        si, v = read_bin_f32(feat / f"{f}.1min.bin")
        arrays[f] = (si, v)
    si0 = arrays["close"][0]
    if si0 is None:
        return None
    n_days_total = len(min_cal_days)
    n = arrays["close"][1].size // SLOTS
    if n < 20:
        return None
    c = arrays["close"][1][: n * SLOTS].reshape(n, SLOTS)
    h = arrays["high"][1][: n * SLOTS].reshape(n, SLOTS)
    l = arrays["low"][1][: n * SLOTS].reshape(n, SLOTS)
    v = arrays["volume"][1][: n * SLOTS].reshape(n, SLOTS)
    bad = np.isnan(c).any(axis=1) | np.isnan(v).any(axis=1)
    prev_close = np.concatenate([[np.nan], c[:, -1][:-1]])
    ret1m = c / prev_close[:, None] - 1.0
    lr = limit_rate(stock)
    out = {
        "day_ret": c[:, -1] / c[:, 0] - 1.0,
        "tail30_mom": c[:, -1] / c[:, -31] - 1.0,
        "tail60_mom": c[:, -1] / c[:, -61] - 1.0,
        "max_gain": np.nanmax(h, axis=1) / prev_close - 1.0,
        "near_limit": (ret1m >= 0.97 * lr).sum(axis=1).astype(float),
        "vol_ratio_d1": v.sum(axis=1) / np.concatenate([[np.nan], v.sum(axis=1)[:-1]]),
        "up_bars": (np.diff(c, axis=1) > 0).mean(axis=1),
        "close_pos": (c[:, -1] - np.nanmin(l, axis=1)) /
                     np.maximum(np.nanmax(h, axis=1) - np.nanmin(l, axis=1), 1e-12),
        "amp": (np.nanmax(h, axis=1) - np.nanmin(l, axis=1)) / prev_close,
    }
    cummax = np.maximum.accumulate(c, axis=1)
    out["path_dd"] = ((cummax - c) / np.maximum(cummax, 1e-12)).max(axis=1)
    for k in out:
        out[k][bad] = np.nan
        out[k][:2] = np.nan  # prev_close 预热
    # 对齐到分钟日历日
    first_day = si0 // SLOTS
    full = {k: np.full(n_days_total, np.nan) for k in out}
    for j in range(n):
        di = min_day_to_cal.get(first_day + j)
        if di is not None:
            for k in out:
                full[k][di] = out[k][j]
    return full


def build() -> pd.DataFrame:
    global min_day_to_cal
    day_cal = [d.strip() for d in (OVERLAY_ROOT / "calendars" / "day.txt").read_text().splitlines() if d.strip()]
    day_pos = {d: i for i, d in enumerate(day_cal)}
    min_cal = [t.strip() for t in (CN_DATA_1MIN / "calendars" / "1min.txt").read_text().splitlines() if t.strip()]
    min_days = [min_cal[i * SLOTS][:10] for i in range(len(min_cal) // SLOTS)]
    min_day_to_cal = {i: day_pos[d] for i, d in enumerate(min_days) if d in day_pos}

    uni_rows = []
    for ln in (OVERLAY_ROOT / "instruments" / "highbeta883926.txt").read_text().splitlines():
        p = ln.split("\t")
        if len(p) >= 3:
            uni_rows.append((p[0], p[1], p[2]))
    stocks = sorted({s for s, _, _ in uni_rows})
    start_i = day_pos["2024-01-02"]
    recs = []
    for s in stocks:
        st = stock_states(s, min_days)
        if st is None:
            continue
        for di in range(start_i, len(day_cal)):
            for k in st:
                recs.append((day_cal[di], s, k, st[k][di]))
    df = pd.DataFrame(recs, columns=["datetime", "instrument", "field", "value"])
    wide = df.pivot_table(index=["datetime", "instrument"], columns="field", values="value")
    wide.to_pickle(OUTDIR / "accel_state_raw.pkl")
    return wide


def _to_ts(ser: pd.Series) -> pd.Series:
    """datetime 层统一为 Timestamp（与 dayf_meta 对齐）。"""
    lv = ser.index.get_level_values("datetime")
    if not isinstance(lv.dtype, pd.DatetimeTZDtype) and not pd.api.types.is_datetime64_any_dtype(lv):
        ser = ser.copy()
        ser.index = ser.index.set_levels(pd.to_datetime(ser.index.levels[0]), level="datetime")
    return ser


def diag(wide: pd.DataFrame) -> None:
    from qlib_ifind_beta.experiment.screen_lib import daily_rank_ic
    meta: pd.DataFrame = pd.read_pickle(OUTDIR / "dayf_meta.pkl")
    meta = meta.reorder_levels(["datetime", "instrument"]).sort_index()
    label = meta["LABEL"]
    champs = meta.drop(columns=["LABEL"])

    # shift1：T 日特征 = T-1 状态
    idx = wide.index.to_frame(index=False)
    day_cal = sorted(idx["datetime"].unique())
    feats = {}
    grp = {s: g for s, g in wide.groupby(level="instrument")}
    for col in wide.columns:
        for k, shift in (("t1", 1), ("t2", 2), ("t3", 3), ("t5", 5)):
            if k != "t1" and col not in ("day_ret", "vol_ratio_d1", "max_gain"):
                continue  # 多日变体只做核心三分量
            shifted = {}
            for s, g in grp.items():
                gg = g[col].droplevel("instrument").reindex(day_cal)
                shifted[s] = gg.shift(shift)
            f = _to_ts(pd.concat(shifted).rename_axis(["instrument", "datetime"]).swaplevel())
            feats[f"{col}_{k}"] = f[f.index.isin(label.index)]

    # accel_score_t1：四分量横截面 rank 均值
    comp = ["day_ret_t1", "tail30_mom_t1", "max_gain_t1", "vol_ratio_d1_t1"]
    frame = pd.DataFrame({k: v for k, v in feats.items() if k in comp})
    ranks = frame.groupby(level="datetime").rank(pct=True)
    score = ranks.mean(axis=1)
    feats["accel_score_t1"] = score
    feats["accel_top_t1"] = (score >= 0.8).astype(float).where(score.notna())

    print("▶ A. 状态分量 IC（对 label v2）")
    for name in ("day_ret_t1", "tail30_mom_t1", "tail60_mom_t1", "max_gain_t1",
                 "near_limit_t1", "vol_ratio_d1_t1", "up_bars_t1", "close_pos_t1",
                 "path_dd_t1", "amp_t1", "accel_score_t1",
                 "day_ret_t2", "max_gain_t2", "vol_ratio_d1_t2"):
        f = feats.get(name)
        if f is None:
            continue
        ics = daily_rank_ic(f, label)
        if len(ics) < 100:
            continue
        half = len(ics) // 2
        print(f"  {name:18s} IC={ics.mean():+.4f} IR={ics.mean()/ics.std():+.3f} "
              f"half1={ics.iloc[:half].mean():+.4f} half2={ics.iloc[half:].mean():+.4f} n={len(ics)}")

    top = feats["accel_top_t1"]
    lab_days = label.index.get_level_values("datetime").unique()
    top_f = top[top.index.get_level_values("datetime").isin(lab_days)]
    print(f"\n▶ B. 状态覆盖: accel_top 比例 {top_f.mean():.2%}")
    # 持续性：T-1 top 且 T-2 top
    score_prev = {}
    cal_ts = pd.to_datetime(pd.Index(day_cal))
    for s, g in score.groupby(level="instrument"):
        gg = g.droplevel("instrument").reindex(cal_ts)
        score_prev[s] = gg.shift(2)
    prev = _to_ts(pd.concat(score_prev).rename_axis(["instrument", "datetime"]).swaplevel())
    prev_top = (prev >= 0.8)
    both = (top_f > 0) & prev_top.reindex(top_f.index).fillna(False)
    print(f"   持续性: T-1 已加速且 T-2 也加速 → {both.sum()/max((top_f>0).sum(),1):.2%}")

    print("\n▶ C. 条件分布 label | accel_top")
    y = label.reindex(top_f.index).dropna()
    m = top_f.reindex(y.index) > 0
    for name, mask in (("已加速", m), ("未加速", ~m)):
        v = y[mask]
        print(f"  {name}: n={len(v):6d} 均值={v.mean():+.4%} 中位={v.median():+.4%} "
              f"P10={v.quantile(0.1):+.4%} P(label<-5%)={(v < -0.05).mean():.2%} "
              f"bottom-decile均值={v.quantile(0.1):+.4%}")

    # D. champion Top10 诊断（pred 从 mlruns 冻结 recorder）
    print("\n▶ D. champion Top10 × accel_top（overlay 前瞻）")
    try:
        import qlib as _q
        _q.init(provider_uri=str(OVERLAY_ROOT), region="cn")
        from qlib.workflow import R
        from qlib_ifind_beta.config import CHAMPION_RECORDER_ID
        rec = R.get_recorder(recorder_id=CHAMPION_RECORDER_ID, experiment_name="minute_enhanced_tk10_nd8")
        pred = rec.load_object("pred.pkl")
        ps = pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred
        ps = ps.reorder_levels(["datetime", "instrument"]).sort_index()
        hits, cnts, ys_top, m_top = 0, 0, [], []
        for d, g in ps.groupby(level="datetime"):
            if d not in set(top_f.index.get_level_values("datetime")):
                continue
            pick = g.sort_values(ascending=False).head(10).index
            t = top_f.xs(d, level="datetime").reindex(pick).fillna(0) > 0
            hits += int(t.sum()); cnts += len(pick)
            yday = label.xs(d, level="datetime").reindex(pick)
            ys_top.append(yday.mean())
            m_top.append(float(yday[t.reindex(pick).fillna(False)].mean()) if t.any() else np.nan)
        pool = top_f.mean()
        print(f"  Top10 内 accel_top 占比 {hits/max(cnts,1):.2%} vs 全池 {pool:.2%}")
        print(f"  Top10 日均 label（含已加速） {np.nanmean(ys_top):+.4%}；"
              f"Top10∩已加速 日均 {np.nanmean(m_top):+.4%}")
    except Exception as e:  # noqa: BLE001
        print(f"  [待确认] champion pred 读取失败: {type(e).__name__}: {str(e)[:100]}")

    pd.DataFrame({k: v for k, v in feats.items()}).to_pickle(OUTDIR / "accel_state.pkl")
    print(f"\nsaved → {OUTDIR/'accel_state.pkl'}")


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    raw = OUTDIR / "accel_state_raw.pkl"
    if raw.exists():
        wide = pd.read_pickle(raw)
        print(f"复用 {raw.name} {wide.shape}")
    else:
        wide = build()
        print(f"raw states {wide.shape}")
    diag(wide)
