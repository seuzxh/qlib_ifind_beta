"""实时信号生成：纯内存因子计算 + 预测（盘中 9:41 调用）。

与物化推理共享同一模型选择和特征缺失口径：
  1. 因子计算不共享：_compute_all_factors 手算 4 个 extra 因子（vol_vs_yest_t2/t3/t5 +
     overnight_gap），与 materialize_minute.py 公式重复。计划提取共享函数。
  2. MinuteEnhancedHandler 无 infer processor；实时矩阵直接执行 shared Dropna(feature)。
  3. 优先使用覆盖目标日的滚动门控模型，任何异常自动回退 HFLGB/FROZEN。
  4. universe 不自动刷新：load_universe(T) 若 T 日未刷新返回空 → RuntimeError。计划自动刷新。

Pipeline（英文原文保留）:
  1. Fetch 9:31-09:41 minute bars via kline-fetcher (parallel)
  2. Compute 18 factors in memory (from minute bars + cn_data_1min T-1 volumes)
  3. Build a one-day feature DataFrame
  4. Use champion handler (fit on 2024-2025 train segment from existing bins)
     to normalize the new data via its already-fit ZScoreNorm
  5. Run HFLGBModel.predict → top10

Key design: NO modification to qlib_data calendars or bins. The handler reads
existing historical bins for fit; T-day factors are injected as an in-memory
DataFrame and processed by the fitted infer_processors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from qlib_ifind_beta.config import (
    FEATURES_DST, FREQ, MINUTE_FACTOR_FIELDS, MINUTE_FACTOR_EXTRA_FIELDS,
    MINUTE_FACTOR_AMT_FIELDS, MINUTE_DEAL_PRICE_FIELD, MINUTE_CHANGE_941_FIELD,
    REAL_BARS_PER_DAY, SLOTS_PER_DAY, OVERLAY_ROOT,
    CHAMPION_TOPK, UNIVERSE_MARKET,
)
from qlib_ifind_beta.minute_factors import compute_champion_factors
from qlib_ifind_beta.model_ensemble import load_model_bundle, predict_bundle_matrix

from .data_fetch import (
    fetch_bars_parallel, get_prev_day_volumes_multi,
    get_daily_close_factor, load_universe,
)

logger = logging.getLogger(__name__)
MORNING_BAR_COUNT = 11  # 09:31-09:41 = 11 bars

# The 18 fields consumed by MinuteEnhancedHandler (order matters for model)
ENHANCED_FIELDS = tuple(MINUTE_FACTOR_FIELDS) + tuple(MINUTE_FACTOR_EXTRA_FIELDS)  # 14 + 4 = 18


def _bars_to_arrays(bars: list[dict]) -> dict[str, np.ndarray] | None:
    """Convert kline-fetcher bars to c/o/h/l/vol/vwap arrays."""
    if len(bars) < MORNING_BAR_COUNT:
        return None
    bars = bars[-MORNING_BAR_COUNT:]
    c = np.array([b["close"] for b in bars], dtype=np.float64)
    o = np.array([b["open"] for b in bars], dtype=np.float64)
    h = np.array([b["high"] for b in bars], dtype=np.float64)
    l = np.array([b["low"] for b in bars], dtype=np.float64)
    vol = np.array([b["volume"] for b in bars], dtype=np.float64)
    amount = np.array([b.get("amount", 0) for b in bars], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(vol > 0, amount / vol, c)
    return {"c": c, "o": o, "h": h, "l": l, "vol": vol, "vwap": vwap, "amount": amount}


def _compute_all_factors(arrs: dict, prev_vols: list[float],
                         daily_info: dict | None) -> dict[str, float]:
    """Compute all 18 champion factors from morning arrays.

    ⚠️ 审查(2026-07-15): 4 个 extra 因子（vol_vs_yest_t2/t3/t5 + overnight_gap）
    在此函数内手算，与 materialize_minute.py 的向量实现公式重复。
    14 个 baseline 因子已通过 compute_day_factors 共享 ✅。
    计划：将 extra 因子提取到 minute_factors.py 共享函数，消除重复。
    """
    c, o, h, l, vol = arrs["c"], arrs["o"], arrs["h"], arrs["l"], arrs["vol"]
    vwap, amount = arrs["vwap"], arrs["amount"]

    daily_info = daily_info or {}
    factors = compute_champion_factors(
        c[:10], o[:10], h[:10], l[:10], vol[:10], prev_vols,
        prev_close=daily_info.get("prev_close"),
        prev_factor=daily_info.get("prev_factor"),
        today_open=daily_info.get("open") or float(o[0]),
        today_factor=daily_info.get("factor") or daily_info.get("prev_factor"),
        execution_close=float(c[10]),
    )

    amt = vol * vwap
    d5_amt = amt[0:4].mean()
    factors["amt_ratio_5m"] = amt[5:10].mean() / d5_amt if d5_amt > 0 else np.nan
    factors["amt_vs_yest"] = np.nan

    return factors


def generate_realtime_signal(
    target_date: str,
    topk: int = CHAMPION_TOPK,
    max_workers: int = 10,
    use_online: bool = True,
) -> dict:
    """Generate top-k signal for target_date using real-time minute data.

    Pure in-memory: no qlib calendar/bin modification.
    """
    logger.info(f"▶ Realtime signal for {target_date}")

    # 1. Load universe
    # ⚠️ 审查(2026-07-15): highbeta883926.txt 的 end_date 可能未到 T 日（需上游 dump_universe
    # 刷新），返回 0 只 → RuntimeError。计划：返回空时自动调 universe.dump_universe 刷新。
    codes = load_universe(target_date)
    logger.info(f"  Universe: {len(codes)} stocks")
    if not codes:
        raise RuntimeError(f"Empty universe for {target_date}")

    # 2. Fetch minute bars
    logger.info(f"  Fetching {MORNING_BAR_COUNT} bars for {len(codes)} stocks...")
    bars_data = fetch_bars_parallel(codes, count=MORNING_BAR_COUNT, max_workers=max_workers)
    logger.info(f"  Fetched: {len(bars_data)}/{len(codes)} stocks")

    # 3. Get prev-day volumes from cn_data_1min (already synced)
    logger.info(f"  Loading prev-day volumes from cn_data_1min...")
    prev_vols = get_prev_day_volumes_multi(codes, target_date)

    # 4. Get daily close/factor for overnight_gap + change_941
    logger.info(f"  Loading daily close/factor...")
    daily_info = get_daily_close_factor(codes, target_date)

    # 4b. Fill T-day daily open + factor from realtime bars.
    # T-day factor not in daily bins → assume factor unchanged (no dividend/split
    # overnight for most stocks). Set today_factor = prev_factor so overnight_gap
    # and change_941 compute correctly using raw prices.
    for code in codes:
        if code not in bars_data or not bars_data[code]:
            continue
        bars = bars_data[code]
        today_open = float(bars[0]["open"])
        di = daily_info.get(code)
        if di:
            di["open"] = today_open
            di["factor"] = di.get("factor") or di.get("prev_factor", 1.0)
        else:
            di = {"open": today_open, "factor": 1.0,
                  "prev_close": np.nan, "prev_factor": 1.0}
            daily_info[code] = di

    # 5. Compute factors for each stock
    logger.info(f"  Computing 18 factors...")
    factor_rows = {}  # code → {factor_name: value}
    price_941_map = {}  # code → price_941
    change_941_map = {}  # code → change_941

    for code in codes:
        if code not in bars_data:
            continue
        arrs = _bars_to_arrays(bars_data[code])
        if arrs is None:
            continue

        pv = prev_vols.get(code, [0.0, 0.0, 0.0, 0.0])
        di = daily_info.get(code)
        factors = _compute_all_factors(arrs, pv, di)

        # Collect the 18 enhanced fields + aux
        row = {}
        for f in ENHANCED_FIELDS:
            row[f] = factors.get(f, np.nan)
        factor_rows[code] = row
        price_941_map[code] = factors.get("price_941", arrs["c"][10])
        change_941_map[code] = factors.get("change_941", np.nan)

    logger.info(f"  Factors computed: {len(factor_rows)}/{len(codes)} stocks")

    if not factor_rows:
        return {"date": target_date, "n_candidates": 0, "candidates": [], "topk": []}

    # 6. Build feature DataFrame + predict via champion handler
    result = _predict_in_memory(
        target_date, factor_rows, topk, change_941_map, use_online=use_online
    )

    # 7. Enrich with price_941 and change_941
    for c in result.get("candidates", []):
        c["price_941"] = price_941_map.get(c["code"], np.nan)
        c["change_941"] = change_941_map.get(c["code"], np.nan)

    for c in result.get("topk", []):
        c["price_941"] = price_941_map.get(c["code"], np.nan)
        c["change_941"] = change_941_map.get(c["code"], np.nan)

    logger.info(f"  ✅ Signal generated: {len(result.get('topk', []))} stocks")
    return result


def _predict_in_memory(target_date: str, factor_rows: dict[str, dict],
                       topk: int, change_941_map: dict = None,
                       use_online: bool = True) -> dict:
    """Predict a T-day feature matrix with the date-matched model bundle."""
    import qlib
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib.data import D

    # 2. Build T-day feature DataFrame
    rows = []
    multi_index = []
    target_ts = pd.Timestamp(target_date)
    for code, factors in factor_rows.items():
        row = {}
        for fname in ENHANCED_FIELDS:
            row[f"${fname}"] = factors.get(fname, np.nan)
        rows.append(row)
        multi_index.append((target_ts, code))

    if not rows:
        return {"date": target_date, "n_candidates": 0, "candidates": [], "topk": []}

    tday_df = pd.DataFrame(rows)
    tday_df.index = pd.MultiIndex.from_tuples(multi_index, names=["datetime", "instrument"])

    # 3. Match MinuteEnhancedHandler.shared DropnaProcessor(feature): any
    # missing/inf feature removes that stock-day from the tradable universe.
    processed_clean = tday_df.replace([np.inf, -np.inf], np.nan).dropna()
    if processed_clean.shape[0] == 0:
        logger.warning(f"No valid features after processing")
        return {"date": target_date, "n_candidates": 0, "candidates": [], "topk": []}

    # 4. Load and predict the frozen/date-matched model bundle.
    bundle = load_model_bundle(target_date, use_online=use_online)
    scores = predict_bundle_matrix(bundle, processed_clean)
    scores = scores.sort_values(ascending=False)

    # 6. Get limit_up/limit_down for buy interception.
    # Primary: read from overlay bins. Fallback: board_limit() per-code constant.
    from qlib_ifind_beta.materialize import board_limit as _board_limit
    aux_day = {}
    try:
        aux = D.features(D.instruments(market=UNIVERSE_MARKET),
                         ["$limit_up", "$limit_down"],
                         start_time=target_date, end_time=target_date)
        if not aux.empty:
            level = "datetime" if "datetime" in aux.index.names else 1
            try:
                a = aux.xs(target_date, level=level)
                for code, r in a.iterrows():
                    aux_day[code] = (float(r.iloc[0]), float(r.iloc[1]))
            except KeyError:
                pass
    except Exception:
        pass

    # 7. Assemble candidates — fill limit_up/down from board_limit if bins missing
    cands = []
    for idx, sc in scores.items():
        code = idx[1] if isinstance(idx, tuple) else idx
        lu, ld = aux_day.get(code, (np.nan, np.nan))
        if not np.isfinite(lu):
            lu, ld = _board_limit(code)
        cands.append({"code": code, "score": float(sc),
                      "limit_up": lu, "limit_down": ld})

    # Limit-up interception (skip if change_941 >= limit_up)
    # If limit_up is NaN, don't filter (conservative: keep the stock)
    if change_941_map is None:
        change_941_map = {}
    for c in cands:
        c["change_941"] = change_941_map.get(c["code"], np.nan)

    tradable = [c for c in cands
                if not (np.isfinite(c.get("limit_up", np.nan)) and
                        c.get("change_941", np.nan) >= c["limit_up"])]

    return {"date": target_date, "n_candidates": len(cands),
            "candidates": cands, "topk": tradable[:topk],
            "model_source": bundle.source, "ensemble_weight": bundle.weight}
