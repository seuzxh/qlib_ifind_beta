"""实时信号生成：纯内存因子计算 + 预测（盘中 9:41 调用）。

⚠️ 审查(2026-07-15): 与 predict_day（盘后路径）有 4 个已知差异待修复：
  1. 因子计算不共享：_compute_all_factors 手算 4 个 extra 因子（vol_vs_yest_t2/t3/t5 +
     overnight_gap），与 materialize_minute.py 公式重复。计划提取共享函数。
  2. ZScoreNorm 死代码：_predict_in_memory 中 ZScoreNorm 查找永远返回 None
     （handler infer_processors=[]），手动标准化分支从不执行。计划删除。
  3. 模型硬编码：硬编码 CHAMPION_RECORDER_ID，无 rolling use_online 路径。计划接入。
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

Pipeline:
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
    CHAMPION_DATA_START, CHAMPION_EXPERIMENT, CHAMPION_FIT_END, CHAMPION_FIT_START,
    CHAMPION_RECORDER_ID, CHAMPION_TOPK, UNIVERSE_MARKET,
)
from qlib_ifind_beta.minute_factors import compute_day_factors

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

    factors = compute_day_factors(c, o, h, l, vol,
                                  prev_day_minute_vol=prev_vols[0] if prev_vols[0] > 0 else None)

    morning_vol_sum = vol[0:10].sum()
    for k, fname in [(2, "vol_vs_yest_t2"), (3, "vol_vs_yest_t3"), (5, "vol_vs_yest_t5")]:
        idx = {2: 1, 3: 2, 5: 3}[k]
        pv = prev_vols[idx] if idx < len(prev_vols) else 0.0
        factors[fname] = morning_vol_sum / (pv / float(REAL_BARS_PER_DAY)) if pv > 0 else np.nan

    if daily_info and daily_info.get("prev_close") and daily_info.get("prev_factor"):
        prev_close = daily_info["prev_close"]
        prev_factor = daily_info["prev_factor"]
        # today_open: prefer daily_info (filled from kline bars); fallback to bar[0] open
        today_open = daily_info.get("open") or float(c[0])
        today_factor = daily_info.get("factor") or prev_factor
        raw_prev_close = prev_close / prev_factor
        raw_open = today_open / today_factor
        factors["overnight_gap"] = raw_open / raw_prev_close - 1.0 if raw_prev_close > 0 else np.nan

        price_941 = c[10]
        raw_p941 = price_941 / today_factor
        factors["change_941"] = raw_p941 / raw_prev_close - 1.0 if raw_prev_close > 0 else np.nan
    else:
        factors["overnight_gap"] = np.nan
        factors["change_941"] = np.nan

    amt = vol * vwap
    d5_amt = amt[0:4].mean()
    factors["amt_ratio_5m"] = amt[5:10].mean() / d5_amt if d5_amt > 0 else np.nan
    factors["amt_vs_yest"] = np.nan

    return factors


def generate_realtime_signal(
    target_date: str,
    topk: int = CHAMPION_TOPK,
    max_workers: int = 10,
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
    result = _predict_in_memory(target_date, factor_rows, topk, change_941_map)

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
                       topk: int, change_941_map: dict = None) -> dict:
    """Predict using champion handler + model, pure in-memory.

    Strategy: build a handler that reads historical bins (up to last synced day)
    for ZScoreNorm fit, then manually inject T-day factor values as a new row
    into the handler's cache, and run predict.
    """
    import qlib
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    from qlib.workflow import R
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler

    # 1. Build handler on historical data (fit ZScoreNorm on train segment)
    # Use the last synced date as end_time (don't need target_date in calendar)
    handler = MinuteEnhancedHandler(
        instruments=UNIVERSE_MARKET,
        start_time=CHAMPION_DATA_START, end_time=CHAMPION_FIT_END,
        fit_start_time=CHAMPION_FIT_START, fit_end_time=CHAMPION_FIT_END,
        label=["$close"],  # dummy label
    )

    # Force the handler to load + fit its processors
    # We use a dummy DatasetH to trigger fit_process_data
    # Get the last trading date from calendar
    cal = D.calendar(freq="day")
    last_date = cal[-1]
    dataset = DatasetH(handler=handler, segments={"test": (last_date, last_date)})

    # Now handler's infer_processors (ZScoreNorm) are fitted on train data.
    # Get the ZScoreNorm fit parameters (mean_train, std_train, cols).
    # ⚠️ 审查(2026-07-15): 死代码！MinuteEnhancedHandler 的 infer_processors=[]（空），
    # 此 for 循环永远不进入，zscore_proc 永远为 None。下游 ZScoreNorm 手动标准化分支
    # （~lines 260-280）从不执行，实际只执行 replace([inf,-inf],NaN).fillna(0)。
    # 计划：删除 zscore_proc 查找 + 手动标准化分支，只保留 fillna(0)。
    zscore_proc = None
    for p in handler.infer_processors:
        if type(p).__name__ == "ZScoreNorm":
            zscore_proc = p
            break

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

    # 3. Apply infer processors manually (ProcessInf → ZScoreNorm → Fillna)
    # ⚠️ 审查(2026-07-15): champion handler 的 infer_processors=[]，无标准化。
    # 下方 ZScoreNorm 分支因 zscore_proc=None 永远跳过，实际只执行：
    #   replace([inf,-inf], NaN) → fillna(0)
    # 这与 handler 的 DropnaProcessor（drop NaN 行）行为不同——路径 A drop，路径 B fillna(0)。
    # 对历史数据无差异（无 NaN），对实时数据（overnight_gap 可能 NaN）会产生微小差异。
    processed = tday_df.copy()
    # ProcessInf: replace inf with NaN
    processed = processed.replace([np.inf, -np.inf], np.nan)

    # ZScoreNorm: (x - mean_train) / std_train, per-column
    if zscore_proc is not None and hasattr(zscore_proc, 'mean_train'):
        cols = zscore_proc.cols  # list of column names to normalize
        mean = zscore_proc.mean_train  # numpy array
        std = zscore_proc.std_train    # numpy array
        # Normalize only the columns that exist in both
        valid_cols = [c for c in cols if c in processed.columns]
        if valid_cols:
            col_idx = [cols.index(c) for c in valid_cols]
            vals = processed[valid_cols].values
            m = mean[col_idx]
            s = std[col_idx]
            normalized = (vals - m) / s
            processed.loc[:, valid_cols] = normalized

    # Fillna: replace NaN with 0
    processed = processed.fillna(0)

    # 4. Drop rows with all-NaN features (matching DropnaProcessor behavior)
    # After Fillna, this shouldn't drop any rows
    processed_clean = processed.dropna()
    if processed_clean.shape[0] == 0:
        logger.warning(f"No valid features after processing")
        return {"date": target_date, "n_candidates": 0, "candidates": [], "topk": []}

    # 5. Load model and predict
    # ⚠️ 审查(2026-07-15): 硬编码 CHAMPION_RECORDER_ID（FROZEN champion），无 rolling
    # use_online 路径。计划：优先从 rolling 实验加载 online 模型，fallback 到 FROZEN。
    rec = R.get_recorder(recorder_id=CHAMPION_RECORDER_ID,
                         experiment_name=CHAMPION_EXPERIMENT)
    model = rec.load_object("params.pkl")

    # HFLGBModel: model.model.predict(x.values)
    scores = pd.Series(model.model.predict(processed_clean.values),
                       index=processed_clean.index)
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
            "candidates": cands, "topk": tradable[:topk]}
