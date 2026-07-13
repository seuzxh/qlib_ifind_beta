"""Real-time signal generation: fetch minute bars → compute factors → write temp bins → predict.

This module bridges real-time kline-fetcher data into the existing champion
prediction pipeline. At 9:41 AM, it:
  1. Fetches 9:31-09:41 minute bars for ~100 universe stocks via kline-fetcher
  2. Computes 18 factors in memory (reusing minute_factors.py + extra formulas)
  3. Writes T-day factor rows into overlay day.bins (appending to existing data)
  4. Calls predict_day(T) which loads the FROZEN champion model and predicts
  5. Returns top10 signal list with scores + prices

The key insight: rather than bypassing qlib's preprocessing (which would require
reproducing RobustZScoreNorm/Fillna fit parameters), we write the T-day factor
values into the existing overlay bins and let predict_day's DatasetH/Handler
pipeline handle normalization. The handler reads from bins, so we just need
to ensure the T-day row exists.
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
    REAL_BARS_PER_DAY, SLOTS_PER_DAY,
)
from qlib_ifind_beta.binio import read_bin, write_bin
from qlib_ifind_beta.minute_factors import compute_day_factors

from .data_fetch import (
    fetch_bars_parallel, get_prev_day_volumes_multi,
    get_daily_close_factor, load_universe,
)

logger = logging.getLogger(__name__)

MORNING_BAR_COUNT = 11  # 09:31-09:41 = 11 bars


def _bars_to_arrays(bars: list[dict]) -> dict[str, np.ndarray] | None:
    """Convert kline-fetcher bars to c/o/h/l/vol/vwap arrays.

    bars: list of dicts with keys date, time, open, high, low, close, volume, amount.
    Returns dict with arrays of length MORNING_BAR_COUNT, or None if insufficient.
    """
    if len(bars) < MORNING_BAR_COUNT:
        return None

    # Take the last 11 bars (should be 09:31-09:41)
    bars = bars[-MORNING_BAR_COUNT:]

    c = np.array([b["close"] for b in bars], dtype=np.float64)
    o = np.array([b["open"] for b in bars], dtype=np.float64)
    h = np.array([b["high"] for b in bars], dtype=np.float64)
    l = np.array([b["low"] for b in bars], dtype=np.float64)
    vol = np.array([b["volume"] for b in bars], dtype=np.float64)
    amount = np.array([b.get("amount", 0) for b in bars], dtype=np.float64)

    # vwap = amount / volume (when volume > 0), else use close
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(vol > 0, amount / vol, c)

    return {"c": c, "o": o, "h": h, "l": l, "vol": vol, "vwap": vwap, "amount": amount}


def _compute_all_factors(arrs: dict, prev_vols: list[float],
                         daily_info: dict | None) -> dict[str, float]:
    """Compute all 18 champion factors from morning arrays.

    Args:
        arrs: dict with c/o/h/l/vol/vwap/amount arrays (length 11)
        prev_vols: [vol_T-1, vol_T-2, vol_T-3, vol_T-5] full-day minute volumes
        daily_info: dict with prev_close, prev_factor, open, factor (for overnight_gap/change_941)

    Returns:
        dict of 18 factor values + price_941 + change_941
    """
    c, o, h, l, vol = arrs["c"], arrs["o"], arrs["h"], arrs["l"], arrs["vol"]
    vwap, amount = arrs["vwap"], arrs["amount"]

    # 14 baseline factors via compute_day_factors
    factors = compute_day_factors(c, o, h, l, vol,
                                  prev_day_minute_vol=prev_vols[0] if prev_vols[0] > 0 else None)

    # Extra 1: vol_vs_yest_t2/t3/t5
    morning_vol_sum = vol[0:10].sum()
    for k, fname in [(2, "vol_vs_yest_t2"), (3, "vol_vs_yest_t3"), (5, "vol_vs_yest_t5")]:
        idx = {2: 1, 3: 2, 5: 3}[k]
        pv = prev_vols[idx] if idx < len(prev_vols) else 0.0
        if pv > 0:
            factors[fname] = morning_vol_sum / (pv / float(REAL_BARS_PER_DAY))
        else:
            factors[fname] = np.nan

    # Extra 2: overnight_gap (unadjusted open gap)
    if daily_info and daily_info.get("prev_close") and daily_info.get("prev_factor"):
        prev_close = daily_info["prev_close"]
        prev_factor = daily_info["prev_factor"]
        today_open = daily_info["open"]
        today_factor = daily_info["factor"]
        raw_prev_close = prev_close / prev_factor
        raw_open = today_open / today_factor
        factors["overnight_gap"] = raw_open / raw_prev_close - 1.0 if raw_prev_close > 0 else np.nan

        # change_941 (unadjusted)
        price_941 = c[10]
        raw_p941 = price_941 / today_factor
        factors["change_941"] = raw_p941 / raw_prev_close - 1.0 if raw_prev_close > 0 else np.nan
    else:
        factors["overnight_gap"] = np.nan
        factors["change_941"] = np.nan

    # Amount factors (materialized but NOT consumed by handler's 18 fields)
    # Compute for completeness but they won't affect prediction
    amt = vol * vwap
    d5_amt = amt[0:4].mean()
    factors["amt_ratio_5m"] = amt[5:10].mean() / d5_amt if d5_amt > 0 else np.nan
    morning_amt_sum = amt[0:10].sum()
    # amt_vs_yest uses T-1 full day amt; we don't have that easily, skip
    factors["amt_vs_yest"] = np.nan

    return factors


def _write_factor_row(code: str, factors: dict, target_date: str,
                      daily_close_path: Path) -> bool:
    """Write a single stock's T-day factor values into overlay day.bins.

    Appends/updates the T-day row in each factor bin file.
    """
    from qlib_ifind_beta.config import DAY_CAL

    # Find the row index for target_date in the daily calendar
    with open(DAY_CAL) as f:
        cal = [line.strip() for line in f if line.strip()]
    target_ts = pd.Timestamp(target_date)
    target_row = None
    for i, d in enumerate(cal):
        if pd.Timestamp(d) == target_ts:
            target_row = i
            break
    if target_row is None:
        logger.warning(f"{target_date} not in day calendar")
        return False

    # Write each factor as a row update in the bin
    dst_dir = Path(FEATURES_DST) / code.lower()

    # All factor fields to write
    all_fields = (
        list(MINUTE_FACTOR_FIELDS) +      # 14 baseline
        list(MINUTE_FACTOR_EXTRA_FIELDS) +  # 4 extra
        list(MINUTE_FACTOR_AMT_FIELDS) +    # 2 amount
        [MINUTE_DEAL_PRICE_FIELD, MINUTE_CHANGE_941_FIELD]  # price_941, change_941
    )

    # Map factor dict keys to bin field names
    field_values = {}
    for f in MINUTE_FACTOR_FIELDS:
        field_values[f] = factors.get(f, np.nan)
    for f in MINUTE_FACTOR_EXTRA_FIELDS:
        field_values[f] = factors.get(f, np.nan)
    for f in MINUTE_FACTOR_AMT_FIELDS:
        field_values[f] = factors.get(f, np.nan)
    field_values[MINUTE_DEAL_PRICE_FIELD] = factors.get("price_941", np.nan)
    field_values[MINUTE_CHANGE_941_FIELD] = factors.get("change_941", np.nan)

    # Get start_index + length from daily close.bin
    si_dc, close_d = read_bin(daily_close_path)
    if close_d.size == 0 or si_dc is None:
        return False
    rel_idx = target_row - si_dc
    if rel_idx < 0 or rel_idx >= close_d.size:
        # Need to extend the bin — write a new row
        # For simplicity, pad with NaN up to target_row
        # Actually this shouldn't happen if calendar is up to date
        logger.warning(f"{code}: target_row {target_row} out of bin range "
                       f"(si={si_dc}, len={close_d.size})")
        return False

    # Write each field
    for fname in all_fields:
        bin_path = dst_dir / f"{fname}.{FREQ}.bin"
        val = field_values.get(fname, np.nan)
        val_f32 = np.float32(val) if np.isfinite(val) else np.float32(np.nan)

        if bin_path.exists():
            si, arr = read_bin(bin_path)
            if si is not None and si == si_dc and arr.size == close_d.size:
                arr[rel_idx] = val_f32
                write_bin(bin_path, si, arr)
            else:
                # Bin doesn't align — skip (will be NaN in prediction)
                logger.debug(f"{code}.{fname}: bin misaligned, skipping")
        # If bin doesn't exist, the factor will be NaN in prediction (acceptable)

    return True


def generate_realtime_signal(
    target_date: str,
    topk: int = 10,
    max_workers: int = 10,
    skip_fetch: bool = False,
    mock_bars: dict | None = None,
) -> dict:
    """Generate top-k signal for target_date using real-time minute data.

    Pipeline:
    1. Load universe (~100 codes)
    2. Fetch 9:31-09:41 minute bars via kline-fetcher (parallel)
    3. Get prev-day volumes from cn_data_1min (for vol_vs_yest family)
    4. Get daily close/factor (for overnight_gap/change_941)
    5. Compute 18 factors per stock
    6. Write factor rows to overlay bins
    7. Call predict_day(target_date) → top10

    Args:
        target_date: YYYY-MM-DD
        topk: number of top stocks to return
        max_workers: parallel fetch workers
        skip_fetch: if True, skip kline fetch (for testing)
        mock_bars: pre-loaded bars dict for testing

    Returns:
        predict_day result dict: {date, n_candidates, candidates, topk}
    """
    from qlib_ifind_beta.config import FEATURES_SRC, FREQ

    logger.info(f"▶ Realtime signal for {target_date}")

    # 1. Load universe
    codes = load_universe(target_date)
    logger.info(f"  Universe: {len(codes)} stocks")
    if not codes:
        raise RuntimeError(f"Empty universe for {target_date}")

    # 2. Fetch minute bars
    if skip_fetch and mock_bars:
        bars_data = mock_bars
    else:
        logger.info(f"  Fetching {MORNING_BAR_COUNT} bars for {len(codes)} stocks...")
        bars_data = fetch_bars_parallel(codes, count=MORNING_BAR_COUNT, max_workers=max_workers)
    logger.info(f"  Fetched: {len(bars_data)}/{len(codes)} stocks")

    # 3. Get prev-day volumes for vol_vs_yest family (k=1,2,3,5)
    logger.info(f"  Loading prev-day volumes from cn_data_1min...")
    prev_vols = get_prev_day_volumes_multi(codes, target_date)

    # 4. Get daily close/factor for overnight_gap + change_941
    logger.info(f"  Loading daily close/factor...")
    daily_info = get_daily_close_factor(codes, target_date)

    # 4b. Fill T-day daily open/close/factor from realtime bars (for overnight_gap)
    # When daily bins don't have T-day data yet (intraday), derive from minute bars:
    #   open = first bar's open, close = last bar's close, factor from T-1 (unchanged intraday)
    logger.info(f"  Filling T-day daily open/close from realtime bars...")
    from qlib_ifind_beta.realtime.data_fetch import _prev_trading_day, _load_day_calendar
    prev_date_str = _prev_trading_day(target_date)
    for code in codes:
        if code not in bars_data:
            continue
        bars = bars_data[code]
        if not bars:
            continue
        # Read T-1 factor from daily bin (should exist)
        ddir = Path(FEATURES_SRC) / code.lower()
        close_path = ddir / f"close.{FREQ}.bin"
        if not close_path.exists():
            continue
        si, close_d = read_bin(close_path)
        si_f, factor_d = read_bin(ddir / f"factor.{FREQ}.bin")
        if factor_d.size == 0 or si_f != si:
            continue

        # Find target_row and prev_row in calendar
        cal = _load_day_calendar()
        target_ts = pd.Timestamp(target_date)
        target_row = None
        prev_row = None
        for i, d in enumerate(cal):
            if prev_date_str and d == prev_date_str:
                prev_row = i
            if d == target_ts.strftime("%Y-%m-%d") or pd.Timestamp(d) == target_ts:
                target_row = i
                break
        if target_row is None:
            continue
        t_idx = target_row - si
        p_idx = (prev_row - si) if prev_row is not None else None
        if t_idx < 0 or t_idx >= close_d.size:
            continue

        # Today's open = first minute bar's open; close = last bar's close
        today_open = float(bars[0]["open"])
        today_close = float(bars[-1]["close"])
        # Factor is unchanged intraday (no ex-div during the session for most stocks)
        # Use T-1 factor if available
        if p_idx is not None and 0 <= p_idx < factor_d.size:
            today_factor = float(factor_d[p_idx])
        else:
            today_factor = float(factor_d[t_idx]) if not np.isnan(factor_d[t_idx]) else 1.0

        # Write to daily bins
        si_o, open_d = read_bin(ddir / f"open.{FREQ}.bin")
        if si_o == si and open_d.size == close_d.size:
            open_d[t_idx] = np.float32(today_open)
            write_bin(ddir / f"open.{FREQ}.bin", si_o, open_d)
        close_d[t_idx] = np.float32(today_close)
        write_bin(close_path, si, close_d)
        factor_d[t_idx] = np.float32(today_factor)
        write_bin(ddir / f"factor.{FREQ}.bin", si_f, factor_d)

        # Update daily_info for factor computation
        if code not in daily_info and p_idx is not None and 0 <= p_idx < close_d.size:
            daily_info[code] = {
                "prev_close": float(close_d[p_idx]),
                "prev_factor": float(factor_d[p_idx]),
                "open": today_open,
                "factor": today_factor,
            }
        elif code in daily_info:
            daily_info[code]["open"] = today_open
            daily_info[code]["factor"] = today_factor

    # 5-6. Compute factors and write to bins
    logger.info(f"  Computing factors + writing bins...")
    success = 0
    failed = []
    for code in codes:
        if code not in bars_data:
            failed.append((code, "no bars"))
            continue
        arrs = _bars_to_arrays(bars_data[code])
        if arrs is None:
            failed.append((code, "insufficient bars"))
            continue

        pv = prev_vols.get(code, [0.0, 0.0, 0.0, 0.0])
        di = daily_info.get(code)
        factors = _compute_all_factors(arrs, pv, di)

        daily_close_path = Path(FEATURES_SRC) / code.lower() / f"close.{FREQ}.bin"
        if daily_close_path.exists():
            if _write_factor_row(code, factors, target_date, daily_close_path):
                success += 1
            else:
                failed.append((code, "write failed"))
        else:
            failed.append((code, "no daily close bin"))

    logger.info(f"  Factors written: {success}/{len(codes)} "
                f"(failed: {len(failed)})")
    if failed:
        sample_fail = failed[:5]
        logger.debug(f"  Sample failures: {sample_fail}")

    # 7. Predict via FROZEN champion model
    # Use dummy label ($close) to bypass DropnaLabel (which would drop T-day rows
    # since Ref($close,-1) needs T+1 close that doesn't exist yet).
    # Then manually prepare features and predict.
    logger.info(f"  Running champion model predict...")
    result = _predict_realtime(target_date, topk)

    logger.info(f"  ✅ Signal generated: {len(result.get('topk', []))} stocks")
    return result


def _predict_realtime(date: str, topk: int = 10) -> dict:
    """Predict using FROZEN champion model, bypassing label T+1 dependency.

    Uses $close as dummy label (avoids DropnaLabel dropping T-day rows where
    Ref($close,-1) is NaN because T+1 close doesn't exist yet).
    Manually prepares features and calls model.model.predict() directly.
    """
    from qlib_ifind_beta.config import OVERLAY_ROOT as _OVERLAY_ROOT
    import qlib
    qlib.init(provider_uri=str(_OVERLAY_ROOT), region="cn")
    from qlib.workflow import R
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler
    from qlib_ifind_beta.config import (
        CHAMPION_DATA_START, CHAMPION_EXPERIMENT, CHAMPION_FIT_END,
        CHAMPION_FIT_START, CHAMPION_RECORDER_ID, UNIVERSE_MARKET,
    )

    # Build handler with dummy label
    handler = MinuteEnhancedHandler(
        instruments=UNIVERSE_MARKET,
        start_time=CHAMPION_DATA_START, end_time=date,
        fit_start_time=CHAMPION_FIT_START, fit_end_time=CHAMPION_FIT_END,
        label=["$close"],  # dummy — always exists, avoids DropnaLabel
    )
    dataset = DatasetH(handler=handler, segments={"test": (date, date)})

    # Prepare features (bypass model.predict which reads "test" segment via DatasetH)
    test_data = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    test_clean = test_data.dropna()
    if test_clean.shape[0] == 0:
        logger.warning(f"No valid features for {date}")
        return {"date": date, "n_candidates": 0, "candidates": [], "topk": []}

    # Load FROZEN champion model
    rec = R.get_recorder(recorder_id=CHAMPION_RECORDER_ID,
                         experiment_name=CHAMPION_EXPERIMENT)
    model = rec.load_object("params.pkl")

    # Predict directly (HFLGBModel.model.predict on raw features)
    scores = pd.Series(model.model.predict(test_clean.values), index=test_clean.index)
    scores = scores.sort_values(ascending=False)

    # Get aux fields (price_941, change_941, limit_up, limit_down)
    aux = D.features(D.instruments(market=UNIVERSE_MARKET),
                     ["$price_941", "$change_941", "$limit_up", "$limit_down"],
                     start_time=date, end_time=date)
    aux_day = {}
    if not aux.empty:
        level = "datetime" if "datetime" in aux.index.names else 1
        try:
            a = aux.xs(date, level=level)
            for code, r in a.iterrows():
                aux_day[code] = (float(r.iloc[0]), float(r.iloc[1]),
                                 float(r.iloc[2]), float(r.iloc[3]))
        except KeyError:
            pass

    # Assemble candidates
    cands = []
    for code, sc in scores.items():
        code_str = code if isinstance(code, str) else (code[1] if isinstance(code, tuple) else str(code))
        if code_str not in aux_day:
            continue
        price_941, change_941, limit_up, limit_down = aux_day[code_str]
        cands.append({"code": code_str, "score": float(sc),
                      "price_941": price_941, "change_941": change_941,
                      "limit_up": limit_up, "limit_down": limit_down})

    # Limit-up interception
    tradable = [c for c in cands if not (c["change_941"] >= c["limit_up"])]
    return {"date": date, "n_candidates": len(cands),
            "candidates": cands, "topk": tradable[:topk]}
