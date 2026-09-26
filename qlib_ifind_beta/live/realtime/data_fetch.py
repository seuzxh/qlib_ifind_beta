"""Real-time minute bar fetching via kline-fetcher + prev-day cache from cn_data_1min.

Provides:
  - fetch_realtime_bars(code, count=11) → list of 11 bars (09:31-09:41)
  - get_prev_day_volumes(codes, target_date) → dict code → full-day minute vol
  - load_universe(date) → list of codes for that trading day

kline-fetcher env: KLINE_API_BASE_URL=http://183.242.5.14:7778
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from qlib_ifind_beta.config import (
    CN_DATA_1MIN, DAY_CAL, OVERLAY_ROOT, SLOTS_PER_DAY, UNIVERSE_MARKET,
)

logger = logging.getLogger(__name__)

# kline-fetcher path (must be importable)
KLINE_FETCHER_PATH = "/home/zxh/quant_projects/kline-fetcher"
KLINE_API_BASE_URL = "http://183.242.5.14:7778"


def _ensure_kline_env():
    """Ensure KLINE_API_BASE_URL is set before importing kline_fetcher."""
    os.environ.setdefault("KLINE_API_BASE_URL", KLINE_API_BASE_URL)
    if KLINE_FETCHER_PATH not in sys.path:
        sys.path.insert(0, KLINE_FETCHER_PATH)


def _get_fetcher():
    """Lazy-init MinKLineFetcher singleton."""
    _ensure_kline_env()
    from kline_fetcher.min_kline import MinKLineFetcher
    if not hasattr(_get_fetcher, "_inst"):
        _get_fetcher._inst = MinKLineFetcher()
    return _get_fetcher._inst


# ─── real-time bar fetching ──────────────────────────────────────────────────

def fetch_realtime_bars(code: str, count: int = 11) -> list[dict] | None:
    """Fetch the latest `count` minute bars for a stock via kline-fetcher.

    Returns list of dicts with keys: date, time, open, high, low, close,
    volume, amount. Returns None on error or empty.

    count=11 → bars ending at the most recent minute (09:31-09:41 at 9:41).
    """
    try:
        fetcher = _get_fetcher()
        bars = fetcher.fetch_min_kline(code, count=-count)
        if not bars:
            return None
        return bars
    except Exception as e:
        logger.warning(f"fetch_min_kline({code}) failed: {e}")
        return None


def fetch_bars_parallel(codes: list[str], count: int = 11,
                        max_workers: int = 10) -> dict[str, list[dict]]:
    """Fetch minute bars for multiple stocks in parallel.

    Returns dict code → list of bars (None values omitted).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_realtime_bars, c, count): c for c in codes}
        for future in as_completed(futures):
            code = futures[future]
            try:
                bars = future.result()
                if bars:
                    results[code] = bars
            except Exception as e:
                logger.warning(f"fetch {code}: {e}")
    return results


# ─── prev-day data from cn_data_1min (already synced) ────────────────────────

def _load_day_calendar():
    """Return list of trading dates from day.txt."""
    with open(DAY_CAL) as f:
        return [line.strip() for line in f if line.strip()]


def _prev_trading_day(target_date: str) -> str | None:
    """Find the previous trading day before target_date."""
    cal = _load_day_calendar()
    target = pd.Timestamp(target_date)
    prev = None
    for d in cal:
        ts = pd.Timestamp(d)
        if ts < target:
            prev = d
        else:
            break
    return prev


def _load_day_cal_1min():
    """Return (dates_ord, slots) arrays from cn_data_1min 1min calendar."""
    min_cal = CN_DATA_1MIN / "calendars" / "1min.txt"
    dts = []
    with open(min_cal) as f:
        for line in f:
            line = line.strip()
            if line:
                dts.append(datetime.strptime(line, "%Y-%m-%d %H:%M:%S"))
    dates = np.empty(len(dts), dtype=np.int64)
    slots = np.empty(len(dts), dtype=np.int16)
    cur_date = None
    slot = 0
    for i, dt in enumerate(dts):
        if dt.date() != cur_date:
            cur_date = dt.date()
            slot = 0
        dates[i] = dt.toordinal()
        slots[i] = slot
        slot += 1
    return dates, slots


def get_prev_day_volumes(codes: list[str], target_date: str) -> dict[str, float]:
    """Get T-1 full-day minute volume for each code from cn_data_1min.

    Returns dict code → total minute volume (sum of 240 bars) for T-1.
    """
    from qlib_ifind_beta.binio import read_bin

    prev_date = _prev_trading_day(target_date)
    if prev_date is None:
        logger.warning(f"No prev trading day for {target_date}")
        return {}

    # Find prev_date's slot range in 1min calendar
    min_dates, min_slots = _load_day_cal_1min()
    prev_ord = pd.Timestamp(prev_date).toordinal()
    day_mask = min_dates == prev_ord
    if not day_mask.any():
        logger.warning(f"{prev_date} not in 1min calendar")
        return {}

    results = {}
    features_1min = CN_DATA_1MIN / "features"
    for code in codes:
        vol_path = features_1min / code.lower() / "volume.1min.bin"
        if not vol_path.exists():
            continue
        si, vol = read_bin(vol_path)
        if vol.size == 0 or si is None:
            continue
        # The bin's row i corresponds to calendar row si+i.
        # We need rows where calendar date == prev_date.
        # Calendar rows for prev_date: min_dates == prev_ord
        cal_rows = np.where(day_mask)[0]
        # Map to bin indices
        bin_indices = cal_rows - si
        valid = (bin_indices >= 0) & (bin_indices < vol.size)
        if not valid.any():
            continue
        vals = vol[bin_indices[valid]].astype(np.float64)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        results[code] = float(vals.sum())

    return results


def get_prev_day_volumes_multi(codes: list[str], target_date: str,
                               days_back: int = 5) -> dict[str, list[float]]:
    """Get T-1, T-2, T-3, T-5 full-day volumes for vol_vs_yest_t{k}.

    Primary: read from cn_data_1min minute bins (sum of 240 bars).
    Fallback: read daily volume from qlib_data day bins (equivalent: daily vol
    ≈ sum of minute vols). Used when 1min bins haven't been synced for recent
    dates (common during live trading before EOD sync).

    Returns dict code → [vol_T-1, vol_T-2, vol_T-3, vol_T-5] (0 if missing).
    """
    from qlib_ifind_beta.binio import read_bin

    cal = _load_day_calendar()
    target = pd.Timestamp(target_date)
    # Find target index, then get k trading days back
    cal_ts = [pd.Timestamp(d) for d in cal]
    target_idx = None
    for i, ts in enumerate(cal_ts):
        if ts >= target:
            target_idx = i
            break
    if target_idx is None:
        target_idx = len(cal_ts) - 1

    # Indices for k=1,2,3,5 back
    k_indices = {}
    for k in [1, 2, 3, 5]:
        idx = target_idx - k
        k_indices[k] = cal[idx] if idx >= 0 else None

    # Try cn_data_1min first
    min_dates, _ = _load_day_cal_1min()
    features_1min = CN_DATA_1MIN / "features"

    results = {}
    missing_codes = []

    for code in codes:
        vol_path = features_1min / code.lower() / "volume.1min.bin"
        if not vol_path.exists():
            missing_codes.append(code)
            continue
        si, vol = read_bin(vol_path)
        if vol.size == 0 or si is None:
            missing_codes.append(code)
            continue

        vols_by_k = []
        bin_has_data = True
        for k in [1, 2, 3, 5]:
            date_str = k_indices.get(k)
            if date_str is None:
                vols_by_k.append(0.0)
                bin_has_data = False
                continue
            prev_ord = pd.Timestamp(date_str).toordinal()
            day_mask = min_dates == prev_ord
            if not day_mask.any():
                vols_by_k.append(0.0)
                bin_has_data = False
                continue
            cal_rows = np.where(day_mask)[0]
            bin_indices = cal_rows - si
            valid = (bin_indices >= 0) & (bin_indices < vol.size)
            if not valid.any():
                vols_by_k.append(0.0)
                bin_has_data = False
                continue
            vals = vol[bin_indices[valid]].astype(np.float64)
            vals = np.where(np.isfinite(vals), vals, 0.0)
            vols_by_k.append(float(vals.sum()))
        if bin_has_data:
            results[code] = vols_by_k
        else:
            missing_codes.append(code)

    # Fallback: for codes where 1min bin missing T-k data, fetch from kline-fetcher.
    # ⚠️ 审查(2026-07-15): 不能用日频 volume 替代——实测日频 volume 与分钟 volume 之和的
    # 比值在 0.005-0.99 之间（单位/口径完全不同），替代会导致 vol_vs_yest 因子值偏移 10-10000 倍。
    # 必须用 kline-fetcher 拉真实分钟 bar 求和（与 cn_data_1min bin 偏差 ~14%，可接受）。
    if missing_codes:
        logger.info(f"  1min fallback to kline-fetcher for {len(missing_codes)} codes")
        _fill_volumes_from_kline(missing_codes, k_indices, results)

    return results


def _fill_volumes_from_kline(codes: list[str], k_indices: dict,
                             results: dict):
    """Fill vol_vs_yest denominators from kline-fetcher minute bars.

    Fetches T-1/T-2/T-3/T-5 full-day minute bars and sums volumes.
    Uses KLineFetcher.fetch_min_kline(count=-N) which returns last N bars.
    """
    _ensure_kline_env()
    from kline_fetcher.min_kline import MinKLineFetcher
    fetcher = _get_fetcher()

    # How many bars to fetch? We need up to T-5 (5 trading days × 240 bars = 1200)
    # plus today's bars (up to 240). Fetch last 1500 bars and filter by date.
    FETCH_COUNT = 1500

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def fetch_one(code):
        try:
            bars = fetcher.fetch_min_kline(code, count=-FETCH_COUNT)
            if not bars:
                return code, {}
            from collections import defaultdict
            by_date = defaultdict(float)
            for b in bars:
                d = b.get("date", "")
                vol = b.get("volume", 0)
                if d and vol:
                    by_date[d] += vol
            return code, dict(by_date)
        except Exception:
            return code, {}

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_one, c): c for c in codes}
        for fut in as_completed(futures):
            code, daily_vols = fut.result()
            vols_by_k = []
            for k in [1, 2, 3, 5]:
                date_str = k_indices.get(k)
                if date_str and date_str in daily_vols:
                    vols_by_k.append(daily_vols[date_str])
                else:
                    vols_by_k.append(0.0)
            results[code] = vols_by_k

    return results


# ─── universe ─────────────────────────────────────────────────────────────────

def load_universe(date: str, market: str = UNIVERSE_MARKET,
                  filter_st: bool = True) -> list[str]:
    """Load universe codes for a given date from instruments file.

    Format: code\\tstart_date\\tend_date (TSV). Returns codes where
    start_date <= date <= end_date.

    filter_st: if True, exclude ST/*ST stocks (±5% limit, different risk profile).
    ST status is checked via kline-fetcher get_stock_info name lookup.

    highbeta883926.txt 的 end_date 取决于上次成分快照刷新时间。若 T 日未刷新，
    返回空列表；调用方必须 fail closed，不能沿用过期股票池。
    """
    instr_path = OVERLAY_ROOT / "instruments" / f"{market}.txt"
    if not instr_path.exists():
        raise FileNotFoundError(f"Universe file not found: {instr_path}")

    target = pd.Timestamp(date)
    codes = []
    with open(instr_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            code, start, end = parts[0], parts[1], parts[2]
            try:
                if pd.Timestamp(start) <= target <= pd.Timestamp(end):
                    codes.append(code)
            except Exception:
                continue

    if filter_st and codes:
        codes = _filter_st_stocks(codes)

    return codes


def _filter_st_stocks(codes: list[str]) -> list[str]:
    """Remove ST/*ST stocks from the list using kline-fetcher name lookup."""
    _ensure_kline_env()
    try:
        from kline_fetcher.min_kline import MinKLineFetcher
        fetcher = MinKLineFetcher()
        filtered = []
        st_found = []
        for code in codes:
            try:
                info = fetcher.get_stock_info(code)
                name = ""
                if isinstance(info, dict):
                    name = info.get("name", "")
                elif isinstance(info, str):
                    name = info
                if "ST" in name.upper():
                    st_found.append((code, name))
                    continue
                filtered.append(code)
            except Exception:
                # If we can't check, keep the stock (don't penalize API errors)
                filtered.append(code)
        if st_found:
            logger.info(f"  Filtered {len(st_found)} ST stocks: "
                        + ", ".join(f"{c}({n})" for c, n in st_found))
        return filtered
    except Exception as e:
        logger.warning(f"ST filter failed ({e}), skipping filter")
        return codes


# ─── overnight_gap / change_941 inputs from daily bins ──────────────────────

def get_daily_close_factor(codes: list[str], target_date: str) -> dict[str, dict]:
    """Get T-1 daily close + factor for overnight_gap / change_941 computation.

    Returns dict code → {prev_close, prev_factor, open, factor} for target_date.
    Reads from qlib_data (read-only daily bins).
    """
    from qlib_ifind_beta.config import FEATURES_SRC, FREQ
    from qlib_ifind_beta.binio import read_bin

    prev_date = _prev_trading_day(target_date)
    if prev_date is None:
        return {}

    cal = _load_day_calendar()
    cal_ts = [pd.Timestamp(d) for d in cal]
    # Find row indices for target_date and prev_date
    target_ts = pd.Timestamp(target_date)
    prev_ts = pd.Timestamp(prev_date)

    results = {}
    for code in codes:
        ddir = Path(FEATURES_SRC) / code.lower()
        close_path = ddir / f"close.{FREQ}.bin"
        if not close_path.exists():
            continue
        si, close_d = read_bin(close_path)
        if close_d.size == 0 or si is None:
            continue
        si_f, factor_d = read_bin(ddir / f"factor.{FREQ}.bin")
        if factor_d.size != close_d.size or si_f != si:
            continue
        si_o, open_d = read_bin(ddir / f"open.{FREQ}.bin")
        if open_d.size != close_d.size or si_o != si:
            continue

        # Find row indices in the daily array
        # cal row for a date = index in cal_ts
        # bin row = cal_row - si (where cal_row is the position in day.txt)
        target_row = None
        prev_row = None
        for i, ts in enumerate(cal_ts):
            if ts == prev_ts:
                prev_row = i
            if ts == target_ts:
                target_row = i
                break

        if target_row is None or prev_row is None:
            continue
        t_idx = target_row - si
        p_idx = prev_row - si
        if p_idx < 0 or p_idx >= close_d.size:
            continue
        # T-day bins may not exist yet (intraday, before EOD sync).
        # Return T-1 data + None for T-day fields; caller fills from kline bars.
        has_today = 0 <= t_idx < close_d.size
        if not has_today:
            results[code] = {
                "prev_close": float(close_d[p_idx]),
                "prev_factor": float(factor_d[p_idx]),
                "open": None,
                "factor": float(factor_d[p_idx]),  # assume unchanged
                "factor_confirmed": False,
            }
            continue

        results[code] = {
            "prev_close": float(close_d[p_idx]),
            "prev_factor": float(factor_d[p_idx]),
            "open": float(open_d[t_idx]),
            "factor": float(factor_d[t_idx]),
            "factor_confirmed": True,
        }

    return results
