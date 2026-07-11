"""Materialize minute factors + $price_941 as day.bin into the overlay.

Writes 20 day.bins/stock: 14 baseline minute factors (MINUTE_FACTOR_FIELDS) + 4
enhanced extras (MINUTE_FACTOR_EXTRA_FIELDS: vol_vs_yest_t2/t3/t5 +
overnight_gap) + $price_941 (deal price) + $change_941 (涨跌停拦截用). The 14
baseline are frozen for m14 reproducibility; the 4 extras feed MinuteEnhancedHandler
(14 + 4 = 18-factor champion, backtest-log §22).

Reads cn_data_1min 1min bins (close/open/high/low/volume) per stock, slices each
trading day's morning window (slots 1-11 = 09:31-09:41: 10 feature bars slots 1-10
+ 1 buy bar slot 10), computes the 14 factors + 09:41 close, and writes them as
day.bin aligned to qlib_data's day
calendar (same start_index and length as the stock's daily close.bin, so
$price_941[T] row-aligns with $close[T]).

vol_vs_yest denominator = previous min-cal trading day's TOTAL minute volume / REAL_BARS_PER_DAY
(= 240 real bars/day; computed from cn_data_1min itself — minute volume on BOTH sides). The
vol_vs_yest_t2/t3/t5 family reuses the same full_day_vol, shifting by 2/3/5 min-cal days.
overnight_gap is day-space (daily open/close, 不复权 caliber). The
minute-vs-daily volume unit mismatch (per-stock ratio 1.0-192.7, ≈ cumulative
adjustment factor: 科创板≈1, 茅台 5.84, 平安银行 192.7) ruled out the prior
qlib_data daily-volume denominator.

Vectorized per stock: scatter the 1min bin onto the global (day, slot) calendar
grid by absolute start_index, take morning slots 1-11 (09:31-09:41) →
(n_min_days, 11), compute factors column-wise, scatter into the day-aligned
output. Calendar-grid alignment is robust to the dataset's UNIVERSAL slot-0 NaN
(every day's bars are all real since 2026-07-11 rebuild, no NaN placeholders) and
to stock-local holes; missing cells become NaN. A naive reshape(240) is
off-by-one if the calendar changes. See spec §物化架构.

Run: see scripts/materialize_minute.py (full universe) or call
materialize_minute_instrument(code) directly.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from .binio import read_bin, write_bin
from .config import (
    BUY_SLOT, DAY_CAL, FEATURES_1MIN_SRC, FEATURES_DST, FEATURES_SRC, FREQ,
    FIRST_FEATURE_SLOT, MIN_CAL,
    MINUTE_CHANGE_941_FIELD, MINUTE_DEAL_PRICE_FIELD,
    MINUTE_FACTOR_AMT_FIELDS, MINUTE_FACTOR_EXTRA_FIELDS, MINUTE_FACTOR_FIELDS,
    REAL_BARS_PER_DAY, SLOTS_PER_DAY,
)

_1MIN_FIELDS = ("close", "open", "high", "low", "volume", "vwap")

# Of MINUTE_FACTOR_EXTRA_FIELDS（enhanced extras），vol_vs_yest_t2/t3/t5 在 min-cal 空间（随 fac
# scatter，复用 full_day_vol shift 2/3/5）；overnight_gap 在 day-cal 空间（与 change_941 同处
# 直接算，输入是日频 open/close）。两组并集 == MINUTE_FACTOR_EXTRA_FIELDS，否则 config 改动未同步。
_EXTRA_MINUTE_SPACE = ("vol_vs_yest_t2", "vol_vs_yest_t3", "vol_vs_yest_t5")
_EXTRA_DAY_SPACE = ("overnight_gap",)
assert set(_EXTRA_MINUTE_SPACE) | set(_EXTRA_DAY_SPACE) == set(MINUTE_FACTOR_EXTRA_FIELDS), (
    "materialize 内部 _EXTRA_* 空间划分与 config.MINUTE_FACTOR_EXTRA_FIELDS 不一致，请同步")
_OVERNIGHT_GAP_FIELD = "overnight_gap"
# Morning window: slots FIRST_FEATURE_SLOT..BUY_SLOT inclusive = 11 calendar rows.
_MORNING_WINDOW = BUY_SLOT - FIRST_FEATURE_SLOT + 1   # 11
_cal_cache: dict = {}


def _load_min_calendar():
    """Return (min_dates_ord, min_slots) as int arrays over the 1min calendar.

    min_dates_ord[i] = date.toordinal() of 1min row i; min_slots[i] = slot-within-day.
    Cached (1.55M rows, ~once per process).
    """
    if "min" not in _cal_cache:
        dts = []
        with open(MIN_CAL) as fp:
            for line in fp:
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
        _cal_cache["min"] = (dates, slots)
    return _cal_cache["min"]


def _load_day_calendar_lookup():
    """Return (n_days, date_to_row) where date_to_row[ordinal] = day-calendar row.

    date_to_row is a numpy int32 array indexed by date.toordinal(); -1 if the date
    is not a trading day. Sized to the max ordinal in day.txt.
    """
    if "day" not in _cal_cache:
        with open(DAY_CAL) as fp:
            rows = [line.strip() for line in fp if line.strip()]
        ordinals = [
            datetime.strptime(s, "%Y-%m-%d").toordinal() for s in rows
        ]
        max_ord = max(ordinals)
        lookup = np.full(max_ord + 1, -1, dtype=np.int32)
        for i, ord_ in enumerate(ordinals):
            lookup[ord_] = i
        _cal_cache["day"] = (len(rows), lookup)
    return _cal_cache["day"]


def _read_1min_fields(code: str):
    """Return (start_index, dict field→float32 array) or (None, None) if missing/misaligned."""
    d = Path(FEATURES_1MIN_SRC) / code.lower()
    arrs = {}
    si = None
    for f in _1MIN_FIELDS:
        p = d / f"{f}.1min.bin"
        if not p.exists():
            return None, None
        s, a = read_bin(p)
        if a.size == 0 or s is None:
            return None, None
        arrs[f] = a
        if si is None:
            si = s
        elif s != si:
            return None, None   # fields disagree on start_index
    # Length-consistency guard: a truncated bin (e.g. partial write) would scatter
    # with mismatched sizes → silent field misalignment. Treat as "no minute data".
    if len({a.size for a in arrs.values()}) > 1:
        return None, None
    return si, arrs


def materialize_minute_instrument(code: str) -> bool:
    """Read 1min + daily bins for `code`, write 20 day.bins: 14 baseline minute
    factors + 4 enhanced extras (vol_vs_yest_t2/t3/t5 + overnight_gap) +
    $price_941 + $change_941.

    Returns True on success; False if the 1min source is missing/misaligned, or the
    daily close/factor/open bin is missing/misaligned (caller treats as "no minute
    data for this stock" — qlib reads NaN).
    """
    si_m, m = _read_1min_fields(code)
    if si_m is None:
        return False

    # daily close.bin = alignment reference (output start_index + length). The
    # previous implementation also read daily volume.bin for vol_vs_yest, but
    # that crossed minute↔daily units — now both sides use cn_data_1min minute
    # volume, so the daily-volume read is dropped.
    ddir = Path(FEATURES_SRC) / code.lower()
    si_dc, close_d = read_bin(ddir / f"close.{FREQ}.bin")
    if close_d.size == 0 or si_dc is None:
        return False
    si_df, factor_d = read_bin(ddir / f"factor.{FREQ}.bin")
    if factor_d.size != close_d.size or si_df != si_dc:
        return False   # factor 必须与 close 同对齐（同属 qlib_data 日频 bin）
    # overnight_gap 输入：日频 open.bin（后复权），与 close/factor 同对齐校验。
    si_do, open_d = read_bin(ddir / f"open.{FREQ}.bin")
    if open_d.size != close_d.size or si_do != si_dc:
        return False   # open 必须与 close 同对齐（overnight_gap 口径依赖）

    # Calendar-grid alignment: scatter each 1min bin onto the global (day, slot)
    # grid by absolute calendar row, then take morning slots FIRST_FEATURE_SLOT
    # (0) .. BUY_SLOT (10) → 09:31-09:41. Robust to stock-local holes;
    # missing cells become NaN. A naive reshape(240) is
    # off-by-one. See spec §物化架构.
    min_dates, min_slots = _load_min_calendar()
    _, date_to_row = _load_day_calendar_lookup()

    morning_rows = np.where(
        (min_slots >= FIRST_FEATURE_SLOT) & (min_slots <= BUY_SLOT)
    )[0]   # 11 rows/day (slots 1-11), ordered
    assert morning_rows.size % _MORNING_WINDOW == 0, (
        f"1min calendar malformed: morning_rows={morning_rows.size} not divisible "
        f"by {_MORNING_WINDOW} (slots/day={SLOTS_PER_DAY})"
    )
    n_min_days = morning_rows.size // _MORNING_WINDOW   # total trading days in 1min cal

    def morning2d(field):
        """Stock's morning bars on the global grid → (n_min_days, _MORNING_WINDOW).

        Column j corresponds to calendar slot FIRST_FEATURE_SLOT+j (0..10 =
        09:31-09:41). NaN where the stock has no bar at that calendar row
        (suspensions and pre-listing produce NaN).
        """
        arr = m[field]
        flat = np.full(morning_rows.size, np.nan, dtype=np.float64)
        valid = (morning_rows >= si_m) & (morning_rows < si_m + arr.size)
        flat[valid] = arr[morning_rows[valid] - si_m].astype(np.float64)
        return flat.reshape(n_min_days, _MORNING_WINDOW)

    c, o, h, l, v = (morning2d("close"), morning2d("open"), morning2d("high"),
                     morning2d("low"), morning2d("volume"))

    # vol_vs_yest denominator (minute-only both sides): per-day FULL-DAY minute
    # volume from cn_data_1min, summed across all real (non-NaN) slots that day.
    # Computed from the scattered grid (uses the WHOLE volume bin, not just the
    # morning window) → per-min-day total. Reused by the whole vol_vs_yest family
    # (shift 1/2/3/5，见下方 E 段)；NaN-safe：NaN 占位槽对求和贡献 0。
    n_total_min_rows = min_slots.size
    vol_bin = m["volume"]
    bin_rows = si_m + np.arange(vol_bin.size, dtype=np.int64)
    in_range = (bin_rows >= 0) & (bin_rows < n_total_min_rows)
    day_idx_all = np.where(in_range, bin_rows, 0) // SLOTS_PER_DAY
    in_range &= day_idx_all < n_min_days
    src_vals = vol_bin[in_range].astype(np.float64)
    src_vals = np.where(np.isfinite(src_vals), src_vals, 0.0)
    full_day_vol = np.zeros(n_min_days, dtype=np.float64)
    np.add.at(full_day_vol, day_idx_all[in_range], src_vals)

    # per-min-day date → day-calendar row → output-relative index.
    # morning_rows[::_MORNING_WINDOW] = each day's slot-FIRST_FEATURE_SLOT (slot 0)
    # cal row (date is constant in-day). clip ordinal before fancy-index
    # (defensive: a stray 1min date outside the day calendar — e.g. the extra
    # 2026-07-03 min-cal day not in day-cal — must not crash the whole stock;
    # mark it invalid instead).
    first_rows = morning_rows[::_MORNING_WINDOW]
    day_dates = min_dates[first_rows]
    max_ord = date_to_row.size - 1
    safe = np.clip(day_dates, 0, max_ord)
    oob = (day_dates < 0) | (day_dates > max_ord)
    day_rows = np.where(oob, -1, date_to_row[safe])   # -1 where not a trading day
    rel = day_rows - si_dc

    # vectorized factors over n_min_days; features use columns 0-9 (slots 1-10),
    # buy price uses column 10 (slot 11). Mirror compute_day_factors exactly.
    fac = {}
    n = c.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        # A. startup momentum
        fac["startup_mom_1m"] = c[:, 9] / c[:, 8] - 1.0
        fac["startup_mom_3m"] = c[:, 9] / c[:, 6] - 1.0
        fac["startup_mom_5m"] = c[:, 9] / c[:, 4] - 1.0
        fac["startup_total"] = c[:, 9] / o[:, 0] - 1.0
        # B. acceleration
        fac["accel_1m"] = (c[:, 9] / o[:, 9] - 1.0) - (c[:, 0] / o[:, 0] - 1.0)
        fac["accel_3m"] = (c[:, 9] / o[:, 7] - 1.0) - (c[:, 1] / o[:, 0] - 1.0)
        fac["accel_5m"] = (c[:, 9] / o[:, 5] - 1.0) - (c[:, 3] / o[:, 0] - 1.0)
        # C. close position in window
        fac["close_pos_1m"] = (c[:, 9] - l[:, 9]) / (h[:, 9] - l[:, 9])
        h3, l3 = h[:, 7:10].max(axis=1), l[:, 7:10].min(axis=1)
        fac["close_pos_3m"] = (c[:, 9] - l3) / (h3 - l3)
        h5, l5 = h[:, 5:10].max(axis=1), l[:, 5:10].min(axis=1)
        fac["close_pos_5m"] = (c[:, 9] - l5) / (h5 - l5)
        # D. volume ratio (NaN where front-seg mean ≤ 0 — match compute_day_factors guard)
        d1 = v[:, 0:9].mean(axis=1)
        fac["vol_ratio_1m"] = np.full(n, np.nan)
        np.divide(v[:, 9], d1, out=fac["vol_ratio_1m"], where=(d1 > 0))
        d3 = v[:, 0:2].mean(axis=1)
        fac["vol_ratio_3m"] = np.full(n, np.nan)
        np.divide(v[:, 7:10].mean(axis=1), d3, out=fac["vol_ratio_3m"], where=(d3 > 0))
        d5 = v[:, 0:4].mean(axis=1)
        fac["vol_ratio_5m"] = np.full(n, np.nan)
        np.divide(v[:, 5:10].mean(axis=1), d5, out=fac["vol_ratio_5m"], where=(d5 > 0))
        # E. cross-day volume family (minute-only both sides): T 日 9:30-9:40 累积量 /
        #    (T-k 日全天分钟量 / REAL_BARS_PER_DAY)。k=1 → vol_vs_yest（主力反转因子，与
        #    compute_day_factors oracle 数值一致）；k=2/3/5 → 反转族强化（连续放量更稳）。
        #    shift k → 首 k 个 min-cal 日 NaN；分母 ≤0 → NaN（NaN-safe）。
        morning_vol_sum = v[:, 0:10].sum(axis=1)
        for k, fname in ((1, "vol_vs_yest"), (2, "vol_vs_yest_t2"),
                         (3, "vol_vs_yest_t3"), (5, "vol_vs_yest_t5")):
            pv = np.full(n_min_days, np.nan, dtype=np.float64)
            if n_min_days > k:
                pv[k:] = full_day_vol[:-k]
            pv_safe = np.where(pv > 0, pv, np.nan)
            fac[fname] = morning_vol_sum / (pv_safe / float(REAL_BARS_PER_DAY))
        # F. amount（成交额）量能因子（item 7, 2026-07-12）：amount = volume × vwap。
        #    价格加权的量能信号，与 D/E 族的 raw volume 正交（vol 不变但价变 → amt 变）。
        #    amt_ratio_5m：日内后段成交额比前段（对标 vol_ratio_5m）。
        #    amt_vs_yest：开盘成交额 vs 昨日全天均（对标 vol_vs_yest，alpha #1）。
        vwp = morning2d("vwap")
        amt = v * vwp  # (n_min_days, 11) per-bar 成交额
        d5_amt = amt[:, 0:4].mean(axis=1)
        fac["amt_ratio_5m"] = np.full(n, np.nan)
        np.divide(amt[:, 5:10].mean(axis=1), d5_amt, out=fac["amt_ratio_5m"],
                  where=(d5_amt > 0))
        morning_amt_sum = amt[:, 0:10].sum(axis=1)
        # full_day_amt：T-1 全天总成交额（scatter 全天 vol×vwap）
        vwp_bin = m["vwap"]
        amt_bin_vals = vol_bin.astype(np.float64) * vwp_bin.astype(np.float64)
        amt_bin_vals = np.where(np.isfinite(amt_bin_vals), amt_bin_vals, 0.0)
        src_amt = amt_bin_vals[in_range]
        src_amt = np.where(np.isfinite(src_amt), src_amt, 0.0)
        full_day_amt = np.zeros(n_min_days, dtype=np.float64)
        np.add.at(full_day_amt, day_idx_all[in_range], src_amt)
        pv_amt = np.full(n_min_days, np.nan, dtype=np.float64)
        if n_min_days > 1:
            pv_amt[1:] = full_day_amt[:-1]
        pv_amt_safe = np.where(pv_amt > 0, pv_amt, np.nan)
        fac["amt_vs_yest"] = morning_amt_sum / (pv_amt_safe / float(REAL_BARS_PER_DAY))
    price_941 = c[:, 10]

    # scatter into day-aligned output (length = daily close.bin length)
    n_out = close_d.size
    _scatter_fields = list(MINUTE_FACTOR_FIELDS) + list(_EXTRA_MINUTE_SPACE) + list(MINUTE_FACTOR_AMT_FIELDS)
    out = {name: np.full(n_out, np.nan, dtype=np.float32) for name in _scatter_fields}
    out_p941 = np.full(n_out, np.nan, dtype=np.float32)
    valid = (rel >= 0) & (rel < n_out) & (day_rows >= 0)
    rel_v = rel[valid]
    for name in _scatter_fields:
        out[name][rel_v] = fac[name][valid].astype(np.float32)
    out_p941[rel_v] = price_941[valid].astype(np.float32)

    # D 方案 V4 校正（2026-07-06）：~3% 异常票的 1min close bin 被记为不复权 raw 价
    # （1min_factor 误填 1.0），与 day 后复权 close 口径冲突 → label 爆炸（如 SH600608
    # factor=51.86 → label=+44.69）。判据：p941 靠近不复权 close/factor 而非后复权 close，
    # 且 factor>1.5（V1 距离判定 + factor 门槛；排除小 factor 票 unadj≈close 时距离判定
    # 不稳的误判）。校正：p941 *= factor（仅异常票），还原成后复权 9:41 价。校正后
    # change_941 自洽、deal_price 同口径；14 分钟因子全为比率（异常票 raw 价分子分母同
    # 口径）自洽不受影响。详见 backtest-log 2026-07-06-l1-full-backtest.md §13。
    with np.errstate(invalid="ignore", divide="ignore"):
        p941_f = out_p941.astype(np.float64)
        close_f = close_d.astype(np.float64)
        factor_f = factor_d.astype(np.float64)
        unadj_d = close_f / factor_f
        abnormal = (np.abs(p941_f - unadj_d) < np.abs(p941_f - close_f)) & (factor_f > 1.5)
    out_p941[abnormal] = (p941_f[abnormal] * factor_f[abnormal]).astype(np.float32)

    # $change_941 (v2): 9:41 时刻涨跌幅（不复权）vs T-1 不复权收盘 —— 涨跌停 buy 表达式用。
    # day-aligned 空间算：change_941[T]=(price_941[T]/factor[T])/(close[T-1]/factor[T-1])-1。
    # 首日无昨收 → NaN；停牌日 out_p941=NaN → change_941=NaN（NaN-safe）。
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_p941 = out_p941.astype(np.float64) / factor_d.astype(np.float64)
        raw_close = close_d.astype(np.float64) / factor_d.astype(np.float64)
        raw_prev_close = np.full(n_out, np.nan, dtype=np.float64)
        if n_out > 1:
            raw_prev_close[1:] = raw_close[:-1]
        out_change941 = (raw_p941 / raw_prev_close - 1.0).astype(np.float32)

        # overnight_gap（enhanced extra，day 空间反转因子）：不复权开盘跳空 =
        # (open[T]/factor[T]) / (close[T-1]/factor[T-1]) - 1。复用上方 raw_prev_close。
        # 不复权口径（同 change_941 源）：除权日 factor 跳变会被分母分子同步抵消 → 反映真实
        # 开盘情绪；若用后复权 ($open/Ref($close,1)-1) 除权缺口被复权抹平 → 口径错。
        # 首日无昨收 → NaN；停牌日 open=NaN → NaN（NaN-safe）。9:30 集合竞价 < 9:41 买入，无前视。
        raw_open = open_d.astype(np.float64) / factor_d.astype(np.float64)
        out_overnight_gap = (raw_open / raw_prev_close - 1.0).astype(np.float32)

    dst_dir = Path(FEATURES_DST) / code.lower()
    for name in _scatter_fields:
        write_bin(dst_dir / f"{name}.{FREQ}.bin", si_dc, out[name])
    write_bin(dst_dir / f"{MINUTE_DEAL_PRICE_FIELD}.{FREQ}.bin", si_dc, out_p941)
    write_bin(dst_dir / f"{MINUTE_CHANGE_941_FIELD}.{FREQ}.bin", si_dc, out_change941)
    write_bin(dst_dir / f"{_OVERNIGHT_GAP_FIELD}.{FREQ}.bin", si_dc, out_overnight_gap)
    return True
