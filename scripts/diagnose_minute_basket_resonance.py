"""Screen richer 09:31-09:40 paths and constituent-basket resonance factors.

Unlike the rejected SH000001 broadcast factors, the basket is reconstructed
from each date's actual 883926 members and every candidate varies across the
cross-section.  This is a cheap screen only: promotion still requires purged
walk-forward retraining and the exact TD0 index-enhancement backtest.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from qlib_ifind_beta.config import OVERLAY_ROOT
from qlib_ifind_beta.materialize_minute import _load_min_calendar, _read_1min_fields
from scripts.validate_factor_challengers import WINDOWS, _daily_corr
from scripts.validate_prediction_blend import _series

RESULT_PATH = ROOT / "data" / "minute_basket_resonance_diagnostic.json"
CHAMPION_RECORDERS = {
    "W4_2024Q4": "cef0056252f64ac2ae81092683d7ea63",
    "W2_2025Q2": "beaa3a394ae04b04818e92fb6d663d51",
    "W3_2025Q4": "f4d6572fa4c44e789da3c2f96b8f3717",
    "W1_2026Q2": "e0ca66e6b67745ccb4ef244119609e7c",
}


def _safe_corr_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - np.nanmean(left, axis=1, keepdims=True)
    right = right - np.nanmean(right, axis=1, keepdims=True)
    numerator = np.nansum(left * right, axis=1)
    denominator = np.sqrt(np.nansum(left * left, axis=1) * np.nansum(right * right, axis=1))
    return np.divide(numerator, denominator, out=np.full(len(left), np.nan), where=denominator > 0)


def load_minute_paths(index: pd.MultiIndex) -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Load ten real opening bars for the requested (date, instrument) rows."""
    min_dates, min_slots = _load_min_calendar()
    day_starts = np.flatnonzero(min_slots == 0)
    start_by_date = {int(min_dates[row]): int(row) for row in day_starts}
    frame = index.to_frame(index=False)
    frame["ordinal"] = frame["datetime"].map(lambda value: pd.Timestamp(value).to_pydatetime().date().toordinal())
    n = len(frame)
    bar_ret = np.full((n, 10), np.nan)
    cum_ret = np.full((n, 10), np.nan)
    vol_share = np.full((n, 10), np.nan)
    ranges = np.full((n, 10), np.nan)
    vwap_gap = np.full((n, 10), np.nan)
    for code, positions in frame.groupby("instrument", sort=False).groups.items():
        start_index, fields = _read_1min_fields(str(code))
        if start_index is None:
            continue
        positions = np.asarray(list(positions), dtype=int)
        global_start = frame.loc[positions, "ordinal"].map(start_by_date).to_numpy(dtype=float)
        valid_day = np.isfinite(global_start)
        if not valid_day.any():
            continue
        selected = positions[valid_day]
        offsets = global_start[valid_day].astype(np.int64)[:, None] + np.arange(10) - int(start_index)
        valid = (offsets >= 0) & (offsets < len(fields["close"]))
        if not valid.all(axis=1).any():
            continue
        selected = selected[valid.all(axis=1)]
        offsets = offsets[valid.all(axis=1)]
        close = fields["close"][offsets].astype(float)
        opening = fields["open"][offsets].astype(float)
        high = fields["high"][offsets].astype(float)
        low = fields["low"][offsets].astype(float)
        volume = fields["volume"][offsets].astype(float)
        vwap = fields["vwap"][offsets].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            bar_ret[selected] = close / opening - 1
            cum_ret[selected] = close / opening[:, :1] - 1
            total_volume = np.nansum(volume, axis=1, keepdims=True)
            vol_share[selected] = np.divide(volume, total_volume, where=total_volume > 0)
            ranges[selected] = high / low - 1
            vwap_gap[selected] = close / vwap - 1
    return frame.set_index(["datetime", "instrument"]), bar_ret, cum_ret, vol_share, ranges, vwap_gap


def build_candidates(index: pd.MultiIndex) -> pd.DataFrame:
    """Build compact path-shape and member-basket resonance candidates."""
    frame, bar_ret, cum_ret, vol_share, ranges, vwap_gap = load_minute_paths(index)
    dates = frame.index.get_level_values("datetime")
    result = pd.DataFrame(index=frame.index)
    time = np.arange(10, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        result["minute_return_vol"] = np.nanstd(bar_ret, axis=1)
        result["minute_range_mean"] = np.nanmean(ranges, axis=1)
        result["minute_vwap_gap_mean"] = np.nanmean(vwap_gap, axis=1)
        result["minute_vwap_gap_last"] = vwap_gap[:, -1]
        result["minute_volume_back_share"] = np.nansum(vol_share[:, 5:], axis=1)
        result["minute_volume_concentration"] = np.nansum(vol_share * vol_share, axis=1)
        result["minute_path_slope"] = np.nansum((time - time.mean()) * cum_ret, axis=1)
        nav = 1 + cum_ret
        running_max = np.maximum.accumulate(nav, axis=1)
        result["minute_path_max_drawdown"] = np.nanmin(nav / running_max - 1, axis=1)

    # Construct the investable high-beta basket path from the same day's actual
    # 100 constituents. Median is robust to limit moves and bad single-stock bars.
    bar_df = pd.DataFrame(bar_ret, index=frame.index)
    cum_df = pd.DataFrame(cum_ret, index=frame.index)
    vol_df = pd.DataFrame(vol_share, index=frame.index)
    basket_bar = bar_df.groupby(level="datetime").transform("median").to_numpy()
    basket_cum = cum_df.groupby(level="datetime").transform("median").to_numpy()
    basket_vol = vol_df.groupby(level="datetime").transform("median").to_numpy()
    residual_bar = bar_ret - basket_bar
    result["basket_relative_total"] = cum_ret[:, -1] - basket_cum[:, -1]
    result["basket_relative_late5"] = (
        (1 + cum_ret[:, -1]) / (1 + cum_ret[:, 4])
        - (1 + basket_cum[:, -1]) / (1 + basket_cum[:, 4])
    )
    result["basket_resonance_corr"] = _safe_corr_rows(bar_ret, basket_bar)
    basket_centered = basket_bar - np.nanmean(basket_bar, axis=1, keepdims=True)
    stock_centered = bar_ret - np.nanmean(bar_ret, axis=1, keepdims=True)
    variance = np.nansum(basket_centered * basket_centered, axis=1)
    result["basket_beta_10m"] = np.divide(
        np.nansum(stock_centered * basket_centered, axis=1), variance,
        out=np.full(len(frame), np.nan), where=variance > 0,
    )
    result["basket_idio_vol"] = np.nanstd(residual_bar, axis=1)
    result["basket_direction_agreement"] = np.nanmean(
        np.sign(bar_ret) == np.sign(basket_bar), axis=1
    )
    result["basket_volume_profile_corr"] = _safe_corr_rows(vol_share, basket_vol)
    result["basket_resonant_strength"] = cum_ret[:, -1] * basket_cum[:, -1]
    result["basket_down_resilience"] = np.where(
        basket_cum[:, -1] < 0, cum_ret[:, -1] - basket_cum[:, -1], 0.0
    )
    return result.replace([np.inf, -np.inf], np.nan)


def _partial_daily_corr(feature: pd.Series, label: pd.Series,
                        champion: pd.Series, top_n: int | None = None) -> pd.Series:
    values = {}
    for date in feature.index.get_level_values("datetime").unique():
        x, y, p = feature.xs(date), label.xs(date), champion.xs(date)
        common = x.dropna().index.intersection(y.dropna().index).intersection(p.dropna().index)
        if top_n is not None:
            common = p.loc[common].nlargest(top_n).index
        if len(common) < 10 or x.loc[common].nunique() < 2:
            continue
        control = np.column_stack([np.ones(len(common)), p.loc[common].rank(pct=True).to_numpy()])
        rx = x.loc[common].rank(pct=True).to_numpy() - control @ np.linalg.lstsq(
            control, x.loc[common].rank(pct=True).to_numpy(), rcond=None
        )[0]
        ry = y.loc[common].rank(pct=True).to_numpy() - control @ np.linalg.lstsq(
            control, y.loc[common].rank(pct=True).to_numpy(), rcond=None
        )[0]
        if np.std(rx) > 0 and np.std(ry) > 0:
            values[str(pd.Timestamp(date).date())] = float(np.corrcoef(rx, ry)[0, 1])
    return pd.Series(values, dtype=float)


def main() -> None:
    import qlib
    from qlib.config import C
    from qlib.workflow import R

    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(ROOT / "mlruns")
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn", exp_manager=exp_manager)
    rows = {}
    for window, segments in WINDOWS.items():
        recorder_id = CHAMPION_RECORDERS[window]
        recorder = R.get_recorder(recorder_id=recorder_id, experiment_name="factor_challenger_ab")
        champion = _series(recorder.load_object("pred.pkl"))
        label = _series(recorder.load_object("label.pkl"))
        common = champion.dropna().index.intersection(label.dropna().index)
        champion, label = champion.loc[common], label.loc[common]
        candidates = build_candidates(common).reindex(common)
        rows[window] = {}
        for name in candidates:
            factor = candidates[name]
            coverage = float(factor.notna().mean())
            partial = _partial_daily_corr(factor, label, champion)
            top20 = _partial_daily_corr(factor, label, champion, top_n=20)
            raw_rank = _daily_corr(factor, label, "spearman")
            rows[window][name] = {
                "coverage": coverage,
                "raw_rank_ic": float(raw_rank.mean()),
                "partial_rank_ic": float(partial.mean()),
                "partial_rank_icir": float(partial.mean() / partial.std()) if partial.std() else None,
                "top20_partial_rank_ic": float(top20.mean()),
                "top20_partial_rank_icir": float(top20.mean() / top20.std()) if top20.std() else None,
            }
        RESULT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        print(f"{window}: {len(common)} rows, {len(candidates.columns)} candidates", flush=True)

    print("\nmean partial IC across four disjoint windows")
    for name in next(iter(rows.values())):
        values = [rows[w][name]["partial_rank_ic"] for w in rows]
        top = [rows[w][name]["top20_partial_rank_ic"] for w in rows]
        signs = sum(value > 0 for value in values)
        print(f"{name:<32} all={np.mean(values):+.4f} ({signs}/4+) "
              f"top20={np.mean(top):+.4f} ({sum(value > 0 for value in top)}/4+)")
    print(f"Saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
