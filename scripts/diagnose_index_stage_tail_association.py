"""Relate 883926 index stages to high-profit and high-loss constituents.

This is a discovery diagnostic, not a production signal.  Every online stage
and explanatory feature is available by 09:40 on T.  Outcomes use the frozen
champion label (T 09:41 -> T+1 close), and the analysis is restricted to the
361-day purged OOS score panel so champion-control statistics are honest.
"""
from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qlib")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from qlib_ifind_beta.config import UNIVERSE_MARKET
from scripts.diagnose_minute_basket_resonance import build_candidates, load_minute_paths
from scripts.validate_risk_overlay_purged import _test_score

RESULT_PATH = ROOT / "data" / "index_stage_tail_association.json"
READ_ROOT = Path(os.environ.get("QLIB_IFIND_RESEARCH_SOURCE", ROOT)).resolve()
SOURCE_PATH = READ_ROOT / "data" / "xgb_purged_rolling_gate_ab.json"
PROVIDER_ROOT = READ_ROOT / "data" / "qlib_root"
MLRUNS_ROOT = READ_ROOT / "mlruns"
MIN_STAGE_DAYS = 8
MIN_TAIL_ROWS = 20
BOOTSTRAP_DRAWS = 1000


def _datetime_instrument(obj: pd.Series | pd.DataFrame):
    if obj.index.names == ["instrument", "datetime"]:
        return obj.reorder_levels(["datetime", "instrument"]).sort_index()
    return obj.sort_index()


def _rolling_quantile(values: pd.Series, q: float, window: int = 60) -> pd.Series:
    """Past-only threshold: current T value never enters its own threshold."""
    return values.rolling(window, min_periods=20).quantile(q).shift(1)


def classify_online_stages(daily: pd.DataFrame) -> pd.Series:
    """Classify stages using only columns observable by T 09:40."""
    needed = {
        "prior_return_3", "prior_return_5", "prior_down_days_3",
        "prior_up_days_3", "morning_total", "morning_late5",
        "morning_acceleration", "morning_breadth",
    }
    missing = needed.difference(daily.columns)
    if missing:
        raise KeyError(f"missing stage columns: {sorted(missing)}")

    breadth_mid = _rolling_quantile(daily["morning_breadth"], .50)
    breadth_hot = _rolling_quantile(daily["morning_breadth"], .80)
    total_hot = _rolling_quantile(daily["morning_total"], .80)
    prior_hot = _rolling_quantile(daily["prior_return_5"], .70)

    pullback = (daily["prior_return_3"] < 0) & (daily["prior_down_days_3"] >= 2)
    prior_up = (daily["prior_return_3"] > 0) & (daily["prior_up_days_3"] >= 2)
    rebound = (
        pullback & (daily["morning_total"] > 0) & (daily["morning_late5"] > 0)
        & (daily["morning_acceleration"] > 0)
        & (daily["morning_breadth"] >= breadth_mid)
    )
    repair = pullback & (daily["morning_total"] > 0) & ~rebound
    continuation = pullback & ~repair & ~rebound
    boiling = (
        prior_up & (daily["prior_return_5"] >= prior_hot)
        & (daily["morning_total"] >= total_hot)
        & (daily["morning_breadth"] >= breadth_hot)
    )
    retreat = prior_up & ~boiling & (
        (daily["morning_total"] < 0) | (daily["morning_late5"] < 0)
    )
    uptrend = prior_up & ~boiling & ~retreat

    stage = pd.Series("neutral", index=daily.index, dtype="object", name="online_stage")
    stage.loc[continuation] = "pullback_continuation"
    stage.loc[repair] = "pullback_repair"
    stage.loc[rebound] = "rebound_start"
    stage.loc[uptrend] = "uptrend"
    stage.loc[boiling] = "boiling"
    stage.loc[retreat] = "retreat"
    return stage


def _load_oos_panel() -> tuple[pd.Series, pd.Series]:
    source = json.loads(SOURCE_PATH.read_text())
    scores, labels = [], []
    for row in source["segments"]:
        score, label = _test_score(row)
        scores.append(score)
        labels.append(label)
    score = pd.concat(scores).sort_index()
    score = score[~score.index.duplicated(keep="last")]
    label = pd.concat(labels).sort_index().reindex(score.index)
    common = score.dropna().index.intersection(label.dropna().index)
    return score.loc[common], label.loc[common]


def _basket_daily(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from qlib.data import D

    raw = D.features(
        D.instruments(UNIVERSE_MARKET), ["$change"],
        start_time=start, end_time=end, freq="day", disk_cache=0,
    ).iloc[:, 0]
    by_date = raw.groupby(level="datetime")
    return pd.DataFrame({
        "basket_return": by_date.median(),
        "basket_breadth": by_date.apply(lambda x: float((x > 0).mean())),
        "basket_dispersion": by_date.apply(lambda x: float(x.quantile(.75) - x.quantile(.25))),
    }).sort_index()


def _morning_state(index: pd.MultiIndex) -> pd.DataFrame:
    frame, bar_ret, cum_ret, _, _, _ = load_minute_paths(index)
    dates = frame.index.get_level_values("datetime")
    total = pd.Series(cum_ret[:, -1], index=frame.index)
    early = pd.Series(cum_ret[:, 4], index=frame.index)
    late = (1.0 + total) / (1.0 + early) - 1.0
    rows = pd.DataFrame({"total": total, "late": late})
    grouped = rows.groupby(level="datetime")
    out = pd.DataFrame(index=pd.Index(sorted(pd.unique(dates)), name="datetime"))
    out["morning_total"] = grouped["total"].median()
    out["morning_early5"] = early.groupby(level="datetime").median()
    out["morning_late5"] = grouped["late"].median()
    out["morning_acceleration"] = out["morning_late5"] - out["morning_early5"]
    out["morning_breadth"] = grouped["total"].apply(lambda x: float((x > 0).mean()))
    out["morning_dispersion"] = grouped["total"].apply(
        lambda x: float(x.quantile(.75) - x.quantile(.25))
    )
    return out


def _stage_frame(index: pd.MultiIndex) -> pd.DataFrame:
    dates = pd.Index(sorted(index.get_level_values("datetime").unique()), name="datetime")
    basket = _basket_daily(dates.min() - pd.Timedelta(days=20), dates.max())
    morning = _morning_state(index).reindex(dates)
    out = morning.copy()
    shifted = [basket["basket_return"].shift(k).reindex(dates) for k in range(1, 6)]
    lag = pd.concat(shifted, axis=1)
    out["prior_return_3"] = (1.0 + lag.iloc[:, :3]).prod(axis=1) - 1.0
    out["prior_return_5"] = (1.0 + lag).prod(axis=1) - 1.0
    out["prior_down_days_3"] = (lag.iloc[:, :3] < 0).sum(axis=1)
    out["prior_up_days_3"] = (lag.iloc[:, :3] > 0).sum(axis=1)
    out["prior_breadth_1"] = basket["basket_breadth"].shift(1).reindex(dates)
    out["prior_dispersion_1"] = basket["basket_dispersion"].shift(1).reindex(dates)
    out["online_stage"] = classify_online_stages(out)
    return out


def _stock_features(index: pd.MultiIndex, stages: pd.DataFrame,
                    score: pd.Series) -> pd.DataFrame:
    from qlib.data import D

    dates = index.get_level_values("datetime")
    expressions = [f"Ref($change,{k})" for k in range(1, 6)] + [
        "$overnight_gap", "$startup_total", "$startup_mom_5m", "$accel_5m",
    ]
    raw = D.features(
        D.instruments(UNIVERSE_MARKET), expressions,
        start_time=dates.min(), end_time=dates.max(), freq="day", disk_cache=0,
    )
    raw = _datetime_instrument(raw).reindex(index)
    raw.columns = [f"stock_return_lag{k}" for k in range(1, 6)] + [
        "overnight_gap", "startup_total", "startup_mom_5m", "accel_5m",
    ]
    result = raw.copy()
    stage_by_row = stages.reindex(pd.Index(dates)).copy()
    stage_by_row.index = index
    basket_daily = _basket_daily(dates.min() - pd.Timedelta(days=20), dates.max())
    basket_lags = pd.DataFrame({
        f"basket_lag{k}": basket_daily["basket_return"].shift(k).reindex(dates).to_numpy()
        for k in range(1, 6)
    }, index=index)
    stock_lags = result[[f"stock_return_lag{k}" for k in range(1, 6)]]
    result["relative_return_1"] = stock_lags.iloc[:, 0] - basket_lags.iloc[:, 0]
    result["relative_return_3"] = (
        (1.0 + stock_lags.iloc[:, :3]).prod(axis=1)
        - (1.0 + basket_lags.iloc[:, :3]).prod(axis=1)
    )
    result["relative_return_5"] = (
        (1.0 + stock_lags).prod(axis=1) - (1.0 + basket_lags).prod(axis=1)
    )
    residual = stock_lags.to_numpy() - basket_lags.to_numpy()
    down = basket_lags.to_numpy() < 0
    down_count = down.sum(axis=1)
    result["downside_resilience_5"] = np.divide(
        np.where(down, residual, 0.0).sum(axis=1), down_count,
        out=np.full(len(index), np.nan), where=down_count > 0,
    )
    result["positive_days_5"] = (stock_lags > 0).mean(axis=1)
    result["prior_momentum_5"] = (1.0 + stock_lags).prod(axis=1) - 1.0
    minute = build_candidates(index).reindex(index)
    keep = [
        "minute_path_slope", "minute_path_max_drawdown",
        "minute_volume_back_share", "basket_relative_total",
        "basket_relative_late5", "basket_resonance_corr", "basket_beta_10m",
        "basket_idio_vol", "basket_direction_agreement",
        "basket_down_resilience",
    ]
    result = result.join(minute[keep])
    result["champion_rank"] = score.groupby(level="datetime").rank(pct=True)
    for name in ["online_stage", "morning_total", "morning_late5",
                 "morning_breadth", "prior_return_3", "prior_return_5"]:
        result[name] = stage_by_row[name]
    return result.replace([np.inf, -np.inf], np.nan)


def _daily_partial_ic(feature: pd.Series, outcome: pd.Series,
                      control: pd.Series) -> pd.Series:
    values = {}
    dates = feature.index.get_level_values("datetime").unique()
    for date in dates:
        frame = pd.concat([
            feature.xs(date).rename("x"), outcome.xs(date).rename("y"),
            control.xs(date).rename("c"),
        ], axis=1).dropna()
        if len(frame) < 10 or frame["x"].nunique() < 3:
            continue
        ranked = frame.rank(pct=True)
        design = np.column_stack([np.ones(len(ranked)), ranked["c"]])
        rx = ranked["x"] - design @ np.linalg.lstsq(design, ranked["x"], rcond=None)[0]
        ry = ranked["y"] - design @ np.linalg.lstsq(design, ranked["y"], rcond=None)[0]
        if rx.std() > 0 and ry.std() > 0:
            values[pd.Timestamp(date)] = float(rx.corr(ry))
    return pd.Series(values, dtype=float)


def _moving_block_mean_ci(values: pd.Series, seed_key: str,
                          block: int = 5) -> list[float] | None:
    """95% moving-block bootstrap CI over event days."""
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 8:
        return None
    block = min(block, len(clean))
    starts = np.arange(len(clean))
    offsets = np.arange(block)
    rng = np.random.default_rng(zlib.crc32(seed_key.encode("utf-8")))
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    blocks_needed = int(np.ceil(len(clean) / block))
    for draw in range(BOOTSTRAP_DRAWS):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = clean[(chosen[:, None] + offsets) % len(clean)].ravel()[:len(clean)]
        draws[draw] = sample.mean()
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def _association(frame: pd.DataFrame, features: list[str]) -> dict:
    output: dict[str, dict] = {}
    for stage, part in frame.groupby("online_stage"):
        dates = part.index.get_level_values("datetime").nunique()
        if dates < MIN_STAGE_DAYS:
            continue
        daily_rank = part.groupby(level="datetime")["active_return"].rank(pct=True)
        win = daily_rank >= .80
        loss = daily_rank <= .20
        if int(win.sum()) < MIN_TAIL_ROWS or int(loss.sum()) < MIN_TAIL_ROWS:
            continue
        stage_rows = {
            "days": int(dates), "rows": int(len(part)),
            "winner_rows": int(win.sum()), "loser_rows": int(loss.sum()),
            "features": {},
        }
        for name in features:
            values = part[name]
            valid = values.notna()
            if valid.sum() < 50:
                continue
            q20, q80 = values[valid].quantile([.20, .80])
            high, low = values >= q80, values <= q20
            base_win, base_loss = win.mean(), loss.mean()
            pooled_std = values[win | loss].std()
            effect = (
                (values[win].median() - values[loss].median()) / pooled_std
                if pd.notna(pooled_std) and pooled_std > 0 else np.nan
            )
            partial = _daily_partial_ic(values, part["active_return"], part["champion_score"])
            daily_spreads = {}
            for date in part.index.get_level_values("datetime").unique():
                day_values = values.xs(date)
                day_win = win.xs(date)
                day_loss = loss.xs(date)
                scale = day_values.std()
                if pd.notna(scale) and scale > 0 and day_win.any() and day_loss.any():
                    daily_spreads[pd.Timestamp(date)] = float(
                        (day_values[day_win].median() - day_values[day_loss].median()) / scale
                    )
            daily_spreads = pd.Series(daily_spreads, dtype=float)
            quarterly = {}
            quarters = part.index.get_level_values("datetime").to_period("Q")
            for quarter in pd.unique(quarters):
                mask = quarters == quarter
                subset = part.loc[mask]
                if len(subset) >= 50:
                    value = subset[name].corr(subset["active_return"], method="spearman")
                    if pd.notna(value):
                        quarterly[str(quarter)] = float(value)
            stage_rows["features"][name] = {
                "winner_median": float(values[win].median()),
                "loser_median": float(values[loss].median()),
                "tail_effect_size": float(effect) if pd.notna(effect) else None,
                "winner_lift_high": float(win[high].mean() / base_win) if high.any() else None,
                "winner_lift_low": float(win[low].mean() / base_win) if low.any() else None,
                "loser_lift_high": float(loss[high].mean() / base_loss) if high.any() else None,
                "loser_lift_low": float(loss[low].mean() / base_loss) if low.any() else None,
                "partial_rank_ic_mean": float(partial.mean()) if len(partial) else None,
                "partial_rank_ic_days": int(len(partial)),
                "partial_rank_ic_ci95": _moving_block_mean_ci(
                    partial, f"{stage}:{name}:partial"
                ),
                "daily_tail_spread_mean": float(daily_spreads.mean()) if len(daily_spreads) else None,
                "daily_tail_spread_ci95": _moving_block_mean_ci(
                    daily_spreads, f"{stage}:{name}:spread"
                ),
                "quarterly_rank_ic": quarterly,
                "quarter_sign_consistency": int(abs(sum(np.sign(list(quarterly.values())))))
                if quarterly else 0,
            }
        output[str(stage)] = stage_rows
    return output


def _candidate_summary(associations: dict) -> list[dict]:
    rows = []
    for scope, stages in associations.items():
        for stage, stage_result in stages.items():
            for feature, result in stage_result["features"].items():
                effect = result["tail_effect_size"]
                partial = result["partial_rank_ic_mean"]
                quarters = result["quarterly_rank_ic"]
                if effect is None or partial is None or len(quarters) < 3:
                    continue
                signs = np.sign(list(quarters.values()))
                consistency = max((signs > 0).mean(), (signs < 0).mean())
                rows.append({
                    "scope": scope, "stage": stage, "feature": feature,
                    "tail_effect_size": effect, "partial_rank_ic": partial,
                    "quarter_consistency": float(consistency),
                    "winner_best_lift": max(result["winner_lift_high"], result["winner_lift_low"]),
                    "loser_best_lift": max(result["loser_lift_high"], result["loser_lift_low"]),
                    "partial_rank_ic_ci95": result["partial_rank_ic_ci95"],
                    "daily_tail_spread_ci95": result["daily_tail_spread_ci95"],
                    "stable_ci": bool(
                        result["partial_rank_ic_ci95"]
                        and np.prod(result["partial_rank_ic_ci95"]) > 0
                    ),
                    "discovery_score": float(abs(effect) * abs(partial) * consistency),
                })
    return sorted(rows, key=lambda row: row["discovery_score"], reverse=True)


def main() -> None:
    import qlib
    from qlib.config import C

    C["exp_manager"]["kwargs"]["uri"] = "file:" + str(MLRUNS_ROOT)
    qlib.init(provider_uri=str(PROVIDER_ROOT), region="cn", exp_manager=C["exp_manager"])
    score, label = _load_oos_panel()
    print(f"loaded purged OOS panel: {len(score)} rows", flush=True)
    stages = _stage_frame(score.index)
    features = _stock_features(score.index, stages, score)
    panel = features.copy()
    panel["label"] = label
    panel["active_return"] = label - label.groupby(level="datetime").transform("median")
    panel["champion_score"] = score
    feature_names = [
        name for name in features.columns
        if name not in {"online_stage", "champion_rank", "morning_total",
                        "morning_late5", "morning_breadth", "prior_return_3",
                        "prior_return_5"}
    ]
    rank = score.groupby(level="datetime").rank(method="first", ascending=False)
    associations = {
        "full_pool": _association(panel, feature_names),
        "champion_top20": _association(panel.loc[rank <= 20], feature_names),
    }
    stage_counts = stages["online_stage"].value_counts().to_dict()
    output = {
        "period": [str(score.index.get_level_values("datetime").min().date()),
                   str(score.index.get_level_values("datetime").max().date())],
        "rows": int(len(score)), "days": int(len(stages)),
        "stage_counts": {str(k): int(v) for k, v in stage_counts.items()},
        "tail_definition": "daily within-scope top/bottom 20% of active label return",
        "associations": associations,
        "candidate_summary": _candidate_summary(associations),
    }
    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"stage_counts": output["stage_counts"],
                      "top_candidates": output["candidate_summary"][:15]},
                     ensure_ascii=False, indent=2))
    print(f"Saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
