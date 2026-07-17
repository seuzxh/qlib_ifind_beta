"""Validation-gated risk overlays for 883926 index enhancement.

These helpers never change the alpha model.  They turn information known by
09:40 on date T into a soft penalty applied only inside the model's Top20 pool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_percentile(values: pd.Series) -> pd.Series:
    if "datetime" not in values.index.names:
        return values.rank(method="average", pct=True)
    return values.groupby(level="datetime").rank(method="average", pct=True)


def compute_chase_risk(features: pd.DataFrame) -> pd.Series:
    """Joint risk of an established uptrend accelerating again by T 09:40.

    Required columns are all decision-time-safe:
    ``mom_5d`` and ``positive_days_5`` end at T-1; ``overnight_gap``,
    ``startup_total`` and ``accel_5m`` use information no later than 09:40 T.
    """
    required = {
        "mom_5d", "positive_days_5", "overnight_gap",
        "startup_total", "accel_5m",
    }
    missing = required.difference(features.columns)
    if missing:
        raise KeyError(f"missing chase-risk columns: {sorted(missing)}")
    extension = (1.0 + features["overnight_gap"]) * (
        1.0 + features["startup_total"]
    ) - 1.0
    components = pd.DataFrame({
        "momentum": _daily_percentile(features["mom_5d"]),
        "persistence": _daily_percentile(features["positive_days_5"]),
        "extension": _daily_percentile(extension.clip(lower=0)),
        "acceleration": _daily_percentile(features["accel_5m"].clip(lower=0)),
    }, index=features.index)
    risk = components.prod(axis=1, min_count=4).pow(0.25)
    active = (
        (features["mom_5d"] > 0)
        & (features["positive_days_5"] >= 0.6)
        & (extension > 0)
        & (features["accel_5m"] > 0)
    )
    return risk.where(active, 0.0).rename("chase_risk")


def compute_tracking_risk(features: pd.DataFrame) -> pd.Series:
    """Cross-sectional risk of not tracking the contemporaneous member basket."""
    required = {"basket_beta_10m", "basket_resonance_corr", "basket_idio_vol"}
    missing = required.difference(features.columns)
    if missing:
        raise KeyError(f"missing basket-risk columns: {sorted(missing)}")
    parts = pd.DataFrame({
        "beta_deviation": _daily_percentile((features["basket_beta_10m"] - 1).abs()),
        "low_resonance": _daily_percentile(-features["basket_resonance_corr"]),
        "idio_volatility": _daily_percentile(features["basket_idio_vol"]),
    }, index=features.index)
    return parts.mean(axis=1, skipna=False).rename("tracking_risk")


def apply_top_pool_penalty(
    base_score: pd.Series,
    risk: pd.Series,
    threshold: float,
    penalty_weight: float,
    top_pool: int = 20,
) -> pd.Series:
    """Softly demote high-risk names while keeping selection inside Top20."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0,1)")
    if not 0 <= penalty_weight <= 1:
        raise ValueError("penalty_weight must be in [0,1]")
    if top_pool < 1:
        raise ValueError("top_pool must be positive")
    if penalty_weight == 0:
        return base_score.copy()

    common = base_score.index.intersection(risk.index)
    base = base_score.loc[common].dropna()
    aligned_risk = risk.reindex(base.index).fillna(0.0)
    descending = base.groupby(level="datetime").rank(
        method="first", ascending=False
    ) if "datetime" in base.index.names else base.rank(method="first", ascending=False)
    inside = descending <= top_pool
    excess = ((aligned_risk - threshold) / (1.0 - threshold)).clip(0, 1)
    if "datetime" in base.index.names:
        scale = base.groupby(level="datetime").transform("std").fillna(0.0)
        outside_max = base.where(~inside).groupby(level="datetime").transform("max")
    else:
        scale = pd.Series(base.std(), index=base.index).fillna(0.0)
        outside_max = pd.Series(base.where(~inside).max(), index=base.index)
    adjusted = base.copy()
    adjusted.loc[inside] -= penalty_weight * excess.loc[inside] * scale.loc[inside]
    # A penalized rank-20 must not be replaced by rank-21. Clip only when the
    # soft penalty would leave the alpha-approved candidate pool.
    floor = outside_max + scale * 1e-9
    adjusted.loc[inside] = np.maximum(adjusted.loc[inside], floor.loc[inside])
    return adjusted.rename(base_score.name)
