"""Features are indexed by forecast ORIGIN t; the label is observed PM2.5 at t+h."""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .config import Config
from .data import WEATHER_NUMERIC


@dataclass
class Supervised:
    X: pd.DataFrame
    y: pd.Series
    target_time: pd.Series
    baselines: pd.DataFrame
    warm: pd.Series


def build_features(frame, config: Config, feature_set="full"):
    if feature_set not in {"lags_only", "temporal", "full"}:
        raise ValueError("feature_set must be lags_only, temporal, or full")
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing or not frame.index.equals(expected):
        raise ValueError("Features require a sorted, unique, regular hourly index")
    pm = frame["pm2.5"]
    values = {}
    for lag in config.lags:
        values[f"pm25_lag_{lag}"] = pm.shift(lag)
        values[f"pm25_lag_{lag}_missing"] = pm.shift(lag).isna().astype(float)
    if feature_set in {"temporal", "full"}:
        for window in config.rolling_windows:
            rolling = pm.rolling(window, min_periods=1)
            values[f"pm25_mean_{window}"] = rolling.mean()
            values[f"pm25_std_{window}"] = rolling.std(ddof=0)
            values[f"pm25_observed_fraction_{window}"] = rolling.count() / window
        target_clock = frame.index + pd.to_timedelta(config.horizon_hours, unit="h")
        for label, number, period in (
            ("hour", target_clock.hour, 24),
            ("weekday", target_clock.dayofweek, 7),
            ("month", target_clock.month - 1, 12),
        ):
            values[f"target_{label}_sin"] = np.sin(2 * np.pi * number / period)
            values[f"target_{label}_cos"] = np.cos(2 * np.pi * number / period)
        values["target_weekend"] = (target_clock.dayofweek >= 5).astype(float)
    if feature_set == "full":
        for name in WEATHER_NUMERIC:
            for lag in (0, 1, 24):
                values[f"weather_{name}_lag_{lag}"] = frame[name].shift(lag)
            values[f"weather_{name}_missing"] = frame[name].isna().astype(float)
        values["wind_direction"] = frame["cbwd"].fillna("MISSING").astype(str)
    X = pd.DataFrame(values, index=frame.index)
    # For any h, repeat the latest observed seasonal cycle; never address future data.
    seasonal_shift = math.ceil(config.horizon_hours / config.seasonal_period_hours) * config.seasonal_period_hours - config.horizon_hours
    baselines = pd.DataFrame({"persistence": pm, "seasonal_naive": pm.shift(seasonal_shift)}, index=frame.index)
    target_time = pd.Series(frame.index + pd.to_timedelta(config.horizon_hours, unit="h"), index=frame.index, name="target_time")
    warm = pd.Series(frame.index >= frame.index.min() + pd.to_timedelta(config.warmup_hours, unit="h"), index=frame.index)
    return Supervised(X, pm.shift(-config.horizon_hours).rename("actual"), target_time, baselines, warm)


def common_evaluation_mask(data: Supervised):
    """Compare every model on identical origins, with real targets and both baselines."""
    return data.warm & data.y.notna() & data.baselines.notna().all(axis=1)
