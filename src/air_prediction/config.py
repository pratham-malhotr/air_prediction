"""Immutable experiment settings; no hidden random splitting."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib

import pandas as pd


@dataclass(frozen=True)
class Config:
    horizon_hours: int = 1
    seasonal_period_hours: int = 24
    lags: tuple[int, ...] = (0, 1, 2, 3, 6, 12, 24, 48, 72, 168)
    rolling_windows: tuple[int, ...] = (3, 6, 12, 24, 72, 168)
    train_start: str = "2010-01-01"
    validation_start: str = "2013-01-01"
    holdout_start: str = "2014-01-01"
    holdout_end: str = "2015-01-01"
    fold_months: int = 3
    minimum_training_rows: int = 1000
    random_seed: int = 42
    threads: int = 2
    xgboost_candidates: tuple[dict, ...] = field(default_factory=lambda: (
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
    ))

    def __post_init__(self):
        for name in ("horizon_hours", "seasonal_period_hours", "fold_months", "minimum_training_rows", "threads"):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.lags or any(not isinstance(x, int) or x < 0 for x in self.lags):
            raise ValueError("lags must contain nonnegative integer hour offsets")
        if len(set(self.lags)) != len(self.lags):
            raise ValueError("lags must be unique")
        if not self.rolling_windows or any(not isinstance(x, int) or x < 2 for x in self.rolling_windows):
            raise ValueError("rolling_windows must contain integers >= 2")
        if len(set(self.rolling_windows)) != len(self.rolling_windows):
            raise ValueError("rolling_windows must be unique")
        dates = [pd.Timestamp(getattr(self, name)) for name in ("train_start", "validation_start", "holdout_start", "holdout_end")]
        if not all(a < b for a, b in zip(dates, dates[1:])):
            raise ValueError("Require train_start < validation_start < holdout_start < holdout_end")
        if any(d.tzinfo is not None for d in dates):
            raise ValueError("Use naive Beijing local time in configuration dates")
        if not self.xgboost_candidates:
            raise ValueError("At least one XGBoost candidate is required")

    @property
    def warmup_hours(self):
        return max(max(self.lags), max(self.rolling_windows) - 1, self.seasonal_period_hours)

    def to_dict(self):
        return asdict(self)


def load_config(path):
    with Path(path).open("rb") as stream:
        values = tomllib.load(stream)
    for key in ("lags", "rolling_windows", "xgboost_candidates"):
        if key in values:
            values[key] = tuple(values[key])
    return Config(**values)
