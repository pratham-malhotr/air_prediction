"""Calendar backtests with a training-label availability boundary."""

from dataclasses import dataclass
import pandas as pd

from .config import Config
from .features import common_evaluation_mask


@dataclass(frozen=True)
class Fold:
    phase: str
    number: int
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def name(self):
        return f"{self.phase}_{self.number:02d}"


def make_folds(config: Config, phase):
    if phase == "validation":
        start, end = pd.Timestamp(config.validation_start), pd.Timestamp(config.holdout_start)
    elif phase == "holdout":
        start, end = pd.Timestamp(config.holdout_start), pd.Timestamp(config.holdout_end)
    else:
        raise ValueError("phase must be validation or holdout")
    folds = []
    while start < end:
        stop = min(start + pd.DateOffset(months=config.fold_months), end)
        folds.append(Fold(phase, len(folds) + 1, start, stop))
        start = stop
    return folds


def fold_masks(data, fold: Fold, config: Config):
    # Fit once, immediately before the first forecast origin in this block.
    first_origin = fold.start - pd.to_timedelta(config.horizon_hours, unit="h")
    train = (data.target_time <= first_origin) & (data.X.index >= pd.Timestamp(config.train_start)) & data.warm & data.y.notna()
    period = (data.target_time >= fold.start) & (data.target_time < fold.end)
    test = period & common_evaluation_mask(data)
    if int(train.sum()) < config.minimum_training_rows:
        raise ValueError(f"{fold.name}: only {int(train.sum())} eligible training rows")
    if int(test.sum()) < 2:
        raise ValueError(f"{fold.name}: fewer than two common evaluation rows; inspect dates and missingness")
    # A full hourly input grid includes origins whose future labels do not exist at the end.
    expected = len(pd.date_range(fold.start, fold.end, freq="h", inclusive="left"))
    audit = {
        "fold": fold.name, "phase": fold.phase, "target_start": str(fold.start), "target_end_exclusive": str(fold.end),
        "fit_time": str(first_origin), "latest_training_target": str(data.target_time[train].max()),
        "training_rows": int(train.sum()), "expected_target_hours": expected,
        "origins_in_input": int(period.sum()), "common_evaluation_rows": int(test.sum()),
        "coverage_fraction": int(test.sum()) / expected,
        "missing_targets": int((period & data.y.isna()).sum()),
        "missing_persistence": int((period & data.baselines["persistence"].isna()).sum()),
        "missing_seasonal": int((period & data.baselines["seasonal_naive"].isna()).sum()),
        "note": "Missingness counts can overlap; coverage denominator includes all expected target hours.",
    }
    return train, test, audit
