from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from air_prediction.config import Config
from air_prediction.features import build_features, common_evaluation_mask
from air_prediction.metrics import regression_metrics
from air_prediction.models import make_model
from air_prediction.splits import Fold, fold_masks, make_folds


@pytest.mark.parametrize("horizon", [1, 6, 24, 25, 49])
def test_targets_and_seasonal_baseline_are_aligned(hourly_frame, horizon):
    cfg = replace(Config(), horizon_hours=horizon)
    data = build_features(hourly_frame, cfg)
    t = 400
    seasonal_source = t + horizon - 24 * int(np.ceil(horizon / 24))
    assert data.y.iloc[t] == t + horizon
    assert data.baselines.persistence.iloc[t] == t
    assert data.baselines.seasonal_naive.iloc[t] == seasonal_source
    assert seasonal_source <= t
    assert data.X.pm25_lag_24.iloc[t] == t - 24
    assert data.target_time.iloc[t] == data.X.index[t] + pd.Timedelta(hours=horizon)


def test_rolling_window_ends_at_origin(hourly_frame):
    data = build_features(hourly_frame, Config())
    assert data.X.pm25_mean_3.iloc[400] == np.mean([398, 399, 400])


def test_future_values_cannot_change_existing_features(hourly_frame):
    original = build_features(hourly_frame, Config())
    changed = hourly_frame.copy()
    changed.loc[changed.index[501]:, ["pm2.5", "TEMP", "DEWP", "PRES", "Iws", "Is", "Ir"]] = 999999
    changed.loc[changed.index[501]:, "cbwd"] = "FUTURE_ONLY_CATEGORY"
    new = build_features(changed, Config())
    pd.testing.assert_frame_equal(original.X.iloc[:501], new.X.iloc[:501])
    pd.testing.assert_frame_equal(original.baselines.iloc[:501], new.baselines.iloc[:501])


def test_missing_measurements_are_not_filled_from_the_future(hourly_frame):
    hourly_frame.loc[hourly_frame.index[400], "pm2.5"] = np.nan
    data = build_features(hourly_frame, Config())
    assert np.isnan(data.X.pm25_lag_0.iloc[400])
    assert data.X.pm25_lag_0_missing.iloc[400] == 1
    assert not common_evaluation_mask(data).iloc[400]
    assert not common_evaluation_mask(data).iloc[399]  # missing target
    assert data.X.pm25_observed_fraction_3.iloc[400] == pytest.approx(2 / 3)


def test_irregular_grid_is_rejected_before_lags(hourly_frame):
    with pytest.raises(ValueError, match="regular hourly"):
        build_features(hourly_frame.drop(hourly_frame.index[400]), Config())


@pytest.mark.parametrize("horizon", [1, 6, 24])
def test_no_training_label_extends_beyond_first_forecast_origin(hourly_frame, horizon):
    cfg = replace(Config(), horizon_hours=horizon, minimum_training_rows=10)
    data = build_features(hourly_frame, cfg)
    fold = Fold("validation", 1, hourly_frame.index[600], hourly_frame.index[800])
    train, test, audit = fold_masks(data, fold, cfg)
    first_origin = fold.start - pd.Timedelta(hours=horizon)
    assert data.target_time[train].max() <= first_origin
    assert data.X.index[test].min() == first_origin
    assert not (train & test).any()
    assert audit["common_evaluation_rows"] == 200
    assert audit["coverage_fraction"] == 1.0


def test_calendar_folds_are_nonoverlapping_and_cover_each_phase():
    cfg = Config()
    validation, held = make_folds(cfg, "validation"), make_folds(cfg, "holdout")
    assert len(validation) == len(held) == 4
    assert validation[-1].end == held[0].start == pd.Timestamp("2014-01-01")
    assert all(a.end == b.start for a, b in zip(held, held[1:]))
    assert held[-1].end == pd.Timestamp("2015-01-01")


def test_fold_local_imputation_scaling_and_unseen_categories(hourly_frame):
    data = build_features(hourly_frame, Config())
    X = data.X.iloc[200:500].copy()
    X.loc[X.index[:10], "weather_TEMP_lag_0"] = np.nan
    model = make_model("ridge", X, Config())
    model.fit(X, data.y.loc[X.index])
    statistics = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_.copy()
    test = data.X.iloc[[600]].copy()
    test["wind_direction"] = "UNSEEN"
    test["weather_TEMP_lag_0"] = 999999
    prediction = model.predict(test)
    assert np.isfinite(prediction).all()
    np.testing.assert_equal(statistics, model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_)


def test_metrics_use_rmse_and_permit_negative_r_squared():
    values = regression_metrics([1, 2, 3], [2, 2, 5])
    assert values["mae"] == 1
    assert values["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert values["r2"] == -1.5
    assert regression_metrics([2, 2], [2, 3])["r2"] is None
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([1, np.nan], [1, 2])


@pytest.mark.parametrize("change", [
    {"horizon_hours": 0}, {"lags": (-1,)}, {"rolling_windows": (1,)},
    {"holdout_start": "2012-01-01"}, {"validation_start": "2013-01-01 00:30"},
    {"validation_start": "2013-01-01T00:00:00+00:00"},
    {"lags": (True,)}, {"rolling_windows": (True,)},
])
def test_invalid_config_is_rejected(change):
    with pytest.raises(ValueError):
        replace(Config(), **change)
