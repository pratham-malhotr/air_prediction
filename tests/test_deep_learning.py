"""Unit and integration tests for the Deep Learning & Hybrid architectures."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="PyTorch is required for deep learning tests")
import torch

from air_prediction.deep_learning import (
    DLConfig,
    DeepLearningPipeline,
    HybridResidualModel,
    SequenceAirDataset,
    TemporalForecaster,
)
from air_prediction.models import make_model


@pytest.fixture
def synthetic_series():
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    # Synthetic smooth signal + random weather features
    t = np.linspace(0, 10, n)
    pm = 50 + 20 * np.sin(t) + np.random.normal(0, 5, n)
    pm = np.clip(pm, 0, None)
    
    df = pd.DataFrame({
        "pm25_lag_0": pm,
        "pm25_lag_1": np.roll(pm, 1),
        "weather_TEMP_lag_0": 15 + 5 * np.cos(t),
        "weather_DEWP_lag_0": 5 + 3 * np.sin(t),
        "target_hour_sin": np.sin(2 * np.pi * dates.hour / 24),
        "target_hour_cos": np.cos(2 * np.pi * dates.hour / 24),
        "wind_direction": ["NW" if i % 2 == 0 else "SE" for i in range(n)]
    }, index=dates)
    y = pd.Series(np.roll(pm, -1), index=dates)
    mask = pd.Series(True, index=dates)
    mask.iloc[:24] = False  # Warmup
    return df, y, mask


def test_sequence_air_dataset():
    data = np.random.randn(50, 4).astype(np.float32)
    targets = np.random.randn(50).astype(np.float32)
    ds = SequenceAirDataset(data, targets, seq_len=10)
    assert len(ds) == 41  # 50 - 10 + 1
    
    x_win, y_tgt = ds[0]
    assert x_win.shape == (10, 4)
    assert y_tgt.shape == (1,)
    # Target should be at window end index (idx + 9)
    assert np.isclose(y_tgt.item(), targets[9])


def test_temporal_forecaster_architectures():
    batch_size = 8
    seq_len = 16
    input_dim = 6
    x = torch.randn(batch_size, seq_len, input_dim)

    for arch in ["lstm", "gru", "bilstm"]:
        model = TemporalForecaster(input_dim=input_dim, hidden_dim=32, num_layers=2, architecture=arch)
        out, attn = model(x)
        assert out.shape == (batch_size, 1)
        assert attn.shape == (batch_size, seq_len)
        # Check attention sums to 1
        assert torch.allclose(attn.sum(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_deep_learning_pipeline_fit_predict(synthetic_series):
    X, y, mask = synthetic_series
    cfg = DLConfig(seq_len=12, hidden_dim=16, num_layers=1, epochs=3, batch_size=16)
    pipeline = DeepLearningPipeline(cfg)
    pipeline.fit(X, y, valid_mask=mask)

    preds = pipeline.predict(X)
    assert len(preds) == len(X)
    # Check warmup has NaNs and valid range has nonnegative predictions
    assert np.isnan(preds[:11]).all()
    valid_preds = preds[11:]
    assert not np.isnan(valid_preds).any()
    assert (valid_preds >= 0).all()


def test_hybrid_residual_model(synthetic_series):
    X, y, mask = synthetic_series
    dl_cfg = DLConfig(seq_len=12, hidden_dim=16, num_layers=1, epochs=3, batch_size=16)
    dl_pipe = DeepLearningPipeline(dl_cfg)

    # Use Ridge as the fast test regressor for the residual head
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer

    numeric = list(X.select_dtypes(include="number").columns)
    tabular_reg = Pipeline([
        ("pre", ColumnTransformer([("num", SimpleImputer(strategy="mean"), numeric)], remainder="drop")),
        ("reg", Ridge(alpha=1.0))
    ])

    hybrid = HybridResidualModel(dl_pipeline=dl_pipe, tabular_model=tabular_reg)
    hybrid.fit(X, y, valid_mask=mask)

    hybrid_pred, dl_base, residual = hybrid.predict(X)
    assert len(hybrid_pred) == len(X)
    assert len(dl_base) == len(X)
    assert len(residual) == len(X)

    # Check nonnegativity of hybrid output
    valid_idx = ~np.isnan(hybrid_pred)
    assert (hybrid_pred[valid_idx] >= 0).all()


def tiny_pipeline(**kwargs):
    return DeepLearningPipeline(DLConfig(seq_len=4, hidden_dim=4, num_layers=1,
                                         epochs=1, batch_size=32, **kwargs))


@pytest.mark.parametrize('target_mode', ['level', 'delta'])
def test_future_features_and_labels_do_not_change_training(synthetic_series, target_mode):
    X, y, _ = synthetic_series
    mask = pd.Series(np.arange(len(X)) < 80, index=X.index)
    mask.iloc[:4] = False
    X = X.copy()
    X.iloc[20:25, 0] = np.nan
    changed_X, changed_y = X.copy(), y.copy()
    changed_X.iloc[80:, :6] = 1e6
    changed_X.loc[changed_X.index[80:], 'wind_direction'] = 'FUTURE_ONLY'
    changed_y.iloc[80:] = 1e6
    original = tiny_pipeline(target_mode=target_mode).fit(X, y, mask)
    changed = tiny_pipeline(target_mode=target_mode).fit(changed_X, changed_y, mask)
    np.testing.assert_array_equal(original.feature_medians_, changed.feature_medians_)
    np.testing.assert_array_equal(original.feature_means, changed.feature_means)
    assert original.category_levels_ == changed.category_levels_
    assert original.preprocessing_fit_end_ == str(X.index[79])
    assert original.loss_history_ == changed.loss_history_
    np.testing.assert_array_equal(original.predict(X.iloc[:80]), changed.predict(X.iloc[:80]))


def test_prediction_is_causal_and_uses_frozen_imputation(synthetic_series):
    X, y, mask = synthetic_series
    pipeline = tiny_pipeline().fit(X, y, mask)
    X = X.copy()
    X.iloc[70:75, 0] = np.nan
    prefix = pipeline.predict(X.iloc[:75])
    # Adding extreme future rows must not change earlier missing-value fills.
    X.iloc[75:, :6] = 1e6
    np.testing.assert_allclose(pipeline.predict(X)[:75], prefix, equal_nan=True, atol=1e-6)
    np.testing.assert_allclose(pipeline.predict(X.iloc[71:75])[-1], prefix[-1], atol=1e-6)
    weights = pipeline.get_attention_weights(X.iloc[71:75])
    assert weights.shape == (4,)
    assert weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize('kind', ['empty', 'length', 'index', 'nonboolean', 'short'])
def test_invalid_training_masks_never_fall_back_to_other_labels(synthetic_series, kind):
    X, y, mask = synthetic_series
    if kind == 'empty':
        mask[:] = False
    elif kind == 'length':
        mask = np.ones(len(X) - 1, dtype=bool)
    elif kind == 'index':
        mask.index = mask.index + pd.Timedelta(hours=1)
    elif kind == 'nonboolean':
        mask = mask.astype(int)
    else:
        X, y, mask = X.head(2), y.head(2), mask.head(2)
    with pytest.raises(ValueError):
        tiny_pipeline().fit(X, y, mask)


def test_sequence_pipeline_rejects_gaps_and_misaligned_targets(synthetic_series):
    X, y, mask = synthetic_series
    with pytest.raises(ValueError, match='regular hourly'):
        tiny_pipeline().fit(X.drop(X.index[50]), y.drop(y.index[50]))
    with pytest.raises(ValueError, match='same index'):
        tiny_pipeline().fit(X, y.set_axis(y.index + pd.Timedelta(hours=1)), mask)


def test_hybrid_calibration_is_out_of_sample_with_label_gap(synthetic_series):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import make_column_selector, make_column_transformer

    X, y, mask = synthetic_series
    head = make_pipeline(make_column_transformer(
        (SimpleImputer(), make_column_selector(dtype_include=np.number))), Ridge())
    hybrid = HybridResidualModel(tiny_pipeline(), head)
    target_time = pd.Series(X.index + pd.Timedelta(hours=3), index=X.index)
    hybrid.fit(X, y, mask, target_time)
    audit = hybrid.fit_audit_
    cutoff = pd.Timestamp(audit['calibration_start'])
    assert pd.Timestamp(audit['base_latest_target']) <= cutoff
    assert hybrid.dl.training_origins_.max() < cutoff
    assert pd.Timestamp(audit['preprocessing_fit_end']) < cutoff
    assert audit['calibration_rows'] == int(np.ceil(mask.sum() * .2))
    # Errors in the correction head must reach the caller.
    def fail(_):
        raise RuntimeError('broken correction head')
    head.predict = fail
    with pytest.raises(RuntimeError, match='broken correction head'):
        hybrid.predict(X)


def test_neural_serialization_and_schema_are_stable(synthetic_series, tmp_path):
    import joblib

    X, y, mask = synthetic_series
    model = tiny_pipeline().fit(X, y, mask)
    joblib.dump(model, tmp_path / 'model.joblib')
    restored = joblib.load(tmp_path / 'model.joblib')
    np.testing.assert_array_equal(model.predict(X), restored.predict(X[X.columns[::-1]]))
    assert np.isnan(model.predict(X.head(2))).all()
    with pytest.raises(ValueError, match='Missing fitted'):
        restored.predict(X.drop(columns=X.columns[0]))


@pytest.mark.parametrize('settings', [dict(seq_len=0), dict(epochs=0), dict(hidden_dim=1),
                                     dict(architecture='cnn'), dict(dropout=1), dict(learning_rate=float('nan'))])
def test_invalid_neural_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        DLConfig(**settings)


def test_bidirectional_without_attention_uses_both_final_states():
    model = TemporalForecaster(input_dim=3, hidden_dim=4, num_layers=1,
                                dropout=0.0, architecture='bilstm', use_attention=False).eval()
    values = torch.randn(2, 5, 3)
    with torch.no_grad():
        _, (hidden, _) = model.rnn(values)
        expected = model.head(torch.cat((hidden[-2], hidden[-1]), dim=1))
        actual, attention = model(values)
    torch.testing.assert_close(actual, expected)
    assert attention is None


def test_delta_target_reconstructs_concentrations_and_preserves_missingness(synthetic_series):
    X, y, mask = synthetic_series
    model = tiny_pipeline(target_mode='delta', use_attention=False).fit(X, y, mask)
    assert model.model.attention is None
    endpoint = model.training_origins_
    expected_mean = (y.loc[endpoint] - X.loc[endpoint, 'pm25_lag_0']).mean()
    assert model.target_mean_ == pytest.approx(expected_mean, abs=1e-5)
    values = X.copy()
    values.iloc[-1, 0] = np.nan
    predictions = model.predict(values)
    assert np.isfinite(predictions[-1]) and predictions[-1] >= 0
    np.testing.assert_allclose(model.predict(values.tail(4))[-1], predictions[-1], atol=1e-5)
    with pytest.raises(ValueError, match='Attention is disabled'):
        model.get_attention_weights(values.tail(4))
