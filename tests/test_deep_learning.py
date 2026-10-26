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
