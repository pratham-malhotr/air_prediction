"""Metrics on common rows; undefined R squared is recorded as null."""

import numpy as np
import pandas as pd


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual):
        raise ValueError("actual and predicted must be nonempty vectors of equal length")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite observations and predictions")
    residual = predicted - actual
    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((actual - actual.mean()) ** 2))
    return {"n": len(actual), "mae": float(np.mean(np.abs(residual))), "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "r2": 1.0 - sse / sst if len(actual) > 1 and sst > 0 else None,
            "bias": float(residual.mean())}


def score_predictions(predictions, group_columns=("phase", "model")):
    rows = []
    for key, group in predictions.groupby(list(group_columns), sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({**dict(zip(group_columns, key)), **regression_metrics(group.actual, group.prediction)})
    return pd.DataFrame(rows)
