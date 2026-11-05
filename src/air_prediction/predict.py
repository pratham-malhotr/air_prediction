"""Forecast from a saved deployment model and a CSV of observed history."""

import joblib
import numpy as np
import pandas as pd

from .data import load_dataset
from .features import build_features


def required_history_hours(artifact):
    """Raw hourly slots needed to construct every feature in the last sequence."""
    hours = artifact["config"].warmup_hours + 1
    if artifact["model_name"] in {"bilstm", "lstm", "gru", "hybrid"}:
        pipeline = artifact["pipeline"]
        neural = pipeline.dl if artifact["model_name"] == "hybrid" else pipeline
        hours += neural.config.seq_len - 1
    return max(hours, artifact.get("required_history_hours", hours))


def predict_last(artifact, data):
    if len(data.X) < required_history_hours(artifact):
        raise ValueError("Not enough hourly history to construct all sequence features")
    X = data.X.reindex(columns=artifact["columns"])
    if artifact["model_name"] in {"bilstm", "lstm", "gru", "hybrid"}:
        pipeline = artifact["pipeline"]
        neural = pipeline.dl if artifact["model_name"] == "hybrid" else pipeline
        X = X.tail(neural.config.seq_len)
    else:
        X = X.iloc[[-1]]
    output = artifact["pipeline"].predict(X)
    if isinstance(output, tuple):
        output = output[0]
    forecast = float(np.maximum(output[-1], 0.0))
    if not np.isfinite(forecast):
        raise RuntimeError("The saved model returned a nonfinite prediction")
    return forecast


def predict_latest(model_path, csv_path):
    artifact = joblib.load(model_path)
    dataset = load_dataset(csv_path)
    origin = dataset.frame.index.max()
    if origin < pd.Timestamp(artifact["fit_through"]):
        raise ValueError("History ends before model fit time; using this model would leak future training data")
    data = build_features(dataset.frame, artifact["config"])
    if not data.warm.iloc[-1]:
        raise ValueError("Not enough hourly history for the configured lags and rolling windows")
    forecast = predict_last(artifact, data)
    return {"model": artifact["model_name"], "origin_time": str(origin), "target_time": str(data.target_time.iloc[-1]),
            "horizon_hours": artifact["config"].horizon_hours, "pm25_ug_m3": forecast,
            "latest_pm25_missing": bool(dataset.frame["pm2.5"].isna().iloc[-1]), "source": dataset.source}
