"""Forecast from a saved deployment model and a CSV of observed history."""

import joblib
import numpy as np

from .data import load_dataset
from .features import build_features


def predict_latest(model_path, csv_path):
    artifact = joblib.load(model_path)
    dataset = load_dataset(csv_path)
    origin = dataset.frame.index.max()
    if origin < __import__("pandas").Timestamp(artifact["fit_through"]):
        raise ValueError("History ends before model fit time; using this model would leak future training data")
    data = build_features(dataset.frame, artifact["config"])
    if not data.warm.iloc[-1]:
        raise ValueError("Not enough hourly history for the configured lags and rolling windows")
    X = data.X.iloc[[-1]].reindex(columns=artifact["columns"])
    forecast = float(np.maximum(artifact["pipeline"].predict(X)[0], 0))
    return {"model": artifact["model_name"], "origin_time": str(origin), "target_time": str(data.target_time.iloc[-1]),
            "horizon_hours": artifact["config"].horizon_hours, "pm25_ug_m3": forecast,
            "latest_pm25_missing": bool(dataset.frame["pm2.5"].isna().iloc[-1]), "source": dataset.source}
