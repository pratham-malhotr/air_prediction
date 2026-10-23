"""Exercises the actual estimator when installed (including Colab); explicit local skip otherwise."""

from pathlib import Path

import pytest

pytest.importorskip("xgboost", reason="XGBoost is not installed; run this smoke test in Colab or after installing requirements")

from air_prediction.config import load_config
from air_prediction.demo import make_demo
from air_prediction.experiment import run_experiment
from air_prediction.predict import predict_latest


def test_actual_xgboost_backtests_ablation_and_saved_forecast(tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = make_demo(tmp_path / "synthetic.csv")
    result = run_experiment(data, load_config(root / "configs/demo.toml"), tmp_path / "demo", ("xgboost", "ridge"), ablations=True)
    output = result["output"]
    assert set(result["metrics"].model) == {"xgboost", "ridge", "persistence", "seasonal_naive"}
    assert (output / "xgboost.json").exists()
    assert (output / "validation_ablation.csv").exists()
    assert predict_latest(output / "xgboost.joblib", data)["pm25_ug_m3"] >= 0
