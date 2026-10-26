"""Exercise the dashboard's input boundaries and its use of the saved pipelines."""

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("flask", reason="The dashboard tests require the web extra")

from air_prediction.config import Config
from air_prediction.data import FILENAME
from air_prediction.features import build_features
from air_prediction.models import make_model
from air_prediction.predict import predict_latest
from frontend.app import create_app
from frontend.service import ForecastService, InputError


def csv_bytes(frame):
    raw = frame.copy()
    for key in ("year", "month", "day", "hour"):
        raw[key] = getattr(raw.index, key)
    return raw.to_csv(index=False).encode()


@pytest.fixture()
def web_setup(tmp_path, hourly_frame):
    run = tmp_path / "artifacts/phase1"
    run.mkdir(parents=True)
    data_path = tmp_path / "data/raw" / FILENAME
    data_path.parent.mkdir(parents=True)
    data_path.write_bytes(csv_bytes(hourly_frame))
    config = replace(Config(), minimum_training_rows=10)
    data = build_features(hourly_frame, config)
    fit_through = hourly_frame.index[700]
    mask = data.warm & data.y.notna() & (data.target_time <= fit_through)
    for name in ("ridge", "xgboost"):
        if name == "xgboost":
            pytest.importorskip("xgboost")
        pipeline = make_model(name, data.X, config, {"n_estimators": 5, "max_depth": 2} if name == "xgboost" else None)
        pipeline.fit(data.X.loc[mask], data.y.loc[mask])
        joblib.dump({"pipeline": pipeline, "config": config, "columns": list(data.X),
                     "model_name": name, "fit_through": str(fit_through)}, run / f"{name}.joblib")
    # Synthetic fixture scores are deliberately separate from all measured project results.
    rows = [{"phase": "holdout", "model": name, "n": 10, "mae": 1.0, "rmse": 2.0, "r2": .5}
            for name in ("ridge", "xgboost", "persistence", "seasonal_naive")]
    pd.DataFrame(rows).to_csv(run / "metrics.csv", index=False)
    pd.DataFrame([{**row, "fold": f"holdout_0{quarter}"} for row in rows for quarter in range(1, 5)]).to_csv(run / "fold_metrics.csv", index=False)
    pd.DataFrame([{**row, "pollution_regime": "above_training_p90"} for row in rows]).to_csv(run / "errors_by_pollution_regime.csv", index=False)
    (run / "run_manifest.json").write_text(json.dumps({"training_p90_pm25": 900}))
    service = ForecastService(tmp_path)
    app = create_app(tmp_path, service)
    app.config.update(TESTING=True)
    return app.test_client(), service, data_path, run, hourly_frame


def test_dashboard_and_upload_predictions_match_the_existing_cli(web_setup):
    client, service, data_path, run, frame = web_setup
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    response = client.get("/api/bootstrap")
    assert response.status_code == 200 and response.json["ready"] is True
    upload = client.post("/api/history", data={"file": (BytesIO(data_path.read_bytes()), "readings.csv")})
    assert upload.status_code == 200
    context = upload.json
    for name in ("ridge", "xgboost"):
        result = client.post("/api/forecast", json={"history_id": context["id"], "history": context["history"], "mode": "history", "model": name})
        assert result.status_code == 200
        expected = predict_latest(run / f"{name}.joblib", data_path)
        assert result.json["forecast"] == pytest.approx(expected["pm25_ug_m3"], abs=1e-8)
        assert result.json["historical"] is False


def test_manual_inputs_advance_the_clock_and_do_not_change_observed_history(web_setup, tmp_path):
    client, service, _, run, frame = web_setup
    original = service.sample.copy()
    values = service.context()["defaults"]
    values["pm2.5"] = 125.0
    response = client.post("/api/forecast", json={"mode": "manual", "inputs": values, "model": "ridge"})
    assert response.status_code == 200
    result = response.json
    expected_origin = frame.index.max() + pd.to_timedelta(1, unit="h")
    assert pd.Timestamp(result["origin_time"]).tz_localize(None) == expected_origin
    assert pd.Timestamp(result["target_time"]).tz_localize(None) == expected_origin + pd.to_timedelta(1, unit="h")
    appended = pd.concat([frame, pd.DataFrame([values], index=[expected_origin])])
    csv = tmp_path / "appended.csv"
    csv.write_bytes(csv_bytes(appended))
    expected = predict_latest(run / "ridge.joblib", csv)
    assert result["forecast"] == pytest.approx(expected["pm25_ug_m3"], abs=1e-8)
    values["pm2.5"] = 350.0
    changed = client.post("/api/forecast", json={"inputs": values, "model": "ridge"})
    assert abs(changed.json["forecast"] - result["forecast"]) > .1
    pd.testing.assert_frame_equal(original, service.sample)


@pytest.mark.parametrize("key,value", [("pm2.5", None), ("pm2.5", -1), ("pm2.5", True),
                                      ("pm2.5", "nan"), ("pm2.5", "inf"), ("pm2.5", 1e100),
                                      ("PRES", 0), ("Iws", -1), ("cbwd", ["NW"])])
def test_invalid_measurements_return_specific_field_errors(web_setup, key, value):
    client, service, *_ = web_setup
    inputs = service.context()["defaults"]
    inputs[key] = value
    response = client.post("/api/forecast", json={"inputs": inputs})
    assert response.status_code == 400
    assert key in response.json["error"]["fields"]


def test_missing_weather_remains_unknown_and_can_still_be_forecast(web_setup):
    client, service, *_ = web_setup
    inputs = {key: None for key in service.context()["defaults"]}
    inputs["pm2.5"] = 50
    response = client.post("/api/forecast", json={"inputs": inputs})
    assert response.status_code == 200
    assert np.isfinite(response.json["forecast"])
    assert any("weather" in warning for warning in response.json["warnings"])


@pytest.mark.parametrize("change", ["short", "old", "duplicate", "span"])
def test_csv_history_limits_preserve_the_time_contract(web_setup, change):
    client, _, _, _, frame = web_setup
    if change == "short":
        raw = frame.tail(100)
    elif change == "old":
        raw = frame.head(400)
    elif change == "duplicate":
        raw = pd.concat([frame, frame.tail(1)])
    else:
        raw = frame.iloc[[0, -1]].copy()
        raw.index = pd.to_datetime(["2010-01-01", "2050-01-01"])
    response = client.post("/api/history", data={"file": (BytesIO(csv_bytes(raw)), "invalid.csv")})
    assert response.status_code == 400
    assert response.json["error"]["message"]


def test_csv_gaps_are_retained_in_the_chart_and_features(web_setup):
    client, _, _, _, frame = web_setup
    gapped = frame.drop(frame.index[-3])
    upload = client.post("/api/history", data={"file": (BytesIO(csv_bytes(gapped)), "gapped.csv")})
    assert upload.status_code == 200 and upload.json["inserted_hours"] == 1
    response = client.post("/api/forecast", json={"mode": "history", "history_id": upload.json["id"], "history": upload.json["history"]})
    assert response.status_code == 200
    assert len(response.json["series"]) == 169
    assert response.json["series"][-3]["value"] is None


def test_bad_payloads_unknown_history_and_cross_origin_requests_are_rejected(web_setup):
    client, service, *_ = web_setup
    assert client.post("/api/forecast", json=[]).status_code == 400
    assert client.post("/api/forecast", json={"model": []}).status_code == 400
    assert client.post("/api/forecast", json={"history_id": "unknown-token"}).status_code == 404
    assert client.post("/api/forecast", json={"inputs": service.context()["defaults"]}, headers={"Origin": "https://another-site.example"}).status_code == 403
    assert client.get("/api/bootstrap").headers["Cache-Control"] == "no-store"


def test_uploaded_context_survives_a_new_server_instance(web_setup):
    client, service, data_path, run, _ = web_setup
    upload = client.post("/api/history", data={"file": (BytesIO(data_path.read_bytes()), "readings.csv")})
    fresh = create_app(run.parents[1]).test_client()
    response = fresh.post("/api/forecast", json={"mode": "history", "history_id": upload.json["id"], "history": upload.json["history"]})
    assert response.status_code == 200
    expected = predict_latest(run / "xgboost.joblib", data_path)
    assert response.json["forecast"] == pytest.approx(expected["pm25_ug_m3"], abs=1e-8)
    assert not hasattr(service, "_uploads")


def test_oversized_upload_returns_a_readable_error(web_setup):
    client, *_ = web_setup
    response = client.post("/api/history", data={"file": (BytesIO(b"x" * (4 * 1024 * 1024)), "large.csv")})
    assert response.status_code == 413
    assert "4 MB" in response.json["error"]["message"]


def test_stateless_history_is_revalidated_instead_of_trusted(web_setup):
    client, _, data_path, *_ = web_setup
    upload = client.post("/api/history", data={"file": (BytesIO(data_path.read_bytes()), "readings.csv")})
    forged = dict(upload.json["history"])
    forged["csv"] = "year,month,day,hour,pm2.5\n2010,1,1,0,10\n"
    response = client.post("/api/forecast", json={"mode": "history", "history_id": upload.json["id"], "history": forged})
    assert response.status_code == 400


def test_forecast_reports_measured_reference_and_data_quality(web_setup):
    client, service, *_ = web_setup
    inputs = service.context()["defaults"]
    inputs["pm2.5"] = 5000
    result = client.post("/api/forecast", json={"inputs": inputs}).json
    assert result["data_quality"] == {"observed": 169, "total": 169, "fraction": 1.0}
    assert result["backtest"]["mae"] == 1.0
    assert any("extrapolation" in warning for warning in result["warnings"])
    assert any("changes sharply" in warning for warning in result["warnings"])


def test_trusted_vercel_proxy_preserves_same_origin_check(web_setup, monkeypatch):
    _, service, _, run, _ = web_setup
    monkeypatch.setenv("VERCEL", "1")
    client = create_app(run.parents[1], service).test_client()
    response = client.post("/api/forecast", json={"inputs": service.context()["defaults"]},
                           headers={"Origin": "https://aircast.example", "X-Forwarded-Proto": "https", "X-Forwarded-Host": "aircast.example"})
    assert response.status_code == 200


def test_bootstrap_returns_only_available_models(web_setup):
    client, service, *_ = web_setup
    response = client.get("/api/bootstrap")
    assert response.status_code == 200
    data = response.json
    assert data["ready"] is True
    # In web_setup only ridge and xgboost are trained
    assert set(data["models"].keys()) == {"xgboost", "ridge"}
    assert "hybrid" not in data["models"]
    assert data["forecast"]["selected_model"] in {"xgboost", "ridge"}


def test_forecast_rejects_unavailable_model_with_informative_error(web_setup):
    client, service, *_ = web_setup
    inputs = service.context()["defaults"]
    response = client.post("/api/forecast", json={"inputs": inputs, "model": "unavailable_model"})
    assert response.status_code == 400
    error_msg = response.json["error"]["message"]
    assert "unavailable_model" in error_msg
    assert "XGBoost" in error_msg or "Ridge" in error_msg
