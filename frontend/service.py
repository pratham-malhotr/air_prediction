"""Use the saved Python pipelines with observed hourly history and new readings."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from air_prediction.data import FILENAME, REQUIRED, WEATHER_NUMERIC, load_dataset, validate_timestamps
from air_prediction.features import build_features

MODEL_LABELS = {
    "hybrid": "Hybrid (BiLSTM + XGBoost)",
    "bilstm": "Bi-LSTM (Deep Learning)",
    "xgboost": "XGBoost",
    "ridge": "Ridge regression",
}
MAX_UPLOAD_ROWS = 100_000
MAX_UPLOAD_BYTES = 4 * 1024 * 1024


class InputError(ValueError):
    def __init__(self, message, fields=None, status=400):
        super().__init__(message)
        self.fields = fields or {}
        self.status = status


def clock(value):
    return pd.Timestamp(value).tz_localize("Asia/Shanghai").isoformat()


def number(value):
    return None if pd.isna(value) else float(value)


class ForecastService:
    def __init__(self, project_root):
        self.root = Path(project_root)
        run = self.root / "artifacts/phase1"
        self.models = {}
        # Core required models
        for name in ("xgboost", "ridge"):
            model_path = run / f"{name}.joblib"
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
        # Phase 2 DL & Hybrid models (loaded when available)
        for name in ("hybrid", "bilstm"):
            model_path = run / f"{name}.joblib"
            if model_path.exists():
                self.models[name] = joblib.load(model_path)

        if "xgboost" not in self.models or "ridge" not in self.models:
            raise ValueError("Aircast needs at least the one-hour XGBoost and Ridge deployment models.")
        self.history_size = max(model["config"].warmup_hours for model in self.models.values()) + 1
        self.fit_through = max(pd.Timestamp(model["fit_through"]) for model in self.models.values())
        sample = load_dataset(self.root / "data/raw" / FILENAME)
        self.sample = sample.frame.tail(self.history_size).copy()
        config = self.models["xgboost"]["config"]
        training = sample.frame.loc[(sample.frame.index >= pd.Timestamp(config.train_start)) &
                                    (sample.frame.index < pd.Timestamp(config.validation_start)) &
                                    (sample.frame.index <= self.fit_through)]
        self.training_ranges = {key: {"min": number(training[key].min()), "max": number(training[key].max())}
                                for key in ("pm2.5", *WEATHER_NUMERIC)}
        self.change_threshold = number(training["pm2.5"].diff().abs().quantile(.95))
        self.metrics = pd.read_csv(run / "metrics.csv").query("phase == 'holdout'").to_dict("records")
        self.quarters = pd.read_csv(run / "fold_metrics.csv").query("phase == 'holdout'").to_dict("records")
        self.regimes = pd.read_csv(run / "errors_by_pollution_regime.csv").to_dict("records")
        self.manifest = json.loads((run / "run_manifest.json").read_text())

    def _history(self, history_id, history=None):
        if history_id == "sample" and history is None:
            return self.sample.copy(), {"label": "Beijing sample", "kind": "sample", "rows": len(self.sample)}
        if not isinstance(history_id, str):
            raise InputError("Choose the Beijing sample or upload an hourly CSV first.")
        if not isinstance(history, dict) or not isinstance(history.get("csv"), str):
            raise InputError("Please select the hourly CSV again. Its recent readings are missing from this request.", status=404)
        encoded = history["csv"].encode("utf-8")
        if len(encoded) > 128_000:
            raise InputError("The recent-history payload is too large. Please select the original CSV again.")
        dataset = self._parse_history(encoded)
        label = history.get("label", "Uploaded history.csv")
        if not isinstance(label, str):
            raise InputError("The history filename must be text.")
        return dataset.frame.tail(self.history_size), {"label": Path(label).name[:100], "kind": "upload", "rows": dataset.audit["input_rows"]}

    def _context(self, frame, info, history_id):
        last = frame.iloc[-1]
        return {"id": history_id, **info, "history_hours": len(frame),
                "start": clock(frame.index.min()), "end": clock(frame.index.max()),
                "next_observation_time": clock(frame.index.max() + pd.to_timedelta(1, unit="h")),
                "defaults": {"pm2.5": number(last["pm2.5"]), **{key: number(last[key]) for key in WEATHER_NUMERIC},
                             "cbwd": None if pd.isna(last["cbwd"]) else str(last["cbwd"])}}

    def context(self, history_id="sample"):
        frame, info = self._history(history_id)
        return self._context(frame, info, history_id)

    def _parse_history(self, payload):
        if len(payload) > MAX_UPLOAD_BYTES:
            raise InputError("This CSV is too large. Please keep uploads below 4 MB.", status=413)
        try:
            raw = pd.read_csv(BytesIO(payload), nrows=MAX_UPLOAD_ROWS + 1, na_values=["NA", "nan", ""])
            if len(raw) > MAX_UPLOAD_ROWS:
                raise InputError("Please upload at most 100,000 hourly records.")
            missing = sorted(set(REQUIRED) - set(raw.columns))
            if missing:
                raise InputError("The CSV is missing these columns: " + ", ".join(missing))
            if raw.empty:
                raise InputError("The CSV has no readings.")
            timestamps = validate_timestamps(raw)
            span = int((timestamps.max() - timestamps.min()) / pd.to_timedelta(1, unit="h")) + 1
            if span > MAX_UPLOAD_ROWS:
                raise InputError("The CSV spans too many hours. Please check its date columns.")
            if span < self.history_size:
                raise InputError(f"We need at least {self.history_size} hourly slots (just over seven days).")
            if timestamps.max() < self.fit_through:
                raise InputError("This history ends before the saved models' training period. Use data ending on or after "
                                 + self.fit_through.strftime("%d %b %Y, %H:%M") + " Beijing time.")
            with TemporaryDirectory(prefix="aircast-") as directory:
                path = Path(directory) / "history.csv"
                path.write_bytes(payload)
                dataset = load_dataset(path)
            for column in ("Iws", "Is", "Ir"):
                if dataset.frame[column].dropna().lt(0).any():
                    raise InputError(f"{column} cannot contain negative measurements.")
            if dataset.frame["PRES"].dropna().le(0).any():
                raise InputError("Pressure measurements must be greater than zero.")
            if dataset.frame[["pm2.5", *WEATHER_NUMERIC]].abs().gt(1_000_000).any().any():
                raise InputError("A measurement is outside the supported numeric range. Please check the units in your CSV.")
            if dataset.frame["pm2.5"].tail(self.history_size).notna().sum() < 2:
                raise InputError("The recent window needs at least two measured PM2.5 readings.")
        except InputError:
            raise
        except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
            raise InputError("We could not read this hourly CSV. " + str(exc)) from exc
        return dataset

    def upload(self, payload, filename):
        dataset = self._parse_history(payload)
        identifier = "upload_" + sha256(payload).hexdigest()[:16]
        info = {"label": Path(filename or "Uploaded history.csv").name[:100], "kind": "upload",
                "rows": int(dataset.audit["input_rows"]), "inserted_hours": int(dataset.audit["inserted_missing_hours"])}
        frame = dataset.frame.tail(self.history_size).copy()
        normalized = frame.copy()
        for key in ("year", "month", "day", "hour"):
            normalized[key] = getattr(frame.index, key)
        context = self._context(frame, info, identifier)
        # Carry only the recent window, so another server instance can serve the next request.
        context["history"] = {"csv": normalized.to_csv(index=False), "label": info["label"]}
        return context

    def _read_inputs(self, values):
        if not isinstance(values, dict):
            raise InputError("Please enter the latest measurements.")
        errors, row = {}, {}
        for key in ("pm2.5", *WEATHER_NUMERIC):
            value = values.get(key)
            if value is None or value == "":
                if key == "pm2.5":
                    errors[key] = "Enter the latest PM2.5 concentration."
                row[key] = np.nan
                continue
            try:
                if isinstance(value, bool):
                    raise ValueError
                parsed = float(value)
                if not np.isfinite(parsed):
                    raise ValueError
                if abs(parsed) > 1_000_000:
                    errors[key] = "Please check this value and its units."
                elif key in ("pm2.5", "Iws", "Is", "Ir") and parsed < 0:
                    errors[key] = "This measurement cannot be negative."
                elif key == "PRES" and parsed <= 0:
                    errors[key] = "Pressure must be greater than zero."
                row[key] = parsed
            except (TypeError, ValueError):
                errors[key] = "Enter a finite number."
        wind = values.get("cbwd")
        if wind is not None and not isinstance(wind, str):
            errors["cbwd"] = "Choose a wind direction."
        elif wind and len(wind.strip()) > 32:
            errors["cbwd"] = "Wind direction must be at most 32 characters."
        row["cbwd"] = wind.strip() if isinstance(wind, str) and wind.strip() else pd.NA
        if errors:
            raise InputError("Please check the highlighted measurements.", errors)
        return row

    def forecast(self, history_id="sample", mode="manual", inputs=None, selected=None, history=None):
        if selected is None:
            selected = "hybrid" if "hybrid" in self.models else ("xgboost" if "xgboost" in self.models else next(iter(self.models)))
        if not isinstance(selected, str) or selected not in self.models:
            valid_names = ", ".join(MODEL_LABELS.get(m, m) for m in self.models)
            raise InputError(f"Selected model '{selected}' is unavailable. Choose from: {valid_names}.")
        if mode not in ("manual", "history"):
            raise InputError("Choose manual readings or a CSV history.")
        frame, info = self._history(history_id, history)
        if mode == "manual":
            row = self._read_inputs(inputs)
            origin = frame.index.max() + pd.to_timedelta(1, unit="h")
            frame = frame.reindex(pd.date_range(frame.index.min(), origin, freq="h", name=frame.index.name))
            for key, value in row.items():
                frame.at[origin, key] = value
            frame = frame.tail(self.history_size)
        origin = frame.index.max()
        if origin < self.fit_through:
            raise InputError("The forecast must start after the saved models' training period.")
        predictions = {}
        for name, artifact in self.models.items():
            data = build_features(frame, artifact["config"])
            if not data.warm.iloc[-1]:
                raise InputError(f"Please provide at least {self.history_size} consecutive hourly slots.")
            if name in ("hybrid", "bilstm"):
                raw_out = artifact["pipeline"].predict(data.X.reindex(columns=artifact["columns"]))
            else:
                X = data.X.iloc[[-1]].reindex(columns=artifact["columns"])
                raw_out = artifact["pipeline"].predict(X)
            # If HybridResidualModel returns (hybrid, dl_base, residual)
            if isinstance(raw_out, tuple):
                val = raw_out[0]
            else:
                val = raw_out
            if hasattr(val, "__len__"):
                val = val[-1] if len(val) > 0 else 0.0
            value = float(np.maximum(val, 0))
            if not np.isfinite(value):
                raise RuntimeError("The saved model returned a nonfinite prediction.")
            predictions[name] = value
        chosen = self.models[selected]
        data = build_features(frame, chosen["config"])
        latest = number(frame["pm2.5"].iloc[-1])
        warnings = []
        if latest is None:
            warnings.append("The latest PM2.5 reading is missing; the forecast uses the available history.")
        if frame[[*WEATHER_NUMERIC, "cbwd"]].iloc[-1].isna().any():
            warnings.append("Some current weather readings are missing. They have been left unknown.")
        observed = int(frame["pm2.5"].notna().sum())
        if observed < len(frame):
            warnings.append(f"{len(frame) - observed} PM2.5 readings are missing in the recent history.")
        for key, limits in self.training_ranges.items():
            value = number(frame[key].iloc[-1])
            if value is not None and limits["min"] is not None and (value < limits["min"] or value > limits["max"]):
                label = {"pm2.5": "PM2.5", "TEMP": "Temperature", "DEWP": "Dew point", "PRES": "Pressure",
                         "Iws": "Accumulated wind", "Is": "Snow duration", "Ir": "Rain duration"}[key]
                warnings.append(f"{label} is outside the initial training range ({limits['min']:g}–{limits['max']:g}). Treat this forecast as an extrapolation.")
        previous = number(frame["pm2.5"].iloc[-2])
        if latest is not None and previous is not None and self.change_threshold and abs(latest - previous) > self.change_threshold:
            warnings.append("The latest concentration changes sharply from the preceding hour. Abrupt pollution changes are a known weakness of these models.")
        reference = next((row for row in self.metrics if row["model"] == selected),
                         next(row for row in self.metrics if row["model"] == "xgboost"))
        return {"selected_model": selected, "forecast": predictions[selected], "predictions": predictions,
                "latest": latest, "change": None if latest is None else predictions[selected] - latest,
                "origin_time": clock(origin), "target_time": clock(data.target_time.iloc[-1]),
                "horizon_hours": chosen["config"].horizon_hours, "mode": mode, "context": {**info, "id": history_id},
                "baselines": {key: number(value) for key, value in data.baselines.iloc[-1].items()},
                "series": [{"time": clock(timestamp), "value": number(value),
                            "entered": bool(mode == "manual" and timestamp == origin)}
                           for timestamp, value in frame["pm2.5"].items()],
                "warnings": warnings, "historical": info["kind"] == "sample",
                "data_quality": {"observed": observed, "total": len(frame), "fraction": observed / len(frame)},
                "backtest": {key: reference.get(key, 0.0) for key in ("mae", "rmse", "r2", "n")}}

    def bootstrap(self):
        context = self.context()
        available_models = {k: MODEL_LABELS.get(k, k.replace("_", " ").title()) for k in self.models}
        default_model = "xgboost" if "xgboost" in self.models else next(iter(self.models))
        return {"ready": True, "context": context, "models": available_models,
                "required_hours": self.history_size, "metrics": self.metrics,
                "quarters": self.quarters, "regimes": self.regimes,
                "fit_through": clock(self.fit_through), "training_p90": self.manifest["training_p90_pm25"],
                "forecast": self.forecast(inputs=context["defaults"], selected=default_model)}
