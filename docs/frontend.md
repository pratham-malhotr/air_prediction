# Aircast forecast workspace

Aircast is the interface for the project's saved XGBoost and Ridge pipelines and, locally, completed Phase 2 LSTM, Bi-LSTM and hybrid models. It can run locally or use the prepared Vercel deployment. It uses the original Python feature construction and preprocessing; the browser does not approximate the models or replace their predictions with a formula.

## Open the app

On this Mac, double-click `Start_Aircast.command`, or run:

```bash
bash scripts/start_frontend.sh
```

The launcher opens `http://127.0.0.1:8000` in the default browser. Leave its Terminal window running; Ctrl+C stops the app. Another port can be selected with `--port 8001`.

On a fresh checkout, install the Python package with its optional web dependencies first:

```bash
python -S scripts/restore_phase2_runtime.py
python -m pip install -r requirements-phase2.txt
python scripts/serve_frontend.py
```

Use Python 3.13 for the saved Phase 2 weights. The restorer supplies the dataset, classical and neural artifacts, comparison tables and report after verifying the pinned release. It preserves conflicting existing runs. The full downloadable research ZIP also includes training/evaluation evidence; the inference restore is smaller. Training and the measured scores are separate from running this interface.

## Enter new readings

The form adds one new hour to the selected history. PM2.5 is required; unknown weather fields may stay blank. Numeric inputs are checked before prediction, and missing measurements remain explicit.

The initial context is the final 169 hourly slots (192 when the 24-step Phase 2 models are loaded) of the Beijing archive. The form starts with example values copied from the last recorded hour. Since the archive ends at 23:00 on 31 December 2014, those example inputs represent a new scenario at 00:00 on 1 January 2015. Its forecast is for 01:00. The interface labels this as a historical scenario, rather than live air quality.

All available trained models run on the same observed inputs. The selected model supplies the main forecast, with another available model shown for comparison. Editing a field marks the old result as stale until another forecast is generated. Repeated manual requests explore the next hour after the selected history; they do not advance a stored observation stream or retrain the models.

## Upload a history

The CSV upload uses the same schema as the selected dataset:

```text
year,month,day,hour,pm2.5,DEWP,TEMP,PRES,cbwd,Iws,Is,Ir
```

The `No` column is optional and is not used for prediction. Times follow the Beijing local clock. `NA` or empty measurements remain missing. The upload needs at least 169 hourly slots (192 with the Phase 2 sequence models), unique valid timestamps, and enough recent observed PM2.5 values. Missing timestamp rows are restored before lagging.

The upload is limited to 4 MB and 100,000 rows/hourly slots. Its last timestamp must be on or after the models' fit date, 31 December 2014 at 23:00. This prevents a model trained on later data from being used to present an earlier forecast as a valid historical prediction.

Uploading forecasts directly from the file's last hour. Switching to the form lets us append the next hour's readings to that uploaded context. After parsing, temporary files are removed. The recent window is returned to this browser tab and travels with later forecast requests. The server revalidates it, so another instance can handle the next request without a shared upload session. Reloading or resetting the tab clears the selected context.

## Read and export the result

The forecast card shows PM2.5 in µg/m³, the target time, and the change from the latest reading. The chart shows 24 hours, 72 hours, or seven days of history. Missing observations break the line; the app does not draw an interpolated measurement. Hovering shows the timestamp and value.

The export contains all available model predictions and the two naive references, together with the origin, target, source and input mode. It is a forecast file, not a measured accuracy result for the new hour. The forecast screen shows the history's completeness and selected model's measured 2014 MAE/RMSE. These past errors are not an uncertainty interval. Inputs outside the initial training range or following a sharp change are flagged. The performance view shows the full comparisons and peak-error limitation. The report link opens the compiled IEEE paper.

The app does not assign AQI categories or confidence intervals. Historical Beijing performance does not establish accuracy for another city or a present-day deployment.

## Implementation and verification

- `frontend/service.py` loads trusted local model artifacts, validates inputs, preserves timing and builds the original features.
- `frontend/app.py` serves the dashboard, forecast API, CSV upload/download and report.
- `frontend/static/` contains the responsive interface and its interactive SVG chart. Assets are local; no CDN or JavaScript build step is required.
- `tests/test_frontend.py` checks prediction parity with the existing CLI, edited measurements, missingness, date boundaries, upload limits and readable API errors.

The optional browser check uses an available Playwright installation and Chromium browser:

```bash
node scripts/check_frontend_browser.cjs
```

It starts a temporary service on port 8016, uses the actual saved models, exercises the main interactions, and saves screenshots/results under `artifacts/frontend_review/`. The temporary service is stopped afterward.

The dashboard has 28 Python tests within the complete project suite. Phase 2 checks verify CLI parity for all three saved neural/hybrid models, complete feature history, optional PyTorch behavior and artifact integrity. The browser check exercises the available models and all seven comparison methods when Phase 2 is loaded. Performance counts, default-model badges, summary scores and peak errors follow the loaded run. The classical deployment has four comparison methods.

See [the Vercel deployment guide](deployment.md) for the inference-only bundle and production command.

The dashboard prefers a completed, verified real-data run in `artifacts/phase2_research/` when present, exposing the independent LSTM, Bi-LSTM and calibrated hybrid. Until that run is complete, it uses the preserved `artifacts/phase2/` baseline. Required history is derived from the loaded models, including a longer selected sequence if applicable.

The completed research run defaults to the model selected from mean 2013 validation RMSE (LSTM in the recorded run). The report link opens the Phase 2 report when that run and report are available.
