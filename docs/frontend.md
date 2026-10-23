# Aircast forecast workspace

Aircast is the interface for the project's saved XGBoost and Ridge pipelines. It can run locally or use the prepared Vercel deployment. It uses the original Python feature construction and preprocessing; the browser does not approximate the models or replace their predictions with a formula.

## Open the app

On this Mac, double-click `Start_Aircast.command`, or run:

```bash
bash /Users/prathammalhotra/Desktop/air_pred/scripts/start_frontend.sh
```

The launcher opens `http://127.0.0.1:8000` in the default browser. Leave its Terminal window running; Ctrl+C stops the app. Another port can be selected with `--port 8001`.

On a fresh checkout, install the Python package with its optional web dependencies first:

```bash
python -m pip install -e '.[web]'
python scripts/serve_frontend.py
```

The real dataset and trained artifacts must be present under `data/raw/` and `artifacts/phase1/`. They are included in the downloadable project ZIP. Training and the measured Phase 1 scores are separate from running this interface.

## Enter new readings

The form adds one new hour to the selected history. PM2.5 is required; unknown weather fields may stay blank. Numeric inputs are checked before prediction, and missing measurements remain explicit.

The initial context is the final 169 hourly slots of the Beijing archive. The form starts with example values copied from the last recorded hour. Since the archive ends at 23:00 on 31 December 2014, those example inputs represent a new scenario at 00:00 on 1 January 2015. Its forecast is for 01:00. The interface labels this as a historical scenario, rather than live air quality.

Both trained models run on the same inputs. The selected model supplies the main forecast; the other appears as a comparison. Editing a field marks the old result as stale until another forecast is generated. The app does not alter the recorded history or retrain the models.

## Upload a history

The CSV upload uses the same schema as the selected dataset:

```text
year,month,day,hour,pm2.5,DEWP,TEMP,PRES,cbwd,Iws,Is,Ir
```

The `No` column is optional and is not used for prediction. Times follow the Beijing local clock. `NA` or empty measurements remain missing. The upload needs at least 169 hourly slots, unique valid timestamps, and enough recent observed PM2.5 values. Missing timestamp rows are restored before lagging.

The upload is limited to 4 MB and 100,000 rows/hourly slots. Its last timestamp must be on or after the models' fit date, 31 December 2014 at 23:00. This prevents a model trained on later data from being used to present an earlier forecast as a valid historical prediction.

Uploading forecasts directly from the file's last hour. Switching to the form lets us append the next hour's readings to that uploaded context. After parsing, temporary files are removed. The recent window is returned to this browser tab and travels with later forecast requests. The server revalidates it, so another instance can handle the next request without a shared upload session. Reloading or resetting the tab clears the selected context.

## Read and export the result

The forecast card shows PM2.5 in µg/m³, the target time, and the change from the latest reading. The chart shows 24 hours, 72 hours, or seven days of history. Missing observations break the line; the app does not draw an interpolated measurement. Hovering shows the timestamp and value.

The export contains both model predictions and the two naive references, together with the origin, target, source and input mode. It is a forecast file, not a measured accuracy result for the new hour. The forecast screen shows the history's completeness and selected model's measured 2014 MAE/RMSE. These past errors are not an uncertainty interval. Inputs outside the initial training range or following a sharp change are flagged. The performance view shows the full comparisons and peak-error limitation. The report link opens the compiled IEEE paper.

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

The dashboard has 23 passing Python tests, alongside 38 original ML checks and five deployment-package checks. Browser execution is blocked by the restricted development workspace; the browser check above is provided for execution in a normal local Terminal.

See [the Vercel deployment guide](deployment.md) for the inference-only bundle and production command.
