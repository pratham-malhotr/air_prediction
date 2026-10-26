# Deploying Aircast to Vercel

The deployment keeps the actual Python inference pipelines. The frontend assets are served from Vercel's CDN, while the Flask application runs as a Python function. No model training occurs during a request.

## Deployment status

The Vercel CLI reported a successful production deployment on 23 October 2026 and assigned [aircast-pm25.vercel.app](https://aircast-pm25.vercel.app). This confirms the build and publication reported by Vercel; hosted inference and visual rendering still need independent checks. The local prepared folder is linked to the `aircast-pm25` project.

The CLI also reports Vercel Authentication protection. That message alone does not establish whether the production alias requires login: Standard Protection leaves production domains public, while All Deployments protects them too. The [project's Deployment Protection settings](https://vercel.com/pratham-malhotras-projects/aircast-pm25/settings/deployment-protection) show the actual scope. Keep the existing settings while testing through the authenticated CLI.

## Deploy the prepared application

From this Mac's Terminal:

```bash
bash scripts/deploy_vercel.sh
```

The script prepares `dist/aircast-pm25`, reuses an available Vercel CLI, and requests a production deployment. Vercel account authentication must be available to that CLI. The printed deployment URL is the result to verify; preparing the folder alone does not publish anything.

To prepare the files without publishing:

```bash
python scripts/prepare_vercel.py
```

An existing `.vercel` project link inside the generated folder is preserved when rebuilding. The generator verifies the dataset and forecasting-source hashes before it packages the evaluated models.

## Runtime files and dependencies

The prepared folder contains the Flask entrypoint, frontend templates, public assets, Python forecasting source, both saved models, measured comparison tables, the original dataset and its attribution, and the compiled report. It excludes the training outputs, local environment, caches, notebook snapshots and test suite.

The Python runtime is 3.13. The inference requirements pin the ML package versions used to save the models. The Linux deployment uses `xgboost-cpu==3.4.1`, the official CPU-only build, rather than installing unused GPU support. Matplotlib is not needed to serve forecasts.

The original ML requirements and local frontend installation remain separate from this inference-only dependency list. The prepared Linux requirements should not be installed directly into the Mac environment.

## Uploads across server instances

A serverless application cannot rely on one process keeping every uploaded history. The upload route therefore returns only the normalized recent window to the browser. Each subsequent prediction carries that window and the server validates it again. A different server instance can produce the same result without a shared session store.

Upload requests are limited to 4 MB, below Vercel's request-body limit. Temporary parsing files are removed after validation. The app does not permanently store CSV uploads. HTTPS proxy headers are handled when running on Vercel so the same-origin check accepts the deployed hostname.

## Verify the deployment

Run the hosted checks from this Mac's Terminal:

```bash
python scripts/check_vercel.py
```

The checker reuses the authenticated Vercel CLI and handles protected deployment requests through `vercel curl`. It checks model readiness, current HTML/static assets, bootstrap predictions and comparison tables, edited inputs with each model, the original CSV upload and subsequent stateless forecast, invalid-input handling, and report/example downloads. Remote predictions must match the actual saved local pipelines within a small floating-point tolerance. POST requests include the production Origin so HTTPS handling is checked too.

Only a successful hosted check writes `artifacts/deployment_verification.json`. This record is separate from the offline test suite and does not claim a browser visual review. A failed check exits with the endpoint and HTTP status; inspect that deployment's function logs to diagnose it.

- `/api/health` should report both models and a one-hour horizon.
- `/api/bootstrap` should return the sample forecast and measured comparison data.
- Enter a new PM2.5 value and confirm the forecast changes.
- Upload the original Beijing CSV, then forecast from that history. XGBoost should be approximately **11.92273** and Ridge **13.63978** µg/m³ for 1 January 2015 at 00:00 Beijing time.
- Confirm the chart, model selection, CSV export, example download and report link work.

Those sample values verify inference parity, not accuracy for an unobserved target. Forecast quality is established by the chronological 2014 backtests. Small platform-dependent floating-point differences are possible.

The restricted development workspace cannot resolve Vercel's API host or start a browser. A successful local preflight and the CLI's Ready status do not independently verify hosted inference or browser rendering. Open the production alias in a browser to check layout, chart interactions, model switching and export after the hosted checks pass. Test the alias in a private window to confirm the access available to someone without your Vercel session.

## Hosting references

- [Flask on Vercel](https://vercel.com/docs/frameworks/backend/flask)
- [Python runtime and bundle limits](https://vercel.com/docs/functions/runtimes/python)
- [Function request limits](https://vercel.com/docs/functions/limitations)
- [Deployment Protection scopes](https://vercel.com/docs/deployment-protection)
- [Testing protected deployments with Vercel CLI](https://vercel.com/docs/cli/curl)
- [Official XGBoost CPU-only installation](https://xgboost.readthedocs.io/en/stable/install.html#minimal-installation-cpu-only)
