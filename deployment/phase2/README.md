# Phase 2 production runtime

This runtime serves all five saved models with Gunicorn. Startup checks the forecasting source, original dataset and three neural model hashes against the completed research run. PyTorch initializes inside a single worker after Gunicorn forks. There is no request-time fitting.

The runnable release ZIP supplies the ignored `artifacts/phase1/*.joblib` and `artifacts/phase2_research/*.joblib` files. Extract that bundle into a clean directory before running these commands. A source-only Git clone does not include trained weights.

## Gunicorn

Use Python 3.13 and a virtual environment:

```bash
python -m pip install -r deployment/phase2/requirements.txt
python scripts/serve_phase2_production.py
```

`PORT` defaults to `10000`; `AIRCAST_HOST` defaults to `0.0.0.0`. One worker and one thread bound the model memory and preserve the tested inference configuration. On macOS, XGBoost may require `brew install libomp`.

Verify the actual server against the saved reference predictions:

```bash
python scripts/check_phase2_production.py --url http://127.0.0.1:10000
```

For a temporary local smoke test that starts and stops Gunicorn:

```bash
python scripts/check_phase2_production.py --launch
```

## Container

Run from the extracted release root:

```bash
docker build -f deployment/phase2/Dockerfile -t aircast-phase2 .
docker run --publish 10000:10000 aircast-phase2
python scripts/check_phase2_production.py --url http://127.0.0.1:10000
```

The image uses Python 3.13, CPU PyTorch, a non-root user, a writable temporary Matplotlib cache and an HTTP health check. Its context excludes local environments, Git metadata, checkpoints and external raw data. The manual GitHub CI workflow downloads and verifies the release model bundle, builds this image, and compares Linux HTTP results for all five models against the saved references. The uploaded CI evidence is named `phase2-container`.

A container build and smoke test verify hosting readiness. A public neural deployment still needs a host and resource allocation. The existing classical Vercel deployment is separate.

## Reverse proxy

Set `AIRCAST_TRUST_PROXY=1` only when exactly one trusted proxy terminates HTTPS and replaces incoming `X-Forwarded-Proto` and `X-Forwarded-Host` headers. This lets the application's same-origin check recognize its public HTTPS URL. Keep the backend inaccessible except through that proxy. The default does not trust forwarded headers.

## Evidence and references

The [local HTTP verification record](../../docs/results/phase2_production/verification.json) is also included as `artifacts/phase2_production/verification.json` in the runnable release. Remote container evidence is available from the [manual CI runs](https://github.com/pratham-malhotr/air_prediction/actions/workflows/ci.yml). These records report observed checks; they do not indicate a public deployment.

[Gunicorn application factories and command options](https://gunicorn.org/run/) · [Flask production deployment](https://flask.palletsprojects.com/en/stable/deploying/) · [Werkzeug proxy-header trust](https://werkzeug.palletsprojects.com/en/stable/middleware/proxy_fix/)
