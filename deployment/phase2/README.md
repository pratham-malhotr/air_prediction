# Phase 2 production runtime

This runtime serves all five saved models with Gunicorn. Startup checks the forecasting source, original dataset and three neural model hashes against the completed research run. PyTorch initializes inside a single worker after Gunicorn forks. There is no request-time fitting.

A clean Git checkout can build the container directly. The build restores the evaluated runtime files from the public `phase2-research-2026-11-05-r1` release, pinned to SHA-256 `d20646ee62de7aa8fcd7a4895e978e6150cb6faa717123cdb17d7b48b136faf8`. Local models, training outputs and credentials are excluded from the Docker build context.

The standard-library restorer verifies the whole archive, selected payload hashes, current forecasting source, dataset, neural model hashes and verified report before writing files. It restores 18 runtime files and never unpickles models. An existing file must match; a different experiment is preserved and causes an error. Use a fresh checkout when deploying another version. No retraining is required.

## Gunicorn

Use Python 3.13 and a virtual environment:

```bash
python -S scripts/restore_phase2_runtime.py
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

Run from the repository root, including a source-only clone:

```bash
docker build -f deployment/phase2/Dockerfile -t aircast-phase2 .
docker run --publish 10000:10000 aircast-phase2
python scripts/check_phase2_production.py --url http://127.0.0.1:10000
```

The image uses Python 3.13, CPU PyTorch, a non-root user, a writable temporary Matplotlib cache and an HTTP health check. Its build downloads the pinned public release over HTTPS and verifies it before installing inference dependencies. The manual GitHub CI workflow starts without local artifacts, builds this image directly from checkout, and compares Linux HTTP results for all five models against the recorded references. The uploaded CI evidence is named `phase2-container`.

For an offline local restore, supply the exact published ZIP:

```bash
python -S scripts/restore_phase2_runtime.py --bundle /path/to/air_prediction_phase2.zip
```

The restorer has no package dependencies. It rejects an unexpected archive digest, missing or duplicate selected members, mismatched training provenance, incompatible source, conflicting files and destinations outside the checkout. Unselected archive members are never extracted. Updating the pinned release requires a reviewed source change; there is no mutable “latest” model download.

A container build and smoke test verify hosting readiness. A public neural deployment still needs a host and resource allocation. The existing classical Vercel deployment is separate.

## Reverse proxy

Set `AIRCAST_TRUST_PROXY=1` only when exactly one trusted proxy terminates HTTPS and replaces incoming `X-Forwarded-Proto` and `X-Forwarded-Host` headers. This lets the application's same-origin check recognize its public HTTPS URL. Keep the backend inaccessible except through that proxy. The default does not trust forwarded headers.

## Evidence and references

The [local HTTP verification record](../../docs/results/phase2_production/verification.json) is also included as `artifacts/phase2_production/verification.json` in the runnable release. Remote container evidence is available from the [manual CI runs](https://github.com/pratham-malhotr/air_prediction/actions/workflows/ci.yml). These records report observed checks; they do not indicate a public deployment.

[Gunicorn application factories and command options](https://gunicorn.org/run/) · [Flask production deployment](https://flask.palletsprojects.com/en/stable/deploying/) · [Werkzeug proxy-header trust](https://werkzeug.palletsprojects.com/en/stable/middleware/proxy_fix/)
