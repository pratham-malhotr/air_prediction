# Changelog

All notable changes to the Air Quality Prediction project are documented here.

## 2026-11-05 — Deployment from a clean checkout

- Added a standard-library runtime restorer pinned to the published release digest, with selective extraction and checks for source, dataset, model and report provenance.
- Preserved existing differing experiments and excluded local artifacts and credentials from the Docker context.
- Made the production image restore its own runtime during build; manual CI now builds directly from a checkout without preloading model files.
- Verified all five forecasts through Gunicorn from a fresh restored checkout; the strict suite has 129 passing tests, including 13 restoration regressions.

## 2026-11-05 — Post-publication Phase 2 continuation

- Fixed exact decimal CSV round trips exposed by Linux CPU CI, added a regression and repeated the full research run under the corrected source.
- Completed a frozen external transfer test on all 12 UCI 501 stations: 217,156 common station-hours, no fitting or model reselection, preserved reference bytes, and independently recomputed pooled/station metrics.
- Added Gunicorn serving, startup provenance checks, a non-root CPU Docker recipe, HTTP forecast parity checks and a manual Linux container CI job.
- Hardened container readiness against transient TCP resets during worker startup, with a regression.
- Expanded the strict local suite to 116 passing tests and included external evidence and production instructions in the runnable release.

## 2026-11-05 — Independent Phase 2 research and submission

- Added full-window independent recurrent benchmarks, an eight-configuration validation search, matched LSTM/Bi-LSTM attention ablations, and three-seed stability checks.
- Added level/delta targets, corrected bidirectional terminal-state extraction, and selected hybrid calibration/regularization/correction weight using validation only.
- Frozen selection prefers LSTM; exploratory 2014 RMSE is 20.726 (LSTM), 20.799 (Bi-LSTM), 20.919 (hybrid) and 21.063 (Ridge). The correction still worsens peaks relative to its own frozen base.
- Added the IEEE report, twelve-slide editable deck and PDF, executed notebook, scientific figures and verified package builder.
- Local demo prefers the validation-selected model, exposes all five saved models, and opens the Phase 2 report. Expanded verification to 101 passing tests and completed browser checks.

## 2026-11-04 — Phase 2 audit and measured baseline

- Corrected future-data leakage in neural preprocessing, batch-dependent imputation, permissive training-mask fallback and omitted wind encoding.
- Added chronological out-of-sample residual calibration, seeded/standardized neural training, explicit correction failures and isolated Phase 2 experiment provenance.
- Completed real-data quarterly validation/comparison, saved both models and published compact results and five report figures. Ridge remains better overall; the hybrid does not improve pooled RMSE.
- Fixed sequence history and CLI/dashboard prediction parity; added dynamic Phase 2 comparison tables and mobile table containment.
- Expanded verification to 95 passing tests and added Phase 2 browser scenarios.

## [0.2.0] - 2026-10-25

### Fixed
- **PyTorch Import Guarding (`src/air_prediction/__init__.py`)**: Resolved critical `ModuleNotFoundError: No module named 'torch'` by guarding deep learning imports with graceful fallbacks. Lean environments (including Vercel serverless deployment) and baseline test suites can now import `air_prediction` without requiring heavy PyTorch installations.
- **macOS OpenMP Collision Deadlock (`tests/conftest.py`)**: Configured single-thread environment variables (`OMP_NUM_THREADS="1"`, `OPENBLAS_NUM_THREADS="1"`, `MKL_NUM_THREADS="1"`) in test fixtures to resolve the dual-OpenMP runtime deadlock between PyTorch (`libomp.dylib`) and system/XGBoost on Apple Silicon macOS ARM64.
- **Frontend Model Availability & Selection Disconnect (`frontend/service.py`, `frontend/static/app.js`)**:
  - `ForecastService.bootstrap()` now returns only models that are actually loaded and present in `self.models`.
  - `app.js` dynamically populates the `<select id="model">` options to reflect available models instead of hardcoding unavailable Phase 2 options.
  - Corrected forecast card badge (`#result-model`) to show the actual selected model name (`Hybrid (BiLSTM + XGBoost)`, `Bi-LSTM`, `XGBoost`, or `Ridge`) rather than hardcoding "XGBoost".
  - Made side-by-side comparison text robust when alternate models are selected.
  - `ForecastService.forecast()` provides an informative, actionable error message if an unknown or unavailable model is requested.
- **NumPy 2.x & Pandas 2.x Timedelta Deprecation Warnings**: Replaced ambiguous `pd.Timedelta(hours=...)` additions against `pd.DatetimeIndex` with explicit `pd.to_timedelta(..., unit="h")` and `pd.to_timedelta(..., unit="D")` across `src/air_prediction/features.py`, `splits.py`, `results.py`, `experiment.py`, `frontend/service.py`, and test suites, eliminating 311 `DeprecationWarning` instances.
- **Hardcoded Local Paths in Documentation (`docs/frontend.md`, `docs/deployment.md`)**: Replaced personal developer machine paths (`/Users/prathammalhotra/...`, `/opt/anaconda3/...`) with clean, portable relative commands.
- **Vercel Deployment Verification Model Assertion (`scripts/check_vercel.py`)**: Relaxed strict equality check on available models to subset check `{"xgboost", "ridge"}.issubset(...)` so future model additions do not fail health checks.

### Added
- **PyPI Extras Configuration (`pyproject.toml`)**: Added `dl = ["torch>=2.0,<3"]` and `all = ["pytest>=8,<9", "Flask>=3.1,<4", "torch>=2.0,<3"]` optional dependencies. Synchronized version string to `0.2.0`.
- **Regression Test Coverage**:
  - `tests/test_data_and_integration.py::test_graceful_import_when_torch_unavailable`: Verifies `air_prediction` imports cleanly when PyTorch is not installed.
  - `tests/test_data_and_integration.py::test_build_features_no_timedelta_deprecation_warnings`: Asserts zero `DeprecationWarning` during feature generation.
  - `tests/test_frontend.py::test_bootstrap_returns_only_available_models`: Validates bootstrap payload filters unavailable models.
  - `tests/test_frontend.py::test_forecast_rejects_unavailable_model_with_informative_error`: Verifies HTTP 400 with helpful model options on invalid selection.
  - `tests/test_deep_learning.py`: Added `pytest.importorskip("torch")` to allow graceful test skipping on systems without deep learning packages.
- **Technical Documentation**:
  - `docs/ARCHITECTURE.md`: Complete system architecture guide covering data, ML, deep learning, frontend, and deployment layers.
  - `docs/TESTING.md`: QA and testing methodology, commands, test catalog, and concurrency handling.
  - `docs/KNOWN_ISSUES.md`: Transparent record of scientific limitations, dataset specifics, and runtime considerations.
