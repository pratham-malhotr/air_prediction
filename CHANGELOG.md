# Changelog

All notable changes to the Air Quality Prediction project are documented here.

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
