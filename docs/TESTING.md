# Testing & Quality Assurance Guide

## 1. Test Suite Overview

The repository features 116 automated unit, integration, and regression tests ensuring data integrity, chronological timing validity, machine learning accuracy, frontend correctness, and deployment safety.

### Test Catalog Summary

| Test File | Test Count | Scope |
|---|---|---|
| `tests/test_data_and_integration.py` | 14 | Dataset loading, calendar validation, provenance checksums, full synthetic backtests, corruption detection, PyTorch graceful degradation, and timedelta warnings. |
| `tests/test_deep_learning.py` | 23 | Sequence dataset 3D shape generation, LSTM/GRU/Bi-LSTM forecaster architectures, temporal attention, future-data invariance, frozen preprocessing, strict masks, chronological calibration, serialization and error propagation. |
| `tests/test_deployment.py` | 5 | Vercel package bundling, asset isolation, secret omission, and project link preservation. |
| `tests/test_eda_notes.py` | 2 | EDA metric calculation within training bounds without future leakage. |
| `tests/test_forecasting.py` | 23 | Target alignment, rolling window boundaries, missing value non-leakage, chronological fold boundaries, and configuration validation. |
| `tests/test_frontend.py` | 28 | Web dashboard endpoints, manual form validation, CSV upload limits, stateless context revalidation, available model discovery, and error reporting. |
| `tests/test_xgboost_optional.py` | 1 | End-to-end XGBoost integration backtest, parameter selection, and saved model inference. |
| `tests/test_phase2.py` | 4 | Complete synthetic Phase 2 runs, baseline preservation, identical-hour comparisons, provenance, saved-model inference, overwrite refusal and direct CLI inference without launcher thread settings. |
| `tests/test_phase2_research.py` | 2 | Selection freezes before comparison, independent full-window training, calibration boundaries, seed summaries and saved LSTM/Bi-LSTM/hybrid inference. |
| `tests/test_phase2_external.py` | 6 | Archive selection, field compatibility, declared target boundaries, common rows, immutable pre-download protocol and cached provenance. |
| `tests/test_phase2_production.py` | 8 | Container startup-reset retry, post-fork Gunicorn factory, port validation, HTTPS proxy/origin enforcement and rejection of source/data/model drift or synthetic bundles. |
| **Total** | **116** | **100% Pass Rate** |

---

## 2. Running Tests

### Standard Test Execution
Run the complete suite using pytest:
```bash
pytest -v
```

### Targeted Test Runs
```bash
# Run only core ML pipeline tests
pytest tests/test_forecasting.py tests/test_data_and_integration.py -v

# Run only web dashboard tests
pytest tests/test_frontend.py -v

# Run deep learning tests
pytest tests/test_deep_learning.py -v

# Run with fail-on-warning mode to verify zero deprecation warnings
pytest -W error::DeprecationWarning -v
```

---

## 3. macOS ARM64 & OpenMP Considerations

When running PyTorch alongside XGBoost on macOS, multiple OpenMP runtimes can cause thread deadlocks if multi-threading is unconstrained. `tests/conftest.py` automatically initializes the environment with:
```python
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
```
No manual flags are required when running through `pytest`.

---

## 4. Browser End-to-End Testing (Optional)

To execute automated browser QA against a live local instance:
```bash
node scripts/check_frontend_browser.cjs
```
Requirements:
- Node.js 18+
- Playwright (`npm install --no-save playwright && npx playwright install chromium`)

The script launches a temporary service on port 8016, simulates user interactions (range toggle, manual PM2.5 entry, model selection, CSV upload), verifies server responses, and saves screenshots to `artifacts/frontend_review/`.

## GitHub CI

The classical matrix checks Python 3.11/3.12 without optional PyTorch. A dedicated Python 3.13 job installs the recorded CPU PyTorch environment, runs the full suite with deprecation warnings as errors, and rejects skipped neural tests. Both jobs upload JUnit evidence. Remote job results are reviewed after publication.

## Phase 2 production and transfer evidence

`python scripts/check_phase2_production.py --launch` exercises a real local Gunicorn worker, all five saved forecasts, model readiness, the selected default, measured comparisons, PDF delivery, invalid-input handling and HTTP headers. It writes its result only after all checks pass. The manual CI workflow additionally builds the release-backed Docker image and repeats HTTP parity on Linux.

`python scripts/build_phase2_external_report.py` independently recomputes pooled and station scores from the frozen UCI 501 transfer predictions and checks common target hours, protocol/model/evaluator hashes, timestamps and the pre-download declaration. The external observations are evaluation data and are never used for fitting or selection.
