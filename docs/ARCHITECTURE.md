# Architecture & System Design

## 1. System Overview

CSAI Project 19 (`air-prediction`) forecasts hourly PM2.5 concentrations ($\mu\text{g/m}^3$) one hour ahead based on historical air quality and weather telemetry.

```
       +-------------------------------------------------------------+
       |             UCI Beijing PM2.5 Dataset (UCI 381)             |
       |               43,824 Hourly Records (2010–2014)             |
       +-------------------------------------------------------------+
                                      |
                                      v
                       +-----------------------------+
                       |    data.py: Regularize &    |
                       | Validate (Zero Imputation)  |
                       +-----------------------------+
                                      |
                                      v
                       +-----------------------------+
                       |   features.py: 70+ Lags,    |
                       | Rolling Stats, Weather, Cal |
                       +-----------------------------+
                                      |
                 +--------------------+--------------------+
                 |                                         |
                 v                                         v
   +---------------------------+             +---------------------------+
   |   Classical ML Pipeline   |             |   Phase 2 Deep Learning   |
   |      (models.py)          |             |     (deep_learning.py)    |
   |  - Persistence            |             |  - 3D Sequence Window     |
   |  - Seasonal Naive (24h)   |             |  - Bi-LSTM + Attention    |
   |  - Ridge (alpha=10)       |             |  - Hybrid Residual Model  |
   |  - XGBoost (hist)         |             |    (DL Base + XGB Resid)  |
   +---------------------------+             +---------------------------+
                 |                                         |
                 +--------------------+--------------------+
                                      |
                                      v
                       +-----------------------------+
                       |   splits.py: Chronological  |
                       | Expanding Quarterly Folds   |
                       | (Train 10-12, Val 13, Test) |
                       +-----------------------------+
                                      |
                                      v
                       +-----------------------------+
                       |    Saved Model Artifacts    |
                       |    (artifacts/phase1/*.joblib)|
                       +-----------------------------+
                                      |
                 +--------------------+--------------------+
                 |                                         |
                 v                                         v
   +---------------------------+             +---------------------------+
   |       Aircast Web UI      |             |      Vercel Serverless    |
   | (frontend/app.py, service)|             | (deployment/app.py, json) |
   |  - Interactive SVG Chart  |             |  - Stateless window flow  |
   |  - Manual / CSV Input     |             |  - Lean CPU inference     |
   |  - Dark/Light Theme       |             |  - CDN-backed public asset|
   +---------------------------+             +---------------------------+
```

---

## 2. Core Modules (`src/air_prediction/`)

### `data.py`
- Enforces strict calendar timestamp validation (checking year, month, day, hour bounds).
- Restores missing hourly slots without imputing pollution measurements, preserving true data gaps.
- Verifies SHA256 hashes and provenance metadata sidecars (`.provenance.json`).

### `features.py`
- Constructs features indexed strictly by forecast origin $t$:
  - **PM2.5 Lags**: $t=0, 1, 2, 3, 6, 12, 24, 48, 72, 168$ hours and missingness indicators.
  - **Rolling Statistics**: Mean, standard deviation (ddof=0), and observed data fraction over windows of $3, 6, 12, 24, 72, 168$ hours.
  - **Calendar Encodings**: Cyclic sine/cosine representations for target hour (period 24), weekday (period 7), and month (period 12), plus weekend indicator.
  - **Meteorological Lags**: Temperature (`TEMP`), dew point (`DEWP`), pressure (`PRES`), accumulated wind (`Iws`), snow (`Is`), and rain (`Ir`) at lags 0, 1, and 24 hours.
  - **Wind Direction**: Encoded `cbwd` with explicit missingness category.
- Establishes non-leaking naive baselines:
  - **Persistence**: Forecast equals $y(t)$.
  - **Daily Seasonal Naive**: Forecast equals $y(t - 23)$ for target $t + 1$.

### `splits.py`
- Implements strict chronological quarterly boundaries:
  - 2010–2012: Initial training and Exploratory Data Analysis.
  - 2013: Four expanding quarterly folds for hyperparameter selection.
  - 2014: Four expanding quarterly backtests with fixed configurations.
- Enforces label boundary constraint: no training row extends past the origin of the first evaluation hour.

### `models.py`
- Fold-local transformation pipelines:
  - **Ridge**: Fold-fitted median imputer and StandardScaler with regularizer $\alpha=10.0$.
  - **XGBoost**: Histogram tree method (`hist`) with native missing-value path traversal, preserving raw distribution gaps.

### `deep_learning.py`
- **SequenceAirDataset**: PyTorch sliding window generator producing 3D sequence tensors of shape `(batch, seq_len, features)`.
- **TemporalForecaster**: Modular neural forecaster supporting LSTM, GRU, and Bidirectional LSTM (`Bi-LSTM`) coupled with additive Temporal Attention.
- **HybridResidualModel**: Dual-stage architecture:
  1. Deep learning sequence model predicts base trajectory $y_{\text{dl}}$.
  2. Tabular gradient boosted trees fit the residual errors $r = y_{\text{actual}} - y_{\text{dl}}$.
  3. Combined forecast: $\hat{y} = \max(y_{\text{dl}} + \hat{r}, 0)$.

---

## 3. Web Service & Frontend Architecture (`frontend/`)

### `frontend/service.py`
- Encapsulates prediction, history management, and validation.
- Loads only available models from disk, reporting active models via `/api/bootstrap`.
- Validates manual numeric inputs against physical bounds (nonnegative PM2.5, positive pressure, finite measurements).
- Stateless history protocol: returns the normalized recent 169-hour window to the browser; subsequent predictions submit this window, enabling seamless multi-instance horizontal scaling without shared sessions.

### `frontend/static/` & `frontend/templates/`
- Zero-build vanilla architecture: Semantic HTML5, pure CSS3 with custom properties and dark/light themes, and vanilla ECMAScript.
- Reactive SVG chart: renders historical line, data gaps (discontinuous path segments), forecast point, and interactive hover tooltip without heavy external chart libraries.

---

## 4. Concurrency & Runtime Safety

On macOS ARM64 systems, simultaneous use of PyTorch and XGBoost multi-threading can cause OpenMP thread pool deadlocks due to conflicting `libomp.dylib` shared libraries. The system prevents this by setting:
```bash
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
```
in test fixtures (`tests/conftest.py`) and application launch scripts (`scripts/serve_frontend.py`, `scripts/train_phase2_hybrid.py`).
