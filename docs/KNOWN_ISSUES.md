# Known Issues & Scientific Limitations

This document records technical limitations, scientific assumptions, and known system boundaries in the Air Quality Prediction project.

---

## 1. Domain & Dataset Scope

1. **Historical Beijing Specificity**:
   - The models were trained and backtested on the historical Beijing dataset (2010–2014).
   - Findings reflect meteorology, emission profiles, and sensor characteristics of Beijing during that five-year window.
   - Results cannot be assumed to generalize to other cities, topographies, or modern emission regimes without localized retraining.

2. **Sensor Location Disparity**:
   - In the UCI 381 dataset, PM2.5 readings were measured at the US Embassy in Chaoyang District, while weather observations were recorded at Beijing Capital International Airport (~20 km northeast).
   - Local microclimatic variations between the two sites represent an inherent data limitation.

3. **Absence of AQI Categorization**:
   - Official Air Quality Index (AQI) standards (such as US EPA or China MEP) require 24-hour truncated running averages or 1-hour NowCast calculations with specific breakpoint scales.
   - Aircast forecasts raw 1-hour PM2.5 concentrations in $\mu\text{g/m}^3$. It intentionally avoids mapping to arbitrary AQI categories to preserve scientific integrity.

---

## 2. Model Performance Characteristics

1. **Extreme High-Pollution Tail**:
   - For severe pollution events ($\ge 218\ \mu\text{g/m}^3$, the training 90th percentile), XGBoost RMSE is 39.959, compared with 38.557 for the naive Persistence baseline.
   - Decision-tree partitioning tends to underestimate abrupt, unprecedented pollution spikes outside the training distribution.

2. **Ridge vs. XGBoost Marginal Performance**:
   - On the 2014 holdout backtest (8,531 hours), Ridge achieved an RMSE of 21.063, while XGBoost achieved 21.088.
   - No significance test was performed; the small observed difference does not establish statistical significance. Ridge's competitive performance shows that regularized linear lags are a strong reference for this series.

3. **Phase 2 historical comparisons and hybrid limitations**:
   - The completed independent-model continuation has 2014 RMSE 20.726 for LSTM, 20.799 for Bi-LSTM and 20.919 for the hybrid, compared with 21.063 for Ridge on the same 8,531 hours.
   - LSTM is selected using mean 2013 validation RMSE across three neural seeds. The new 2014 ranking is exploratory and is not a statistical-significance claim.
   - Against its own frozen base, the selected correction reduces below-threshold error but increases peak error; pooled RMSE is nearly unchanged (20.917 to 20.919).
   - The independently trained networks and hybrid base have different training windows. Comparing them alone does not isolate the correction's effect.
   - The earlier shared-base baseline is preserved separately. See [the selected research report](phase2_research.md) and [the first corrected audit](phase2.md).

---

## 3. Runtime & Platform Boundaries

1. **Mixed numerical runtimes on macOS ARM64**:
   - XGBoost requires the external OpenMP runtime (`brew install libomp` on this Mac).
   - Loading a hybrid with unrestricted numerical thread settings caused a reproducible native crash during XGBoost deserialization in the audit environment.
   - Package startup now initializes single-thread CPU defaults before numerical imports, and optional neural exports load lazily. Training and dashboard launchers use the same policy; a clean-subprocess CLI test covers this failure.

2. **Serverless Deployment Size Limits**:
   - The existing Vercel bundle intentionally contains only the classical inference dependencies (`deployment/requirements.txt`). Its Python 3.13 runtime and resolved dependencies are pinned in `deployment/pyproject.toml` and `deployment/uv.lock`.
   - Vercel currently documents a 500 MB standard uncompressed Python function limit; larger-function availability has separate account/feature requirements. Check the [current runtime documentation](https://vercel.com/docs/functions/runtimes/python) before changing the deployment architecture.
   - The tested Phase 2 deployment path is the Gunicorn/container runtime with CPU PyTorch. It serves all five models locally and in Linux container CI; public neural hosting still requires an account connection and resource allocation.
