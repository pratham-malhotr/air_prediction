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
   - The 0.025 $\mu\text{g/m}^3$ difference is not statistically significant. Ridge's competitive performance demonstrates that linear regularized lags capture the majority of the short-term autoregressive signal.

---

## 3. Runtime & Platform Boundaries

1. **Dual OpenMP Runtime on macOS ARM64**:
   - PyTorch and XGBoost bundle separate `libomp.dylib` libraries.
   - Unconstrained multithreading can cause `libomp` barrier deadlocks in child worker threads on macOS.
   - Single-thread environment flags (`OMP_NUM_THREADS=1`) must be maintained when running both frameworks in a single process.

2. **Serverless Deployment Size Limits**:
   - Vercel serverless Python functions enforce a 250 MB unzipped bundle limit.
   - PyTorch packages (>120 MB) are omitted from the Vercel production deployment (`deployment/requirements.txt`), which runs lean inference using XGBoost and Ridge.
   - Full deep learning and hybrid models require local execution or containerized hosting with the `[dl]` optional dependency.
