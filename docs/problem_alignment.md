# Alignment with Project 19 and the Phase 1 instructions

Faculty title: **Air Quality Forecasting via Gradient Boosting and LSTM Models**. The user requested only the current **Machine Learning Phase 1**. The local project therefore implements the XGBoost stage and its validation. The title's LSTM component is deferred.

| Project statement | Implementation |
|---|---|
| Short-horizon PM2.5 concentration forecast | Direct one-hour-ahead regression, explicitly configurable horizon |
| Historical pollutant and weather series | User-selected UCI 381 PM2.5 and its weather variables |
| Lags, rolling means, calendar, weather covariates | Explicit causal features, missingness flags, fold-fitted wind encoding |
| Train XGBoost regression | Three fixed candidates; squared-error objective, regularization, historical validation |
| Rolling chronological backtesting | Quarterly expanding fits with hourly forecast origins; no random split |
| Fixed-horizon MAE, RMSE and R squared | Pooled and per-quarter outputs on common observed targets |
| Persistence and seasonal naive comparisons | Both required models, with mathematically aligned seasonal source |
| Regular interval, sort before lagging | Hourly reindexing, duplicate rejection and sorting before feature construction |
| Scaling fitted on training dates only | Ridge's fold-local pipeline; tree model needs no numeric scaling |
| State AQI standard if mapping | AQI is optional and omitted; no invalid hourly-to-daily category conversion |
| Reproducible project and report | Modular source, checksums, tests, Colab, IEEE paper and presentation source |

Additional implementation choices are clearly design choices: Ridge as a linear comparator, h=1 hour, daily period m=24, 2013 selection, 2014 evaluation, quarterly refits, a fixed small grid, and optional validation-only feature ablations. The faculty did not prescribe those exact boundaries or candidate counts. They can be changed before final evaluation with a documented rationale.

GitHub creation and the invitation were authorized previously but could not be completed from this environment. The user later explicitly asked to leave GitHub aside and start locally. No remote repository or invitation is claimed. Exactly five real team members are required; identities are still pending. There is no supplied final evaluation date.
