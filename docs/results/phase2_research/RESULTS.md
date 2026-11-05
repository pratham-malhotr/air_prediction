# Phase 2 selected research results

Exploratory historical comparison; the 2014 Phase 1 results were already viewed.

Standalone Bi-LSTM and LSTM use complete eligible training folds. The hybrid has a separately calibrated base.
Validation preference: **lstm**. Comparison seed: **42**.

| Phase | Model | Hours | MAE | RMSE | R2 |
|---|---|---:|---:|---:|---:|
| holdout | bilstm | 8531 | 11.457 | 20.799 | 0.9503 |
| holdout | hybrid | 8531 | 11.483 | 20.919 | 0.9497 |
| holdout | lstm | 8531 | 11.299 | 20.726 | 0.9506 |
| holdout | persistence | 8531 | 11.926 | 22.033 | 0.9442 |
| holdout | ridge | 8531 | 11.825 | 21.063 | 0.9490 |
| holdout | seasonal_naive | 8531 | 67.296 | 98.754 | -0.1205 |
| holdout | xgboost | 8531 | 11.833 | 21.088 | 0.9489 |
| validation | bilstm | 8563 | 12.817 | 23.596 | 0.9422 |
| validation | hybrid | 8563 | 12.792 | 23.547 | 0.9424 |
| validation | lstm | 8563 | 12.684 | 23.462 | 0.9429 |
| validation | persistence | 8563 | 13.184 | 24.870 | 0.9358 |
| validation | ridge | 8563 | 13.141 | 23.809 | 0.9412 |
| validation | seasonal_naive | 8563 | 76.079 | 110.183 | -0.2604 |
| validation | xgboost | 8563 | 13.397 | 26.772 | 0.9256 |
