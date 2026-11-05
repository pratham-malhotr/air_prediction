# Phase 2 results

Exploratory historical comparison: 2014 results were previously viewed in Phase 1.

The standalone Bi-LSTM and hybrid share a frozen base trained before calibration.
The correction head uses only the later calibration block. Classical results reuse audited Phase 1 predictions.

| Phase | Model | Hours | MAE | RMSE | R2 |
|---|---|---:|---:|---:|---:|
| holdout | bilstm | 8531 | 12.697 | 21.655 | 0.9461 |
| holdout | hybrid | 8531 | 12.805 | 22.038 | 0.9442 |
| holdout | persistence | 8531 | 11.926 | 22.033 | 0.9442 |
| holdout | ridge | 8531 | 11.825 | 21.063 | 0.9490 |
| holdout | seasonal_naive | 8531 | 67.296 | 98.754 | -0.1205 |
| holdout | xgboost | 8531 | 11.833 | 21.088 | 0.9489 |
| validation | bilstm | 8563 | 15.211 | 26.892 | 0.9249 |
| validation | hybrid | 8563 | 14.855 | 27.141 | 0.9235 |
| validation | persistence | 8563 | 13.184 | 24.870 | 0.9358 |
| validation | ridge | 8563 | 13.141 | 23.809 | 0.9412 |
| validation | seasonal_naive | 8563 | 76.079 | 110.183 | -0.2604 |
| validation | xgboost | 8563 | 13.397 | 26.772 | 0.9256 |
