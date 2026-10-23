# Phase 1 experimental protocol

Our task is to estimate the next PM2.5 concentration from the pollution and weather history available when the forecast is made. We use direct regression with a fixed horizon of one hour. The code also supports other integer horizons, but each horizon needs its own reported comparison.

## Time convention and features

We use the dataset's Beijing local clock (Asia/Shanghai, UTC+8). At origin t, our input includes y(t), weather(t), and earlier records; the label is y(t+h). We assume the current hourly measurement is available before issuance. The archive does not provide publication delays, so we cannot verify that assumption for a live deployment.

1. Parse the four date columns, reject duplicates, sort, and reindex to the complete hourly grid. Reindexing preserves the meaning of an hour offset. ID `No` is excluded.
2. Keep PM2.5 lags at offsets 0, 1, 2, 3, 6, 12, 24, 48, 72, and 168 hours, with missingness indicators.
3. Compute trailing means, population standard deviations, and observed fractions over 3, 6, 12, 24, 72, and 168 hours, ending at t. Missing PM2.5 remains missing; the rolling mean averages available observations, with the observed fraction exposing reduced support. An entirely missing window has a missing mean and standard deviation.
4. Encode target hour, weekday, and month using sine/cosine pairs; add a weekend indicator. Future calendar values are known at issuance. Month encoding is a coarse annual proxy; it does not encode a smooth day-of-year cycle.
5. Use the six numeric weather variables at offsets 0, 1, and 24 and current weather-missingness flags. Encode wind direction with fold-fitted one-hot encoding; missing wind is a separate category. Unseen test categories map to all-zero indicators without changing training encoders.

For a trailing window w, mean(t,w) is the sum of observed y values in [t−w+1,t], divided by their observed count. We use trailing windows and keep missing observations explicit. A perturbation test changes future pollutant/weather values and checks that earlier features stay identical. [Our analysis](analysis.md) connects these choices to the measured EDA findings.

## Baselines and linear comparator

Persistence: yhat(t+h | t)=y(t).

Seasonal naive with period m: yhat(t+h | t)=y(t+h−m ceil(h/m)). At h=1 and m=24 the source is t−23, rather than t−24. The general formula repeats the latest observed seasonal cycle for any positive horizon.

Ridge regression minimizes sum_i (y_i−b−x_i' beta)^2 + alpha ||beta||², with alpha fixed at 10. Correlated lag features motivate regularization. Training-only median imputation and standardization support the linear model. XGBoost does not need numeric scaling; it uses native missing-value handling. Encoders and learned preprocessing are newly fitted in every fold.

## XGBoost theory

Boosting builds an additive predictor F_K(x)=b+sum_k eta f_k(x), where each tree refines the current prediction. For squared error, the gradients point toward residual corrections. The XGBoost objective adds tree complexity and leaf-weight penalties to the loss. A second-order expansion uses gradient g_i and Hessian h_i; a leaf's regularized optimum is −G/(H+lambda), where G and H sum derivatives in that leaf. Candidate splits trade loss reduction against complexity. Depth, minimum child weight, learning rate, subsampling, and L2 regularization limit model complexity. This gives us a model that can represent nonlinear feature relationships while controlling tree complexity. See [Chen and Guestrin](https://arxiv.org/abs/1603.02754) and the [official estimator documentation](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html).

We compare three fixed parameter candidates using pooled chronological validation RMSE. The number of trees is included in the grid. We keep the grid small so each comparison is easy to inspect and reproduce; we are not claiming a globally optimal setting. Random seeds and thread counts are recorded. Final holdout data is not used for early stopping.

## Chronological validation and rolling backtesting

Our initial training period is 2010–2012. We use four expanding quarterly folds in 2013 to select parameters, then keep those settings fixed for the four 2014 evaluation blocks. For a block whose first target time is B, the first origin is B−h. A training example is eligible only if its target time is at most B−h. That ensures its label is already known when the model is fitted, including when the horizon is greater than one.

We fit once per quarter and issue a new fixed-horizon prediction each hour using the updated observed history. Later quarters can include observations from earlier quarters because those readings are already available by then. The parameter choices remain fixed. This lets us evaluate the same retraining rule throughout the test year.

All models share the mask: enough timestamp history, observed target, observed persistence input, and observed seasonal-naive input. This excludes hours with unavailable baseline observations and can bias evaluation toward better-observed periods; report coverage. Native XGBoost could predict some excluded hours, but those predictions are outside this fair-comparison metric table. Missingness exclusion counts overlap. Targets are never synthesized for scoring. The first 168 elapsed hours are a warm-up even if some measurements within that history are missing.

Optional feature ablations compare lags only; lags plus rolling/calendar features; and the complete environmental feature set. They use 2013 only with the selected parameters held fixed. This isolates changes under one parameter setting, but does not claim each feature group was separately optimized.

## Metrics and error analysis

For n common observed targets and errors e_i=yhat_i−y_i:

- MAE = mean(|e_i|), in µg/m³.
- RMSE = sqrt(mean(e_i²)), in µg/m³. It weights large misses more heavily.
- R² = 1−sum(e_i²)/sum((y_i−mean(y))²), dimensionless. It may be negative; it is undefined for constant ground truth and saved as null/blank in that case.
- Bias = mean(e_i), where positive means overprediction.

Pooled metrics use all out-of-fold predictions rather than averaging quarterly RMSE or R². Per-fold metrics expose seasonal variability. Percent RMSE improvement over persistence is 100×(RMSE_persistence−RMSE_model)/RMSE_persistence, when the baseline denominator is positive. Negative improvement is reported honestly.

Inspect prediction scatter, hourly residual histograms, and residual autocorrelation on the regular hourly grid. Stratify errors by target month, hour, and whether actual PM2.5 exceeds the initial training period's 90th percentile. That percentile is fixed without consulting the holdout distribution. The historical plots and importance scores are descriptive; this implementation does not supply statistical significance or calibrated prediction intervals.

## Freeze, reproduce, and interpret

We keep the initial EDA scope and candidate grid fixed before reviewing the holdout. If we revise the approach after seeing those results, we need to record that as exploratory work and reconsider the evaluation. Keeping separate output directories and a run manifest helps us track which settings produced each result.

Every completed run writes its data hash, source, configuration, parameters, package versions, split audits, predictions, metrics, model artifacts, and narrative summary. Final deployment models are refitted on all observed archive labels only after the backtest outputs have been computed. Their feature importance describes that deployment fit, not a held-out feature selection exercise. Inference rejects a history ending earlier than the deployment model's training endpoint.

This Phase 1 does not implement AQI or deep learning. In particular, an hourly forecast cannot be directly relabeled as a valid daily AQI. Any later AQI extension must state its jurisdiction, standard version, pollutant averaging period, and treatment of incomplete periods.
