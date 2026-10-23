# Phase 1 experimental protocol

The task is direct supervised regression: estimate a future PM2.5 concentration from the environmental history available at issuance. The standard configuration fixes h=1 hour. The implementation also aligns larger integer horizons correctly; results at different horizons must be reported as separate experiments.

## Time convention and features

Use the dataset's Beijing local clock (Asia/Shanghai, UTC+8). At origin t, the input may include y(t), weather(t), and earlier records; the label is y(t+h). The origin's hourly measurement is assumed published before issuance. Operational sensor latency is not provided and remains a limitation.

1. Parse the four date columns, reject duplicates, sort, and reindex to the complete hourly grid. Reindexing preserves the meaning of an hour offset. ID `No` is excluded.
2. Keep PM2.5 lags at offsets 0, 1, 2, 3, 6, 12, 24, 48, 72, and 168 hours, with missingness indicators.
3. Compute trailing means, population standard deviations, and observed fractions over 3, 6, 12, 24, 72, and 168 hours, ending at t. Missing PM2.5 remains missing; the rolling mean averages available observations, with the observed fraction exposing reduced support. An entirely missing window has a missing mean and standard deviation.
4. Encode target hour, weekday, and month using sine/cosine pairs; add a weekend indicator. Future calendar values are known at issuance. Month encoding is a coarse annual proxy; it does not encode a smooth day-of-year cycle.
5. Use the six numeric weather variables at offsets 0, 1, and 24 and current weather-missingness flags. Encode wind direction with fold-fitted one-hot encoding; missing wind is a separate category. Unseen test categories map to all-zero indicators without changing training encoders.

For a trailing window w, mean(t,w)=sum of observed y values in [t−w+1,t], divided by their observed count. No centered rolling window, backward fill, global interpolation, target imputation, or future observed weather is allowed. A perturbation test replaces every future pollutant/weather value and verifies that earlier features remain identical.

## Baselines and linear comparator

Persistence: yhat(t+h | t)=y(t).

Seasonal naive with period m: yhat(t+h | t)=y(t+h−m ceil(h/m)). At h=1 and m=24 the source is t−23, rather than t−24. The general formula repeats the latest observed seasonal cycle for any positive horizon.

Ridge regression minimizes sum_i (y_i−b−x_i' beta)^2 + alpha ||beta||², with alpha fixed at 10. Correlated lag features motivate regularization. Training-only median imputation and standardization support the linear model. XGBoost does not need numeric scaling; it uses native missing-value handling. Encoders and learned preprocessing are newly fitted in every fold.

## XGBoost theory

Boosting builds an additive predictor F_K(x)=b+sum_k eta f_k(x), where each tree refines the current prediction. For squared error, the gradients point toward residual corrections. The XGBoost objective adds tree complexity and leaf-weight penalties to the loss. A second-order expansion uses gradient g_i and Hessian h_i; a leaf's regularized optimum is −G/(H+lambda), where G and H sum derivatives in that leaf. Candidate splits trade loss reduction against complexity. Depth, minimum child weight, learning rate, subsampling, and L2 regularization limit model complexity. This is an explanation of the model class, not a derivation of a claim that it will win on this dataset. See [Chen and Guestrin](https://arxiv.org/abs/1603.02754) and the [official estimator documentation](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html).

The project selects among three fixed parameter candidates by pooled chronological validation RMSE. Estimator counts are part of that grid. There is no early stopping using final holdout data. Random seeds and CPU thread counts are recorded. A small fixed grid is transparent but does not establish a global optimum.

## Chronological validation and rolling backtesting

Initial training spans 2010–2012. Model selection evaluates four 2013 quarters with expanding training. After selecting parameters, four 2014 quarterly evaluations use those frozen parameters. For a block whose first target time is B, the first origin is B−h. Include a training example only if its target time is at most B−h, so every label is already observed at fit time. This boundary matters for horizons greater than one.

Fit once per quarter, then issue a new fixed-horizon prediction each hour using updated observed histories. Later quarters may include previously observed validation/holdout records in training. This is the specified operational rolling policy, not a static model evaluated all year without updates. No model is trained on observations later than the start of its evaluation block.

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

Keep the initial training-only EDA and candidate grid fixed before reviewing final holdout results. If you change them afterward, call that a new exploratory analysis and obtain a new evaluation strategy rather than claiming the same holdout remains untouched. The code cannot enforce a person's research decisions; the run manifest and output directories preserve evidence.

Every completed run writes its data hash, source, configuration, parameters, package versions, split audits, predictions, metrics, model artifacts, and narrative summary. Final deployment models are refitted on all observed archive labels only after the backtest outputs have been computed. Their feature importance describes that deployment fit, not a held-out feature selection exercise. Inference rejects a history ending earlier than the deployment model's training endpoint.

This Phase 1 does not implement AQI or deep learning. In particular, an hourly forecast cannot be directly relabeled as a valid daily AQI. Any later AQI extension must state its jurisdiction, standard version, pollutant averaging period, and treatment of incomplete periods.
