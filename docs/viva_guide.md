# Individual viva preparation

Each of the five members should be able to explain the entire pipeline, rerun it, and interpret its real results. These prompts are study aids, not a substitute for understanding or evidence of anyone's contribution.

1. **What exactly do we predict?** A numeric PM2.5 concentration in µg/m³ at t+h, with h fixed at one hour by default. This is regression, not AQI classification.
2. **What information exists at issuance?** PM2.5 and weather through hour t, assuming the hourly records are already published. The future weather record is unavailable; the future calendar is known.
3. **Why is lag zero valid?** Rows are indexed by origin t and the label is y(t+1). y(t) is historical relative to the label. If rows were indexed by target time, the offsets would need a different convention.
4. **What is the seasonal naive forecast at h=1?** y(t−23), which is 24 hours before the target time t+1. Explain why y(t−24) would address the wrong target hour.
5. **Why regularize the timestamp grid first?** After deleting a missing hour, a one-row shift no longer means a one-hour lag. Reindexing inserts missing slots before shifts.
6. **How are missing values handled?** Targets remain unfilled; evaluation excludes missing truth and missing baseline inputs. Features retain missingness. XGBoost handles numeric NaNs; Ridge fits medians and scaling on training data only. Explain the observed fraction in rolling windows.
7. **Why is a random train/test split unsuitable?** It can mix future regimes and measurements into training for older forecast origins. Our chronological splits respect label availability and issuance time.
8. **What happens at a quarterly boundary?** Fit before the first origin; use only training labels observed by that time. Keep the fitted model fixed for that quarter, while updating input histories hourly.
9. **Can earlier 2014 observations train a later 2014 fold?** Yes, under the explicitly specified rolling retraining policy. They were observed before those later predictions. Parameters remain frozen after 2013.
10. **Why persistence and seasonal naive?** They test whether complexity adds value beyond recent pollution and daily repetition. A trained model is not automatically better.
11. **Why Ridge?** Lagged predictors are correlated; L2 penalization stabilizes a linear comparator. Scaling and imputation are fitted separately in each fold.
12. **How does XGBoost work?** Explain additive trees, squared-error gradients, Hessians, regularized leaf scores, learning rate, tree depth, minimum child weight, row/column sampling, and lambda.
13. **Why no scaling for XGBoost?** Tree partitions do not rely on Euclidean distances or coefficient penalties on the raw feature scale. Ridge's coefficient regularization does depend on scale.
14. **How were parameters selected?** A fixed three-candidate grid evaluated on four chronological 2013 folds, choosing pooled RMSE. No test quarter is an early-stopping set.
15. **What do MAE, RMSE and R² tell us?** MAE is typical absolute error; RMSE emphasizes large misses; R² is relative squared-error reduction against the evaluation targets' mean. Negative R² is possible and should be retained.
16. **Why pooled rather than average quarterly RMSE?** Square roots and quarterly variances make simple averaging a different quantity. Pooling gives one metric over all common scored hours; report quarterly metrics alongside it.
17. **Why might peak pollution errors be worse?** Extremes can be rare, shifts can be abrupt, and squared-error models may smooth changes. Check actual error strata before asserting this dataset exhibits the pattern.
18. **Why not delete every IQR outlier?** Pollution peaks can be valid events that matter for forecasting. Audit them; do not automatically remove important targets.
19. **What does feature importance prove?** It summarizes splits in a fitted model, not causal effects or stability across all folds. Correlated features can share or obscure importance.
20. **What do residual ACF plots reveal?** Remaining temporal structure can indicate missed dynamics. Use the complete hourly grid so absent targets do not compress time. ACF alone is not a calibrated uncertainty interval.
21. **Why not output an AQI category?** The forecast is hourly concentration; an AQI standard has specific averaging and calculation rules. An optional AQI extension needs a valid standard and aggregation procedure.
22. **Does this forecast today's air in India?** No. It is evaluated on one historical Beijing series. Present-day, other-city, and sensor-location transfer need new data and validation.
23. **What work is Phase 2?** LSTM/temporal CNN and a technically justified hybrid. There are no Phase 2 performance claims in this submission.
24. **What evidence makes a run reproducible?** CSV checksum, provenance, configuration, versions, forecast rows, fold boundaries, selection record and outputs. Demonstrate how to run the notebook and locate those files.
25. **What if XGBoost loses to persistence?** Report it. Analyze errors, seasonality and horizon strength. Do not replace splits or cherry-pick a quarter to create a better headline.

Practice: trace one concrete row through feature creation, baseline alignment, train-label eligibility, preprocessing, prediction, and metric computation. Then show your own contribution and a result or design decision you can defend from the actual run.
