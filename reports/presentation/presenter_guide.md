# Phase 1 presentation guide

Use the Beamer slides as a starting presentation and replace pending content after the real Colab run. The faculty provided no fixed slide count or time limit. Adjust to the announced evaluation slot.

1. Define the origin, horizon, target units and information available at issuance.
2. Cite the selected dataset and literature; explain the weather/PM2.5 location mismatch.
3. Present actual training EDA: missingness, distributions, temporal profiles, correlations and ACF. Distinguish association from causation.
4. Trace one forecast row through lags and rolling windows. Explain why lag zero is historical and the seasonal source is t−23.
5. Explain the model objectives and regularization. Justify the linear comparator and simple references.
6. Draw the chronological train/validation/holdout timeline; describe quarterly retraining and hourly observation updates.
7. Show common-target MAE, RMSE and R², the evaluated coverage, and per-quarter variation. Report baseline-relative changes without cherry-picking.
8. Show peak-error strata and residual diagnostics. Explain any validation-only ablations if run.
9. Demonstrate the notebook, reproducibility files and a saved-model forecast. Explain that the archive's next-hour forecast is historical, not today's weather.
10. State measured conclusions and limitations. Complete five actual contribution records and prepare every member for questions across the whole pipeline.

Do not substitute the synthetic demo's metrics for the real experiment, invent a model improvement, or claim that the report/presentation is submission-ready before names, measured findings and compiled files have been reviewed.
