# Phase 1 requirement mapping

We use this checklist to keep the implementation, report, and presentation aligned with the project brief.

## Project 19 problem statement

| Requirement | Implementation |
|---|---|
| Short-horizon PM2.5 regression | Direct one-hour-ahead forecast from observations available at issuance |
| Selected dataset | UCI Beijing PM2.5, dataset 381, with historical weather variables |
| Lags, rolling statistics, calendar and weather covariates | Explicit causal features, missingness indicators and fold-fitted wind encoding |
| XGBoost regression | Regularized tree ensemble with historical parameter selection |
| Rolling chronological backtesting | Expanding quarterly fits and hourly forecast origins |
| Persistence and seasonal naive | Both reference forecasts aligned to the same target horizon |
| MAE, RMSE and R² | Pooled and quarterly metrics on identical observed targets |
| Regular interval and training-only preprocessing | Hourly reindexing before lagging; fold-local encoding/imputation/scaling |
| Optional AQI visualization | Omitted; no hourly-to-daily AQI conversion is assumed |

## Faculty evaluation

| Criterion | Project material | Status |
|---|---|---|
| Literature review | `literature_review.md` and report bibliography | Written review and citations complete; understanding assessed in the viva |
| Dataset quality and EDA | Schema/timestamp audit, missingness, distributions, calendar profiles, correlations and autocorrelation | Real CSV analysed; findings in `analysis.md` |
| Feature engineering | Historical features, observed fractions and explicit target alignment | 70 documented input columns; worked preview and validation feature comparison |
| Theoretical rigor | Baseline formulas, Ridge/boosting objectives, regularization, metric definitions and split logic | Explanations complete; individual justification assessed in the viva |
| Model application | XGBoost, required naive references, Ridge comparator and chronological selection/backtests | Real four-model backtests, quarterly/peak errors and interpretation complete |
| Repository and code quality | Modular Python source, configuration, tests, CI and provenance | Local implementation verified; GitHub publication and contribution evidence not verified |
| Presentation | Editable PowerPoint, matching PDF, thirteen graph assets, Python sources and presenter guide | Ten detailed slides covering completed Phase 1 work, with team names/enrollment numbers on the final thank-you slide |
| Individual viva | `viva_guide.md` | Preparation guide complete; individual assessment remains to be done |
| LaTeX/IEEE report | IEEEtran source, bibliography and print-sized figures | Expanded to seven pages with measured findings, implementation details and all five team names/enrollment numbers |

## Submission checks

- Verify the actual input data, its checksum, fixed horizon, and all chronological folds.
- Keep parameter selection separate from final holdout results.
- Report both naive references, all required metrics, quarterly variation, and scored-hour coverage.
- Use observed targets; exclude synthetic fixtures from project findings.
- Interpret missingness, peak errors, residual behavior, and limitations using measured outputs.
- Ensure the report, presentation, repository and individual explanations agree.

The one-hour horizon, daily seasonal period, 2013 validation, 2014 evaluation, quarterly refits, Ridge comparator and three-candidate grid are design choices. The faculty did not prescribe those exact values or a minimum paper/model/slide count.

Aircast is an additional local demonstration interface for the saved models. Its interactive forecasts are separate from the chronological backtests used as evaluation evidence.
