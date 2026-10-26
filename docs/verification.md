# Phase 1 validation

These notes record the implementation checks and the real-data experiment completed on 23 October 2026.

## Validation coverage

| Area | Checks |
|---|---|
| Data integrity | Required columns, finite measurements, valid integer calendar fields, unique timestamps, chronological sorting, missing hourly slots, and file checksums |
| Forecast alignment | Future target timestamps, persistence and seasonal naive alignment at multiple horizons |
| Feature causality | Changing future pollution/weather observations cannot alter earlier features |
| Missingness | Missing targets remain unfilled; comparison masks require common observed targets and baseline inputs |
| Chronological evaluation | Nonoverlapping calendar folds; training labels available by first issuance; complete configured evaluation period |
| Preprocessing | Training-only imputation/scaling and handling of unseen wind categories |
| EDA interpretation | Notes reflect calculated training statistics; later observations cannot change them; undefined correlations are explained |
| Metrics | MAE, RMSE, negative R², undefined constant-target R², and rejection of nonfinite predictions |
| Experiment artifacts | Saved forecasts/models, inference timing, coverage, source labels, and completion records |

Synthetic fixtures exercise these behaviors without being treated as Beijing observations. The actual XGBoost estimator is included in the integration tests.

Audit on 23 October 2026: **66 tests passed, none skipped**. This includes the original 38 ML checks, 23 dashboard checks and five deployment-package checks. The suite covers XGBoost backtests, feature ablation, saved-model loading, inference, HTTP routes, stateless uploads and packaging of the evaluated artifacts. These tests verify implementation behavior. Forecasting scores come from the separate real-data experiment below.

## Real-data validation

- The supplied CSV has 43,824 hourly rows with complete timestamp coverage, no duplicates, and 2,067 missing PM2.5 values.
- Its SHA256 is `4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c`.
- EDA uses only 2010–2012. Cleaning preserves missing measurements and peaks.
- All three XGBoost candidates use four chronological 2013 folds. Candidate 1 is selected before the four 2014 backtests.
- All four models share 8,563 validation targets and 8,531 holdout targets. Target timestamps are exactly one hour after forecast origins.
- Forecast labels and both naive references were checked against the raw series. Pooled and quarterly MAE, RMSE and R² were independently recalculated from the saved predictions using scikit-learn.
- Holdout RMSE: persistence 22.033, Ridge 21.063, XGBoost 21.088, seasonal naive 98.754 µg/m³.
- Saved models load and forecast the archive's next historical hour; they are separate refits from the backtest models.
- The experiment source fingerprint matches the current forecasting package. Report-only scripts generate supplementary tables and figures from saved outputs.

Full predictions, coverage, selection, models, and provenance are in `artifacts/phase1/`. [The analysis](analysis.md) explains the measured strengths and weaknesses. The report and slides use those same results.

A subsequent prediction audit refitted XGBoost and Ridge for all four 2014 quarters using the frozen parameters and original training boundaries. All 17,062 learned-model predictions matched the saved backtests; the largest floating-point difference was less than `6e-14`. The audit checked actual targets against the raw CSV, both naive references, common scoring dates, lag/rolling alignment, and recalculated every pooled validation/holdout MAE, RMSE and R² with scikit-learn. This reproduces the original experiment without selecting new parameters from the holdout.

The same audit checked that manual concentrations of 12, 65 and 150 change the actual model output, and that uploading the original CSV into one app instance can be used by a fresh instance to reproduce `predict_latest`. The prepared Vercel model files and inference service are byte-identical to those checked locally. The record is [prediction_audit.json](../artifacts/prediction_audit.json). These are historical and local HTTP checks; they do not establish that the hosted requests have passed.

## Final presentation checks

The report was expanded to seven IEEE-format pages, and the presentation was further developed into ten widescreen PowerPoint slides with thirteen graph panels and detailed speaker notes. The graphs cover the concentration distribution, missingness by year, autocorrelation, hourly profile, weather correlations, validation ablation, annual RMSE, quarterly RMSE, pollution regimes, a final-week forecast trace, residual distribution, largest spike miss and deployment-refit gain importance. Plots are embedded at 300 dpi with standalone vector PDFs retained. The report includes all five supplied names and enrollment numbers; the presentation repeats them on the closing thank-you slide. No email addresses or individual contribution allocations are included. Tectonic 0.17.0 compiles the report. A shared Python layout produces the PowerPoint and matching ten-page presentation PDF; the PDF is not a Microsoft PowerPoint export. All figures and tables use the completed real-data experiment, with no retraining or changes to model selection. The residual histogram explicitly discloses the displayed ±80 range and counts outside that range; its percentages retain all common target hours as the denominator.

Repository publication and the team's individual viva preparation remain separate submission tasks. Individual task allocations have not been supplied. Document sources, reporting dependencies and rebuild instructions are in `reports/README.md`.

## Interactive frontend

Aircast uses the same saved pipelines and feature builder. Its tests verify manual input changes, forecast timing, prediction parity with the existing CLI, CSV parsing and gaps, upload limits, missing weather, stateless history revalidation and readable errors. The upload flow was checked across independent app instances. Additional checks using the actual Beijing CSV confirmed that both web predictions match `predict_latest`. The ML package source hash, selected parameters and measured backtest scores remain unchanged.

JavaScript syntax, CSS parsing, HTML IDs and controller selectors have been checked. This restricted workspace cannot bind a local listening port or launch Chromium, so browser rendering and screenshots have not been verified here. `scripts/check_frontend_browser.cjs` provides the actual-model desktop/mobile checks for execution from an unrestricted local Terminal.

The inference-only Vercel snapshot starts independently from the main checkout and serves the application, health/bootstrap endpoints and report. Deployment tests verify CDN assets, model inclusion, source/data checksums, exclusion of training outputs and preservation of an existing project link. The Vercel CLI subsequently reported the production deployment Ready at `https://aircast-pm25.vercel.app`. This workspace could not fetch the live endpoints or resolve the host, so hosted inference is still unverified. `scripts/check_vercel.py` checks the real deployment through the user's authenticated CLI and writes a separate verification record only after all hosted checks pass.
