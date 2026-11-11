# Project completion review — 7 November 2026

**Implementation follow-up:** the notebook, fresh-clone instructions, dashboard wording, deployment packaging and public report findings below have now been addressed. The updated public deployment passed hosted API and anonymous desktop/mobile checks. This document preserves the initial audit findings; [completion_handoff.md](completion_handoff.md) records the final verification and account/faculty-dependent work.

The core Phase 1 and Phase 2 forecasting work is implemented and has measured results. The remaining academic handoff work is mainly notebook reproducibility, deployment/document consistency, and checking the actual college submission requirements. Public access to the neural models and operation on current observations are separate levels of completion.

This review covers the checkout at `ed6696beb4eb6098c57830499c1d453009d5d67d`. It examines the Python package, configurations, tests, frontend, deployment/restoration scripts, notebooks, reports, presentations, saved predictions and model manifests, packaged release, public classical endpoints, and current GitHub Actions results. The phase mapping comes from the repository's documented scope; a separate faculty rubric is not present.

**Verification performed for this review.**

- Ran `.venv/bin/python -m pytest -q -W error::DeprecationWarning`: **129 passed**, with no skips reported, in 14.16 seconds.
- Ran `.venv/bin/python -m pip check`: no broken requirements.
- Independently recomputed pooled validation/comparison metrics from the saved Phase 1, first Phase 2, and selected Phase 2 research predictions. Checked duplicate keys, common origins, targets, folds, and the one-hour horizon.
- Checked the selected research source/data/model hashes, frozen selection evidence, preservation of Phase 1 predictions, label/calibration boundaries, and complete eligible training counts for the independent networks.
- Compared all five actual saved-model forecasts through the CLI prediction function and Flask endpoints. The local default is LSTM; the service requires 192 hourly slots when these sequence models are loaded.
- Independently recomputed the external pooled and station metrics from **1,520,092 prediction rows**: seven methods on **217,156 common station-hours** across 12 stations. Checked the frozen model/evaluator/protocol hashes and pre-download declaration.
- Verified all **331 payload checksums** in the 52.6 MB Phase 2 ZIP, its ZIP integrity, and its SHA-256 against the published release metadata.
- Opened both local report PDFs and PowerPoint files, confirmed page/slide counts and speaker notes, and checked Phase 2 document hashes against the saved verification record.
- Checked the public health/bootstrap endpoints, dashboard HTML, an actual historical forecast response, and report delivery. The hosted Ridge and XGBoost historical forecasts match the local saved models within `1e-4`.
- Confirmed the latest push and manual CI runs at this commit succeeded, including Python 3.11/3.12 classical jobs, the Python 3.13 neural job, and the clean-checkout Linux container job. [Latest complete CI run](https://github.com/pratham-malhotr/air_prediction/actions/runs/41998855653).

Existing browser screenshots/checks were inspected; browser automation and full model retraining were not rerun during this review. The fresh test suite, artifact checks, metric recomputation, and public HTTP checks are distinct from that earlier browser evidence. Detailed local evidence is saved in [verification.json](../artifacts/completion_review/verification.json) and [github_verification.json](../artifacts/completion_review/github_verification.json); `artifacts/` is ignored by Git.

**What Phase 1 should deliver, and its current state.**

| Work | State | Evidence |
|---|---|---|
| Problem definition, one-hour target, dataset attribution | Complete | [README](../README.md), [methodology](methodology.md), [data guide](../data/README.md) |
| Calendar/schema checks, hourly regularization, missing-data handling | Complete | `data.py`, preparation outputs and data tests |
| Training-period EDA and written interpretation | Complete | `eda.py`, [analysis](analysis.md), saved EDA tables/plots |
| Causal lags, trailing statistics, calendar/weather features | Complete | `features.py`, 70-column feature dictionary, alignment/invariance tests |
| Persistence, daily seasonal naive, Ridge and XGBoost | Complete | `models.py`, `experiment.py`, completed real-data runs |
| Validation selection and chronological quarterly backtests | Complete | Four 2013 validation folds, four 2014 comparison folds, saved selection and fold audits |
| MAE, RMSE, R², coverage, quarter/peak/residual diagnostics | Complete | Recomputed metrics on 8,531 common 2014 hours; [analysis](analysis.md) |
| Saved-model inference and usable classical demonstration | Complete | CLI/Flask parity and responding public classical service |
| Report and presentation | Complete locally | Seven-page IEEE report; ten-slide editable deck, matching PDF and notes |
| Notebook showing the current implementation | Needs correction | Root Phase 1 notebook has stale embedded source and incomplete/error outputs |
| Final submission/rubric/contribution evidence | Needs final check | Faculty rubric and individual contribution records are not established by this checkout |

Phase 1 does not need another model simply to meet its documented scope. Ridge slightly outperforms XGBoost in the measured pooled comparison; the report should retain that finding.

**What Phase 2 should deliver, and its current state.**

| Work | State | Evidence |
|---|---|---|
| Sequence construction and training-only neural preprocessing | Complete | `deep_learning.py`; frozen preprocessing, causality and sequence tests |
| Independent LSTM and Bi-LSTM benchmarks | Complete | Standalone fits use all eligible outer-training rows |
| Validation search and attention/directionality comparisons | Complete | Eight screening configurations; matched attention variants; seeds 42, 137 and 2026 |
| A technically defensible hybrid | Complete | Frozen Bi-LSTM base, later residual calibration, validation-selected correction weight |
| Fair classical/neural comparison and error investigation | Complete | Common-hour checks, quarterly/peak scores and correction-versus-own-base analysis |
| Reproducible weights, inference, provenance and package | Complete | Model/source/data/selection hashes; verified release ZIP; five-model CLI/Flask parity |
| External transfer evaluation | Complete | All 12 UCI 501 stations, no external fitting or reselection, independently verified scores |
| Phase 2 academic documents and notebook | Complete locally | Five-page report, twelve-slide editable deck/PDF/notes; seven of seven code cells executed without error outputs |
| Local demonstration and deployable runtime | Complete | Validation-selected LSTM default, five saved models, Gunicorn factory, release restoration and successful Linux container CI |
| Public demonstration of neural models | Pending if required | The public Vercel service currently exposes only Ridge and XGBoost |
| New blind or present-day performance claim | Not established | The 2014 research comparison is exploratory; the external evidence covers historical station/feature transfer |

The earlier `artifacts/phase2/` experiment and the selected `artifacts/phase2_research/` experiment intentionally have different scores. Use the selected research run for the final Phase 2 story and identify the earlier run as an archived baseline.

**What the current results actually support.**

The following values were independently checked against the saved predictions. Every row uses the same 8,531 observed 2014 target hours; errors are in µg/m³.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| LSTM | 11.299 | 20.726 | 0.9506 |
| Bi-LSTM | 11.457 | 20.799 | 0.9503 |
| Bi-LSTM + XGBoost | 11.483 | 20.919 | 0.9497 |
| Ridge | 11.825 | 21.063 | 0.9490 |
| XGBoost | 11.833 | 21.088 | 0.9489 |
| Persistence | 11.926 | 22.033 | 0.9442 |
| Daily seasonal naive | 67.296 | 98.754 | −0.1205 |

LSTM has the lowest observed historical RMSE and was selected from validation evidence. Its RMSE is about **1.60% lower than Ridge**, so the advantage should be described as modest. R² is a regression score; `0.9506` is not “95.06% forecast accuracy.”

The hybrid does not beat the independent recurrent models. Its correction changes its own frozen base's pooled RMSE from about **20.917 to 20.919** and worsens peak RMSE from **37.921 to 38.587**. This is a valid negative finding, not unfinished implementation. Any improvement experiment should preserve this result and use a fresh evaluation protocol.

The external test already exists: Bi-LSTM has the lowest external RMSE, **19.293**, while persistence has the lowest MAE, **10.203**. The original validation-selected LSTM remains the default. The official [UCI 501 description](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) confirms that these are different monitoring sites and that the available wind/rain fields differ from the legacy inputs. The evaluator appropriately leaves incompatible legacy fields unknown. This evidence does not establish current accuracy in India or elsewhere.

**Initial findings, in priority order (implementation status is in the handoff).**

1. **Repair and execute the Phase 1 notebook before submitting it.** [Air_Prediction_Phase1_Colab.ipynb](../Air_Prediction_Phase1_Colab.ipynb) has 14 code cells, only two execution counts, and a saved `ModuleNotFoundError: No module named 'pandas'`. Its embedded ZIP contains 44 entries: only nine match the current checkout, 30 differ, and five refer to files absent from the checkout. The stored error does not mean pandas is missing from the current verified environment; it means the notebook is not clean execution evidence. Replace/regenerate its embedded source or load a pinned current checkout, make dependency setup work in a fresh runtime, handle completed output directories explicitly, and execute the full notebook without error outputs. Completion means its results agree with the audited Phase 1 metrics and its source agrees with the submitted version.

2. **Update the report served by the public classical site.** [The hosted report](https://aircast-pm25.vercel.app/report) is currently an older **four-page** PDF with no team enrollment numbers. [The local final Phase 1 report](../reports/ieee/main.pdf) is **seven pages** and includes all five team members/enrollment numbers. Forecast parity passes; the report handoff is stale. Prepare a deployment containing the approved final report, publish it when authorized, and verify the hosted PDF hash/content afterward. If the existing public site remains Phase 1, its report should remain clearly labelled Phase 1.

3. **Make the README quick start work from a fresh clone.** Installing dependencies does not create the ignored `artifacts/phase1/*.joblib` files. The current quick start reaches dashboard launch without an explicit model restore/training step; its research command also requires saved Phase 1 comparison predictions. Document two complete paths: use the full runnable ZIP for research reproduction, or use the pinned runtime restorer for inference. The runtime restorer intentionally does not restore `artifacts/phase1/predictions.csv`; it is insufficient on its own for rerunning Phase 2 research. A training path must first produce the Phase 1 comparison run. Completion means each documented path succeeds independently in a clean directory.

4. **Align active UI wording and documentation with the selected research run.** [The performance page](../frontend/templates/index.html) still says “Four methods” when the local API supplies seven comparison methods. `app.js` still assigns XGBoost the “MAIN MODEL” tag although LSTM is the local validation-selected default; its headline and peak explanation remain classical-specific. Make the active model role and model count explicit. [The frontend guide](frontend.md) still mentions a 101-test suite and six-model comparison, and [the Phase 1 checklist](phase1_checklist.md) still says GitHub publication is unverified even though the repository, release and current CI are accessible. Update active status documents while preserving dated historical run evidence such as the release's original 116-test record.

5. **Finish the submission handoff against the actual faculty requirements.** Compare the official rubric with both phase checklists; confirm report/deck templates, required signatures or certificates, notebook expectations, filenames and submission destination. Record truthful individual contributions if required, link the exact final release/commit, and rehearse the individual viva using the supplied guides. These are outstanding verification/human tasks; they cannot be certified from code and test results.

6. **Publish the neural backend if an online Phase 2 demonstration is part of the required outcome.** The container, clean-checkout model restoration, Gunicorn startup and five-model Linux HTTP checks are already implemented. Remaining work is selecting and connecting a host, allocating measured CPU/memory capacity, configuring its public HTTPS/proxy route, and running the existing HTTP checker against that URL. Then verify model selection, manual/CSV forecasts, report delivery and mobile use on the public deployment. No specific hosting account has been provisioned by this review.

**Further research and product work, conditional on the intended finish.**

- For a stronger research paper, add relevant PM2.5 recurrent/hybrid studies to the current method-focused related work, explicitly comparing target, horizon, chronological split, inputs and metrics. A fixed paper count is not established by the repository's brief.
- For a new generalization claim, freeze the protocol before accessing an untouched period or appropriate new dataset. The already-inspected 2014 year and already-scored UCI 501 observations cannot become new blind tests by rerunning them. Any significance analysis must account for temporal dependence; three-seed standard deviations are not confidence intervals for predictive superiority.
- Calibrated prediction intervals, deeper peak-error research, additional horizons, GRU/TCN/Transformer experiments and richer explainability are optional extensions. GRU support already exists in code but is not part of the recorded benchmark. More architectures are not a prerequisite for completing the documented phases.
- A continuously operating forecasting product needs a current data feed, field/unit/timezone mapping, observation-availability rules, a maintained history, forecast/outcome storage, measured latency/memory/load capacity, operational monitoring, and a defined retraining/model-promotion process. The present service uses uploaded histories or archived scenarios and predicts with fixed saved models. Each manual request appends one scenario hour to the chosen history; it does not create a continuously advancing stored observation stream.
- AQI is optional and deliberately absent from the documented core scope. If added, choose the relevant standard and implement its averaging/missing-data requirements; an hourly concentration forecast alone is insufficient.

The implementation follow-up resolves the actionable repository findings and updates the existing public classical deployment. Official faculty requirements, truthful individual contribution confirmation and a public neural hosting account still require external information. A present-day operational forecasting product is additional data and operating scope. See the [completion handoff](completion_handoff.md) for current evidence.
