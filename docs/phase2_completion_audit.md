# Phase 2 completion audit and continuation

This audit uses the repository's stated Phase 2 scope and the work requested in this session. No separate college grading rubric was found in the checkout or the repository's GitHub issues, so it does not certify requirements that have not been supplied.

## Requirement-to-evidence review

| Documented requirement | Status | Evidence |
|---|---|---|
| One-hour PM2.5 forecasts from observed history only | Verified | Shared origin-indexed features; label-availability guards and forecasting tests. |
| Preserve missing hourly slots and exclude invented scoring targets | Verified | Loader/grid checks, common evaluation mask and finite-observation scoring. |
| Training-only neural preprocessing | Verified | Medians, category levels and scaling use the union of training windows; future-feature/label perturbation tests cover both level and delta targets. |
| Independently trained recurrent benchmarks | Verified | LSTM and Bi-LSTM use every eligible outer-training row; saved training counts are checked against the dataset. |
| Technically justified hybrid | Verified | Frozen neural base, later chronological residual calibration, explicit correction shrinkage and surfaced inference failures. |
| Validation-only parameter selection | Verified | Eight screening candidates, matched architecture/attention comparisons and hybrid selection on 2013; selection/evidence hashes persist before the 2014 comparison. |
| Repeated-seed stability | Verified | Seeds 42, 137 and 2026; mean and sample SD retained for matched ablations and selected models. |
| Fair classical/neural comparison | Verified | Identical 8,563 validation and 8,531 historical comparison hours; original Phase 1 predictions/configuration/hash checks. |
| Pollution-peak and correction analysis | Verified | Training-only 218 µg/m³ threshold; peak/bias tables and isolated comparison of the correction against its own frozen base. |
| Reproducible saved inference | Verified | Source/data/model hashes, fixed deployment seed, original Phase 1 preservation and real CLI/dashboard parity. |
| Working local demonstration | Verified | Five available saved models, validation-selected LSTM default, sequence history requirements, report selection and browser checks. |
| Academic submission | Verified | Five-page IEEE report, editable source, twelve-slide PowerPoint with notes, slide PDF, speaker guide and figures. |
| Reproducible notebook and package | Verified | Seven executed notebook cells without error outputs; package checksum verification and excluded credentials/environments/checkpoints. |
| CI actually exercises deep learning | Implemented and locally verified | Python 3.13 CPU PyTorch job rejects skipped neural tests. The first remote run exposed a decimal round-trip issue; strict CSV loading and a regression now preserve exact target alignment. Final remote status is recorded in GitHub Actions. |
| GitHub availability of the runnable weights | Release handoff | Runnable ZIP and its SHA-256 checksum are published with the Phase 2 release; model binaries remain outside source commits under the repository workflow. |
| External validation | Completed | Pre-download protocol; five immutable models and two baselines evaluated on all 12 UCI 501 stations, 217,156 common station-hours; independently recomputed scores and transfer limitations. |
| Production runtime | Implemented | Gunicorn factory, source/data/weight preflight, non-root CPU container recipe and an actual HTTP forecast checker. Manual CI builds the published bundle on Linux; public hosting is separate. |

## Verification used for publication

- `python -m pytest -q -W error::DeprecationWarning`: 116 passing tests.
- `python scripts/verify_phase2_submission.py`: source/input/model hashes, frozen selection, common rows, chronological boundaries, independent training counts, CLI/dashboard parity, document consistency and notebook execution.
- `node scripts/check_frontend_browser.cjs`: model selection, numerical forecasts, measured score tables, manual inputs, CSV upload/export, report link, chart behavior, mobile layout and no JavaScript errors.
- `git diff --check`, workflow parsing, CPU wheel/version compatibility, targeted publication credential scan, and archive-to-workspace checksums.

The preferred model remains the validation-selected LSTM. Its exploratory 2014 RMSE is 20.726; Bi-LSTM is 20.799, hybrid 20.919, Ridge 21.063 and XGBoost 21.088. These are historical rankings, not significance tests or a fresh blind evaluation.

## Work completed after the initial push

The first three commits published the causal pipeline, independent research, submission materials and dashboard. The first Linux CI run passed the Python 3.11/3.12 classical jobs but exposed one-bit decimal CSV parsing differences in the neural fixtures. Reading the comparison CSV with `float_precision='round_trip'` fixes that portability issue without relaxing exact target checks. A full research rerun under the corrected source preserves honest artifact provenance; the earlier run is retained locally.

The fourth commit adds the portability regression, external transfer supplement, production runtime and additional verification. The external protocol freezes the original published model bytes and records both training and runtime source hashes. It never changes its model selection based on the new results. Bi-LSTM has the lowest observed external RMSE (19.293), while persistence has the lowest MAE (10.203); the original validation-selected LSTM remains the demo default.

The first release-backed Docker build succeeded, then its readiness checker encountered a transient connection reset before worker initialization completed. A small follow-up commit adds that socket failure to bounded startup retries and a regression, without changing forecast or training code. The revised runnable release preserves the original publication.

## Clean checkout deployment continuation

The original Docker recipe depended on ignored model files already being present. CI previously restored those files into its runner before building, so it did not establish that a hosting service could deploy directly from Git. The new `scripts/restore_phase2_runtime.py` fixes that gap: it uses only the standard library to verify the pinned public release, current forecasting source and all selected runtime payloads before creating files. It refuses to replace a different existing experiment. The Docker build now invokes this restorer itself; its context excludes local artifacts and training data.

A real source checkout with no saved models was restored and served through Gunicorn. All five HTTP forecasts matched the release references, the selected default remained LSTM, and measured comparisons, report delivery, response headers and invalid-input rejection passed. The current strict suite has **129 passing tests**, including thirteen restoration regressions. [Local clean-checkout evidence](results/phase2_production/clean_checkout_audit.json) is distinct from the original publication evidence above. The manual CI workflow also builds from source without preloading artifacts; its remote status is recorded in Actions.

Public hosting still awaits a hosting account choice and connection. No paid service has been created. The frozen research models, selection and numerical findings remain unchanged.

## Remaining scope

The local Phase 2 implementation, documented research and submission are complete. A public neural deployment needs an appropriate host and resource allocation. This work supplies a production runtime and container verification; it does not provision or charge a hosting account. A separate college rubric has not been supplied. Any requirements unique to that rubric still need comparison with this audit.

The external test uses different stations and lacks three legacy inputs, so it cannot replace untouched same-station temporal validation. The work makes no significance, causal-weather, prediction-interval or present-day accuracy claim. [External supplement](phase2_external_validation.md) and [production instructions](../deployment/phase2/README.md) describe the evidence and limits.
