# Phase 2 completion audit before GitHub publication

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
| CI actually exercises deep learning | Added for publication | Dedicated Python 3.13 job installs pinned CPU PyTorch, runs all tests with deprecation warnings as errors, and rejects skipped neural tests. Remote results must be checked after pushing. |
| GitHub availability of the runnable weights | Prepared for publication | Publish the verified package as a release asset; model binaries remain outside source commits under the repository workflow. |

## Verification used for publication

- `python -m pytest -q -W error::DeprecationWarning`: 101 passing tests.
- `python scripts/verify_phase2_submission.py`: source/input/model hashes, frozen selection, common rows, chronological boundaries, independent training counts, CLI/dashboard parity, document consistency and notebook execution.
- `node scripts/check_frontend_browser.cjs`: model selection, numerical forecasts, measured score tables, manual inputs, CSV upload/export, report link, chart behavior, mobile layout and no JavaScript errors.
- `git diff --check`, workflow parsing, CPU wheel/version compatibility, targeted publication credential scan, and archive-to-workspace checksums.

The preferred model remains the validation-selected LSTM. Its exploratory 2014 RMSE is 20.726; Bi-LSTM is 20.799, hybrid 20.919, Ridge 21.063 and XGBoost 21.088. These are historical rankings, not significance tests or a fresh blind evaluation.

## Scope left for the next work

The core Phase 2 local implementation and submission are complete against the documented scope. The next work is production neural-hosting readiness and a separately declared external validation. Actual hosting needs a suitable runtime; no paid service or live deployment is implied by pushing source. New data must be checked for station and feature compatibility before it can support a blind temporal claim. Cross-station or changed-feature results must be labeled as external transfer evidence rather than presented as equivalent to the original Beijing station evaluation.
