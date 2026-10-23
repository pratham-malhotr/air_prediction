# Mapping to the faculty's Phase 1 evaluation

| Criterion | Implemented support | Work requiring real execution or team completion |
|---|---|---|
| Literature review | `literature_review.md`, report bibliography, original sources and limitations | Read sources; explain each source's relevance in your own words |
| Dataset quality & EDA | Schema checks, timestamp audit, missingness, distributions, calendar profiles, correlations and ACF; EDA notebook | Execute on UCI data; interpret actual tables and plots |
| Feature engineering | Causal lags, trailing statistics, missing flags, known calendar and historical weather | Explain offset alignment, warm-up and observed fractions; review optional validation ablations |
| Theoretical rigor | Baseline equations, Ridge objective, boosting objective, split logic and metric definitions | Every member must understand assumptions, regularization, bias/variance and limitations |
| Model application | XGBoost grid, linear comparator, required naive models, chronological model selection and holdout | Run XGBoost on real data; inspect baseline improvements and errors; retain negative outcomes |
| GitHub & code quality | Modular package, tests, configuration, provenance, reproducible Colab source | GitHub creation/invitation/push intentionally deferred by user; add individual commit history |
| Presentation | Beamer slides and presenter guide; imports measured result figures | Compile after the real run, add names and discuss actual findings |
| Individual viva | `viva_guide.md` and contribution template | Five real members must understand the full project and complete their own records |
| LaTeX/IEEE report | IEEEtran paper, bibliography, honest pending-results placeholders | Fill names, measured EDA findings, result interpretation, final limitations; compile and proofread |

## Before submission

- Run the real Colab notebook from a clean runtime and keep its outputs. `run_manifest.json` must identify real data, not `synthetic`.
- Confirm the official CSV, 2010–2014 dates, fixed horizon, all chronological folds, common scored targets, and baseline coverage in the audits.
- Inspect per-quarter results; do not present only the best quarter or only R².
- Explain real missingness and peak-error behavior. Do not copy synthetic demonstration metrics into the report.
- Confirm selection uses validation, not final holdout errors. Record any post-holdout protocol changes honestly.
- Run tests; save the executed notebook and record dependency versions.
- Replace all team placeholders using exactly five actual members and honest contribution evidence.
- Publish the authorized repository and invitation when GitHub access becomes available; never commit credentials or cached tokens.
- Compile the report and slides; add the actual experiment's outputs, conclusion, and limitations.

The faculty did not specify a mandatory paper count, model count, slide count, report length, or evaluation date in the provided instructions. This project does not invent such requirements. Deep learning, hybrid innovation, and DL ablations are outside the current phase.
