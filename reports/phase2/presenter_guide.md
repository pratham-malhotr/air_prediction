# Phase 2 presentation speaker guide

The same notes are embedded in the editable PowerPoint. Explain the 2014 exploratory status, validation-only selection, seed variation and peak errors. No individual contributions or email addresses are inferred.

## 1. Scope and team

Phase 2 asks whether independent recurrent networks and a calibrated correction improve on strong classical references. The team is carried forward exactly from the existing project. All 2014 results are exploratory because the year was already inspected in Phase 1.

## 2. Audit findings

The first audit fixed full-dataset scaling, batch-dependent imputation, ignored masks, dropped wind direction, in-sample residual fitting and suppressed errors. This continuation also fixed bidirectional state extraction without attention and added independent full-window benchmarks. A zero or invalid training mask fails rather than silently relaxing the boundary.

## 3. Data and timing

UCI 381 has 43,824 hourly rows and 2,067 missing concentrations. The same engineered features and common evaluation mask are used for all models. At each quarterly boundary, only labels available at the first forecast origin enter training. 2014 remains exploratory even though individual fits respect chronology.

## 4. Neural architecture

Input and target scaling are fitted only on training windows. Delta targets use current pollution as a fixed offset; missing current pollution uses its saved training median. Standardized Huber loss, AdamW, gradient clipping and fixed epochs train the model. The bidirectional network reads backward within the observed window, never beyond the forecast origin. Directionality comparisons have matching hidden width but different parameter counts.

## 5. Candidate search

The complete grid is saved before results are inspected. Initial screening uses seed 42 on all four 2013 quarters. This is a finite staged search, not exhaustive optimization or nested cross-validation. Unsuccessful candidates remain in the CSV and checkpoint outputs. The winning template supplies the matched ablations.

## 6. Ablations and seed variability

Compare LSTM and Bi-LSTM with attention enabled and disabled at the selected template settings. Selection uses mean pooled validation RMSE across seeds. Error bars show sample standard deviation across seeds; they are not confidence intervals. Attention weights do not establish causal feature importance.

## 7. Hybrid calibration

Reserve the last 10% or 20% of training origins for calibration. The base is trained earlier with a label-availability gap and then frozen. Two XGBoost settings and weights 0, .25, .5 and 1 are scored using three validation seeds. The zero-weight control can reject an unhelpful correction. The hybrid intentionally has less base-training data than the independent neural benchmark.

## 8. Final score table

These rows use the identical observed 2014 targets. Classical predictions are reused from the audited Phase 1 experiment and checked for exact origins, labels and fold alignment. The preferred model comes from validation, not this final ranking. Small differences do not establish significance, and a lower RMSE does not guarantee a lower MAE or better peak behavior.

## 9. Failure analysis

The high-pollution threshold is fixed using initial training data only. There are 865 peak hours and 7,666 lower-pollution hours in the common 2014 set. Compare peak RMSE and bias with Ridge and persistence, not just with a weaker neural base. Seasonal changes and differences between full and reduced training windows complicate attribution; the diagnostics are descriptive, not causal.

## 10. Seed stability and example trace

The left panel shows variation across the three predefined validation seeds for selected model configurations. The right panel shows a time-selected forecast interval rather than a hand-picked success. We do not choose a deployment seed using holdout performance. No calibrated prediction intervals are supplied.

## 11. Demo and reproducibility

The local dashboard verifies model hashes from a completed real-data manifest before exposing models. The hosted classical deployment does not include PyTorch. The notebook loads measured results by default and has an explicit switch to rerun the complete experiment. The archive excludes environments, credentials and training checkpoints, and includes source, model weights, the attributed dataset and comparison evidence.

## 12. Conclusions and viva prompts

Explain why the standalone model is independent, why calibration residuals must be out of sample, why the bidirectional window is causal, why all models share a target mask, and why 2014 remains exploratory. Discuss seed variation without claiming a confidence interval. Do not assume a hybrid must beat Ridge. All five team members and enrollment numbers are listed on the opening slide, with no invented contribution allocations.
