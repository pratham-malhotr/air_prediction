# Phase 2 selected research and submission

The independent-model continuation is complete. It preserves the [first corrected baseline](phase2.md) and Phase 1 results while adding validation-based tuning, matched recurrent/attention ablations, seed stability, calibrated hybrid selection, submission documents and an executed notebook.

## What was completed

- **Independent benchmarks:** standalone Bi-LSTM and LSTM use every eligible outer training row. They do not reuse the hybrid's reduced-training base.
- **Candidate screening:** eight Bi-LSTM configurations compare level/delta targets, 24/48-hour sequences, width 32/64, six/ten epochs, and attention. Screening uses seed 42 and the four 2013 folds.
- **Matched ablations:** LSTM/Bi-LSTM crossed with attention enabled/disabled, at seeds 42, 137 and 2026. Hidden width and training settings match; parameter counts differ between directions.
- **Hybrid tuning:** two calibration fractions, two residual-tree configurations and four correction weights, scored across all three validation seeds. The zero-weight control permits rejecting the correction.
- **Frozen selection:** the settings and evidence hashes are persisted before new 2014 comparisons. Seed 42 is predeclared for comparison and deployment, rather than selected from final-year performance.
- **Saved models and demo:** independent LSTM, Bi-LSTM and hybrid weights are saved. The local dashboard loads the completed research run, defaults to the validation-preferred LSTM, and links to the Phase 2 report.
- **Submission:** IEEE report, editable slides, slide PDF, speaker guide, figures, executed training/inspection notebook and a package with weights and data attribution.

## Selected method

The screening winner is `delta_24h_no_attention`. Both selected standalone networks use a 24-hour sequence, 32 hidden units per direction, one layer, dropout 0.1, six epochs, batch size 128 and learning rate 0.001. They predict a standardized change from the current concentration, reconstruct concentration and clip at zero. Attention is disabled in the selected configurations; its tested alternatives remain in the ablation evidence.

Input medians, category levels and scaling are fitted only on actual neural training windows. Delta offsets use current measured concentration or its saved training-median fill when missing. Without attention, the Bi-LSTM uses both complete terminal hidden states, rather than only the last output row.

The hybrid uses a separately trained Bi-LSTM base and reserves the final 20% of eligible training origins for out-of-sample residual calibration. Validation selects 80 trees, depth 2, learning rate 0.03, minimum child weight 20, L2 regularization 30, row/column subsampling 0.8, and correction weight 0.5. Base labels are available before calibration starts; calibration labels are available before evaluation begins. The neural base stays frozen during residual fitting and deployment.

## Results on identical historical hours

Every model uses the same **8,531 observed 2014 target hours**. Errors are in µg/m³.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| LSTM | 11.299 | 20.726 | 0.9506 |
| Bi-LSTM | 11.457 | 20.799 | 0.9503 |
| Bi-LSTM + XGBoost | 11.483 | 20.919 | 0.9497 |
| Ridge | 11.825 | 21.063 | 0.9490 |
| XGBoost | 11.833 | 21.088 | 0.9489 |
| Persistence | 11.926 | 22.033 | 0.9442 |
| Daily seasonal naive | 67.296 | 98.754 | -0.1205 |

The validation-selected LSTM also has the lowest observed 2014 RMSE in this run. This is a small historical difference, not a statistical-significance claim. The selected hybrid is better than the classical references on pooled RMSE but does not beat either independent recurrent model.

**All new 2014 comparisons remain exploratory:** Phase 1 results for that year had already been inspected. No fresh blind evaluation is claimed. Parameters and the deployment seed were chosen from 2013 evidence only.

## Seed stability and attention findings

Each seed uses the same 8,563 validation target hours. Values below are the mean and sample standard deviation of pooled 2013 RMSE across three neural seeds.

| Selected model | RMSE mean | RMSE SD |
|---|---:|---:|
| LSTM | 23.502 | 0.131 |
| Bi-LSTM | 23.613 | 0.132 |
| Bi-LSTM + XGBoost | 23.614 | 0.073 |

Ridge's recorded validation RMSE is 23.809. Mean validation RMSE prefers LSTM without attention. Attention does not improve mean RMSE for either architecture in this matched template. The errors and parameter counts do not establish universal architectural superiority. These standard deviations describe initialization/batch-order variation, not confidence intervals; the residual head's random state is fixed at 42.

## Peak-error investigation

The high-pollution threshold remains **218 µg/m³**, estimated only from 2010–2012. The same 865 high-pollution hours are scored for every model. Peak RMSE is **37.132 for LSTM**, **37.902 for Bi-LSTM**, **38.587 for the hybrid**, and **38.522 for Ridge**.

The hybrid correction is also compared with its own frozen base, separating correction behavior from the independent model's larger training window. On 2014, the correction lowers below-threshold RMSE from 18.018 to 17.860 but raises peak RMSE from 37.921 to 38.587. Pooled RMSE changes from 20.917 to 20.919. It reduces pooled bias but provides no pooled RMSE improvement on that year. Selection is retained as originally frozen; these findings do not trigger another round of 2014-driven tuning.

[Correction-effect data](results/phase2_research/correction_effect.csv) and [figure](results/phase2_research/correction_effect.png) retain that comparison. The diagnostics describe errors and do not identify a causal pollution mechanism.

## Inspect and reproduce

- [IEEE-format report](../reports/phase2/Aircast_Phase2_Report.pdf) and [editable LaTeX](../reports/phase2/Aircast_Phase2_Report.tex).
- [Editable PowerPoint](../reports/phase2/Aircast_Phase2_Presentation.pptx), [slide PDF](../reports/phase2/slides.pdf), and [presenter guide](../reports/phase2/presenter_guide.md).
- [Executed notebook](../notebooks/Air_Prediction_Phase2.ipynb); full retraining is an explicit switch.
- [Compact results and scientific figures](results/phase2_research/), including every candidate and seed score.
- Local package: `dist/air_prediction_phase2.zip`, with a payload checksum manifest.

```bash
python -m pip install -r requirements-phase2.txt
python scripts/run_phase2_research.py --output artifacts/phase2-new-run
python scripts/build_phase2_submission.py --run artifacts/phase2_research
python scripts/package_phase2.py
bash scripts/start_frontend.sh
```

The dashboard prefers a completed real-data `artifacts/phase2_research/` run and verifies model hashes before loading. Until it is complete, it can use the preserved `artifacts/phase2/` baseline. Saved 24-step models require 192 raw hourly slots. Alternate experiment directories are not automatically activated.

## Limits and optional follow-up

The finite search reuses one validation year, the final comparison uses one predeclared seed, and the data covers one historical city. This work does not provide causal weather effects, calibrated prediction intervals, official AQI mapping or present-day accuracy. A fresh untouched period or another appropriate dataset is needed for new blind performance claims.

The Phase 2 local implementation and submission materials are complete. Public hosting of the neural backend is optional separate work; the existing Vercel app remains the classical deployment. Original Phase 1 documents and both earlier result directories are preserved.
