# Final project handoff

Both phases have completed historical forecasting experiments. Use this list to assemble the submission; the college's official rubric, required certificates and submission destination must still be checked by the team.

| Deliverable | Phase 1 | Phase 2 |
|---|---|---|
| Report PDF | [Seven-page report](../reports/ieee/main.pdf) | [Five-page report](../reports/phase2/Aircast_Phase2_Report.pdf) |
| Editable report | [LaTeX source](../reports/ieee/main.tex) | [LaTeX source](../reports/phase2/Aircast_Phase2_Report.tex) |
| Editable presentation | [Ten-slide PowerPoint](../reports/presentation/Aircast_Phase1_Presentation.pptx) | [Twelve-slide PowerPoint](../reports/phase2/Aircast_Phase2_Presentation.pptx) |
| Presentation PDF | [Phase 1 slides](../reports/presentation/slides.pdf) | [Phase 2 slides](../reports/phase2/slides.pdf) |
| Presenter notes | [Phase 1 guide](../reports/presentation/presenter_guide.md) | [Phase 2 guide](../reports/phase2/presenter_guide.md) |
| Executed notebook | [Phase 1 Colab notebook](../Air_Prediction_Phase1_Colab.ipynb) | [Phase 2 notebook](../notebooks/Air_Prediction_Phase2.ipynb) |
| Results and limits | [Classical analysis](analysis.md) | [Selected research](phase2_research.md), [external supplement](phase2_external_validation.md) |
| Reproduction | [README setup paths](../README.md#installation-and-quick-start) | [Research instructions](phase2_research.md), [runtime instructions](../deployment/phase2/README.md) |

- [x] Preserve observed targets, chronological splits, required baselines, coverage and MAE/RMSE/R².
- [x] Keep classical results, independent neural comparisons and the hybrid correction analysis traceable to their respective runs.
- [x] Include original dataset attribution and distinguish historical scenarios from present-day observations.
- [x] Supply both editable and PDF documents, speaker notes, notebooks, saved models and reproduction instructions.
- [ ] Compare this inventory with the official faculty rubric and required report/declaration/certificate formats.
- [ ] Confirm the five names/enrollment numbers, truthful individual contributions, required signatures and submission destination with the team.
- [ ] Rehearse the individual viva, including why Ridge is competitive, why the hybrid correction did not improve pooled RMSE, and why Phase 2's 2014 comparison is exploratory.
- [ ] Connect a suitable hosting account if a public neural demonstration is required; the current Vercel deployment is classical. The five-model local/container demo is available.

No contribution allocations, signatures, faculty approvals or new performance claims are inferred from Git history. A new live-data product or blind evaluation is additional scope beyond this completed historical study.

Local handoff archives can be regenerated without replacing the checksum-pinned public runtime release:

```bash
python scripts/package_project.py
python scripts/package_phase2.py --output dist/air_prediction_phase2_completed.zip
```

The Phase 2 builder writes a payload manifest and a SHA-256 sidecar. Use the published pinned ZIP, rather than this newly built handoff ZIP, with `restore_phase2_runtime.py --bundle`.
