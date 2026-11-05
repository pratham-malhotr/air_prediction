# Phase 2 submission

This directory contains the independent recurrent-model and calibrated-hybrid research submission. Phase 1 documents stay in their original locations.

Generated deliverables:

- `Aircast_Phase2_Report.pdf`: IEEE-format research report, with editable `Aircast_Phase2_Report.tex` and `report_template.tex`.
- `Aircast_Phase2_Presentation.pptx`: twelve editable slides with embedded speaker notes.
- `slides.pdf`: the same presentation drawn from the shared layout.
- `presenter_guide.md`: the slide-by-slide explanation and viva prompts.
- `figures/`: scientific figures in PNG and standalone PDF formats.
- `research_summary.md`: concise measured findings generated from saved predictions.
- `../../notebooks/Air_Prediction_Phase2.ipynb`: executed inspection/inference notebook, with an explicit switch for full retraining.

All results come from the completed real-data run in `artifacts/phase2_research/`. Model selection uses the 2013 validation folds. New 2014 comparisons are exploratory because Phase 1 results had already been viewed. Seed error bars are standard deviations across predefined validation seeds, not confidence intervals. Team names and enrollment numbers are reused from Phase 1; no individual contribution assignments or email addresses are invented.

From the repository root:

```bash
python scripts/run_phase2_research.py --output artifacts/phase2_research
# For an interrupted run with the same source, configuration and input:
python scripts/run_phase2_research.py --resume

python scripts/build_phase2_submission.py
python scripts/verify_phase2_submission.py
python scripts/package_phase2.py
```

Completed runs cannot be overwritten. Use a new `--output` for independent experiments. Report generation uses `reports/requirements.txt` and Tectonic; the builder can also reuse document libraries already present under `.dependencies/documents/`. The PowerPoint/PDF share a layout, but the PDF is not an export from Microsoft PowerPoint. Rendering can differ slightly across font installations.

The generated ZIP is `dist/air_prediction_phase2.zip`, also provided as the [verified GitHub release package](https://github.com/pratham-malhotr/air_prediction/releases/download/phase2-research-2026-11-05/air_prediction_phase2.zip). It includes source, data attribution, selected weights, evaluation evidence, reports and the notebook. It excludes environments, credentials, Git metadata and training checkpoints, and includes a checksum manifest for every packaged file.

The recorded handoff has 115 passing automated tests, completed browser checks, a five-page report, twelve presentation slides with notes and seven executed notebook cells. `deliverable_manifest.json` records document hashes and the saved verification evidence.

The later [external transfer supplement](../../docs/phase2_external_validation.md) evaluates all 12 UCI 501 stations under a frozen protocol. It is separate from the five-page historical-study paper. [Production runtime instructions](../../deployment/phase2/README.md) accompany the runnable release.
