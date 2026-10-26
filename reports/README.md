# Phase 1 submission documents

- **Report:** [`ieee/main.pdf`](ieee/main.pdf), exactly seven pages in IEEE conference format. Editable source: [`ieee/main.tex`](ieee/main.tex), with the bibliography and print-sized figures alongside it.
- **PowerPoint:** [`presentation/Aircast_Phase1_Presentation.pptx`](presentation/Aircast_Phase1_Presentation.pptx), ten widescreen slides with thirteen detailed graph panels. Text, tables, and diagrams are editable PowerPoint objects. Scientific plots are embedded at 300 dpi, with standalone vector PDFs and PNGs in [`presentation/figures/`](presentation/figures/). Detailed speaker notes are embedded.
- **Presentation PDF:** [`presentation/slides.pdf`](presentation/slides.pdf), ten pages generated from the same measured layout and graph assets as the PowerPoint.
- **Speaker guide:** [`presentation/presenter_guide.md`](presentation/presenter_guide.md), covering the full Phase 1 explanation and sources.

Both deliverables cover the completed classical ML phase and the Aircast demonstration. All five provided names and enrollment numbers appear in the report and on the closing thank-you slide. No email addresses or invented individual task allocations are included.

The results come from the completed real-data experiment in `artifacts/phase1/`. Reporting does not retrain models, revise selection, or overwrite experiment outputs. Separate deep-learning modules are outside these Phase 1 results.

## Rebuild

From the project root, in the project's Python environment:

```bash
python -m pip install -r reports/requirements.txt
bash scripts/build_pdfs.sh
```

The report build also requires Tectonic and the project's Matplotlib dependency. Tectonic may fetch missing LaTeX resources on the first run. The Python commands use the `python` selected by the active environment. On macOS the deck uses Arial; on Linux it uses DejaVu Sans. Other installations can set `AIRCAST_FONT_REGULAR` and `AIRCAST_FONT_BOLD` to matching TrueType font paths.

[`scripts/build_report_figures.py`](../scripts/build_report_figures.py) reads the saved EDA and predictions to draw the two supplementary article figures. [`scripts/build_presentation.py`](../scripts/build_presentation.py) is the canonical presentation source and produces both the editable `.pptx` and matching PDF. It invokes [`scripts/build_presentation_figures.py`](../scripts/build_presentation_figures.py) to regenerate all thirteen graphs from saved results and the verified original CSV. The figure manifest records provenance, chart scopes, and residuals outside the histogram's displayed range. This ten-slide version replaces the earlier compact deck.

The presentation PDF is drawn directly from the common layout, rather than exported by Microsoft PowerPoint. PowerPoint font rendering can vary slightly by installation.
