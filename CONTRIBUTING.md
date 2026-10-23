# Development workflow

The project uses Python 3.11 or newer. Install the package and development dependencies from the project root:

```bash
python -m pip install -e '.[dev,web]'
```

The XGBoost minimum is 2.1.4, which includes the [scikit-learn 1.6 compatibility fix](https://xgboost.readthedocs.io/en/stable/changes/v2.1.0.html).

The command-line workflow is:

```bash
air-prediction download
air-prediction eda
air-prediction train
```

The preparation and report scripts use the same data/configuration as the pipeline:

```bash
python scripts/prepare_data.py
python scripts/build_analysis.py
```

The preparation script exports the cleaned hourly series, gap dates, feature dictionary and a worked preview. The analysis script requires a completed real-data run and creates supplementary tables/figures. For the full validation feature comparison, train with `air-prediction train --ablations`.

With Tectonic installed, `bash scripts/build_pdfs.sh` builds the IEEE report and presentation. Its first run downloads the TeX support files into the project's cache.

An existing dataset can be supplied with `--data /path/to/file.csv`. The configuration is `configs/phase1.toml`. For a new experiment, use a new `--output` directory; completed runs are preserved.

Run the implementation checks before submitting changes:

```bash
python -m pytest -q
```

Keep training-label availability, target alignment and common evaluation timestamps intact. Fit learned preprocessing on training dates only. Changes to the experimental protocol after viewing holdout results must be recorded as exploratory changes, rather than reusing the same holdout as untouched evidence.

Use the actual dataset for project findings. Synthetic fixtures are for implementation tests. Preserve source attribution, input hashes, parameter settings and measured outputs. Keep datasets, model artifacts, credentials and caches out of source commits.

Keep the report, slides and explanations consistent with the implemented methods and measured results.

The optional dashboard lives in `frontend/` and uses the same saved model artifacts. Start it with `bash scripts/start_frontend.sh`; see `docs/frontend.md` for inputs and verification. Keep web-specific dependencies in the `web` extra so the ML pipeline remains usable independently.
