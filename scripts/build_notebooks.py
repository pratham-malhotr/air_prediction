"""Build readable walkthroughs with an embedded, identical copy of project source.

Only the Python standard library is required. No dataset or credential is embedded.
"""

import base64
from io import BytesIO
import json
from pathlib import Path
import textwrap
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]


def source_files():
    for name in ("README.md", "requirements.txt", "pyproject.toml", ".gitignore", "data/README.md", "data/raw/.gitkeep", "artifacts/.gitkeep"):
        path = ROOT / name
        if path.exists():
            yield path
    for name in ("src", "configs", "docs", "reports", "tests", "scripts"):
        for path in sorted((ROOT / name).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".toml", ".md", ".tex", ".bib"}:
                yield path


def source_bundle():
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        for path in source_files():
            info = ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return base64.b64encode(stream.getvalue()).decode()


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}


def code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(source).strip() + "\n"}


def common_cells(bundle, eda_only=False):
    heading = "Training-period EDA" if eda_only else "Complete Phase 1 Machine Learning project"
    bootstrap = '''
import base64, io, os, sys, zipfile
from pathlib import Path

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    PROJECT_ROOT = Path('/content/air_prediction')
elif (Path.cwd() / 'src' / 'air_prediction').is_dir():
    PROJECT_ROOT = Path.cwd()
elif (Path.cwd().parent / 'src' / 'air_prediction').is_dir():
    PROJECT_ROOT = Path.cwd().parent
else:
    PROJECT_ROOT = Path.cwd() / 'air_prediction'

# This ZIP contains the same readable source/config/docs/tests shipped with the project.
# It contains no dataset, login credentials, trained model, or claimed result.
SOURCE_BUNDLE = "__BUNDLE__"
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
if not (PROJECT_ROOT / 'src' / 'air_prediction').exists():
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_BUNDLE))) as bundle:
        for member in bundle.infolist():
            target = (PROJECT_ROOT / member.filename).resolve()
            if not target.is_relative_to(PROJECT_ROOT.resolve()):
                raise ValueError('Invalid bundled source path')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(member))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.cache' / 'matplotlib'))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
print('Project source:', PROJECT_ROOT)
print('Colab runtime:', IN_COLAB)
'''.replace("__BUNDLE__", bundle)
    return [
        md(f"""
        # Air Quality Prediction — Project 19
        ## {heading}

        **Phase 1 only.** Predict PM2.5 one hour ahead using historical measurements.
        Required comparison: XGBoost, persistence, daily seasonal naive; Ridge adds a linear reference.

        **How to use:** upload this notebook at [Google Colab](https://colab.research.google.com/),
        choose a CPU runtime, and select **Runtime → Run all**. No GitHub connection is required.
        Runtime and internet availability depend on your Colab session.
        The source is included below, so this single notebook is sufficient to start.
        Save an executed copy to your own Drive and download the outputs at the end.

        **Evidence policy:** an unexecuted notebook contains no measured results. This notebook uses the
        actual [UCI Beijing PM2.5 dataset](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data),
        never a silent synthetic replacement. Source: Chen (2015), DOI 10.24432/C5JS49, CC BY 4.0.
        """),
        md("""
        ### 1. Restore the reusable source
        Run this setup cell. In Colab the Files sidebar will show the package under `/content/air_prediction`.
        All feature, split, and model code is readable in `src/air_prediction`; the encoded ZIP merely makes
        the notebook portable without GitHub. When opened in the existing local project, local source is used.
        """),
        code(bootstrap),
        code("""
        import subprocess
        INSTALL_DEPENDENCIES = IN_COLAB
        if INSTALL_DEPENDENCIES:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                            '-r', str(PROJECT_ROOT / 'requirements.txt'), 'pytest>=8,<9'], check=True)
        import numpy as np
        import pandas as pd
        from IPython.display import Image, Markdown, display
        from air_prediction.config import load_config
        from air_prediction.data import FILENAME, download_dataset, load_dataset
        from air_prediction.eda import run_eda
        print('numpy:', np.__version__, 'pandas:', pd.__version__)
        """),
        md("""
        ### 2. Fix the experiment settings before viewing test results
        Forecast origin = t, target = t+1, using Beijing local time. Initial training: 2010–2012.
        Parameter selection: 2013. Final rolling evaluation: 2014. Refits occur quarterly; observed histories
        update hourly. The EDA preview uses only initial training dates. Keep the final holdout for evaluation.
        """),
        code("""
        CONFIG_PATH = PROJECT_ROOT / 'configs' / 'phase1.toml'
        DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / FILENAME
        OUTPUT_ROOT = PROJECT_ROOT / 'artifacts' / 'phase1'
        EDA_DIR = PROJECT_ROOT / 'artifacts' / 'eda_preview'
        config = load_config(CONFIG_PATH)
        display(config.to_dict())
        """),
        code("""
        if not DATA_PATH.is_file():
            download_dataset(DATA_PATH)
        dataset = load_dataset(DATA_PATH)
        display(Markdown('**Data source:** ' + dataset.source))
        display(pd.Series({k: v for k, v in dataset.audit.items()
                           if k not in {'provenance', 'missing_by_column'}}, name='archive audit'))
        display(pd.Series(dataset.audit['missing_by_column'], name='Archive-wide missing count'))
        display(dataset.frame.head())
        """),
        md("""
        ### 3. Training EDA and quality checks
        The full-archive audit above checks timestamps and completeness; the plots and statistics below use
        dates **strictly before 2013**. Distinguish a missing measurement from a true zero. Preserve valid
        pollution peaks; the IQR rule is descriptive, not an automatic deletion rule.
        """),
        code("""
        run_eda(dataset, EDA_DIR, config.validation_start)
        display(pd.read_csv(EDA_DIR / 'missingness.csv', index_col=0))
        display(pd.read_csv(EDA_DIR / 'descriptive_statistics.csv', index_col=0))
        import json
        display(json.loads((EDA_DIR / 'distribution_summary.json').read_text()))
        """),
        md("""
        **Timeline and missingness:** identify changes over time, lengthy gaps, and variables whose
        completeness changes. Describe actual observations after execution; do not infer them from this prompt.
        """),
        code("display(Image(filename=str(EDA_DIR / 'timeline_missingness.png')))"),
        md("""
        **Distribution and calendar:** compare mean and median, tail behavior, and hourly/monthly variation.
        Consider whether peaks are rare but important and whether temporal patterns motivate the chosen lags.
        """),
        code("""
        display(Image(filename=str(EDA_DIR / 'distribution_seasonality.png')))
        display(Image(filename=str(EDA_DIR / 'calendar_profiles.png')))
        """),
        md("""
        **Correlation and autocorrelation:** association does not establish causation. ACF uses the complete
        hourly grid so missing observations do not compress time. Examine lag 1, the daily scale, and the weekly scale.
        """),
        code("""
        display(pd.read_csv(EDA_DIR / 'correlations.csv', index_col=0))
        display(pd.read_csv(EDA_DIR / 'autocorrelation.csv', index_col=0))
        display(Image(filename=str(EDA_DIR / 'correlation_autocorrelation.png')))
        """),
        md("""
        ### 4. Trace the causal features and baseline alignment
        At t we know y(t), so lag zero is valid. Trailing windows end at t. With horizon one,
        daily seasonal naive uses y(t−23), which is 24 hours before the target t+1.
        No evaluation target is imputed. Weather is available only through t.
        """),
        code("""
        from air_prediction.features import build_features, common_evaluation_mask
        supervised = build_features(dataset.frame, config)
        eligible_training = common_evaluation_mask(supervised) & (supervised.target_time < pd.Timestamp(config.validation_start))
        trace = supervised.X.loc[eligible_training, ['pm25_lag_0', 'pm25_lag_24', 'pm25_mean_24', 'weather_TEMP_lag_0']].head()
        trace['target_time'] = supervised.target_time.reindex(trace.index)
        trace['actual_future_target'] = supervised.y.reindex(trace.index)
        trace = trace.join(supervised.baselines)
        display(trace)
        print('Input features before one-hot encoding:', supervised.X.shape[1])
        print('Feature source:', PROJECT_ROOT / 'src' / 'air_prediction' / 'features.py')
        """),
    ]


def training_cells():
    return [
        md("""
        ### 5. Validate implementation before measuring models
        These tests use clearly labeled synthetic fixtures to check logic; they are not Beijing results.
        The optional XGBoost smoke test runs when the dependency is installed.
        """),
        code("""
        RUN_TESTS = True
        if RUN_TESTS:
            subprocess.run([sys.executable, '-m', 'pytest', '-q', str(PROJECT_ROOT / 'tests')],
                           cwd=PROJECT_ROOT, check=True)
        """),
        md("""
        ### 6. Model selection and final rolling backtests
        The code chooses XGBoost settings by pooled RMSE across the four 2013 folds, saves the decision,
        then evaluates frozen settings over the four 2014 folds. Ridge uses a fixed alpha of 10.
        All models score the same observed targets. Coverage and missingness exclusions are saved.
        Optional feature ablations use validation dates only. No LSTM or hybrid model is trained.

        **Run once per output directory.** Completed results are protected from overwriting. To start another
        deliberate experiment, change `OUTPUT_ROOT` and document the change. Report templates default to
        `artifacts/phase1`; update their `resultroot` if you intentionally choose another run directory.
        """),
        code("""
        from air_prediction.experiment import run_experiment
        MODELS = ('xgboost', 'ridge')
        RUN_ABLATIONS = True
        result = run_experiment(DATA_PATH, config, OUTPUT_ROOT, names=MODELS, ablations=RUN_ABLATIONS)
        display(pd.read_csv(OUTPUT_ROOT / 'validation_candidates.csv'))
        display(json.loads((OUTPUT_ROOT / 'selection.json').read_text()))
        display(result['metrics'])
        display(result['fold_metrics'])
        """),
        md("""
        ### 7. Measured comparisons, coverage, and failure modes
        MAE/RMSE are in µg/m³. R² can be negative. Pooled scores weight each scored hour equally; do not
        average quarter-specific R² values. Persistence may be difficult to beat at one hour. Report the outcome
        honestly, then inspect quarter variation, high-pollution errors, bias, and remaining residual structure.
        """),
        code("""
        display(Markdown((OUTPUT_ROOT / 'RESULTS.md').read_text()))
        display(pd.DataFrame(json.loads((OUTPUT_ROOT / 'fold_audit.json').read_text())))
        display(Image(filename=str(OUTPUT_ROOT / 'figures' / 'model_comparison.png')))
        display(Image(filename=str(OUTPUT_ROOT / 'figures' / 'forecast_last_week.png')))
        display(Image(filename=str(OUTPUT_ROOT / 'figures' / 'residual_diagnostics.png')))
        display(pd.read_csv(OUTPUT_ROOT / 'errors_by_pollution_regime.csv'))
        if (OUTPUT_ROOT / 'validation_ablation.csv').is_file():
            display(pd.read_csv(OUTPUT_ROOT / 'validation_ablation.csv'))
        if (OUTPUT_ROOT / 'feature_importance.csv').is_file():
            display(pd.read_csv(OUTPUT_ROOT / 'feature_importance.csv').head(20))
        """),
        md("""
        ### 8. Demonstrate a saved-model forecast
        The deployment model is refitted after backtesting. Forecasting after the archive endpoint is a
        historical next-hour forecast, **not current live air quality**. No AQI category is computed from this
        hourly point prediction. Explain these distinctions in your individual viva.
        """),
        code("""
        from air_prediction.predict import predict_latest
        display(predict_latest(OUTPUT_ROOT / (MODELS[0] + '.joblib'), DATA_PATH))
        display(Markdown((PROJECT_ROOT / 'docs' / 'phase1_checklist.md').read_text()))
        """),
    ]


def export_cells():
    return [
        md("""
        ### Download and finish the Phase 1 submission
        The ZIP includes readable source, the input CSV/provenance, measured outputs, the IEEE report source,
        presentation source, and viva/contribution guides. Also save this executed notebook to your own Drive.
        Fill the five real team members, interpret actual EDA/results, and compile/proofread the report and slides.
        Their templates are intentionally honest about any results that have not yet been generated.

        Dataset: CC BY 4.0; cite UCI dataset DOI 10.24432/C5JS49. See `docs/literature_review.md` for primary sources.
        """),
        code("""
        export_path = PROJECT_ROOT.parent / 'air_prediction_phase1_outputs.zip'
        excluded = {'__pycache__', '.cache', '.pytest_cache', '.git', '.venv', '.ipynb_checkpoints'}
        with zipfile.ZipFile(export_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(PROJECT_ROOT.rglob('*')):
                relative = path.relative_to(PROJECT_ROOT)
                if path.is_file() and not (set(relative.parts) & excluded) and path.suffix not in {'.pyc', '.pyo'}:
                    archive.write(path, relative.as_posix())
        print('Saved:', export_path)
        if IN_COLAB:
            from google.colab import files
            files.download(str(export_path))
        else:
            display(Markdown('Output archive: `' + str(export_path) + '`'))
        """),
    ]


def write_notebook(path, cells):
    for number, cell in enumerate(cells):
        cell["id"] = f"phase1-{number:03d}"
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11"}, "colab": {"provenance": [], "name": path.name}},
                "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    bundle = source_bundle()
    write_notebook(ROOT / "Air_Prediction_Phase1_Colab.ipynb", common_cells(bundle) + training_cells() + export_cells())
    write_notebook(ROOT / "notebooks" / "01_eda.ipynb", common_cells(bundle, eda_only=True) + export_cells())
    print("Built self-contained Phase 1 and EDA notebooks.")


if __name__ == "__main__":
    main()
