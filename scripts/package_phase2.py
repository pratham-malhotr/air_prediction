"""Package source, reports, an executed notebook and verified Phase 2 artifacts."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def package():
    run = ROOT / 'artifacts/phase2_research'
    manifest = json.loads((run / 'run_manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('is_synthetic', True):
        raise ValueError('A completed real-data Phase 2 research run is required')
    verification = json.loads((run / 'verification.json').read_text())
    if verification['source_code_sha256'] != manifest['source_code_sha256']:
        raise ValueError('Verify the active research run before packaging')
    for name, digest in manifest['model_sha256'].items():
        if sha256((run / f'{name}.joblib').read_bytes()).hexdigest() != digest:
            raise ValueError(f'Stale {name} model artifact')
    required = [ROOT / 'reports/phase2/Aircast_Phase2_Report.pdf',
                ROOT / 'reports/phase2/Aircast_Phase2_Presentation.pptx',
                ROOT / 'reports/phase2/slides.pdf', ROOT / 'notebooks/Air_Prediction_Phase2.ipynb']
    if any(not path.is_file() for path in required):
        raise ValueError('Build all Phase 2 submission documents before packaging')
    for path in required[:3]:
        if sha256(path.read_bytes()).hexdigest() != verification['document_sha256'][path.name]:
            raise ValueError(f'Reverify the updated submission document: {path.name}')
    for folder in ('phase2_external', 'phase2_production'):
        evidence = json.loads((ROOT / 'artifacts' / folder / 'verification.json').read_text())
        if evidence['status'] != 'passed':
            raise ValueError(f'Incomplete {folder} verification')
        if folder == 'phase2_production' and evidence['source_code_sha256'] != manifest['source_code_sha256']:
            raise ValueError('Production verification uses a different research source')
    files = {}
    def add(path):
        path = Path(path)
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'Invalid package source: {path}')
        name = path.relative_to(ROOT).as_posix()
        files[name] = path
    for name in ('README.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 'LICENSE', 'pyproject.toml',
                 'requirements.txt', 'requirements-phase2.txt', 'Start_Aircast.command'):
        add(ROOT / name)
    extensions = {'.py', '.sh', '.toml', '.md', '.tex', '.bib', '.yml', '.yaml', '.html',
                  '.css', '.js', '.cjs', '.svg', '.csv', '.json', '.pdf', '.pptx', '.png', '.ipynb', '.txt'}
    for directory in ('src', 'frontend', 'deployment', 'configs', 'docs', 'reports', 'notebooks', 'tests', 'scripts', '.github'):
        for path in sorted((ROOT / directory).rglob('*')):
            if (path.is_file() and '__pycache__' not in path.parts
                    and (path.suffix in extensions or path.name in {'Dockerfile', 'Dockerfile.dockerignore'})):
                add(path)
    phase1 = ROOT / 'artifacts/phase1'
    for name in ('run_manifest.json', 'selection.json', 'predictions.csv', 'metrics.csv', 'fold_metrics.csv',
                 'errors_by_pollution_regime.csv', 'ridge.joblib', 'xgboost.joblib'):
        add(phase1 / name)
    for path in run.iterdir():
        if path.is_file() and path.suffix in {'.json', '.csv', '.gz', '.joblib', '.md', '.xml'}:
            add(path)
    external = ROOT / 'artifacts/phase2_external'
    if (external / 'run_manifest.json').is_file():
        for path in external.rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.csv', '.gz', '.joblib', '.md'}:
                add(path)
    production = ROOT / 'artifacts/phase2_production/verification.json'
    if production.is_file():
        add(production)
    data = ROOT / 'data/raw/PRSA_data_2010.1.1-2014.12.31.csv'
    if sha256(data.read_bytes()).hexdigest() != manifest['dataset_audit']['sha256']:
        raise ValueError('Raw data differs from the evaluated run')
    for path in (data, data.with_suffix('.provenance.json'), ROOT / 'data/README.md'):
        add(path)
    destination = ROOT / 'dist/air_prediction_phase2.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    start = '''# Aircast Phase 2 submission

1. Create a Python 3.13 virtual environment and activate it.
2. Run `python -m pip install -r requirements-phase2.txt` from this directory.
3. On macOS, install XGBoost's runtime with `brew install libomp` if needed.
4. Run `bash scripts/start_frontend.sh` to use the saved local models.
5. Open `notebooks/Air_Prediction_Phase2.ipynb` to inspect verified results and inference.
6. Run `python scripts/run_phase2_research.py --output artifacts/phase2-new-run` to repeat the experiment.
7. Run `pytest -q -W error::DeprecationWarning` for automated verification.

The editable IEEE report, presentation, slide PDF and speaker guide are in `reports/phase2/`.
The saved models and complete candidate evidence are in `artifacts/phase2_research/`.
Production Gunicorn/container instructions are in `deployment/phase2/README.md`.
The frozen cross-station evaluation and its limitations are in `docs/phase2_external_validation.md`.
The local dashboard automatically loads this completed research run. The hosted classical app is separate.
Model selection uses 2013; 2014 comparisons are exploratory because Phase 1 results were already viewed.
Dataset: Chen (2015), UCI Beijing PM2.5, DOI 10.24432/C5JS49, CC BY 4.0; see data/README.md.
Python environments, credentials, Git metadata and training checkpoints are excluded.
'''
    checksums = {name: sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())}
    checksums['START_PHASE2.md'] = sha256(start.encode()).hexdigest()
    metadata = {'format': 1, 'run_source_sha256': manifest['source_code_sha256'],
                'dataset_sha256': manifest['dataset_audit']['sha256'], 'files': checksums}
    with ZipFile(destination, 'w', compression=ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr('START_PHASE2.md', start)
        archive.writestr('PACKAGE_MANIFEST.json', json.dumps(metadata, indent=2) + '\n')
    # Verify every packaged file without extracting or executing archive content.
    with ZipFile(destination) as archive:
        for name, digest in checksums.items():
            if sha256(archive.read(name)).hexdigest() != digest:
                raise RuntimeError(f'Package verification failed: {name}')
        if archive.testzip() is not None:
            raise RuntimeError('Corrupt ZIP entry')
    print(f'Packaged and verified {len(files)} files: {destination} ({destination.stat().st_size / 1e6:.1f} MB)')
    return destination


if __name__ == '__main__':
    package()
