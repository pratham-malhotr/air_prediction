"""Keep extracted handoffs usable independently of the author's checkout."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
import json
import os
import shutil

from scripts import build_phase1_notebook, package_phase2


def test_phase2_package_preserves_notebook_verification_inputs_and_pinned_release(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    repo = Path(__file__).resolve().parents[1]

    def write(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value.encode() if isinstance(value, str) else value)
        return path

    def record(name, value):
        return write(name, json.dumps(value))

    for name in ('README.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 'LICENSE', 'pyproject.toml',
                 'requirements.txt', 'requirements-phase1.txt', 'requirements-phase2.txt',
                 'Air_Prediction_Phase1_Colab.ipynb', 'Start_Aircast.command'):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, destination)
    source = 'synthetic-fixture-source'
    data = write('data/raw/PRSA_data_2010.1.1-2014.12.31.csv', 'Synthetic package fixture\n')
    data_hash = sha256(data.read_bytes()).hexdigest()
    record('data/raw/PRSA_data_2010.1.1-2014.12.31.provenance.json', {'sha256': data_hash})
    write('data/README.md', 'Synthetic packaging fixture, never research evidence.\n')
    document_hashes = {}
    for name in ('Aircast_Phase2_Report.pdf', 'Aircast_Phase2_Presentation.pptx', 'slides.pdf'):
        path = write('reports/phase2/' + name, 'Synthetic document fixture: ' + name)
        document_hashes[name] = sha256(path.read_bytes()).hexdigest()
    write('notebooks/Air_Prediction_Phase2.ipynb', '{"cells":[]}')
    models = {}
    for name in ('lstm', 'bilstm', 'hybrid'):
        path = write('artifacts/phase2_research/' + name + '.joblib', 'Synthetic weight fixture')
        models[name] = sha256(path.read_bytes()).hexdigest()
    record('artifacts/phase2_research/run_manifest.json', {
        'status': 'complete', 'is_synthetic': False, 'source_code_sha256': source,
        'dataset_audit': {'sha256': data_hash}, 'model_sha256': models,
    })
    record('artifacts/phase2_research/verification.json', {
        'source_code_sha256': source, 'document_sha256': document_hashes,
    })
    for name in ('run_manifest.json', 'selection.json', 'predictions.csv', 'metrics.csv',
                 'fold_metrics.csv', 'errors_by_pollution_regime.csv', 'ridge.joblib', 'xgboost.joblib'):
        write('artifacts/phase1/' + name, 'Synthetic classical fixture')
    # Consumers previously failed because these files were absent from the ZIP.
    required = (
        'artifacts/phase1/eda/missingness.csv', 'artifacts/phase1/eda/EDA_NOTES.md',
        'artifacts/phase1/figures/model_comparison.png', 'artifacts/phase1/validation_candidates.csv',
        'artifacts/phase1/validation_ablation.csv', 'artifacts/phase1/fold_audit.json',
        'artifacts/frontend_review/browser_checks.json', 'data/raw/external/uci501.provenance.json',
    )
    for name in required:
        write(name, 'Synthetic evidence fixture: ' + name)
    record('artifacts/phase2_external/verification.json', {'status': 'passed'})
    record('artifacts/phase2_production/verification.json', {'status': 'passed', 'source_code_sha256': source})
    pinned = write('dist/air_prediction_phase2.zip', 'Immutable published release fixture')
    pinned_bytes = pinned.read_bytes()
    monkeypatch.setattr(package_phase2, 'ROOT', root)
    archive = package_phase2.package()
    assert archive.name == 'air_prediction_phase2_completed.zip'
    assert pinned.read_bytes() == pinned_bytes
    with ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read('PACKAGE_MANIFEST.json'))
        for name in required:
            assert bundle.read(name) == (root / name).read_bytes()
            assert manifest['files'][name] == sha256(bundle.read(name)).hexdigest()


def test_phase1_notebook_retrains_separately_after_an_inference_only_restore(tmp_path, monkeypatch):
    monkeypatch.delenv('AIRCAST_PHASE1_RETRAIN', raising=False)
    original = tmp_path / 'artifacts/phase1/run_manifest.json'
    original.parent.mkdir(parents=True)
    original.write_text('{"status":"complete","inference_only":true}')
    original_bytes = original.read_bytes()
    config = {'horizon_hours': 1}
    calls = []

    def train(data, cfg, output, **kwargs):
        calls.append(output)
        output.mkdir(parents=True)
        (output / 'run_manifest.json').write_text(json.dumps({
            'status': 'complete', 'is_synthetic': False, 'dataset_audit': {'sha256': 'fixture'},
            'config': config, 'source_code_sha256': 'fixture-source',
        }))

    notebook = build_phase1_notebook.build()
    cell = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code'
                and 'RUN_FULL_TRAINING =' in ''.join(cell['source']))
    namespace = {'ROOT': tmp_path, 'os': os, 'json': json, 'run_experiment': train,
                 'CONFIG': SimpleNamespace(to_dict=lambda: config), 'DATA': tmp_path / 'data.csv',
                 'dataset': SimpleNamespace(audit={'sha256': 'fixture'})}
    exec(compile(''.join(cell['source']), '<notebook training cell>', 'exec'), namespace)
    assert len(calls) == 1 and calls[0] != original.parent
    assert namespace['RUN'] == calls[0]
    assert original.read_bytes() == original_bytes
