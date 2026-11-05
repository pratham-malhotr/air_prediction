"""Verify the completed Phase 2 research, saved forecasts and submission files."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import os
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
if (ROOT / '.dependencies/documents').exists():
    sys.path.append(str(ROOT / '.dependencies/documents'))

import joblib
import pandas as pd
from air_prediction.data import load_dataset
from air_prediction.experiment import source_fingerprint
from air_prediction.features import build_features
from air_prediction.metrics import score_predictions
from air_prediction.predict import predict_latest
from frontend.service import ForecastService


def verify():
    run = ROOT / 'artifacts/phase2_research'
    report = ROOT / 'reports/phase2'
    manifest = json.loads((run / 'run_manifest.json').read_text())
    assert manifest['status'] == 'complete' and not manifest['is_synthetic']
    assert manifest['source_code_sha256'] == source_fingerprint()
    assert not (run / 'RUNNING.json').exists()
    selection = json.loads((run / 'selection.json').read_text())
    assert selection['holdout_used_for_selection'] is False
    assert sha256((run / 'selection.json').read_bytes()).hexdigest() == manifest['selection_sha256']
    for filename, field in (('search_metrics.csv', 'search_metrics_sha256'),
                             ('ablation_metrics.csv', 'ablation_metrics_sha256'),
                             ('hybrid_search_metrics.csv', 'hybrid_search_metrics_sha256')):
        assert sha256((run / filename).read_bytes()).hexdigest() == selection[field]
    provenance = manifest['baseline_provenance']
    phase1 = ROOT / 'artifacts/phase1'
    assert sha256((phase1 / 'predictions.csv').read_bytes()).hexdigest() == provenance['predictions_sha256']
    assert sha256((phase1 / 'run_manifest.json').read_bytes()).hexdigest() == provenance['manifest_sha256']
    predictions = pd.read_csv(run / 'predictions.csv', parse_dates=['origin_time', 'target_time'], low_memory=False)
    assert not predictions.duplicated(['phase', 'model', 'origin_time']).any()
    pd.testing.assert_frame_equal(score_predictions(predictions), pd.read_csv(run / 'metrics.csv'),
                                  check_exact=False, rtol=1e-10)
    assert predictions.groupby(['phase','model']).size().groupby('phase').nunique().eq(1).all()
    for audit in json.loads((run / 'fold_audit.json').read_text()):
        if audit['phase'] != 'deployment':
            assert pd.Timestamp(audit['latest_training_target']) <= pd.Timestamp(audit['fit_time'])
    for filename in ('hybrid_fold_audit.json', 'hybrid_comparison_audit.json'):
        for audit in json.loads((run / filename).read_text()):
            assert pd.Timestamp(audit['base_latest_target']) <= pd.Timestamp(audit['calibration_start'])
            assert pd.Timestamp(audit['preprocessing_fit_end']) < pd.Timestamp(audit['calibration_start'])
    dataset_path = ROOT / 'data/raw/PRSA_data_2010.1.1-2014.12.31.csv'
    dataset = load_dataset(dataset_path)
    assert dataset.audit['sha256'] == manifest['dataset_audit']['sha256']
    service = ForecastService(ROOT)
    assert service.default_model == selection['preferred_model']
    forecasts = {}
    for name, digest in manifest['model_sha256'].items():
        model_path = run / f'{name}.joblib'
        assert sha256(model_path.read_bytes()).hexdigest() == digest
        artifact = joblib.load(model_path)
        assert artifact['source_code_sha256'] == manifest['source_code_sha256']
        if name != 'hybrid':
            data = build_features(dataset.frame, artifact['config'])
            eligible = (data.warm & data.y.notna()
                        & (data.X.index >= pd.Timestamp(artifact['config'].train_start))
                        & (data.target_time <= dataset.frame.index.max()))
            assert artifact['pipeline'].training_rows_ == eligible.sum()
        cli = predict_latest(model_path, dataset_path)
        live = service.forecast(mode='history', selected=name)
        assert abs(cli['pm25_ug_m3'] - live['forecast']) < 1e-5
        assert live['backtest']['n'] == 8531
        forecasts[name] = cli['pm25_ug_m3']
    import pymupdf
    from pptx import Presentation
    paper = pymupdf.open(report / 'Aircast_Phase2_Report.pdf')
    slides_pdf = pymupdf.open(report / 'slides.pdf')
    text = '\n'.join(page.get_text() for page in paper)
    assert '@@' not in text
    for enrollment in ('2401010407', '2401010343', '2401010042', '2401010478', '2401010188'):
        assert enrollment in text
    for row in pd.read_csv(run / 'metrics.csv').query("phase == 'holdout'").itertuples():
        assert f'{row.rmse:.3f}' in text
    presentation = Presentation(report / 'Aircast_Phase2_Presentation.pptx')
    assert len(presentation.slides) == len(slides_pdf) == 12
    assert all(slide.notes_slide.notes_text_frame.text.strip() for slide in presentation.slides)
    notebook = json.loads((ROOT / 'notebooks/Air_Prediction_Phase2.ipynb').read_text())
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert all(cell['execution_count'] is not None for cell in cells)
    assert not any(item.get('output_type') == 'error' for cell in cells for item in cell['outputs'])
    suite = ET.parse(run / 'test-results.xml').getroot().find('testsuite')
    assert suite is not None and int(suite.attrib['failures']) == int(suite.attrib['errors']) == 0
    browser = json.loads((ROOT / 'artifacts/frontend_review/browser_checks.json').read_text())
    assert browser['status'] == 'passed'
    payloads = {}
    for filename in ('Aircast_Phase2_Report.pdf', 'Aircast_Phase2_Report.tex',
                     'Aircast_Phase2_Presentation.pptx', 'slides.pdf', 'presenter_guide.md'):
        payloads[filename] = sha256((report / filename).read_bytes()).hexdigest()
    evidence = {
        'verified_utc': datetime.now(timezone.utc).isoformat(),
        'source_code_sha256': manifest['source_code_sha256'],
        'tests': {'passed': int(suite.attrib['tests']), 'failures': 0, 'errors': 0},
        'browser': browser, 'models': list(service.models), 'default_model': service.default_model,
        'required_history_hours': service.history_size, 'cli_dashboard_forecasts': forecasts,
        'report_pages': len(paper), 'presentation_slides': len(presentation.slides),
        'executed_notebook_cells': len(cells), 'document_sha256': payloads,
        'checks': ['source/input/model hashes', 'frozen validation selection and evidence',
                   'original Phase 1 preservation', 'common hours and recomputed metrics',
                   'label/calibration boundaries', 'full-window independent training',
                   'real CLI/dashboard forecast parity', 'paper/team/score consistency',
                   'editable slide count and embedded notes', 'executed notebook', 'tests and browser checks'],
    }
    for destination in (run / 'verification.json', ROOT / 'docs/results/phase2_research/verification.json',
                        report / 'deliverable_manifest.json'):
        destination.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({key: evidence[key] for key in ('tests', 'default_model', 'report_pages',
                                                  'presentation_slides', 'executed_notebook_cells')}, indent=2))
    return evidence


if __name__ == '__main__':
    verify()
