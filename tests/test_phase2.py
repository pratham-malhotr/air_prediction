"""Small synthetic experiments exercise Phase 2 provenance and common-row scoring."""
from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')
pytest.importorskip('xgboost')

from air_prediction.config import Config
from air_prediction.data import load_dataset
from air_prediction.deep_learning import DLConfig
from air_prediction.features import build_features
from air_prediction.phase2 import _combine_on_identical_hours, run_phase2
from air_prediction.predict import predict_latest
from air_prediction.splits import fold_masks, make_folds


@pytest.fixture
def phase2_run(tmp_path, hourly_frame):
    frame = hourly_frame.copy()
    frame['pm2.5'] = 30 + 10 * np.sin(np.arange(len(frame)) / 10)
    csv = tmp_path / 'synthetic.csv'
    raw = frame.copy()
    for column in ('year', 'month', 'day', 'hour'):
        raw[column] = getattr(frame.index, column)
    raw.to_csv(csv, index=False)
    csv.with_suffix('.provenance.json').write_text(json.dumps({'source': 'synthetic'}))
    config = replace(Config(), lags=(0, 1, 24), rolling_windows=(3, 24),
                     validation_start='2010-01-20', holdout_start='2010-02-01',
                     holdout_end='2010-02-11', fold_months=1, minimum_training_rows=20)
    data = build_features(load_dataset(csv).frame, config)
    baseline = tmp_path / 'phase1'
    baseline.mkdir()
    rows = []
    for phase in ('validation', 'holdout'):
        for fold in make_folds(config, phase):
            _, test, _ = fold_masks(data, fold, config)
            for name in data.baselines:
                rows.append(pd.DataFrame({'origin_time': data.X.index[test],
                    'target_time': data.target_time.loc[test].values, 'actual': data.y.loc[test].values,
                    'prediction': data.baselines.loc[test, name].values,
                    'model': name, 'phase': phase, 'fold': fold.name}))
    pd.concat(rows).to_csv(baseline / 'predictions.csv', index=False)
    (baseline / 'run_manifest.json').write_text(json.dumps({
        'status': 'complete', 'config': config.to_dict(), 'source_code_sha256': 'fixture',
        'dataset_audit': load_dataset(csv).audit, 'versions': {'fixture': True}}))
    before = {p.name: p.read_bytes() for p in baseline.iterdir()}
    output = tmp_path / 'phase2'
    cfg = DLConfig(architecture='bilstm', seq_len=4, hidden_dim=4, num_layers=1,
                   epochs=1, batch_size=64)
    result = run_phase2(csv, config, output, baseline, cfg, {'n_estimators': 3, 'max_depth': 2})
    return output, baseline, before, result, csv, config, cfg


def test_phase2_experiment_preserves_baselines_and_scores_common_hours(phase2_run):
    output, baseline, before, result, csv, config, cfg = phase2_run
    assert before == {p.name: p.read_bytes() for p in baseline.iterdir()}
    assert not (output / 'RUNNING.json').exists()
    manifest = json.loads((output / 'run_manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['is_synthetic']
    assert manifest['required_history_hours'] == config.warmup_hours + cfg.seq_len
    assert json.loads((output / 'selection.json').read_text())['holdout_used_for_selection'] is False
    predictions = pd.read_csv(output / 'predictions.csv')
    assert predictions.groupby(['phase', 'model']).size().groupby('phase').nunique().eq(1).all()
    scores = result['metrics']
    assert set(scores.model) == {'bilstm', 'hybrid', 'persistence', 'seasonal_naive'}
    for name in ('bilstm', 'hybrid'):
        assert sha256((output / f'{name}.joblib').read_bytes()).hexdigest() == manifest['model_sha256'][name]
        assert np.isfinite(predict_latest(output / f'{name}.joblib', csv)['pm25_ug_m3'])
    with pytest.raises(FileExistsError):
        run_phase2(csv, config, output, baseline, cfg)


def test_phase2_comparison_rejects_different_hours_and_targets(phase2_run):
    output, *_ = phase2_run
    predictions = pd.read_csv(output / 'predictions.csv', parse_dates=['origin_time', 'target_time'])
    neural = predictions.loc[predictions.model.isin(['bilstm', 'hybrid'])]
    baseline = predictions.loc[~predictions.model.isin(['bilstm', 'hybrid'])].copy()
    with pytest.raises(ValueError, match='hours differ'):
        _combine_on_identical_hours(neural, baseline.iloc[1:])
    baseline.iloc[0, baseline.columns.get_loc('actual')] += 1
    with pytest.raises(ValueError, match='targets or folds differ'):
        _combine_on_identical_hours(neural, baseline)


def test_phase2_cli_runs_without_launcher_thread_environment(phase2_run):
    import os
    import subprocess
    import sys

    output, _, _, _, csv, *_ = phase2_run
    environment = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment.pop(name, None)
    result = subprocess.run([sys.executable, '-m', 'air_prediction', 'predict',
                             '--model', str(output / 'hybrid.joblib'), '--data', str(csv)],
                            env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert np.isfinite(json.loads(result.stdout)['pm25_ug_m3'])
