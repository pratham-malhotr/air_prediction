"""Exercise validation selection before comparison and independent neural fitting."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('torch')
pytest.importorskip('xgboost')

from air_prediction.config import Config
from air_prediction.data import load_dataset
from air_prediction.deep_learning import DLConfig
from air_prediction.features import build_features
from air_prediction.phase2_research import (ResearchExperiment, chronological_calibration,
    run_research, summarize_seeds)
from air_prediction.predict import predict_latest
from air_prediction.splits import fold_masks, make_folds


@pytest.fixture
def research_run(tmp_path, hourly_frame, monkeypatch):
    frame = hourly_frame.copy()
    frame['pm2.5'] = 50 + 5 * np.sin(np.arange(len(frame)) / 10)
    for key in ('year', 'month', 'day', 'hour'):
        frame[key] = getattr(frame.index, key)
    csv = tmp_path / 'synthetic.csv'
    frame.to_csv(csv, index=False)
    csv.with_suffix('.provenance.json').write_text(json.dumps({'source': 'synthetic'}))
    config = replace(Config(), lags=(0, 1, 24), rolling_windows=(3, 24),
                     validation_start='2010-01-20', holdout_start='2010-02-01',
                     holdout_end='2010-02-11', fold_months=1, minimum_training_rows=20)
    dataset = load_dataset(csv)
    data = build_features(dataset.frame, config)
    baseline = tmp_path / 'phase1'
    baseline.mkdir()
    tables = []
    for phase in ('validation', 'holdout'):
        for fold in make_folds(config, phase):
            _, mask, _ = fold_masks(data, fold, config)
            for name in data.baselines:
                tables.append(pd.DataFrame({'origin_time': data.X.index[mask],
                    'target_time': data.target_time.loc[mask].values, 'actual': data.y.loc[mask].values,
                    'prediction': data.baselines.loc[mask, name].values, 'model': name, 'phase': phase, 'fold': fold.name}))
    pd.concat(tables).to_csv(baseline / 'predictions.csv', index=False)
    (baseline / 'run_manifest.json').write_text(json.dumps({'status': 'complete',
        'config': config.to_dict(), 'source_code_sha256': 'fixture',
        'dataset_audit': dataset.audit, 'versions': {'fixture': True}}))
    cfg = DLConfig(architecture='bilstm', seq_len=4, hidden_dim=4, num_layers=1,
                   epochs=1, batch_size=64, target_mode='delta')
    settings = {'seeds': [42, 137], 'comparison_seed': 42,
        'candidates': [{'name': 'tiny', 'config': asdict(cfg)}],
        'calibration_fractions': [.1, .2], 'correction_weights': [0.0, .5, 1.0],
        'residual_candidates': [{'n_estimators': 3, 'max_depth': 2}]}
    output = tmp_path / 'research'
    original = ResearchExperiment.neural_cv
    frozen = []
    def check_selection(self, cfg, name, phase='validation', candidate=None):
        if phase == 'holdout':
            path = output / 'selection.json'
            assert path.exists()
            frozen.append(sha256(path.read_bytes()).hexdigest())
        return original(self, cfg, name, phase, candidate)
    monkeypatch.setattr(ResearchExperiment, 'neural_cv', check_selection)
    result = run_research(csv, config, output, baseline, settings)
    return result, output, csv, config, data, frozen


def test_research_freezes_selection_and_uses_complete_standalone_training(research_run):
    result, output, csv, config, data, frozen = research_run
    assert frozen and len(set(frozen)) == 1
    assert not (output / 'RUNNING.json').exists()
    manifest = json.loads((output / 'run_manifest.json').read_text())
    assert manifest['selection_sha256'] == frozen[0]
    assert manifest['is_synthetic']
    assert set(result['metrics'].model) == {'bilstm', 'lstm', 'hybrid', 'persistence', 'seasonal_naive'}
    import joblib
    standalone = joblib.load(output / 'bilstm.joblib')
    hybrid = joblib.load(output / 'hybrid.joblib')
    eligible = data.warm & data.y.notna() & (data.target_time <= data.X.index.max())
    assert standalone['pipeline'].training_rows_ == eligible.sum()
    assert hybrid['pipeline'].dl.training_rows_ < eligible.sum()
    for name in ('bilstm', 'lstm', 'hybrid'):
        assert np.isfinite(predict_latest(output / f'{name}.joblib', csv)['pm25_ug_m3'])
    stability = pd.read_csv(output / 'seed_stability.csv')
    assert stability.seeds.eq(2).all()


def test_calibration_and_seed_summary_do_not_compress_or_pool_away_seeds(hourly_frame):
    config = replace(Config(), lags=(0, 1), rolling_windows=(3,), minimum_training_rows=10)
    data = build_features(hourly_frame, config)
    eligible = data.warm & data.y.notna()
    base, calibration = chronological_calibration(data, eligible, .2, 4)
    assert not (base & calibration).any()
    assert data.target_time.loc[base].max() <= data.X.index[calibration].min()
    rows = pd.DataFrame({'candidate': ['a'] * 4, 'seed': [1, 1, 2, 2],
                         'actual': [0., 0., 0., 0.], 'prediction': [0., 0., 2., 2.]})
    scores, summary = summarize_seeds(rows)
    assert len(scores) == 2
    assert summary.iloc[0].rmse_mean == 1.0
    assert summary.iloc[0].seeds == 2
