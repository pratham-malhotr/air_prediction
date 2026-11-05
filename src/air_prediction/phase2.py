"""Reproducible Phase 2 backtests, isolated from the recorded Phase 1 run."""

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import time
import tomllib

import joblib
import numpy as np
import pandas as pd
import torch

from .data import load_dataset, write_json
from .deep_learning import DLConfig, DeepLearningPipeline, HybridResidualModel
from .experiment import source_fingerprint
from .features import build_features
from .metrics import score_predictions
from .models import make_model
from .splits import fold_masks, make_folds


def load_phase2_settings(path):
    with Path(path).open('rb') as stream:
        settings = tomllib.load(stream)
    unknown = set(settings) - {'deep_learning', 'residual_model', 'calibration_fraction'}
    if unknown:
        raise ValueError(f'Unknown Phase 2 settings: {sorted(unknown)}')
    return (DLConfig(**settings.get('deep_learning', {})),
            settings.get('residual_model', {}), settings.get('calibration_fraction', 0.2))


def _baseline_predictions(directory, dataset, config):
    directory = Path(directory)
    manifest_path = directory / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Phase 1 comparison requires a completed baseline run')
    if manifest['dataset_audit']['sha256'] != dataset.audit['sha256']:
        raise ValueError('Phase 1 and Phase 2 must use the same input CSV')
    if manifest['config'] != json.loads(json.dumps(config.to_dict())):
        raise ValueError('Phase 1 and Phase 2 must use the same feature and split configuration')
    path = directory / 'predictions.csv'
    predictions = pd.read_csv(path, parse_dates=['origin_time', 'target_time'], float_precision='round_trip')
    if predictions.duplicated(['phase', 'model', 'origin_time']).any():
        raise ValueError('Duplicate Phase 1 comparison rows')
    provenance = {
        'directory': str(directory), 'predictions_sha256': sha256(path.read_bytes()).hexdigest(),
        'manifest_sha256': sha256(manifest_path.read_bytes()).hexdigest(),
        'versions': manifest['versions'], 'source_code_sha256': manifest['source_code_sha256'],
        'note': 'Classical metrics reuse saved Phase 1 predictions; classical models were not retrained.',
    }
    return predictions, provenance


def _combine_on_identical_hours(neural, baseline):
    """Fail rather than silently drop hard hours or compare different targets."""
    reference = neural.loc[neural.model == 'bilstm'].sort_values(['phase', 'origin_time'])
    reference = reference.set_index(['phase', 'origin_time'])
    baseline = baseline.loc[baseline.phase.isin(neural.phase.unique())].copy()
    for name, group in baseline.groupby('model'):
        aligned = group.sort_values(['phase', 'origin_time']).set_index(['phase', 'origin_time'])
        if not aligned.index.equals(reference.index):
            raise ValueError(f'{name}: Phase 1 and Phase 2 evaluation hours differ')
        if (not np.array_equal(aligned.actual, reference.actual)
                or not aligned.target_time.equals(reference.target_time)
                or not aligned.fold.equals(reference.fold)):
            raise ValueError(f'{name}: Phase 1 comparison targets or folds differ')
    if baseline.empty:
        raise ValueError('No Phase 1 comparison predictions available')
    baseline['evaluation_source'] = 'saved_phase1_predictions'
    return pd.concat([neural, baseline], ignore_index=True)


def run_phase2(csv_path, config, output, baseline_dir, dl_config=None,
               residual_parameters=None, calibration_fraction=0.2):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Run directory is not empty: {output}; choose a new output directory')
    if output.resolve() == Path(baseline_dir).resolve():
        raise ValueError('Phase 2 must use a separate output directory')
    dl_config = dl_config or DLConfig(architecture='bilstm')
    if dl_config.architecture != 'bilstm':
        raise ValueError('This experiment reports Bi-LSTM; choose architecture="bilstm"')
    if not 0 < calibration_fraction < 1:
        raise ValueError('calibration_fraction must be between zero and one')
    dataset = load_dataset(csv_path)
    frame = dataset.frame
    if (frame.index.min() > pd.Timestamp(config.train_start)
            or frame.index.max() < pd.Timestamp(config.holdout_end) - pd.Timedelta(hours=1)):
        raise ValueError('Input must cover the complete configured training and evaluation periods')
    baseline, baseline_provenance = _baseline_predictions(baseline_dir, dataset, config)
    data = build_features(frame, config)
    parameters = dict(residual_parameters or {
        'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05,
        'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 5.0,
    })
    # One OpenMP worker prevents PyTorch/XGBoost runtime conflicts on macOS.
    parameters['n_jobs'] = 1
    torch.set_num_threads(1)
    make_model('xgboost', data.X, config, parameters)
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    fingerprint = source_fingerprint()
    write_json(output / 'RUNNING.json', {'started_utc': started, 'source': dataset.source})
    protocol = {
        'deep_learning': asdict(dl_config), 'residual_model': parameters,
        'calibration_fraction': calibration_fraction,
        'parameter_policy': 'Single fixed configuration; no holdout tuning or early stopping.',
        'base_policy': 'Both models share the Bi-LSTM fitted before the calibration block.',
        'residual_policy': 'Last fraction of eligible training origins; base labels available at first calibration origin.',
        'evaluation_status': 'Exploratory historical comparison: 2014 results were previously viewed in Phase 1.',
    }
    # Freeze settings on disk before any evaluation is inspected.
    write_json(output / 'protocol.json', protocol)
    tables, audits = [], []
    for phase in ('validation', 'holdout'):
        for fold in make_folds(config, phase):
            tick = time.monotonic()
            train, test, audit = fold_masks(data, fold, config)
            # Keep only history through the last test origin, with no compressed gaps.
            stop = np.flatnonzero(test.to_numpy())[-1] + 1
            X, y = data.X.iloc[:stop], data.y.iloc[:stop]
            hybrid = HybridResidualModel(
                DeepLearningPipeline(dl_config), make_model('xgboost', data.X, config, parameters),
                calibration_fraction=calibration_fraction,
            )
            hybrid.fit(X, y, train.iloc[:stop], data.target_time.iloc[:stop])
            # Infer with enough context for every feature in the first sequence.
            first = np.flatnonzero(test.to_numpy())[0]
            begin = max(0, first - dl_config.seq_len + 1)
            hybrid_values, base_values, corrections = hybrid.predict(data.X.iloc[begin:stop])
            local_test = test.iloc[begin:stop].to_numpy()
            for name, values in (('bilstm', base_values), ('hybrid', hybrid_values)):
                scored = values[local_test]
                if not np.isfinite(scored).all():
                    raise RuntimeError(f'{fold.name}: missing predictions on common evaluation hours')
                table = pd.DataFrame({
                    'origin_time': data.X.index[test], 'target_time': data.target_time.loc[test].values,
                    'actual': data.y.loc[test].values, 'prediction': scored,
                    'model': name, 'fold': fold.name, 'phase': phase,
                    'data_source': dataset.source, 'evaluation_source': 'phase2_backtest',
                })
                if name == 'hybrid':
                    table['base_prediction'] = base_values[local_test]
                    table['residual_prediction'] = corrections[local_test]
                tables.append(table)
            elapsed = time.monotonic() - tick
            audits.append({**audit, **hybrid.fit_audit_, 'elapsed_seconds': elapsed,
                           'training_loss': hybrid.dl.loss_history_})
            pd.concat(tables, ignore_index=True).to_csv(output / 'neural_predictions.csv', index=False)
            write_json(output / 'fold_audit.json', audits)
            print(f'{fold.name}: {int(test.sum())} evaluation hours, {elapsed:.1f}s', flush=True)
        if phase == 'validation':
            validation = _combine_on_identical_hours(pd.concat(tables, ignore_index=True), baseline)
            scores = score_predictions(validation)
            scores.to_csv(output / 'validation_metrics.csv', index=False)
            preferred = scores.sort_values('rmse').iloc[0]['model']
            write_json(output / 'selection.json', {
                'criterion': 'Pooled 2013 validation RMSE', 'preferred_model': preferred,
                'holdout_used_for_selection': False, 'settings': protocol,
            })
    predictions = _combine_on_identical_hours(pd.concat(tables, ignore_index=True), baseline)
    predictions.to_csv(output / 'predictions.csv', index=False)
    scores = score_predictions(predictions)
    scores.to_csv(output / 'metrics.csv', index=False)
    score_predictions(predictions, ('phase', 'fold', 'model')).to_csv(output / 'fold_metrics.csv', index=False)
    initial = frame.loc[(frame.index >= pd.Timestamp(config.train_start)) &
                        (frame.index < pd.Timestamp(config.validation_start)), 'pm2.5']
    threshold = float(initial.quantile(0.9))
    diagnostics = predictions.loc[predictions.phase == 'holdout'].copy()
    diagnostics['pollution_regime'] = np.where(diagnostics.actual >= threshold,
                                               'above_training_p90', 'below_training_p90')
    score_predictions(diagnostics, ('model', 'pollution_regime')).to_csv(
        output / 'errors_by_pollution_regime.csv', index=False)
    eligible = (data.warm & data.y.notna() & (data.X.index >= pd.Timestamp(config.train_start))
                & (data.target_time <= frame.index.max()))
    final = HybridResidualModel(DeepLearningPipeline(dl_config),
                               make_model('xgboost', data.X, config, parameters), calibration_fraction)
    print('Fitting deployment base and calibration correction...', flush=True)
    final.fit(data.X, data.y, eligible, data.target_time)
    required_history = config.warmup_hours + dl_config.seq_len
    for name, pipeline in (('bilstm', final.dl), ('hybrid', final)):
        joblib.dump({
            'pipeline': pipeline, 'config': config, 'model_name': name, 'columns': list(data.X),
            'fit_through': str(frame.index.max()), 'data_source': dataset.source,
            'required_history_hours': required_history, 'phase': 2,
            'fit_audit': final.fit_audit_, 'source_code_sha256': fingerprint,
        }, output / f'{name}.joblib')
    if source_fingerprint() != fingerprint:
        raise RuntimeError('Forecasting source changed during training; run cannot be marked complete')
    write_json(output / 'run_manifest.json', {
        'status': 'complete', 'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'source': dataset.source, 'is_synthetic': dataset.source == 'synthetic',
        'dataset_audit': dataset.audit, 'config': config.to_dict(), 'protocol': protocol,
        'models': ['bilstm', 'hybrid'], 'source_code_sha256': fingerprint,
        'model_sha256': {name: sha256((output / f'{name}.joblib').read_bytes()).hexdigest()
                         for name in ('bilstm', 'hybrid')},
        'baseline_provenance': baseline_provenance, 'training_p90_pm25': threshold,
        'versions': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'torch', 'joblib')},
        'deployment_fit': final.fit_audit_, 'required_history_hours': required_history,
    })
    lines = ['# Phase 2 results', '', protocol['evaluation_status'], '',
             'The standalone Bi-LSTM and hybrid share a frozen base trained before calibration.',
             'The correction head uses only the later calibration block. Classical results reuse audited Phase 1 predictions.',
             '', '| Phase | Model | Hours | MAE | RMSE | R2 |', '|---|---|---:|---:|---:|---:|']
    for row in scores.itertuples():
        lines.append(f'| {row.phase} | {row.model} | {row.n} | {row.mae:.3f} | {row.rmse:.3f} | {row.r2:.4f} |')
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    (output / 'RUNNING.json').unlink()
    return {'output': output, 'metrics': scores}
