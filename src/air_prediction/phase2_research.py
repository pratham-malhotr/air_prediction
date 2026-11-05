"""Validation-selected Phase 2 experiments, paired ablations and seed stability.

The historical 2014 year is opened only after selection.json is persisted.
It remains exploratory because its earlier Phase 1 results are already known.
"""
from dataclasses import asdict, replace
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

from .data import load_dataset, write_json
from .deep_learning import DLConfig, DeepLearningPipeline, HybridResidualModel
from .experiment import source_fingerprint
from .features import build_features
from .metrics import regression_metrics, score_predictions
from .models import make_model
from .phase2 import _baseline_predictions, _combine_on_identical_hours
from .splits import fold_masks, make_folds


def load_research_settings(path):
    with Path(path).open('rb') as stream:
        settings = tomllib.load(stream)
    allowed = {'seeds', 'comparison_seed', 'base', 'candidates', 'calibration_fractions',
               'correction_weights', 'residual_candidates'}
    if set(settings) - allowed:
        raise ValueError(f'Unknown research settings: {sorted(set(settings) - allowed)}')
    base = asdict(DLConfig(**settings.get('base', {})))
    candidates = []
    for entry in settings['candidates']:
        values = dict(entry)
        name = values.pop('name')
        candidates.append({'name': name, 'config': asdict(DLConfig(**{**base, **values}))})
    settings['candidates'] = candidates
    settings.pop('base', None)
    validate_research_settings(settings)
    return settings


def validate_research_settings(settings):
    seeds = settings['seeds']
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)):
        raise ValueError('Use unique nonnegative integer seeds')
    if settings['comparison_seed'] not in seeds:
        raise ValueError('The fixed comparison seed must be among validation seeds')
    candidates = settings['candidates']
    names = [entry['name'] for entry in candidates]
    if not candidates or len(set(names)) != len(names):
        raise ValueError('Use nonempty candidates with unique names')
    for entry in candidates:
        cfg = DLConfig(**entry['config'])
        if cfg.architecture != 'bilstm':
            raise ValueError('Search candidates must be Bi-LSTM; matched LSTM ablations are added automatically')
    if not settings['calibration_fractions'] or any(not 0 < value < 1 for value in settings['calibration_fractions']):
        raise ValueError('Calibration fractions must be between zero and one')
    if not settings['correction_weights'] or any(not np.isfinite(value) or not 0 <= value <= 1
                                                for value in settings['correction_weights']):
        raise ValueError('Correction weights must be finite and in [0, 1]')
    if not settings['residual_candidates']:
        raise ValueError('At least one residual configuration is required')


def summarize_seeds(predictions, identity='candidate'):
    scores = score_predictions(predictions, (identity, 'seed'))
    summary = scores.groupby(identity, sort=True).agg(
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
        seeds=('seed', 'nunique'), hours_per_seed=('n', 'min')).reset_index()
    summary[['rmse_std', 'mae_std']] = summary[['rmse_std', 'mae_std']].fillna(0.0)
    return scores, summary.sort_values(['rmse_mean', identity]).reset_index(drop=True)


def chronological_calibration(data, eligible, fraction, seq_len):
    eligible = eligible.copy()
    eligible.iloc[:seq_len - 1] = False
    positions = np.flatnonzero(eligible.to_numpy())
    count = max(1, int(np.ceil(len(positions) * fraction)))
    if len(positions) - count < 2:
        raise ValueError('Insufficient data for separate base and calibration blocks')
    cutoff = data.X.index[positions[-count]]
    base = eligible & (data.X.index < cutoff) & (data.target_time <= cutoff)
    calibration = eligible & (data.X.index >= cutoff)
    if base.sum() < 2:
        raise ValueError('Insufficient available base labels before calibration')
    return base, calibration


class ResearchExperiment:
    def __init__(self, dataset, config, output, fingerprint):
        self.dataset, self.config, self.output = dataset, config, Path(output)
        self.data = build_features(dataset.frame, config)
        self.fingerprint = fingerprint
        self.audits = []
        self.cache = self.output / 'checkpoints'
        self.cache.mkdir(parents=True, exist_ok=True)

    def fit(self, cfg, mask, role, fold):
        payload = {'config': asdict(cfg), 'source': self.fingerprint,
                   'dataset': self.dataset.audit['sha256'],
                   'mask_sha256': sha256(mask.to_numpy().tobytes()).hexdigest()}
        key = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.cache / f'{key}.joblib'
        tick = time.monotonic()
        if path.exists():
            model = joblib.load(path)
            cached = True
        else:
            stop = np.flatnonzero(mask.to_numpy())[-1] + 1
            model = DeepLearningPipeline(cfg).fit(self.data.X.iloc[:stop], self.data.y.iloc[:stop], mask.iloc[:stop])
            joblib.dump(model, path)
            cached = False
        actual = self.data.X.index.isin(model.training_origins_)
        latest_target = self.data.target_time.loc[actual].max()
        audit = {
            'role': role, 'fold': fold.name if fold else 'deployment',
            'phase': fold.phase if fold else 'deployment', 'seed': cfg.random_seed,
            'config': asdict(cfg), 'training_rows': model.training_rows_,
            'latest_training_target': str(latest_target),
            'preprocessing_fit_end': model.preprocessing_fit_end_,
            'training_loss': model.loss_history_, 'checkpoint': path.name,
            'cached': cached, 'elapsed_seconds': time.monotonic() - tick,
        }
        if fold:
            fit_time = fold.start - pd.Timedelta(hours=self.config.horizon_hours)
            if latest_target > fit_time:
                raise RuntimeError('Training label extends beyond evaluation fit time')
            audit['fit_time'] = str(fit_time)
        self.audits.append(audit)
        write_json(self.output / 'fold_audit.json', self.audits)
        return model

    def predict(self, model, mask):
        positions = np.flatnonzero(mask.to_numpy())
        if not len(positions):
            raise ValueError('No prediction rows requested')
        begin = max(0, positions[0] - model.config.seq_len + 1)
        stop = positions[-1] + 1
        values = model.predict(self.data.X.iloc[begin:stop])[positions - begin]
        if not np.isfinite(values).all():
            raise RuntimeError('Missing predictions on common evaluation origins')
        return values

    def table(self, values, mask, name, fold, seed, candidate):
        return pd.DataFrame({
            'origin_time': self.data.X.index[mask], 'target_time': self.data.target_time.loc[mask].values,
            'actual': self.data.y.loc[mask].values, 'prediction': values,
            'model': name, 'fold': fold.name, 'phase': fold.phase,
            'seed': seed, 'candidate': candidate, 'data_source': self.dataset.source,
            'evaluation_source': 'phase2_research_backtest',
        })

    def neural_cv(self, cfg, name, phase='validation', candidate=None):
        tables = []
        for fold in make_folds(self.config, phase):
            tick = time.monotonic()
            train, test, _ = fold_masks(self.data, fold, self.config)
            model = self.fit(cfg, train, candidate or name, fold)
            values = self.predict(model, test)
            tables.append(self.table(values, test, name, fold, cfg.random_seed, candidate or name))
            print(f'{phase} {candidate or name} seed={cfg.random_seed} {fold.name}: '
                  f'{time.monotonic() - tick:.1f}s', flush=True)
        return pd.concat(tables, ignore_index=True)

    def calibrated_components(self, cfg, fraction, fold=None):
        if fold:
            train, test, _ = fold_masks(self.data, fold, self.config)
        else:
            train = (self.data.warm & self.data.y.notna()
                     & (self.data.X.index >= pd.Timestamp(self.config.train_start))
                     & (self.data.target_time <= self.dataset.frame.index.max()))
            test = None
        base_mask, calibration = chronological_calibration(self.data, train, fraction, cfg.seq_len)
        model = self.fit(cfg, base_mask, f'calibration_{fraction}', fold)
        calibration_prediction = self.predict(model, calibration)
        residual = self.data.y.loc[calibration] - calibration_prediction
        cutoff = self.data.X.index[calibration].min()
        audit = {
            'base_training_rows': int(base_mask.sum()),
            'base_last_origin': str(self.data.X.index[base_mask].max()),
            'base_latest_target': str(self.data.target_time.loc[base_mask].max()),
            'calibration_start': str(cutoff), 'calibration_end': str(self.data.X.index[calibration].max()),
            'calibration_rows': int(calibration.sum()),
            'calibration_latest_target': str(self.data.target_time.loc[calibration].max()),
            'preprocessing_fit_end': model.preprocessing_fit_end_, 'calibration_fraction': fraction,
        }
        if pd.Timestamp(audit['base_latest_target']) > cutoff:
            raise RuntimeError('Base labels extend into the calibration block')
        if fold and pd.Timestamp(audit['calibration_latest_target']) > fold.start - pd.Timedelta(hours=self.config.horizon_hours):
            raise RuntimeError('Calibration labels extend into evaluation')
        return model, calibration, residual, test, audit

    def hybrid_candidates(self, cfg, fractions, heads, weights, seeds):
        tables, descriptions, audits = [], {}, []
        for seed in seeds:
            current = replace(cfg, random_seed=seed)
            for fraction in fractions:
                for fold in make_folds(self.config, 'validation'):
                    tick = time.monotonic()
                    base, calibration, residual, test, audit = self.calibrated_components(current, fraction, fold)
                    base_test = self.predict(base, test)
                    audits.append({**audit, 'seed': seed, 'fold': fold.name})
                    for head_index, parameters in enumerate(heads):
                        head = make_model('xgboost', self.data.X, self.config, {**parameters, 'n_jobs': 1})
                        head.fit(self.data.X.loc[calibration], residual)
                        correction = head.predict(self.data.X.loc[test])
                        for weight in weights:
                            name = f'cal{fraction:g}_head{head_index + 1}_weight{weight:g}'
                            descriptions[name] = {'calibration_fraction': fraction,
                                                  'parameters': parameters, 'correction_weight': weight}
                            values = np.maximum(base_test + weight * correction, 0)
                            table = self.table(values, test, 'hybrid', fold, seed, name)
                            table['base_prediction'] = base_test
                            table['residual_prediction'] = weight * correction
                            tables.append(table)
                    print(f'validation hybrid fraction={fraction} seed={seed} {fold.name}: '
                          f'{time.monotonic() - tick:.1f}s', flush=True)
        write_json(self.output / 'hybrid_fold_audit.json', audits)
        return pd.concat(tables, ignore_index=True), descriptions

    def hybrid_cv(self, cfg, selected, phase, seed):
        tables, audits = [], []
        for fold in make_folds(self.config, phase):
            base, calibration, residual, test, audit = self.calibrated_components(
                replace(cfg, random_seed=seed), selected['calibration_fraction'], fold)
            head = make_model('xgboost', self.data.X, self.config, {**selected['parameters'], 'n_jobs': 1})
            head.fit(self.data.X.loc[calibration], residual)
            base_test = self.predict(base, test)
            correction = head.predict(self.data.X.loc[test]) * selected['correction_weight']
            table = self.table(np.maximum(base_test + correction, 0), test, 'hybrid', fold, seed, 'selected_hybrid')
            table['base_prediction'], table['residual_prediction'] = base_test, correction
            tables.append(table)
            audits.append({**audit, 'phase': phase, 'seed': seed, 'fold': fold.name})
            print(f'{phase} selected hybrid seed={seed} {fold.name}', flush=True)
        return pd.concat(tables, ignore_index=True), audits


def run_research(csv_path, config, output, baseline_dir, settings, resume=False):
    validate_research_settings(settings)
    output = Path(output)
    fingerprint = source_fingerprint()
    dataset = load_dataset(csv_path)
    if (dataset.frame.index.min() > pd.Timestamp(config.train_start)
            or dataset.frame.index.max() < pd.Timestamp(config.holdout_end) - pd.Timedelta(hours=1)):
        raise ValueError('Input must cover all configured training and comparison periods')
    baseline, provenance = _baseline_predictions(baseline_dir, dataset, config)
    protocol = {
        'settings': settings, 'config': config.to_dict(), 'source_code_sha256': fingerprint,
        'dataset_sha256': dataset.audit['sha256'],
        'parameter_policy': 'Search on 2013 only; paired architecture/attention variants averaged across validation seeds.',
        'comparison_policy': 'One predeclared comparison/deployment seed; never select a seed using 2014.',
        'standalone_policy': 'Independent neural models use the complete eligible outer training window.',
        'hybrid_policy': 'Separate chronological calibration; frozen base; validation-selected correction shrinkage.',
        'evaluation_status': 'Exploratory historical comparison; the 2014 Phase 1 results were already viewed.',
    }
    if output.exists() and any(output.iterdir()):
        if not resume or (output / 'run_manifest.json').exists():
            raise FileExistsError('Choose a new output directory, or resume an incomplete matching run')
        if json.loads((output / 'protocol.json').read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError('Cannot resume after settings, source or data changed')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'RUNNING.json', {'started_utc': datetime.now(timezone.utc).isoformat()})
    experiment = ResearchExperiment(dataset, config, output, fingerprint)
    primary = settings['comparison_seed']
    search = []
    configs = {}
    for entry in settings['candidates']:
        cfg = replace(DLConfig(**entry['config']), random_seed=primary)
        configs[entry['name']] = cfg
        search.append(experiment.neural_cv(cfg, 'bilstm', candidate=entry['name']))
        partial = pd.concat(search, ignore_index=True)
        _, summary = summarize_seeds(partial)
        summary.to_csv(output / 'search_metrics.csv', index=False)
    search_predictions = pd.concat(search, ignore_index=True)
    search_predictions.to_csv(output / 'search_predictions.csv.gz', index=False)
    _, search_summary = summarize_seeds(search_predictions)
    winning_search = search_summary.iloc[0]['candidate']
    template = configs[winning_search]
    print(f'Validation search winner: {winning_search}; building matched ablations.', flush=True)

    ablations, ablation_configs = [], {}
    for architecture in ('bilstm', 'lstm'):
        for attention in (True, False):
            name = f'{architecture}_{"attention" if attention else "no_attention"}'
            cfg = replace(template, architecture=architecture, use_attention=attention)
            ablation_configs[name] = cfg
            for seed in settings['seeds']:
                ablations.append(experiment.neural_cv(replace(cfg, random_seed=seed),
                                                      architecture, candidate=name))
            partial = pd.concat(ablations, ignore_index=True)
            seed_scores, summary = summarize_seeds(partial)
            seed_scores.to_csv(output / 'ablation_seed_metrics.csv', index=False)
            summary.to_csv(output / 'ablation_metrics.csv', index=False)
    ablation_predictions = pd.concat(ablations, ignore_index=True)
    ablation_predictions.to_csv(output / 'ablation_predictions.csv.gz', index=False)
    _, ablation_summary = summarize_seeds(ablation_predictions)
    chosen_names = {
        architecture: ablation_summary.loc[ablation_summary.candidate.str.startswith(architecture + '_')]
                      .iloc[0]['candidate'] for architecture in ('bilstm', 'lstm')
    }
    chosen = {name: replace(ablation_configs[candidate], random_seed=primary)
              for name, candidate in chosen_names.items()}
    hybrid_predictions, hybrid_descriptions = experiment.hybrid_candidates(
        chosen['bilstm'], settings['calibration_fractions'], settings['residual_candidates'],
        settings['correction_weights'], settings['seeds'])
    hybrid_predictions.to_csv(output / 'hybrid_search_predictions.csv.gz', index=False)
    hybrid_seed_metrics, hybrid_summary = summarize_seeds(hybrid_predictions)
    hybrid_seed_metrics.to_csv(output / 'hybrid_seed_metrics.csv', index=False)
    hybrid_summary.to_csv(output / 'hybrid_search_metrics.csv', index=False)
    selected_hybrid_name = hybrid_summary.iloc[0]['candidate']
    selected_hybrid = hybrid_descriptions[selected_hybrid_name]
    validation_tables = []
    for name, candidate in chosen_names.items():
        table = ablation_predictions.loc[(ablation_predictions.candidate == candidate)
                                         & (ablation_predictions.seed == primary)].copy()
        table['model'] = name
        validation_tables.append(table)
    validation_tables.append(hybrid_predictions.loc[(hybrid_predictions.candidate == selected_hybrid_name)
                                                    & (hybrid_predictions.seed == primary)].copy())
    validation_neural = pd.concat(validation_tables, ignore_index=True)
    validation = _combine_on_identical_hours(validation_neural, baseline)
    validation_metrics = score_predictions(validation)
    validation_metrics.to_csv(output / 'validation_metrics.csv', index=False)
    # Select the preferred family using mean validation seed RMSE, with classical
    # comparators deterministic because their saved predictions are reused.
    ranking = [{'model': row.model, 'validation_rmse_mean': row.rmse}
               for row in validation_metrics.itertuples() if row.model not in {'bilstm', 'lstm', 'hybrid'}]
    for name, candidate in chosen_names.items():
        value = ablation_summary.loc[ablation_summary.candidate == candidate].iloc[0].rmse_mean
        ranking.append({'model': name, 'validation_rmse_mean': float(value)})
    ranking.append({'model': 'hybrid', 'validation_rmse_mean': float(hybrid_summary.iloc[0].rmse_mean)})
    preferred = min(ranking, key=lambda row: (row['validation_rmse_mean'], row['model']))['model']
    selection = {
        'criterion': 'Mean pooled 2013 RMSE across predefined validation seeds',
        'preferred_model': preferred, 'holdout_used_for_selection': False,
        'search_winner': winning_search, 'neural_candidates': chosen_names,
        'neural_configurations': {name: asdict(cfg) for name, cfg in chosen.items()},
        'hybrid_candidate': selected_hybrid_name, 'hybrid': selected_hybrid,
        'ranking': ranking, 'comparison_seed': primary,
        'search_metrics_sha256': sha256((output / 'search_metrics.csv').read_bytes()).hexdigest(),
        'ablation_metrics_sha256': sha256((output / 'ablation_metrics.csv').read_bytes()).hexdigest(),
        'hybrid_search_metrics_sha256': sha256((output / 'hybrid_search_metrics.csv').read_bytes()).hexdigest(),
    }
    write_json(output / 'selection.json', selection)
    selection_hash = sha256((output / 'selection.json').read_bytes()).hexdigest()
    print(f'Selection frozen: preferred={preferred}; beginning exploratory 2014 comparison.', flush=True)
    holdout_tables = [experiment.neural_cv(cfg, name, 'holdout', f'selected_{name}')
                      for name, cfg in chosen.items()]
    hybrid_holdout, hybrid_audits = experiment.hybrid_cv(chosen['bilstm'], selected_hybrid, 'holdout', primary)
    write_json(output / 'hybrid_comparison_audit.json', hybrid_audits)
    holdout_tables.append(hybrid_holdout)
    neural = pd.concat([validation_neural, *holdout_tables], ignore_index=True)
    predictions = _combine_on_identical_hours(neural, baseline)
    predictions.to_csv(output / 'predictions.csv', index=False)
    scores = score_predictions(predictions)
    scores.to_csv(output / 'metrics.csv', index=False)
    score_predictions(predictions, ('phase', 'fold', 'model')).to_csv(output / 'fold_metrics.csv', index=False)
    training = dataset.frame.loc[(dataset.frame.index >= pd.Timestamp(config.train_start))
                                  & (dataset.frame.index < pd.Timestamp(config.validation_start)), 'pm2.5']
    threshold = float(training.quantile(.9))
    diagnostics = predictions.loc[predictions.phase == 'holdout'].copy()
    diagnostics['pollution_regime'] = np.where(diagnostics.actual >= threshold, 'above_training_p90', 'below_training_p90')
    score_predictions(diagnostics, ('model', 'pollution_regime')).to_csv(output / 'errors_by_pollution_regime.csv', index=False)
    diagnostics['hour'] = diagnostics.target_time.dt.hour
    score_predictions(diagnostics, ('model', 'hour')).to_csv(output / 'errors_by_hour.csv', index=False)
    selected_seed_tables = []
    for name, candidate in chosen_names.items():
        table = ablation_predictions.loc[ablation_predictions.candidate == candidate].copy()
        table['candidate'] = name
        selected_seed_tables.append(table)
    table = hybrid_predictions.loc[hybrid_predictions.candidate == selected_hybrid_name].copy()
    table['candidate'] = 'hybrid'
    selected_seed_tables.append(table)
    stability_scores, stability = summarize_seeds(pd.concat(selected_seed_tables, ignore_index=True))
    stability_scores.to_csv(output / 'seed_metrics.csv', index=False)
    stability.to_csv(output / 'seed_stability.csv', index=False)
    eligible = (experiment.data.warm & experiment.data.y.notna()
                & (experiment.data.X.index >= pd.Timestamp(config.train_start))
                & (experiment.data.target_time <= dataset.frame.index.max()))
    deployment = {}
    for name, cfg in chosen.items():
        print(f'Fitting independent deployment {name} on every eligible training row...', flush=True)
        deployment[name] = experiment.fit(cfg, eligible, f'deployment_{name}', None)
    base, calibration, residual, _, hybrid_fit = experiment.calibrated_components(
        chosen['bilstm'], selected_hybrid['calibration_fraction'])
    head = make_model('xgboost', experiment.data.X, config, {**selected_hybrid['parameters'], 'n_jobs': 1})
    head.fit(experiment.data.X.loc[calibration], residual)
    hybrid = HybridResidualModel(base, head, selected_hybrid['calibration_fraction'], selected_hybrid['correction_weight'])
    hybrid.is_fitted, hybrid.fit_audit_ = True, hybrid_fit
    deployment['hybrid'] = hybrid
    for name, pipeline in deployment.items():
        neural_config = pipeline.dl.config if name == 'hybrid' else pipeline.config
        fit_audit = hybrid_fit if name == 'hybrid' else {
            'training_rows': pipeline.training_rows_, 'latest_training_target': str(experiment.data.target_time.loc[eligible].max()),
            'preprocessing_fit_end': pipeline.preprocessing_fit_end_, 'training_policy': 'complete eligible window'}
        joblib.dump({
            'pipeline': pipeline, 'config': config, 'model_name': name,
            'columns': list(experiment.data.X), 'fit_through': str(dataset.frame.index.max()),
            'data_source': dataset.source, 'required_history_hours': config.warmup_hours + neural_config.seq_len,
            'sequence_length': neural_config.seq_len, 'phase': 2, 'fit_audit': fit_audit,
            'source_code_sha256': fingerprint,
        }, output / f'{name}.joblib')
    if source_fingerprint() != fingerprint or sha256((output / 'selection.json').read_bytes()).hexdigest() != selection_hash:
        raise RuntimeError('Source or selection changed during the experiment')
    lines = ['# Phase 2 selected research results', '', protocol['evaluation_status'], '',
             'Standalone Bi-LSTM and LSTM use complete eligible training folds. The hybrid has a separately calibrated base.',
             f'Validation preference: **{preferred}**. Comparison seed: **{primary}**.', '',
             '| Phase | Model | Hours | MAE | RMSE | R2 |', '|---|---|---:|---:|---:|---:|']
    for row in scores.itertuples():
        lines.append(f'| {row.phase} | {row.model} | {row.n} | {row.mae:.3f} | {row.rmse:.3f} | {row.r2:.4f} |')
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    write_json(output / 'run_manifest.json', {
        'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'source': dataset.source, 'is_synthetic': dataset.source == 'synthetic',
        'dataset_audit': dataset.audit, 'config': config.to_dict(), 'protocol': protocol,
        'models': list(deployment), 'source_code_sha256': fingerprint,
        'baseline_provenance': provenance, 'training_p90_pm25': threshold,
        'selection_sha256': selection_hash, 'selection': selection,
        'model_sha256': {name: sha256((output / f'{name}.joblib').read_bytes()).hexdigest() for name in deployment},
        'versions': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'torch', 'joblib')},
        'required_history_hours': max(config.warmup_hours + cfg.seq_len for cfg in chosen.values()),
    })
    (output / 'RUNNING.json').unlink()
    return {'output': output, 'metrics': scores, 'selection': selection}
