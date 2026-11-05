"""Frozen-model external transfer test on every UCI 501 station after 2014.

This is cross-station/feature-shift evidence, not a same-station blind temporal test.
The protocol and reference model bytes are frozen before downloading observations.
"""
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile
import argparse
import json
import os
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import joblib
import numpy as np
import pandas as pd
from air_prediction.data import load_dataset, write_json
from air_prediction.experiment import source_fingerprint
from air_prediction.features import build_features, common_evaluation_mask
from air_prediction.metrics import score_predictions

DATASET_URL = 'https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data'
DOWNLOAD_URL = 'https://archive.ics.uci.edu/static/public/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip'
STATION_PATTERN = re.compile(r'PRSA_Data_([A-Za-z]+)_20130301-20170228\.csv$')
MAX_ARCHIVE = 20_000_000
CANONICAL_STATIONS = {'Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan',
                      'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan',
                      'Wanliu', 'Wanshouxigong'}


def station_members(payload, expected=12, _depth=0):
    """Read selected members in memory; never extract archive paths."""
    if _depth > 2:
        raise ValueError('Unsupported archive nesting depth')
    found = {}
    with ZipFile(BytesIO(payload)) as archive:
        for member in archive.infolist():
            if '__MACOSX' in member.filename or member.is_dir():
                continue
            if member.file_size > 100_000_000:
                raise ValueError('Archive member exceeds the supported size')
            match = STATION_PATTERN.fullmatch(Path(member.filename).name)
            if match:
                station = match.group(1)
                if station in found:
                    raise ValueError(f'Duplicate station member: {station}')
                found[station] = archive.read(member)
            elif member.filename.lower().endswith('.zip'):
                nested = station_members(archive.read(member), expected=None, _depth=_depth+1)
                if set(found) & set(nested):
                    raise ValueError('Duplicate station members in nested archive')
                found.update(nested)
    if expected is not None and len(found) != expected:
        raise ValueError(f'Expected all {expected} stations, found {len(found)}')
    if expected == 12 and set(found) != CANONICAL_STATIONS:
        raise ValueError('External archive does not contain the canonical 12 stations')
    return dict(sorted(found.items()))


def adapt_station(raw, station):
    required = {'year', 'month', 'day', 'hour', 'PM2.5', 'TEMP', 'DEWP', 'PRES', 'wd', 'station'}
    if required - set(raw):
        raise ValueError(f'Missing external fields: {sorted(required - set(raw))}')
    names = raw.station.dropna().astype(str).unique()
    if len(names) != 1 or names[0] != station or raw.station.isna().any():
        raise ValueError('Station field does not match its declared archive member')
    result = raw[['year', 'month', 'day', 'hour', 'TEMP', 'DEWP', 'PRES']].copy()
    result['pm2.5'] = raw['PM2.5']
    result['cbwd'] = raw['wd']
    # WSPM (instantaneous speed) != Iws (accumulated speed).
    # RAIN (mm) != Ir (rain duration); Is (snow duration) is not supplied.
    for name in ('Iws', 'Ir', 'Is'):
        result[name] = np.nan
    return result


def external_mask(data, start, end, fit_through):
    mask = (common_evaluation_mask(data) & (data.target_time >= pd.Timestamp(start))
            & (data.target_time < pd.Timestamp(end)) & (data.X.index >= pd.Timestamp(fit_through)))
    if not mask.any():
        raise ValueError('No common external target hours after the frozen fit boundary')
    return mask


def predict_frozen(artifact, data, mask):
    positions = np.flatnonzero(mask.to_numpy())
    pipeline = artifact['pipeline']
    columns = artifact['columns']
    name = artifact['model_name']
    if name in {'bilstm', 'lstm', 'hybrid'}:
        neural = pipeline.dl if name == 'hybrid' else pipeline
        begin = max(0, positions[0] - neural.config.seq_len + 1)
        values = pipeline.predict(data.X.iloc[begin:positions[-1]+1].reindex(columns=columns))
        if isinstance(values, tuple):
            values = values[0]
        values = np.asarray(values)[positions-begin]
    else:
        values = pipeline.predict(data.X.loc[mask].reindex(columns=columns))
    values = np.maximum(np.asarray(values, dtype=float), 0)
    if values.shape != (len(positions),) or not np.isfinite(values).all():
        raise RuntimeError('Frozen model returned invalid external predictions')
    return values


def freeze_protocol(output, models_dir, resume=False):
    output = Path(output)
    if (output / 'run_manifest.json').exists():
        raise FileExistsError('Completed external runs are preserved; choose another output')
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError('Use --resume for the already-frozen protocol')
    references = output / 'reference_models'
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol['evaluator_sha256'] != sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError('Evaluator changed after declaration; use a fresh protocol directory')
        if protocol['runtime_source_sha256'] != source_fingerprint():
            raise ValueError('Forecasting source changed after declaration')
        for name, digest in protocol['model_sha256'].items():
            if sha256((references / f'{name}.joblib').read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen reference model changed')
        return protocol
    references.mkdir(parents=True, exist_ok=True)
    for name in ('ridge', 'xgboost'):
        shutil.copyfile(ROOT / 'artifacts/phase1' / f'{name}.joblib', references / f'{name}.joblib')
    for name in ('bilstm', 'lstm', 'hybrid'):
        shutil.copyfile(Path(models_dir) / f'{name}.joblib', references / f'{name}.joblib')
    source_manifest = json.loads((Path(models_dir) / 'run_manifest.json').read_text())
    if source_manifest.get('status') != 'complete' or source_manifest.get('is_synthetic', True):
        raise ValueError('External evaluation requires completed real-data reference models')
    for name, digest in source_manifest['model_sha256'].items():
        if sha256((references / f'{name}.joblib').read_bytes()).hexdigest() != digest:
            raise ValueError(f'Reference {name} differs from its evaluated training manifest')
    shutil.copyfile(Path(models_dir) / 'run_manifest.json', references / 'training_manifest.json')
    protocol = {
        'declared_utc': datetime.now(timezone.utc).isoformat(), 'evaluation_kind': 'external_station_and_feature_transfer',
        'dataset_page': DATASET_URL, 'download_url': DOWNLOAD_URL, 'doi': '10.24432/C5RK5G', 'license': 'CC BY 4.0',
        'target_start': '2015-01-01', 'target_end_exclusive': '2017-03-01',
        'station_policy': 'All 12 canonical station CSV members; no outcome-based station selection',
        'models': ['ridge', 'xgboost', 'bilstm', 'lstm', 'hybrid'],
        'model_sha256': {name: sha256((references / f'{name}.joblib').read_bytes()).hexdigest()
                         for name in ('ridge', 'xgboost', 'bilstm', 'lstm', 'hybrid')},
        'training_source_sha256': source_manifest['source_code_sha256'],
        'runtime_source_sha256': source_fingerprint(), 'evaluator_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'field_mapping': {'PM2.5': 'pm2.5', 'TEMP': 'TEMP', 'DEWP': 'DEWP', 'PRES': 'PRES', 'wd': 'cbwd'},
        'unavailable_inputs': ['Iws', 'Ir', 'Is'],
        'unavailable_policy': 'Leave unknown; use existing fitted handling. Never relabel WSPM as Iws or RAIN as Ir.',
        'wind_policy': 'Keep original compass labels; unseen categories use frozen unknown-category handling.',
        'fit_policy': 'No fitting, recalibration, hyperparameter search or seed selection on external observations.',
        'threshold_pm25': source_manifest['training_p90_pm25'],
        'scoring': 'Common observed target/current/seasonal hours within each station; pooled and station-macro scores',
        'interpretation': 'Different monitoring sites and nearest weather stations plus missing legacy features; not a same-station temporal validation.',
    }
    write_json(protocol_path, protocol)
    return protocol


def acquire_archive(path, protocol):
    path = Path(path)
    provenance_path = path.with_suffix('.provenance.json')
    if path.exists():
        if not provenance_path.exists():
            raise ValueError('Cached external archive lacks provenance')
        metadata = json.loads(provenance_path.read_text())
        payload = path.read_bytes()
        if metadata['sha256'] != sha256(payload).hexdigest() or metadata['url'] != DOWNLOAD_URL:
            raise ValueError('Cached external archive differs from its provenance')
        return payload
    with urlopen(DOWNLOAD_URL, timeout=45) as response:
        payload = response.read(MAX_ARCHIVE+1)
    if len(payload) > MAX_ARCHIVE:
        raise ValueError('External archive is unexpectedly large')
    station_members(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    write_json(provenance_path, {'url': DOWNLOAD_URL, 'dataset_page': DATASET_URL,
        'downloaded_utc': datetime.now(timezone.utc).isoformat(), 'license': protocol['license'],
        'doi': protocol['doi'], 'sha256': sha256(payload).hexdigest()})
    return payload


def evaluate(output, archive_path, protocol):
    output = Path(output)
    protocol_sha = sha256((output / 'protocol.json').read_bytes()).hexdigest()
    payload = acquire_archive(archive_path, protocol)
    stations = station_members(payload)
    artifacts = {name: joblib.load(output / 'reference_models' / f'{name}.joblib') for name in protocol['models']}
    config = artifacts['lstm']['config']
    fit_through = max(pd.Timestamp(artifact['fit_through']) for artifact in artifacts.values())
    if fit_through >= pd.Timestamp(protocol['target_start']):
        raise ValueError('Reference fit boundary overlaps external targets')
    predictions, audits = [], []
    expected_hours = len(pd.date_range(protocol['target_start'], protocol['target_end_exclusive'], freq='h', inclusive='left'))
    for station, content in stations.items():
        raw = pd.read_csv(BytesIO(content), na_values=['NA'], float_precision='round_trip')
        normalized = adapt_station(raw, station)
        with TemporaryDirectory(prefix='aircast-external-') as directory:
            path = Path(directory) / 'station.csv'
            normalized.to_csv(path, index=False)
            dataset = load_dataset(path)
        if (dataset.frame.index.min() > pd.Timestamp('2014-12-01')
                or dataset.frame.index.max() < pd.Timestamp(protocol['target_end_exclusive']) - pd.Timedelta(hours=1)):
            raise ValueError(f'{station} does not cover the declared context and full evaluation period')
        data = build_features(dataset.frame, config)
        mask = external_mask(data, protocol['target_start'], protocol['target_end_exclusive'], fit_through)
        base = {'station': station, 'origin_time': data.X.index[mask],
                'target_time': data.target_time.loc[mask].values, 'actual': data.y.loc[mask].values}
        for name, artifact in artifacts.items():
            if artifact['config'].to_dict() != config.to_dict():
                raise ValueError('Reference models use different feature configurations')
            values = predict_frozen(artifact, data, mask)
            predictions.append(pd.DataFrame({**base, 'model': name, 'prediction': values}))
        for name in ('persistence', 'seasonal_naive'):
            predictions.append(pd.DataFrame({**base, 'model': name, 'prediction': data.baselines.loc[mask, name].values}))
        interval = (data.target_time >= pd.Timestamp(protocol['target_start'])) & (data.target_time < pd.Timestamp(protocol['target_end_exclusive']))
        audit = {'station': station, 'archive_member_sha256': sha256(content).hexdigest(),
                 'source_rows': len(raw), 'expected_target_hours': expected_hours,
                 'common_target_hours': int(mask.sum()), 'coverage_fraction': float(mask.sum()/expected_hours),
                 'missing_targets': int((interval & data.y.isna()).sum()),
                 'unavailable_legacy_inputs': protocol['unavailable_inputs'],
                 'unseen_wind_fraction': float((~dataset.frame.cbwd.isin(['NE','NW','SE','cv']) & interval).sum()/interval.sum())}
        audits.append(audit)
        write_json(output / 'station_audit.json', audits)
        print(f'{station}: {int(mask.sum())}/{expected_hours} common hours', flush=True)
    combined = pd.concat(predictions, ignore_index=True)
    combined.to_csv(output / 'predictions.csv.gz', index=False)
    station_scores = score_predictions(combined, ('station','model'))
    pooled = score_predictions(combined, ('model',))
    station_scores.to_csv(output / 'station_metrics.csv', index=False)
    pooled.to_csv(output / 'pooled_metrics.csv', index=False)
    macro = station_scores.groupby('model').agg(stations=('station','nunique'),
        mae_mean=('mae','mean'), mae_std=('mae','std'), rmse_mean=('rmse','mean'), rmse_std=('rmse','std')).reset_index()
    macro.to_csv(output / 'macro_metrics.csv', index=False)
    combined['year'] = pd.to_datetime(combined.target_time).dt.year
    score_predictions(combined, ('year','model')).to_csv(output / 'year_metrics.csv', index=False)
    combined['pollution_regime'] = np.where(combined.actual >= protocol['threshold_pm25'], 'above_original_training_p90', 'below_original_training_p90')
    score_predictions(combined, ('model','pollution_regime')).to_csv(output / 'regime_metrics.csv', index=False)
    if sha256((output / 'protocol.json').read_bytes()).hexdigest() != protocol_sha:
        raise RuntimeError('External protocol changed during evaluation')
    if (source_fingerprint() != protocol['runtime_source_sha256']
            or sha256(Path(__file__).read_bytes()).hexdigest() != protocol['evaluator_sha256']):
        raise RuntimeError('External evaluation source changed during evaluation')
    for name,digest in protocol['model_sha256'].items():
        if sha256((output/'reference_models'/f'{name}.joblib').read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen model bytes changed during evaluation')
    write_json(output / 'run_manifest.json', {'status':'complete', 'completed_utc':datetime.now(timezone.utc).isoformat(),
        'evaluation_kind':protocol['evaluation_kind'], 'protocol_sha256':protocol_sha,
        'archive_sha256':sha256(payload).hexdigest(), 'stations':list(stations),
        'model_sha256':protocol['model_sha256'], 'runtime_source_sha256':protocol['runtime_source_sha256'],
        'evaluator_sha256':protocol['evaluator_sha256'], 'common_target_hours_per_model':int(pooled.n.iloc[0]),
        'no_external_fitting':True, 'interpretation':protocol['interpretation']})
    print(pooled.sort_values('rmse').to_string(index=False))
    return pooled


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'artifacts/phase2_external'))
    parser.add_argument('--models-dir',default=str(ROOT/'artifacts/phase2_research'))
    parser.add_argument('--archive',default=str(ROOT/'data/raw/external/uci501.zip'))
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    protocol=freeze_protocol(args.output,args.models_dir,args.resume)
    print('Frozen protocol:',Path(args.output)/'protocol.json',flush=True)
    if not args.plan_only:
        evaluate(args.output,args.archive,protocol)


if __name__=='__main__':
    main()
