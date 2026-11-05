"""External transfer checks use synthetic observations, never external labels."""
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from air_prediction.config import Config
from air_prediction.features import build_features
from scripts import validate_phase2_external as external


def zip_bytes(entries):
    buffer = BytesIO()
    with ZipFile(buffer, 'w') as archive:
        for name, value in entries:
            archive.writestr(name, value)
    return buffer.getvalue()


def test_archive_reads_nested_station_members_without_extracting_paths():
    name = 'PRSA_Data_Dongsi_20130301-20170228.csv'
    payload = zip_bytes([('inner.zip', zip_bytes([('../../'+name, b'csv')])),
                         ('__MACOSX/'+name, b'ignored')])
    assert external.station_members(payload, expected=1) == {'Dongsi': b'csv'}
    with pytest.raises(ValueError, match='Expected all 12'):
        external.station_members(payload)
    duplicate = zip_bytes([(name, b'one'), ('folder/'+name, b'two')])
    with pytest.raises(ValueError, match='Duplicate'):
        external.station_members(duplicate, expected=1)


def test_archive_rejects_unknown_station_set_and_deep_nesting():
    payload = zip_bytes([(f'PRSA_Data_Unknown{chr(65+i)}_20130301-20170228.csv', b'csv')
                         for i in range(12)])
    with pytest.raises(ValueError, match='canonical'):
        external.station_members(payload)
    for _ in range(4):
        payload = zip_bytes([('nested.zip', payload)])
    with pytest.raises(ValueError, match='nesting'):
        external.station_members(payload)


def test_external_field_mapping_keeps_incompatible_measurements_unknown():
    raw = pd.DataFrame({'year':[2015], 'month':[1], 'day':[1], 'hour':[0],
                        'PM2.5':[31.0], 'TEMP':[2.0], 'DEWP':[-4.0], 'PRES':[1010.0],
                        'wd':['SSW'], 'station':['Dongsi'], 'WSPM':[5.0], 'RAIN':[2.0]})
    result = external.adapt_station(raw, 'Dongsi')
    assert result['pm2.5'].iloc[0] == 31.0
    assert result.cbwd.iloc[0] == 'SSW'
    assert result[['Iws','Ir','Is']].isna().all().all()
    with pytest.raises(ValueError, match='Station field'):
        external.adapt_station(raw, 'Wanliu')
    with pytest.raises(ValueError, match='Missing external'):
        external.adapt_station(raw.drop(columns='PM2.5'), 'Dongsi')


def test_external_mask_respects_common_observations_and_declared_boundaries(hourly_frame):
    frame = hourly_frame.copy()
    frame.iloc[100, frame.columns.get_loc('pm2.5')] = np.nan
    config = replace(Config(), lags=(0,1,24), rolling_windows=(3,24))
    data = build_features(frame, config)
    start, end = frame.index[80], frame.index[140]
    mask = external.external_mask(data, start, end, frame.index[85])
    assert data.target_time.loc[mask].between(start, end, inclusive='left').all()
    assert (data.X.index[mask] >= frame.index[85]).all()
    assert data.y.loc[mask].notna().all() and data.baselines.loc[mask].notna().all().all()
    assert not mask.loc[frame.index[99]]  # Missing next-hour target.
    assert not mask.loc[frame.index[100]]  # Missing persistence value.
    with pytest.raises(ValueError, match='No common'):
        external.external_mask(data, '2050-01-01', '2051-01-01', frame.index[0])


def test_external_protocol_freezes_reference_bytes_before_data_access(tmp_path, monkeypatch):
    monkeypatch.setattr(external, 'ROOT', tmp_path)
    phase1 = tmp_path/'artifacts/phase1'
    phase1.mkdir(parents=True)
    models = tmp_path/'models'
    models.mkdir()
    for name in ('ridge','xgboost'):
        (phase1/f'{name}.joblib').write_bytes(name.encode())
    hashes = {}
    for name in ('lstm','bilstm','hybrid'):
        path = models/f'{name}.joblib'
        path.write_bytes(name.encode())
        hashes[name] = sha256(path.read_bytes()).hexdigest()
    (models/'run_manifest.json').write_text(json.dumps({'status':'complete',
        'is_synthetic':False, 'source_code_sha256':'training-fixture', 'model_sha256':hashes,
        'training_p90_pm25':218.0}))
    monkeypatch.setattr(external, 'acquire_archive', lambda *args: pytest.fail('Protocol must not read labels'))
    output = tmp_path/'external'
    protocol = external.freeze_protocol(output, models)
    assert protocol['training_source_sha256'] == 'training-fixture'
    assert protocol['target_start'] == '2015-01-01'
    assert len(protocol['model_sha256']) == 5
    assert external.freeze_protocol(output, models, resume=True) == protocol
    (output/'reference_models/lstm.joblib').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Frozen reference model changed'):
        external.freeze_protocol(output, models, resume=True)


def test_cached_external_archive_requires_matching_provenance(tmp_path):
    path = tmp_path/'archive.zip'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='lacks provenance'):
        external.acquire_archive(path, {'license':'CC BY 4.0'})
    path.with_suffix('.provenance.json').write_text(json.dumps({'url':external.DOWNLOAD_URL,'sha256':'wrong'}))
    with pytest.raises(ValueError, match='differs from its provenance'):
        external.acquire_archive(path, {'license':'CC BY 4.0'})
