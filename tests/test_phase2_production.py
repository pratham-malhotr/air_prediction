"""Production preflight rejects drift before any application/model is loaded."""
from hashlib import sha256
import json
from pathlib import Path

import pytest

from deployment.phase2 import wsgi
from scripts.serve_phase2_production import server_arguments


def test_production_server_uses_post_fork_factory_and_one_worker():
    args = server_arguments({'PORT':'8099','AIRCAST_HOST':'127.0.0.1'})
    assert args[args.index('--bind')+1] == '127.0.0.1:8099'
    assert args[args.index('--workers')+1] == '1'
    assert args[-1] == 'deployment.phase2.wsgi:create_application()'
    assert '--preload' not in args
    for value in ('bad','80','65536'):
        with pytest.raises(ValueError, match='PORT'):
            server_arguments({'PORT':value})


@pytest.fixture
def production_bundle(tmp_path, monkeypatch):
    import air_prediction.experiment
    monkeypatch.setattr(air_prediction.experiment, 'source_fingerprint', lambda:'source')
    run = tmp_path/'artifacts/phase2_research'
    run.mkdir(parents=True)
    data = tmp_path/'data/raw/PRSA_data_2010.1.1-2014.12.31.csv'
    data.parent.mkdir(parents=True)
    data.write_bytes(b'raw data')
    phase1 = tmp_path/'artifacts/phase1'
    phase1.mkdir()
    for name in ('ridge','xgboost'):
        (phase1/f'{name}.joblib').write_bytes(name.encode())
    hashes = {}
    for name in ('lstm','bilstm','hybrid'):
        model = run/f'{name}.joblib'
        model.write_bytes(name.encode())
        hashes[name] = sha256(model.read_bytes()).hexdigest()
    manifest = {'status':'complete','is_synthetic':False,'source_code_sha256':'source',
                'dataset_audit':{'sha256':sha256(data.read_bytes()).hexdigest()},'model_sha256':hashes}
    (run/'run_manifest.json').write_text(json.dumps(manifest))
    return tmp_path, run, data, manifest


def test_production_preflight_accepts_only_matching_source_data_and_weights(production_bundle):
    root, run, data, manifest = production_bundle
    assert wsgi.validate_bundle(root) == manifest
    (run/'lstm.joblib').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='lstm model differs'):
        wsgi.validate_bundle(root)


@pytest.mark.parametrize('drift', ['source','data','synthetic'])
def test_production_preflight_rejects_drift(production_bundle, drift):
    root, run, data, manifest = production_bundle
    if drift == 'source':
        manifest['source_code_sha256'] = 'changed'
    elif drift == 'synthetic':
        manifest['is_synthetic'] = True
    else:
        data.write_bytes(b'changed')
    (run/'run_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        wsgi.validate_bundle(root)


@pytest.mark.parametrize('trust, expected', [('0',403),('1',200)])
def test_production_proxy_setting_preserves_https_same_origin_validation(monkeypatch, trust, expected):
    import frontend.app
    original_factory = frontend.app.create_app

    class Service:
        def forecast(self, **kwargs):
            return {'ok':True}

    monkeypatch.setattr(wsgi,'validate_bundle',lambda root:None)
    monkeypatch.setattr(frontend.app,'create_app',lambda root:original_factory(root,service=Service()))
    monkeypatch.setenv('AIRCAST_TRUST_PROXY',trust)
    monkeypatch.delenv('VERCEL',raising=False)
    app = wsgi.create_application()
    headers={'Origin':'https://aircast.example','X-Forwarded-Proto':'https',
             'X-Forwarded-Host':'aircast.example'}
    with app.test_client() as client:
        response = client.post('/api/forecast',json={},headers=headers)
        assert response.status_code==expected
        headers['Origin']='https://different.example'
        assert client.post('/api/forecast',json={},headers=headers).status_code==403
