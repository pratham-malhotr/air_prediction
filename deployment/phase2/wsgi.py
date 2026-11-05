"""Production app factory: validate the completed research bundle after worker fork."""
from pathlib import Path
from hashlib import sha256
import json
import os


def validate_bundle(root):
    root=Path(root)
    from air_prediction.experiment import source_fingerprint
    run=root/'artifacts/phase2_research'
    manifest=json.loads((run/'run_manifest.json').read_text())
    if manifest.get('status')!='complete' or manifest.get('is_synthetic',True):
        raise ValueError('Production requires completed real-data Phase 2 research')
    if manifest['source_code_sha256']!=source_fingerprint():
        raise ValueError('Production source differs from the evaluated research bundle')
    data=root/'data/raw/PRSA_data_2010.1.1-2014.12.31.csv'
    if sha256(data.read_bytes()).hexdigest()!=manifest['dataset_audit']['sha256']:
        raise ValueError('Production input archive differs from the evaluated data')
    for name in ('bilstm','lstm','hybrid'):
        if sha256((run/f'{name}.joblib').read_bytes()).hexdigest()!=manifest['model_sha256'][name]:
            raise ValueError(f'Production {name} model differs from evaluated weights')
    for name in ('ridge','xgboost'):
        if not (root/'artifacts/phase1'/f'{name}.joblib').is_file():
            raise ValueError(f'Missing production comparison model: {name}')
    return manifest


def create_application(project_root=None):
    root=Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    validate_bundle(root)
    from frontend.app import create_app
    application=create_app(root)
    trust=os.environ.get('AIRCAST_TRUST_PROXY','0')
    if trust not in {'0','1'}:
        raise ValueError('AIRCAST_TRUST_PROXY must be 0 or 1')
    if trust=='1' and os.environ.get('VERCEL')!='1':
        from werkzeug.middleware.proxy_fix import ProxyFix
        application.wsgi_app=ProxyFix(application.wsgi_app,x_proto=1,x_host=1)
    return application
