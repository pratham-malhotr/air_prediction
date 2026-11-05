"""Restore only the evaluated runtime files from a checksum-pinned public release.

Uses the Python standard library; it never imports ML packages or unpickles models.
Existing files must match the release. Research outputs are never overwritten.
"""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile
import argparse
import json
import os
import stat

ROOT = Path(__file__).resolve().parents[1]
RELEASE_TAG = 'phase2-research-2026-11-05-r1'
BUNDLE_URL = (f'https://github.com/pratham-malhotr/air_prediction/releases/download/'
              f'{RELEASE_TAG}/air_prediction_phase2.zip')
BUNDLE_SHA256 = 'd20646ee62de7aa8fcd7a4895e978e6150cb6faa717123cdb17d7b48b136faf8'
MAX_BUNDLE_BYTES = 70_000_000
MAX_RUNTIME_BYTES = 70_000_000
NEURAL_MODELS = ('bilstm', 'lstm', 'hybrid')
RUNTIME_FILES = (
    *(f'artifacts/phase1/{name}' for name in (
        'ridge.joblib', 'xgboost.joblib', 'metrics.csv', 'fold_metrics.csv',
        'errors_by_pollution_regime.csv', 'run_manifest.json')),
    *(f'artifacts/phase2_research/{name}' for name in (
        'bilstm.joblib', 'lstm.joblib', 'hybrid.joblib', 'metrics.csv', 'fold_metrics.csv',
        'errors_by_pollution_regime.csv', 'run_manifest.json', 'verification.json')),
    'data/raw/PRSA_data_2010.1.1-2014.12.31.csv',
    'data/raw/PRSA_data_2010.1.1-2014.12.31.provenance.json',
    'data/README.md',
    'reports/phase2/Aircast_Phase2_Report.pdf',
)


def file_sha256(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def forecasting_source_sha256(root):
    """Match experiment.source_fingerprint without importing its dependencies."""
    paths = sorted((Path(root) / 'src/air_prediction').glob('*.py'))
    if not paths:
        raise ValueError('No forecasting source found in the destination checkout')
    digest = sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_runtime(bundle, expected_sha256):
    bundle = Path(bundle)
    if bundle.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError('Release bundle exceeds the supported size')
    if file_sha256(bundle) != expected_sha256:
        raise ValueError('Release bundle SHA-256 differs from the pinned digest')
    try:
        with ZipFile(bundle) as archive:
            selected = ('PACKAGE_MANIFEST.json', *RUNTIME_FILES)
            members = archive.infolist()
            total = 0
            for name in selected:
                matching = [member for member in members if member.filename == name]
                if len(matching) != 1:
                    raise ValueError(f'Missing or duplicate runtime archive member: {name}')
                member = matching[0]
                if member.is_dir() or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f'Runtime archive member must be a regular file: {name}')
                total += member.file_size
            if total > MAX_RUNTIME_BYTES or archive.getinfo('PACKAGE_MANIFEST.json').file_size > 1_000_000:
                raise ValueError('Runtime payload exceeds the supported size')
            metadata = json.loads(archive.read('PACKAGE_MANIFEST.json'))
            if metadata.get('format') != 1:
                raise ValueError('Unsupported release manifest format')
            payloads = {}
            for name in RUNTIME_FILES:
                content = archive.read(name)
                if sha256(content).hexdigest() != metadata['files'].get(name):
                    raise ValueError(f'Runtime payload checksum differs: {name}')
                payloads[name] = content
    except BadZipFile as exc:
        raise ValueError('Release bundle is not a valid ZIP archive') from exc
    return metadata, payloads


def validate_runtime(root, metadata, payloads):
    research = json.loads(payloads['artifacts/phase2_research/run_manifest.json'])
    classical = json.loads(payloads['artifacts/phase1/run_manifest.json'])
    verification = json.loads(payloads['artifacts/phase2_research/verification.json'])
    for manifest in (research, classical):
        if manifest.get('status') != 'complete' or manifest.get('is_synthetic', True):
            raise ValueError('Runtime requires completed real-data research')
        if manifest['dataset_audit']['sha256'] != metadata['dataset_sha256']:
            raise ValueError('Research and release dataset hashes differ')
    source = forecasting_source_sha256(root)
    if not (source == metadata['run_source_sha256'] == research['source_code_sha256']
            == verification['source_code_sha256']):
        raise ValueError('Forecasting source differs from the evaluated release')
    dataset = 'data/raw/PRSA_data_2010.1.1-2014.12.31.csv'
    if sha256(payloads[dataset]).hexdigest() != metadata['dataset_sha256']:
        raise ValueError('Raw data differs from the evaluated release')
    for name in NEURAL_MODELS:
        digest = sha256(payloads[f'artifacts/phase2_research/{name}.joblib']).hexdigest()
        if digest != research['model_sha256'][name]:
            raise ValueError(f'Runtime {name} model differs from evaluated weights')
    report = 'reports/phase2/Aircast_Phase2_Report.pdf'
    if sha256(payloads[report]).hexdigest() != verification['document_sha256'][Path(report).name]:
        raise ValueError('Report differs from the verified submission')
    return source


def restore(root, bundle, *, expected_sha256=BUNDLE_SHA256):
    root = Path(root).resolve()
    metadata, payloads = read_runtime(bundle, expected_sha256)
    source = validate_runtime(root, metadata, payloads)
    missing, existing = [], []
    # Validate every destination before writing any runtime file.
    for name in RUNTIME_FILES:
        destination = root / name
        if not destination.resolve().is_relative_to(root):
            raise ValueError(f'Runtime destination escapes the checkout: {name}')
        for parent in destination.parents:
            if parent == root:
                break
            if parent.exists() and not parent.is_dir():
                raise ValueError(f'Runtime parent is not a directory: {parent}')
        if destination.exists():
            if not destination.is_file() or file_sha256(destination) != metadata['files'][name]:
                raise FileExistsError(f'Existing file differs from release; preserved: {name}')
            existing.append(name)
        else:
            missing.append(name)
    for name in missing:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(dir=destination.parent, prefix='.aircast-', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payloads[name])
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o644)
            # Atomic publication without replacing a concurrently created file.
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if not destination.is_file() or file_sha256(destination) != metadata['files'][name]:
                    raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {'status': 'verified', 'verified_utc': datetime.now(timezone.utc).isoformat(),
            'release_tag': RELEASE_TAG, 'bundle_sha256': expected_sha256,
            'source_code_sha256': source, 'restored_files': missing, 'matching_existing_files': existing,
            'files': {name: metadata['files'][name] for name in RUNTIME_FILES}}


def download_bundle(destination):
    total = 0
    with urlopen(BUNDLE_URL, timeout=60) as response, Path(destination).open('xb') as stream:
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_BUNDLE_BYTES:
                raise ValueError('Downloaded release exceeds the supported size')
            stream.write(chunk)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--bundle', type=Path, help='Use an existing pinned release ZIP without network access')
    args = parser.parse_args(argv)
    if args.bundle is not None:
        result = restore(args.project_root, args.bundle)
    else:
        with TemporaryDirectory(prefix='aircast-runtime-') as directory:
            bundle = Path(directory) / 'release.zip'
            download_bundle(bundle)
            result = restore(args.project_root, bundle)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
