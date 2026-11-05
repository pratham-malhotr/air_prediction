"""Release restoration verifies provenance before creating any runtime files."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json

import pytest

from scripts import restore_phase2_runtime as runtime


@pytest.fixture
def release_fixture(tmp_path):
    root = tmp_path / 'checkout'
    source = root / 'src/air_prediction'
    source.mkdir(parents=True)
    (source / '__init__.py').write_text('# Synthetic source fixture\n')
    source_hash = runtime.forecasting_source_sha256(root)
    payloads = {name: f'Synthetic fixture: {name}\n'.encode() for name in runtime.RUNTIME_FILES}
    dataset = sha256(payloads['data/raw/PRSA_data_2010.1.1-2014.12.31.csv']).hexdigest()
    for directory in ('phase1', 'phase2_research'):
        manifest = {'status': 'complete', 'is_synthetic': False, 'source_code_sha256': source_hash,
                    'dataset_audit': {'sha256': dataset}}
        if directory == 'phase2_research':
            manifest['model_sha256'] = {
                name: sha256(payloads[f'artifacts/phase2_research/{name}.joblib']).hexdigest()
                for name in runtime.NEURAL_MODELS}
        payloads[f'artifacts/{directory}/run_manifest.json'] = json.dumps(manifest).encode()
    report = 'reports/phase2/Aircast_Phase2_Report.pdf'
    payloads['artifacts/phase2_research/verification.json'] = json.dumps({
        'source_code_sha256': source_hash,
        'document_sha256': {Path(report).name: sha256(payloads[report]).hexdigest()}}).encode()

    def bundle(*, modify=None, corrupt=None, missing=None, duplicate=None):
        contents = payloads.copy()
        if modify:
            modify(contents)
        if missing:
            contents.pop(missing)
        metadata = {'format': 1, 'run_source_sha256': source_hash, 'dataset_sha256': dataset,
                    'files': {name: sha256(value).hexdigest() for name, value in contents.items()}}
        if corrupt:
            contents[corrupt] += b'changed after checksum'
        path = tmp_path / 'bundle.zip'
        with ZipFile(path, 'w') as archive:
            for name, value in contents.items():
                archive.writestr(name, value)
            archive.writestr('PACKAGE_MANIFEST.json', json.dumps(metadata))
            archive.writestr('../../outside.txt', 'Must never be extracted')
            archive.writestr('.env', 'Must never be restored')
            if duplicate:
                with pytest.warns(UserWarning, match='Duplicate name'):
                    archive.writestr(duplicate, contents[duplicate])
        return path, runtime.file_sha256(path)

    return root, payloads, bundle


def test_runtime_restore_is_selective_idempotent_and_does_not_unpickle(release_fixture, tmp_path):
    root, payloads, make_bundle = release_fixture
    path, digest = make_bundle()
    result = runtime.restore(root, path, expected_sha256=digest)
    assert result['status'] == 'verified' and len(result['restored_files']) == len(runtime.RUNTIME_FILES)
    for name, content in payloads.items():
        assert (root / name).read_bytes() == content
    assert not (root / '.env').exists() and not (tmp_path / 'outside.txt').exists()
    before = {name: (root / name).stat().st_mtime_ns for name in payloads}
    repeated = runtime.restore(root, path, expected_sha256=digest)
    assert repeated['restored_files'] == []
    assert before == {name: (root / name).stat().st_mtime_ns for name in payloads}


def test_runtime_restore_rejects_wrong_pinned_archive_digest_before_writing(release_fixture):
    root, _, make_bundle = release_fixture
    path, _ = make_bundle()
    with pytest.raises(ValueError, match='pinned digest'):
        runtime.restore(root, path, expected_sha256='0' * 64)
    assert not (root / 'artifacts').exists()


def test_runtime_restore_checks_each_payload_before_writing(release_fixture):
    root, _, make_bundle = release_fixture
    path, digest = make_bundle(corrupt='artifacts/phase1/ridge.joblib')
    with pytest.raises(ValueError, match='payload checksum'):
        runtime.restore(root, path, expected_sha256=digest)
    assert not (root / 'artifacts').exists()


def test_runtime_restore_rejects_changed_forecasting_source(release_fixture):
    root, _, make_bundle = release_fixture
    path, digest = make_bundle()
    (root / 'src/air_prediction/__init__.py').write_text('# Changed source\n')
    with pytest.raises(ValueError, match='Forecasting source differs'):
        runtime.restore(root, path, expected_sha256=digest)
    assert not (root / 'artifacts').exists()


@pytest.mark.parametrize('field,value', [('status', 'running'), ('is_synthetic', True)])
def test_runtime_restore_rejects_unfinished_or_synthetic_research(release_fixture, field, value):
    root, _, make_bundle = release_fixture

    def modify(contents):
        name = 'artifacts/phase2_research/run_manifest.json'
        manifest = json.loads(contents[name])
        manifest[field] = value
        contents[name] = json.dumps(manifest).encode()

    path, digest = make_bundle(modify=modify)
    with pytest.raises(ValueError, match='completed real-data'):
        runtime.restore(root, path, expected_sha256=digest)
    assert not (root / 'artifacts').exists()


@pytest.mark.parametrize('kind', ['weights', 'data', 'report'])
def test_runtime_restore_cross_checks_training_and_submission_provenance(release_fixture, kind):
    root, _, make_bundle = release_fixture

    def modify(contents):
        if kind == 'report':
            contents['reports/phase2/Aircast_Phase2_Report.pdf'] += b'changed'
        else:
            name = 'artifacts/phase2_research/run_manifest.json'
            manifest = json.loads(contents[name])
            if kind == 'weights':
                manifest['model_sha256']['lstm'] = '0' * 64
            else:
                manifest['dataset_audit']['sha256'] = '0' * 64
            contents[name] = json.dumps(manifest).encode()

    path, digest = make_bundle(modify=modify)
    with pytest.raises(ValueError, match='differs|hashes differ'):
        runtime.restore(root, path, expected_sha256=digest)
    assert not (root / 'artifacts').exists()


def test_runtime_restore_preserves_conflicting_existing_outputs(release_fixture):
    root, _, make_bundle = release_fixture
    path, digest = make_bundle()
    existing = root / 'artifacts/phase2_research/lstm.joblib'
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b'Existing experiment must be preserved')
    with pytest.raises(FileExistsError, match='preserved'):
        runtime.restore(root, path, expected_sha256=digest)
    assert existing.read_bytes() == b'Existing experiment must be preserved'
    assert not (root / 'artifacts/phase1').exists()


def test_runtime_restore_rejects_symlink_escape_before_writing(release_fixture, tmp_path):
    root, _, make_bundle = release_fixture
    path, digest = make_bundle()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (root / 'artifacts').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes the checkout'):
        runtime.restore(root, path, expected_sha256=digest)
    assert not list(outside.iterdir())


@pytest.mark.parametrize('problem', ['missing', 'duplicate'])
def test_runtime_restore_rejects_missing_or_duplicate_archive_members(release_fixture, problem):
    root, _, make_bundle = release_fixture
    name = 'artifacts/phase2_research/lstm.joblib'
    path, digest = make_bundle(**{problem: name})
    with pytest.raises(ValueError, match='Missing or duplicate'):
        runtime.restore(root, path, expected_sha256=digest)
    assert not (root / 'artifacts').exists()
