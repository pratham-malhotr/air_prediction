"""Verify that only the evaluated inference files go into the Vercel bundle."""

from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest

from scripts.prepare_vercel import prepare


@pytest.fixture()
def deployment_source(tmp_path):
    root = tmp_path / "project"
    repo = Path(__file__).resolve().parents[1]
    for name in ("deployment", "frontend"):
        shutil.copytree(repo / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
    source = root / "src/air_prediction"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text('"""Deployment fixture."""\n')
    fingerprint = sha256()
    for path in sorted(source.glob("*.py")):
        fingerprint.update(path.name.encode())
        fingerprint.update(path.read_bytes())
    data = root / "data/raw/PRSA_data_2010.1.1-2014.12.31.csv"
    data.parent.mkdir(parents=True)
    data.write_text("year,month,day,hour,pm2.5\n2014,12,31,23,12\n")
    (root / "data/README.md").write_text("Fixture dataset attribution\n")
    run = root / "artifacts/phase1"
    run.mkdir(parents=True)
    for name in ("xgboost.joblib", "ridge.joblib", "metrics.csv", "fold_metrics.csv", "errors_by_pollution_regime.csv"):
        (run / name).write_bytes(b"Unit-test deployment content\n")
    (run / "predictions.csv").write_text("This large training output must not be deployed\n")
    (run / "run_manifest.json").write_text(json.dumps({
        "status": "complete", "is_synthetic": False, "source_code_sha256": fingerprint.hexdigest(),
        "dataset_audit": {"path": str(data.relative_to(root)), "sha256": sha256(data.read_bytes()).hexdigest()},
    }))
    report = root / "reports/ieee/main.pdf"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"%PDF-1.4 fixture\n")
    (root / "LICENSE").write_text("Fixture license\n")
    return root


def test_deployment_includes_models_and_cdn_assets_without_training_outputs(deployment_source, tmp_path):
    output = prepare(deployment_source, tmp_path / "prepared")
    assert (output / "app.py").exists()
    assert (output / "public/static/app.js").exists()
    assert (output / "frontend/templates/index.html").exists()
    assert not (output / "frontend/static/app.js").exists()
    assert (output / "artifacts/phase1/xgboost.joblib").exists()
    assert not (output / "artifacts/phase1/predictions.csv").exists()
    assert "xgboost-cpu==3.4.1" in (output / "requirements.txt").read_text()
    assert "matplotlib" not in (output / "requirements.txt").read_text()
    assert (output / ".python-version").read_text().strip() == "3.13"
    metadata = json.loads((output / "aircast_bundle.json").read_text())
    for name, digest in metadata["files"].items():
        assert sha256((output / name).read_bytes()).hexdigest() == digest


def test_rebuilding_preserves_the_existing_vercel_project_link(deployment_source, tmp_path):
    output = prepare(deployment_source, tmp_path / "prepared")
    link = output / ".vercel/project.json"
    link.parent.mkdir()
    link.write_text('{"projectId":"keep-existing-project"}')
    prepare(deployment_source, output)
    assert json.loads(link.read_text())["projectId"] == "keep-existing-project"


@pytest.mark.parametrize("changed", ["data", "source", "synthetic"])
def test_deployment_rejects_stale_or_synthetic_evaluation_evidence(deployment_source, tmp_path, changed):
    if changed == "data":
        (deployment_source / "data/raw/PRSA_data_2010.1.1-2014.12.31.csv").write_text("Changed data")
    elif changed == "source":
        (deployment_source / "src/air_prediction/__init__.py").write_text("# Changed source\n")
    else:
        path = deployment_source / "artifacts/phase1/run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["is_synthetic"] = True
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        prepare(deployment_source, tmp_path / "prepared")
