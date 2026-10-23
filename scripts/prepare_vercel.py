"""Prepare the small inference-only Aircast application for Vercel."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def prepare(root=ROOT, destination=None):
    root = Path(root).resolve()
    destination = Path(destination).resolve() if destination else root / "dist/aircast-pm25"
    run = root / "artifacts/phase1"
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("is_synthetic"):
        raise ValueError("Deploy only a completed real-data experiment")
    source = sha256()
    for path in sorted((root / "src/air_prediction").glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    if source.hexdigest() != manifest["source_code_sha256"]:
        raise ValueError("The forecasting source differs from the evaluated run")
    data_path = root / manifest["dataset_audit"]["path"]
    if sha256(data_path.read_bytes()).hexdigest() != manifest["dataset_audit"]["sha256"]:
        raise ValueError("The input CSV differs from the evaluated run")
    marker = destination / "aircast_bundle.json"
    if destination.exists() and any(destination.iterdir()) and not marker.exists():
        raise ValueError("Choose an empty output directory; this folder is not a prepared Aircast bundle")
    destination.mkdir(parents=True, exist_ok=True)
    previous = json.loads(marker.read_text()).get("files", {}) if marker.exists() else {}
    generated = {}

    def copy(path, name):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        generated[name] = sha256(target.read_bytes()).hexdigest()

    for name in ("app.py", "requirements.txt", "vercel.json", ".python-version", ".vercelignore"):
        copy(root / "deployment" / name, name)
    for path in sorted((root / "frontend").rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root / "frontend")
        if relative.parts[0] == "static":
            copy(path, "public/" + relative.as_posix())
        elif path.suffix in {".py", ".html"}:
            copy(path, "frontend/" + relative.as_posix())
    for path in sorted((root / "src/air_prediction").glob("*.py")):
        copy(path, "src/air_prediction/" + path.name)
    for name in ("xgboost.joblib", "ridge.joblib", "metrics.csv", "fold_metrics.csv", "errors_by_pollution_regime.csv", "run_manifest.json"):
        copy(run / name, "artifacts/phase1/" + name)
    copy(data_path, "data/raw/" + data_path.name)
    copy(root / "data/README.md", "data/README.md")
    copy(root / "reports/ieee/main.pdf", "reports/ieee/main.pdf")
    copy(root / "LICENSE", "LICENSE")
    # Preserve Vercel's project link and credentials; remove only previously generated files.
    for name in set(previous) - set(generated):
        path = destination / name
        if not path.resolve().is_relative_to(destination):
            raise ValueError("Invalid path in the old deployment manifest")
        if path.is_file():
            path.unlink()
    marker.write_text(json.dumps({"format": 1, "dataset_sha256": manifest["dataset_audit"]["sha256"],
                                   "forecast_source_sha256": manifest["source_code_sha256"], "files": generated}, indent=2) + "\n")
    print(f"Prepared {len(generated)} deployment files at {destination}")
    print("Includes the real saved models and dataset; excludes training dependencies and local caches.")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prepare(destination=args.output)
