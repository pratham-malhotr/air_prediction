"""Package the current source, measured Phase 1 outputs and available report PDFs."""

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def source_files():
    for name in ("README.md", "requirements.txt", "pyproject.toml", ".gitignore", "LICENSE", "CONTRIBUTING.md", "Start_Aircast.command", "data/README.md", "data/raw/.gitkeep", "artifacts/.gitkeep"):
        path = ROOT / name
        if path.exists():
            yield path
    for name in ("src", "frontend", "deployment", "configs", "docs", "reports", "tests", "scripts", ".github"):
        for path in sorted((ROOT / name).rglob("*")):
            text_source = path.suffix in {".py", ".sh", ".toml", ".md", ".tex", ".bib", ".yml", ".yaml", ".html", ".css", ".js", ".cjs", ".svg"}
            measured_summary = name == "docs" and path.parent.name == "results" and path.suffix in {".csv", ".json", ".png"}
            deployment_config = name == "deployment" and (path.suffix == ".json" or path.name in {"requirements.txt", ".python-version", ".vercelignore"})
            if path.is_file() and "__pycache__" not in path.parts and (text_source or measured_summary or deployment_config):
                yield path


def main():
    destination = ROOT / "dist" / "air_prediction_phase1.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = list(source_files())
    run = ROOT / "artifacts/phase1"
    manifest_path = run / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "complete" or manifest["is_synthetic"]:
            raise ValueError("Only a completed real-data run belongs in the Phase 1 package")
        # Combined predictions.csv already contains these validation forecasts.
        files.extend(path for path in sorted(run.rglob("*")) if path.is_file() and path.name != "validation_predictions.csv" and path.suffix in {".csv", ".json", ".md", ".png", ".pdf", ".tex", ".joblib", ".gz"})
        # Redistribute the public dataset with its citation and provenance, not the local runtime.
        data = ROOT / manifest["dataset_audit"]["path"]
        if sha256(data.read_bytes()).hexdigest() != manifest["dataset_audit"]["sha256"]:
            raise ValueError("The dataset no longer matches the completed experiment")
        files.append(data)
        if data.with_suffix(".provenance.json").exists():
            files.append(data.with_suffix(".provenance.json"))
    for path in (ROOT / "reports/ieee/main.pdf", ROOT / "reports/presentation/slides.pdf"):
        if path.exists():
            files.append(path)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(destination)


if __name__ == "__main__":
    main()
