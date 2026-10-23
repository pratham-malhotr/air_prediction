from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from air_prediction.config import load_config
from air_prediction.data import FILENAME, download_dataset, load_dataset, write_json
from air_prediction.demo import make_demo
from air_prediction.experiment import run_experiment
from air_prediction.predict import predict_latest

ROOT = Path(__file__).resolve().parents[1]


def test_loader_regularizes_and_sorts_without_imputing(tmp_path):
    path = make_demo(tmp_path / "demo.csv")
    frame = pd.read_csv(path)
    missing_row_time = frame.iloc[300][["year", "month", "day", "hour"]]
    frame = frame.drop(index=300).sample(frac=1, random_state=1)
    frame.to_csv(path, index=False)
    path.with_suffix(".provenance.json").unlink()
    loaded = load_dataset(path)
    assert loaded.audit["inserted_missing_hours"] == 1
    assert loaded.audit["input_was_sorted"] is False
    timestamp = pd.Timestamp(year=int(missing_row_time.year), month=int(missing_row_time.month), day=int(missing_row_time.day), hour=int(missing_row_time.hour))
    assert loaded.frame.loc[timestamp].isna().all()


def test_loader_rejects_duplicate_timestamps_and_changed_checksum(tmp_path):
    path = make_demo(tmp_path / "demo.csv")
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(path)
    path.with_suffix(".provenance.json").unlink()
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_dataset(path)


def test_loader_rejects_negative_target(tmp_path):
    path = make_demo(tmp_path / "demo.csv")
    frame = pd.read_csv(path)
    frame.loc[100, "pm2.5"] = -1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="nonnegative"):
        load_dataset(path)


def test_download_checks_archive_and_records_provenance(tmp_path, monkeypatch):
    demo = pd.read_csv(make_demo(tmp_path / "demo.csv"))
    # Mock acquisition only. These repeated rows are deliberately NOT a usable time series.
    payload = pd.concat([demo] * 21, ignore_index=True).iloc[:43824].to_csv(index=False).encode()
    bundle = BytesIO()
    with ZipFile(bundle, "w") as stream:
        stream.writestr(FILENAME, payload)
    monkeypatch.setattr("air_prediction.data.urlopen", lambda *a, **kw: BytesIO(bundle.getvalue()))
    path = download_dataset(tmp_path / "download.csv")
    metadata = json.loads(path.with_suffix(".provenance.json").read_text())
    assert metadata["sha256"] == sha256(payload).hexdigest()
    with pytest.raises(FileExistsError):
        download_dataset(path)


def test_complete_synthetic_ridge_workflow(tmp_path):
    path = make_demo(tmp_path / "synthetic.csv")
    cfg = load_config(ROOT / "configs/demo.toml")
    result = run_experiment(path, cfg, tmp_path / "demo", names=("ridge",))
    assert result["manifest"]["is_synthetic"] is True
    output = result["output"]
    assert (output / "figures/model_comparison.png").is_file()
    assert (output / "eda/correlation_autocorrelation.png").is_file()
    assert (output / "run_manifest.json").is_file()
    assert not (output / "RUNNING.json").exists()
    predictions = result["predictions"]
    counts = predictions.groupby(["phase", "model"]).size().unstack()
    assert counts.nunique(axis=1).eq(1).all()
    for phase, part in predictions.groupby("phase"):
        targets = [set(group.target_time) for _, group in part.groupby("model")]
        assert all(target == targets[0] for target in targets)
    forecast = predict_latest(output / "ridge.joblib", path)
    assert forecast["target_time"] == "2010-04-01 00:00:00"
    assert forecast["pm25_ug_m3"] >= 0
    assert "SYNTHETIC" in (output / "RESULTS.md").read_text()
    with pytest.raises(FileExistsError):
        run_experiment(path, cfg, output, names=("ridge",))
    with pytest.raises(ValueError, match="separate demo"):
        run_experiment(path, cfg, tmp_path / "phase1", names=("ridge",))


def test_partial_holdout_is_not_reported_as_complete(tmp_path):
    path = make_demo(tmp_path / "synthetic.csv")
    cfg = replace(load_config(ROOT / "configs/demo.toml"), holdout_end="2010-05-01")
    with pytest.raises(ValueError, match="full configured holdout"):
        run_experiment(path, cfg, tmp_path / "demo", names=("ridge",))
