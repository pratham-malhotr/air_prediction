from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from air_prediction.config import load_config
from air_prediction.data import FILENAME, download_dataset, load_dataset
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
    # Mock acquisition only: synthetic measurements on the documented timestamp grid.
    mocked = pd.concat([demo] * 21, ignore_index=True).iloc[:43824].copy()
    index = pd.date_range("2010-01-01", periods=43824, freq="h")
    for column, values in zip(("year", "month", "day", "hour"), (index.year, index.month, index.day, index.hour)):
        mocked[column] = values
    payload = mocked.to_csv(index=False).encode()
    bundle = BytesIO()
    with ZipFile(bundle, "w") as stream:
        stream.writestr(FILENAME, payload)
    monkeypatch.setattr("air_prediction.data.urlopen", lambda *a, **kw: BytesIO(bundle.getvalue()))
    path = download_dataset(tmp_path / "download.csv")
    metadata = json.loads(path.with_suffix(".provenance.json").read_text())
    assert metadata["sha256"] == sha256(payload).hexdigest()
    with pytest.raises(FileExistsError):
        download_dataset(path)


@pytest.mark.parametrize("hour", [-1, 24, 0.5])
def test_loader_rejects_invalid_clock_hours(tmp_path, hour):
    path = make_demo(tmp_path / "demo.csv")
    frame = pd.read_csv(path).iloc[[0]].copy()
    frame["hour"] = hour
    frame.to_csv(path, index=False)
    path.with_suffix(".provenance.json").unlink()
    with pytest.raises(ValueError, match="hour|integer"):
        load_dataset(path)


def test_download_rejects_invalid_archive_without_publishing(tmp_path, monkeypatch):
    monkeypatch.setattr("air_prediction.data.urlopen", lambda *a, **kw: BytesIO(b"not a ZIP archive"))
    destination = tmp_path / "download.csv"
    with pytest.raises(ValueError, match="valid ZIP"):
        download_dataset(destination)
    assert not destination.exists()
    assert not destination.with_suffix(".provenance.json").exists()


def test_download_rejects_duplicate_timestamps_before_publication(tmp_path, monkeypatch):
    demo = pd.read_csv(make_demo(tmp_path / "demo.csv"))
    payload = pd.concat([demo] * 21, ignore_index=True).iloc[:43824].to_csv(index=False).encode()
    bundle = BytesIO()
    with ZipFile(bundle, "w") as stream:
        stream.writestr(FILENAME, payload)
    monkeypatch.setattr("air_prediction.data.urlopen", lambda *a, **kw: BytesIO(bundle.getvalue()))
    destination = tmp_path / "download.csv"
    with pytest.raises(ValueError, match="unique"):
        download_dataset(destination)
    assert not destination.exists()


def test_complete_synthetic_ridge_workflow(tmp_path):
    path = make_demo(tmp_path / "synthetic.csv")
    cfg = replace(load_config(ROOT / "configs/demo.toml"), train_start="2010-01-05")
    result = run_experiment(path, cfg, tmp_path / "demo", names=("ridge",))
    assert result["manifest"]["is_synthetic"] is True
    output = result["output"]
    assert (output / "figures/model_comparison.png").is_file()
    assert (output / "eda/correlation_autocorrelation.png").is_file()
    assert "SYNTHETIC DATA" in (output / "eda/EDA_NOTES.md").read_text()
    assert (output / "run_manifest.json").is_file()
    assert not (output / "RUNNING.json").exists()
    scope = json.loads((output / "eda/eda_scope.json").read_text())
    assert pd.Timestamp(scope["analysis_start"]) == pd.Timestamp(cfg.train_start)
    observed = load_dataset(path).frame
    initial = observed.loc[(observed.index >= cfg.train_start) & (observed.index < cfg.validation_start), "pm2.5"]
    assert result["manifest"]["training_p90_pm25"] == pytest.approx(initial.quantile(0.9))
    predictions = result["predictions"]
    counts = predictions.groupby(["phase", "model"]).size().unstack()
    assert counts.nunique(axis=1).eq(1).all()
    for phase, part in predictions.groupby("phase"):
        targets = [set(group.target_time) for _, group in part.groupby("model")]
        assert all(target == targets[0] for target in targets)
    forecast = predict_latest(output / "ridge.joblib", path)
    assert forecast["target_time"] == "2010-04-01 00:00:00"
    assert forecast["pm25_ug_m3"] >= 0
    old_history = tmp_path / "old_history.csv"
    pd.read_csv(path).iloc[:-24].to_csv(old_history, index=False)
    with pytest.raises(ValueError, match="before model fit time"):
        predict_latest(output / "ridge.joblib", old_history)
    assert "SYNTHETIC" in (output / "RESULTS.md").read_text()
    assert "Synthetic demonstration only" in (output / "results_table.tex").read_text()
    with pytest.raises(FileExistsError):
        run_experiment(path, cfg, output, names=("ridge",))
    with pytest.raises(ValueError, match="separate demo"):
        run_experiment(path, cfg, tmp_path / "phase1", names=("ridge",))


def test_partial_holdout_is_not_reported_as_complete(tmp_path):
    path = make_demo(tmp_path / "synthetic.csv")
    cfg = replace(load_config(ROOT / "configs/demo.toml"), holdout_end="2010-05-01")
    with pytest.raises(ValueError, match="full configured holdout"):
        run_experiment(path, cfg, tmp_path / "demo", names=("ridge",))


def test_failed_report_generation_does_not_mark_run_complete(tmp_path, monkeypatch):
    path = make_demo(tmp_path / "synthetic.csv")
    config = load_config(ROOT / "configs/demo.toml")
    output = tmp_path / "failed_run"

    def fail_summary(*args, **kwargs):
        raise OSError("Result summary could not be written")

    monkeypatch.setattr("air_prediction.results.write_summary", fail_summary)
    with pytest.raises(OSError, match="could not be written"):
        run_experiment(path, config, output, names=("ridge",))
    assert (output / "RUNNING.json").exists()
    assert not (output / "run_manifest.json").exists()


def test_graceful_import_when_torch_unavailable(monkeypatch):
    import sys
    import importlib
    # Simulate environment where torch is not installed
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "air_prediction.deep_learning", raising=False)
    monkeypatch.delitem(sys.modules, "air_prediction", raising=False)

    # Importing air_prediction root module should succeed without torch
    import air_prediction
    assert hasattr(air_prediction, "__version__")
    assert not air_prediction.DLConfig
    with pytest.raises(ImportError, match="requires PyTorch"):
        air_prediction.DLConfig()
    with pytest.raises(ImportError, match="requires PyTorch"):
        air_prediction.TemporalForecaster()




def test_build_features_no_timedelta_deprecation_warnings(hourly_frame):
    import warnings
    from air_prediction.config import Config
    from air_prediction.features import build_features
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        data = build_features(hourly_frame, Config())
        assert len(data.X) == len(hourly_frame)
