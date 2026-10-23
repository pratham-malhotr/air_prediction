"""UCI acquisition, schema checks, provenance, and hourly regularization."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

FILENAME = "PRSA_data_2010.1.1-2014.12.31.csv"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/381/beijing%2Bpm2%2B5%2Bdata.zip"
WEATHER_NUMERIC = ("DEWP", "TEMP", "PRES", "Iws", "Is", "Ir")
REQUIRED = ("year", "month", "day", "hour", "pm2.5", *WEATHER_NUMERIC, "cbwd")


@dataclass
class Dataset:
    frame: pd.DataFrame
    audit: dict
    source: str


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def download_dataset(destination):
    """Download the exact CSV without extracting arbitrary archive members."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; reuse the existing CSV or choose another path")
    try:
        with urlopen(DOWNLOAD_URL, timeout=60) as response:
            archive = response.read(10_000_001)
    except (URLError, OSError) as exc:
        raise RuntimeError(
            f"UCI download failed: {exc}. Download {FILENAME} from {DATASET_PAGE} "
            f"and save it at {destination}, or use the self-contained Colab notebook."
        ) from exc
    if len(archive) > 10_000_000:
        raise ValueError("Unexpectedly large UCI archive")
    with ZipFile(BytesIO(archive)) as bundle:
        members = [item for item in bundle.infolist() if Path(item.filename).name == FILENAME]
        if len(members) != 1 or members[0].file_size > 10_000_000:
            raise ValueError("The UCI archive does not contain the expected CSV")
        payload = bundle.read(members[0])
    # Check before publishing the file to the caller.
    raw = pd.read_csv(BytesIO(payload))
    if len(raw) != 43824 or not set(REQUIRED).issubset(raw.columns):
        raise ValueError("Downloaded file does not match the documented UCI 381 schema and size")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    write_json(destination.with_suffix(".provenance.json"), {
        "source": "uci381", "url": DOWNLOAD_URL, "dataset_page": DATASET_PAGE,
        "sha256": sha256(payload).hexdigest(), "license": "CC BY 4.0", "doi": "10.24432/C5JS49",
    })
    return destination


def load_dataset(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Run the download command or use Colab.")
    raw = pd.read_csv(path, na_values=["NA", "nan", ""], keep_default_na=True)
    missing_columns = sorted(set(REQUIRED) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    if raw.empty:
        raise ValueError("Dataset is empty")
    timestamp = pd.to_datetime(raw[["year", "month", "day", "hour"]], errors="raise")
    if timestamp.isna().any() or timestamp.duplicated().any():
        raise ValueError("Timestamps must be present and unique; resolve duplicates explicitly")
    if not timestamp.eq(timestamp.dt.floor("h")).all():
        raise ValueError("Only hourly timestamps are supported")
    frame = raw.loc[:, ["pm2.5", *WEATHER_NUMERIC, "cbwd"]].copy()
    for name in ("pm2.5", *WEATHER_NUMERIC):
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
        if np.isinf(frame[name]).any():
            raise ValueError(f"Infinite values in {name}")
    if (frame["pm2.5"].dropna() < 0).any():
        raise ValueError("PM2.5 concentrations must be nonnegative")
    frame["cbwd"] = frame["cbwd"].astype("string").str.strip().replace("", pd.NA)
    frame.index = pd.DatetimeIndex(timestamp, name="origin_time")
    was_sorted = frame.index.is_monotonic_increasing
    frame = frame.sort_index()
    hourly = pd.date_range(frame.index.min(), frame.index.max(), freq="h", name="origin_time")
    inserted = len(hourly) - len(frame)
    frame = frame.reindex(hourly)
    provenance_path = path.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {"source": "user_supplied"}
    digest = sha256(path.read_bytes()).hexdigest()
    if provenance.get("sha256") and provenance["sha256"] != digest:
        raise ValueError("CSV checksum differs from its provenance record")
    source = provenance.get("source", "user_supplied")
    audit = {
        "source": source, "path": str(path), "sha256": digest,
        "input_rows": len(raw), "hourly_rows": len(frame), "inserted_missing_hours": inserted,
        "input_was_sorted": bool(was_sorted), "duplicate_timestamps": 0,
        "start": str(frame.index.min()), "end": str(frame.index.max()), "timezone": "Asia/Shanghai (local clock)",
        "missing_by_column": {key: int(value) for key, value in frame.isna().sum().items()},
        "provenance": provenance,
    }
    return Dataset(frame, audit, source)
