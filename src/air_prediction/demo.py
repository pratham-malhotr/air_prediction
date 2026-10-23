"""Explicitly synthetic integration fixtures, never substitutes for the UCI dataset."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from .data import write_json


def make_demo(path, seed=42):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    rng = np.random.default_rng(seed)
    index = pd.date_range("2010-01-01", "2010-04-01", freq="h", inclusive="left")
    n = len(index)
    temp = 5 + 8 * np.sin(np.arange(n) * 2 * np.pi / (24 * 30)) + rng.normal(0, 2, n)
    pm = np.zeros(n)
    pm[0] = 80
    for i in range(1, n):
        pm[i] = max(0, 0.88 * pm[i - 1] + 0.12 * (65 - temp[i] + 12 * np.sin(index[i].hour * np.pi / 12)) + rng.normal(0, 8))
    frame = pd.DataFrame({"No": np.arange(1, n + 1), "year": index.year, "month": index.month, "day": index.day,
                          "hour": index.hour, "pm2.5": pm, "DEWP": temp - 3, "TEMP": temp,
                          "PRES": 1015 + rng.normal(0, 4, n), "cbwd": rng.choice(["NW", "NE", "SE", "cv"], n),
                          "Iws": rng.gamma(2, 3, n), "Is": np.zeros(n), "Ir": np.zeros(n)})
    frame.loc[rng.choice(n, 60, replace=False), "pm2.5"] = np.nan
    frame.loc[rng.choice(n, 15, replace=False), "TEMP"] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="NA")
    write_json(path.with_suffix(".provenance.json"), {"source": "synthetic", "seed": seed,
               "sha256": sha256(path.read_bytes()).hexdigest(), "purpose": "Integration testing only; not Beijing observations"})
    return path
