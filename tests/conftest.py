import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
# Configure single-threading in test processes to prevent dual OpenMP deadlock between PyTorch and XGBoost on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def hourly_frame():
    index = pd.date_range("2010-01-01", periods=1000, freq="h", name="origin_time")
    return pd.DataFrame({"pm2.5": np.arange(len(index), dtype=float), "DEWP": 2.0, "TEMP": 5.0,
                         "PRES": 1010.0, "Iws": 3.0, "Is": 0.0, "Ir": 0.0, "cbwd": "NW"}, index=index)
