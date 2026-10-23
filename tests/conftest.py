import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def hourly_frame():
    index = pd.date_range("2010-01-01", periods=1000, freq="h", name="origin_time")
    return pd.DataFrame({"pm2.5": np.arange(len(index), dtype=float), "DEWP": 2.0, "TEMP": 5.0,
                         "PRES": 1010.0, "Iws": 3.0, "Is": 0.0, "Ir": 0.0, "cbwd": "NW"}, index=index)
