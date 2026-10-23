import re

import pandas as pd

from air_prediction.data import Dataset
from air_prediction.eda import run_eda, write_eda_notes


def test_eda_notes_use_training_values_and_ignore_later_data(hourly_frame, tmp_path):
    frame = hourly_frame.copy()
    frame.loc[frame.index[200:203], "pm2.5"] = float("nan")
    original = Dataset(frame, {}, "synthetic")
    first = tmp_path / "first"
    run_eda(original, first, until="2010-02-01", since="2010-01-01")
    notes = (first / "EDA_NOTES.md").read_text()
    assert "744 hourly slots" in notes
    assert "3 missing readings (0.40%)" in notes
    assert "741 measured concentrations" in notes
    assert "SYNTHETIC DATA" in notes

    # A later pollution spike and weather change must not become an EDA finding.
    changed = frame.copy()
    changed.loc[changed.index >= "2010-02-01", ["pm2.5", "TEMP"]] = 999999
    later = tmp_path / "later"
    run_eda(Dataset(changed, {}, "synthetic"), later, until="2010-02-01", since="2010-01-01")
    assert (later / "EDA_NOTES.md").read_text() == notes


def test_constant_series_notes_do_not_invent_a_weather_relationship(hourly_frame, tmp_path):
    frame = hourly_frame.iloc[:24].copy()
    frame["pm2.5"] = 10.0
    correlations = frame.select_dtypes(include="number").corr()
    autocorrelation = pd.Series({1: float("nan"), 24: float("nan"), 168: float("nan")})
    destination = tmp_path / "notes.md"
    write_eda_notes(frame, "synthetic", correlations, autocorrelation, destination)
    notes = destination.read_text()
    assert "undefined" in notes
    assert "No finite PM2.5–weather correlation" in notes
    assert "cannot establish an annual pattern" in notes
    assert not re.search(r"\b(?:nan|inf)\b", notes, flags=re.I)
