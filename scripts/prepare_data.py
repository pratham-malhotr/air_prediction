"""Save the cleaned hourly series and an inspectable record of feature construction."""

import argparse
from hashlib import sha256
from pathlib import Path
import re

import pandas as pd

from air_prediction.config import load_config
from air_prediction.data import FILENAME, load_dataset, write_json
from air_prediction.features import build_features
from air_prediction.splits import fold_masks, make_folds


def feature_description(name):
    if match := re.fullmatch(r"pm25_lag_(\d+)(_missing)?", name):
        lag, missing = match.groups()
        return (f"Missing indicator for PM2.5 at t−{lag}" if missing else f"PM2.5 observed at t−{lag}"), "History through origin t"
    if match := re.fullmatch(r"pm25_(mean|std|observed_fraction)_(\d+)", name):
        statistic, window = match.groups()
        return f"Trailing {window}-hour {statistic}; missing values remain explicit", "Window ends at origin t"
    if name.startswith("target_"):
        return "Target calendar value; cycles use sine/cosine and weekend is an indicator", "Calendar known at issuance"
    if match := re.fullmatch(r"weather_(.+)_lag_(\d+)", name):
        variable, lag = match.groups()
        return f"{variable} observed at t−{lag}; original dataset units", "History through origin t"
    if match := re.fullmatch(r"weather_(.+)_missing", name):
        return f"Missing indicator for current {match[1]}", "Observation at origin t"
    if name == "wind_direction":
        return "Current cbwd category; missing has its own label", "Fold-fitted one-hot encoding"
    raise ValueError(f"Undocumented feature: {name}")


def prepare(csv_path, config_path, output, feature_output=None):
    dataset = load_dataset(csv_path)
    config = load_config(config_path)
    frame = dataset.frame
    data = build_features(frame, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "cleaned_hourly.csv.gz", index_label="origin_time")

    missing = frame["pm2.5"].isna()
    groups = missing.ne(missing.shift()).cumsum()
    runs = []
    for _, part in frame.loc[missing].groupby(groups[missing]):
        runs.append({"start": part.index.min(), "end": part.index.max(), "hours": len(part)})
    gaps = pd.DataFrame(runs, columns=["start", "end", "hours"])
    gaps.to_csv(output / "missing_pm25_runs.csv", index=False)

    dictionary = []
    for column in data.X:
        description, availability = feature_description(column)
        dictionary.append({"feature": column, "description": description, "availability": availability})
    dictionary_frame = pd.DataFrame(dictionary)
    dictionary_frame.to_csv(output / "feature_dictionary.csv", index=False)
    supervised = data.X.loc[data.warm & data.y.notna()].copy()
    supervised.insert(0, "target_time", data.target_time.loc[supervised.index])
    supervised.insert(1, "target_pm25", data.y.loc[supervised.index])
    if feature_output is not None:
        feature_output = Path(feature_output)
        feature_output.parent.mkdir(parents=True, exist_ok=True)
        supervised.to_csv(feature_output, index_label="origin_time")
        dictionary_frame.to_csv(feature_output.parent / "feature_dictionary.csv", index=False)
        write_json(feature_output.with_suffix(".provenance.json"), {
            "source": dataset.source,
            "source_csv": str(csv_path),
            "source_sha256": dataset.audit["sha256"],
            "dataset_page": dataset.audit["provenance"].get("dataset_page"),
            "doi": dataset.audit["provenance"].get("doi"),
            "license": dataset.audit["provenance"].get("license"),
            "sha256": sha256(feature_output.read_bytes()).hexdigest(),
            "rows": len(supervised),
            "features_before_wind_encoding": len(data.X.columns),
            "columns": ["origin_time", *supervised.columns],
            "config": config.to_dict(),
            "timezone": dataset.audit["timezone"],
            "row_selection": "History warmup complete and observed target available",
            "preprocessing": "Preserve missing inputs; imputation, scaling and wind encoding remain fitted inside training folds",
        })
        print(f"Saved {len(supervised):,} feature-engineered rows to {feature_output}")
    sample = supervised.head(48).copy()
    sample["reference_persistence"] = data.baselines.loc[sample.index, "persistence"]
    sample["reference_seasonal_naive"] = data.baselines.loc[sample.index, "seasonal_naive"]
    sample.to_csv(output / "feature_preview.csv", index_label="origin_time")
    audits = [fold_masks(data, fold, config)[2] for phase in ("validation", "holdout") for fold in make_folds(config, phase)]
    pd.DataFrame(audits).to_csv(output / "fold_coverage.csv", index=False)
    write_json(output / "preparation_audit.json", {
        "dataset": dataset.audit, "features_before_wind_encoding": len(data.X.columns),
        "missing_pm25_runs": len(gaps),
        "longest_missing_run_hours": int(gaps.hours.max()) if len(gaps) else 0,
        "cleaning": ["Drop row ID and replace calendar columns with the hourly index",
                     "Validate dates, unique timestamps and numeric values before sorting",
                     "Preserve missing hourly slots and missing measurements",
                     "Retain valid extreme pollution values; do not winsorize or interpolate targets",
                     "Fit learned preprocessing inside each training fold"],
        "preview": "Target and reference columns are shown for inspection only; they are not columns of model X",
    })
    lines = ["# Data preparation", "",
             f"The supplied CSV contains {len(frame):,} hourly rows; {int(missing.sum()):,} PM2.5 readings are missing.", "",
             "`cleaned_hourly.csv.gz` preserves the observations and their missing values. Cleaning validates the clock and numeric types, sorts records, and replaces the ID/calendar fields with an hourly timestamp. It does not invent pollution measurements or remove valid peaks.", "",
             f"There are {len(gaps):,} separate PM2.5 gaps. The longest lasts {int(gaps.hours.max()) if len(gaps) else 0} hours. Their dates are in `missing_pm25_runs.csv`.", "",
             f"We construct {len(data.X.columns)} features before wind encoding. `feature_dictionary.csv` explains each one. `feature_preview.csv` shows 48 forecast rows, including the future label and the two references for inspection. Those label/reference columns are not model inputs.", "",
             "Missing values in the exported series are intentional. Ridge imputes numeric inputs using training-fold medians; XGBoost uses missing-value splits. Missing evaluation targets are excluded for every method.", "",
             "`fold_coverage.csv` records the label-availability boundaries and common scored hours. No shuffled split is used."]
    (output / "PREPARATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved data preparation and {len(data.X.columns)} documented features to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=f"data/raw/{FILENAME}")
    parser.add_argument("--config", default="configs/phase1.toml")
    parser.add_argument("--output", default="artifacts/phase1/preparation")
    parser.add_argument("--feature-output", help="Also export the full supervised feature CSV, feature dictionary and provenance")
    args = parser.parse_args()
    prepare(args.data, args.config, args.output, args.feature_output)
