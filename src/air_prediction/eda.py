"""Training-period EDA and an archive-wide schema/missingness audit."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import WEATHER_NUMERIC, write_json


def _save(fig, destination):
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_eda(dataset, output, until):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "data_audit.json", dataset.audit)
    frame = dataset.frame.loc[dataset.frame.index < pd.Timestamp(until)].copy()
    if frame.empty or frame["pm2.5"].notna().sum() < 3:
        raise ValueError("Insufficient observed training-period values for EDA")
    write_json(output / "eda_scope.json", {
        "source": dataset.source, "analysis_start": str(frame.index.min()), "analysis_end": str(frame.index.max()),
        "exclusive_boundary": str(pd.Timestamp(until)),
        "policy": "Plots and descriptive statistics use initial training dates only; the archive audit is descriptive.",
    })
    frame.describe(include="all").to_csv(output / "descriptive_statistics.csv")
    missing = frame.isna().sum().rename("missing_count").to_frame()
    missing["missing_percent"] = missing.missing_count / len(frame) * 100
    missing.to_csv(output / "missingness.csv")
    numeric = frame[["pm2.5", *WEATHER_NUMERIC]]
    numeric.corr().to_csv(output / "correlations.csv")
    pm = frame["pm2.5"]
    # ACF uses the full hourly grid; dropping missing rows first would compress time.
    acf = pd.Series({lag: pm.autocorr(lag) for lag in (1, 2, 3, 6, 12, 24, 48, 72, 168)}, name="autocorrelation")
    acf.to_csv(output / "autocorrelation.csv", index_label="lag_hours")
    observed = pm.dropna()
    q1, q3 = observed.quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    write_json(output / "distribution_summary.json", {
        "n_observed": len(observed), "zero_count": int((observed == 0).sum()),
        "skewness": float(observed.skew()), "iqr_upper_fence": float(upper),
        "above_iqr_fence": int((observed > upper).sum()), "outlier_policy": "Retain valid extremes; pollution peaks are part of the task.",
    })
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    pm.resample("D").mean().plot(ax=axes[0], color="#2563eb", linewidth=0.8)
    axes[0].set(title="Training period: daily mean PM2.5", ylabel="PM2.5 (µg/m³)", xlabel="Beijing local date")
    frame.isna().resample("MS").mean().plot(ax=axes[1], linewidth=1)
    axes[1].set(title="Monthly missing fraction (no interpolation)", ylabel="Fraction missing", xlabel="Month")
    _save(fig, output / "timeline_missingness.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(observed, bins=60, color="#2563eb", alpha=0.85)
    axes[0].set(title="PM2.5 distribution", xlabel="PM2.5 (µg/m³)", ylabel="Observed hours")
    axes[1].boxplot([pm[pm.index.month == month].dropna().to_numpy() for month in range(1, 13)], tick_labels=range(1, 13))
    axes[1].set(title="Monthly distributions; peaks retained", xlabel="Month", ylabel="PM2.5 (µg/m³)")
    _save(fig, output / "distribution_seasonality.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, name, group in zip(axes, ("Hour", "Weekday (Monday=0)", "Month"), (pm.index.hour, pm.index.dayofweek, pm.index.month)):
        summary = pm.groupby(group).agg(["mean", "median", "count"])
        summary.to_csv(output / f"profile_{name.split()[0].lower()}.csv")
        summary[["mean", "median"]].plot(ax=axis)
        axis.set(xlabel=name, ylabel="PM2.5 (µg/m³)")
    _save(fig, output / "calendar_profiles.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    matrix = numeric.corr()
    heat = axes[0].imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(range(len(matrix)), matrix.columns, rotation=45, ha="right")
    axes[0].set_yticks(range(len(matrix)), matrix.columns)
    axes[0].set_title("Pairwise Pearson correlation (not causation)")
    fig.colorbar(heat, ax=axes[0])
    axes[1].bar(range(len(acf)), acf.values, color="#08916b")
    axes[1].set_xticks(range(len(acf)), acf.index)
    axes[1].set(title="PM2.5 autocorrelation on hourly grid", xlabel="Lag (hours)", ylabel="Correlation", ylim=(-1, 1))
    _save(fig, output / "correlation_autocorrelation.png")
    return output
