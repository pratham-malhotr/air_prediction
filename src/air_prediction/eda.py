"""Training-period EDA and an archive-wide schema/missingness audit."""

from pathlib import Path
from calendar import month_name

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data import WEATHER_NUMERIC, write_json


def _save(fig, destination, source=None):
    if source == "synthetic":
        fig.suptitle("SYNTHETIC DATA — INTEGRATION CHECK ONLY", color="darkred", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_eda_notes(frame, source, correlations, autocorrelation, destination):
    """Explain calculated training statistics without drawing on later observations."""
    pm = frame["pm2.5"]
    observed = pm.dropna()
    missing = int(pm.isna().sum())
    q1, median, q3, p90 = observed.quantile([0.25, 0.5, 0.75, 0.9])
    upper_fence = q3 + 1.5 * (q3 - q1)
    peak_count = int((observed > upper_fence).sum())
    lines = ["# Training-data EDA notes", ""]
    if source == "synthetic":
        lines += ["**SYNTHETIC DATA — implementation check only. These are not Beijing findings.**", ""]
    lines += [
        "## Data used", "",
        f"We examine {len(frame):,} hourly slots from {frame.index.min():%Y-%m-%d %H:%M} "
        f"to {frame.index.max():%Y-%m-%d %H:%M}, in Beijing local time. "
        "These notes use the initial training window; later validation and test observations are left out.",
        "", "## Missing readings", "",
        f"PM2.5 has {missing:,} missing readings ({100 * missing / len(frame):.2f}%). "
        f"That leaves {len(observed):,} measured concentrations in this window. "
        "We keep the missing hourly slots so a one-hour lag keeps its meaning, and we leave missing targets out of scoring.",
    ]
    weather_missing = frame[[*WEATHER_NUMERIC, "cbwd"]].isna().sum()
    incomplete = weather_missing[weather_missing > 0]
    if incomplete.empty:
        lines.append("The weather fields have no missing measurements in this training window.")
    else:
        detail = ", ".join(f"{column}: {int(count):,}" for column, count in incomplete.items())
        lines.append(f"The weather fields also contain gaps ({detail}). Missingness flags keep these gaps visible to the models.")
    lines += [
        "", "## Concentration range and peaks", "",
        f"The mean PM2.5 is {observed.mean():.2f} µg/m³ and the median is {median:.2f} µg/m³. "
        f"The observed range is {observed.min():.2f}–{observed.max():.2f} µg/m³, with a 90th percentile of {p90:.2f} µg/m³.",
        f"The upper IQR fence is {upper_fence:.2f} µg/m³; {peak_count:,} observations lie above it. "
        "We use this count to inspect the upper tail, not to remove valid pollution events. "
        "Reporting both MAE and RMSE helps separate typical absolute errors from the effect of large misses.",
        "", "## Time patterns", "",
    ]
    hourly = pm.groupby(pm.index.hour).agg(["mean", "count"]).dropna(subset=["mean"])
    peak_hour = int(hourly["mean"].idxmax())
    lines.append(
        f"The largest hourly average is at {peak_hour:02d}:00: {hourly.loc[peak_hour, 'mean']:.2f} µg/m³ "
        f"from {int(hourly.loc[peak_hour, 'count']):,} observations. "
        "This is a pooled profile of the observed training hours, not a claim that every day follows the same pattern."
    )
    monthly = pm.groupby(pm.index.month).agg(["mean", "count"]).dropna(subset=["mean"])
    if len(monthly) > 1:
        peak_month = int(monthly["mean"].idxmax())
        lines.append(
            f"Among the months represented, {month_name[peak_month]} has the highest average "
            f"({monthly.loc[peak_month, 'mean']:.2f} µg/m³, {int(monthly.loc[peak_month, 'count']):,} observations). "
            "We compare the full monthly distributions as well, because an average can hide large variation."
        )
    else:
        lines.append("Only one calendar month has observed values in this window, so it cannot establish an annual pattern.")
    lines += ["", "## Dependence on earlier readings", ""]
    available_acf = autocorrelation.reindex([1, 24, 168]).dropna()
    if available_acf.empty:
        lines.append("The selected autocorrelations are undefined in this window; there is insufficient variation or overlapping data.")
    else:
        detail = "; ".join(f"{int(lag)} h: {value:.3f}" for lag, value in available_acf.items())
        lines.append(f"PM2.5 autocorrelation is {detail}. It is calculated on the hourly grid without dropping missing slots.")
    lines.append(
        "These checks help us inspect recent, daily, and weekly history. Whether those features improve a forecast "
        "is tested on the chronological validation folds; autocorrelation alone does not answer that question."
    )
    lines += ["", "## Weather relationships", ""]
    associations = correlations.loc[list(WEATHER_NUMERIC), "pm2.5"].dropna()
    if associations.empty:
        lines.append("No finite PM2.5–weather correlation is available in this window. Constant or missing fields do not establish an absence of a weather effect.")
    else:
        strongest = associations.abs().idxmax()
        pairs = len(frame[["pm2.5", strongest]].dropna())
        lines.append(
            f"The largest absolute Pearson correlation among the numeric weather fields is for {strongest}: "
            f"r = {associations[strongest]:.3f}, based on {pairs:,} paired observations. "
            "We treat this as an association, not a causal explanation. A small linear correlation can also miss a nonlinear relationship."
        )
    lines += [
        "", "## What this means for our model comparison", "",
        "We compare recent lags, trailing summaries, known calendar values, and weather history. "
        "The observed fraction tells the model how much of a rolling window is measured. "
        "Ridge gives us a linear reference, while XGBoost can use nonlinear relationships. "
        "Persistence and daily seasonal naive keep the comparison tied to simple, explainable forecasts. "
        "The validation and test results, rather than these descriptive plots, decide how well those choices predict.",
    ]
    Path(destination).write_text("\n\n".join(part for part in lines if part != "") + "\n", encoding="utf-8")


def run_eda(dataset, output, until, since=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "data_audit.json", dataset.audit)
    mask = dataset.frame.index < pd.Timestamp(until)
    if since is not None:
        mask &= dataset.frame.index >= pd.Timestamp(since)
    frame = dataset.frame.loc[mask].copy()
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
    correlations = numeric.corr()
    correlations.to_csv(output / "correlations.csv")
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
    write_eda_notes(frame, dataset.source, correlations, acf, output / "EDA_NOTES.md")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    pm.resample("D").mean().plot(ax=axes[0], color="#2563eb", linewidth=0.8)
    axes[0].set(title="Training period: daily mean PM2.5", ylabel="PM2.5 (µg/m³)", xlabel="Beijing local date")
    monthly_missing = frame.isna().resample("MS").mean()
    for column in monthly_missing:
        axes[1].plot(monthly_missing.index, monthly_missing[column], linewidth=1, marker=".", label=column)
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].set(title="Monthly missing fraction (no interpolation)", ylabel="Fraction missing", xlabel="Month")
    _save(fig, output / "timeline_missingness.png", dataset.source)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(observed, bins=60, color="#2563eb", alpha=0.85)
    axes[0].set(title="PM2.5 distribution", xlabel="PM2.5 (µg/m³)", ylabel="Observed hours")
    axes[1].boxplot([pm[pm.index.month == month].dropna().to_numpy() for month in range(1, 13)], tick_labels=range(1, 13))
    axes[1].set(title="Monthly distributions; peaks retained", xlabel="Month", ylabel="PM2.5 (µg/m³)")
    _save(fig, output / "distribution_seasonality.png", dataset.source)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, name, group in zip(axes, ("Hour", "Weekday (Monday=0)", "Month"), (pm.index.hour, pm.index.dayofweek, pm.index.month)):
        summary = pm.groupby(group).agg(["mean", "median", "count"])
        summary.to_csv(output / f"profile_{name.split()[0].lower()}.csv")
        for statistic in ("mean", "median"):
            axis.plot(summary.index, summary[statistic], marker=".", label=statistic)
        axis.legend()
        axis.set(xlabel=name, ylabel="PM2.5 (µg/m³)")
    _save(fig, output / "calendar_profiles.png", dataset.source)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    matrix = correlations
    heat = axes[0].imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(range(len(matrix)), matrix.columns, rotation=45, ha="right")
    axes[0].set_yticks(range(len(matrix)), matrix.columns)
    axes[0].set_title("Pairwise Pearson correlation (not causation)")
    fig.colorbar(heat, ax=axes[0])
    axes[1].bar(range(len(acf)), acf.values, color="#08916b")
    axes[1].set_xticks(range(len(acf)), acf.index)
    axes[1].set(title="PM2.5 autocorrelation on hourly grid", xlabel="Lag (hours)", ylabel="Correlation", ylim=(-1, 1))
    _save(fig, output / "correlation_autocorrelation.png", dataset.source)
    return output
