"""Saved diagnostic figures and a report based exclusively on measured outputs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eda import _save


def plot_results(predictions, scores, output, source=None):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    holdout_scores = scores.loc[scores.phase == "holdout"].set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    holdout_scores[["mae", "rmse"]].plot.bar(ax=axes[0], rot=15, color=["#2563eb", "#08916b"])
    axes[0].set(title="Rolling holdout errors; common observed targets", ylabel="PM2.5 (µg/m³)", xlabel="")
    holdout_scores["r2"].plot.bar(ax=axes[1], rot=15, color="#2563eb")
    axes[1].set(title="Holdout R squared", ylabel="R²", xlabel="")
    _save(fig, figures / "model_comparison.png", source)
    held = predictions.loc[predictions.phase == "holdout"].copy()
    held["target_time"] = pd.to_datetime(held.target_time)
    models = sorted(set(held.model) - {"persistence", "seasonal_naive"})
    fig, ax = plt.subplots(figsize=(13, 4))
    last_week = held.loc[held.target_time >= held.target_time.max() - pd.Timedelta(days=7)]
    truth = last_week.drop_duplicates("target_time").set_index("target_time").actual.sort_index().asfreq("h")
    truth.plot(ax=ax, color="black", linewidth=1.5, label="observed")
    for name in ["persistence", *models]:
        series = last_week.loc[last_week.model == name].set_index("target_time").prediction.sort_index().asfreq("h")
        series.plot(ax=ax, linewidth=0.9, alpha=0.8, label=name)
    ax.set(title="Final seven days of rolling holdout predictions", ylabel="PM2.5 (µg/m³)", xlabel="Beijing local target time")
    ax.legend()
    _save(fig, figures / "forecast_last_week.png", source)
    fig, axes = plt.subplots(len(models), 3, figsize=(13, 3.5 * len(models)), squeeze=False)
    acf_rows = []
    for row, name in enumerate(models):
        part = held.loc[held.model == name].sort_values("target_time")
        sample = part.iloc[::max(1, len(part) // 3000)]
        axes[row, 0].scatter(sample.actual, sample.prediction, s=5, alpha=0.25)
        limit = max(float(part.actual.max()), float(part.prediction.max()), 1)
        axes[row, 0].plot([0, limit], [0, limit], color="black", linestyle="--")
        axes[row, 0].set(title=name, xlabel="Observed PM2.5", ylabel="Predicted PM2.5")
        residual = pd.Series((part.prediction - part.actual).to_numpy(), index=pd.DatetimeIndex(part.target_time)).asfreq("h")
        axes[row, 1].hist(residual.dropna(), bins=50, color="#2563eb")
        axes[row, 1].set(title="Residual distribution", xlabel="Prediction − observation (µg/m³)", ylabel="Hours")
        lags = [1, 6, 12, 24, 48, 168]
        correlations = [residual.autocorr(lag) for lag in lags]
        axes[row, 2].bar(range(len(lags)), correlations, color="#08916b")
        axes[row, 2].set_xticks(range(len(lags)), lags)
        axes[row, 2].set(title="Residual autocorrelation", xlabel="Lag (hours)", ylabel="Correlation", ylim=(-1, 1))
        acf_rows.extend({"model": name, "lag_hours": lag, "autocorrelation": value} for lag, value in zip(lags, correlations))
    pd.DataFrame(acf_rows).to_csv(output / "residual_autocorrelation.csv", index=False)
    _save(fig, figures / "residual_diagnostics.png", source)


def write_summary(scores, fold_scores, manifest, output):
    held = scores.loc[scores.phase == "holdout"].set_index("model")
    label = "SYNTHETIC INTEGRATION DEMO — NOT PROJECT RESULTS" if manifest["is_synthetic"] else "Measured Phase 1 experiment"
    lines = [f"# {label}", "", f"Data source: {manifest['source']}; SHA256: `{manifest['dataset_audit']['sha256']}`.",
             f"Forecast horizon: {manifest['config']['horizon_hours']} hour(s).", "",
             "Selection used chronological validation RMSE only. Holdout models were refit at each block boundary; histories update hourly.", "",
             "| Model | MAE (µg/m³) | RMSE (µg/m³) | R² | Scored hours |", "|---|---:|---:|---:|---:|"]
    for name, row in held.iterrows():
        r2 = f"{row.r2:.4f}" if pd.notna(row.r2) else "undefined"
        lines.append(f"| {name} | {row.mae:.4f} | {row.rmse:.4f} | {r2} | {int(row.n)} |")
    lines += ["", "## Interpretation", ""]
    persistence_rmse = held.loc["persistence", "rmse"]
    for name in manifest["models"]:
        error = held.loc[name, "rmse"]
        if persistence_rmse > 0:
            delta = 100 * (persistence_rmse - error) / persistence_rmse
            lines.append(f"- {name}: RMSE improvement relative to persistence = {delta:.2f}% (negative means worse).")
    lines += ["", "Compare `fold_metrics.csv` to assess seasonal variation; pooled scores weight each scored hour equally.",
              "Coverage and overlapping exclusion counts are in `fold_audit.json`. Error tables stratify by month, hour, and training-period PM2.5 p90.",
              "Residual autocorrelation and feature importance are descriptive; neither establishes causality or calibrated uncertainty.",
              "", "## Limits", "",
              "One Beijing monitoring location and historical years do not establish transfer to another city or current conditions.",
              "Observed weather is used only through the forecast origin, assuming those measurements are available at issuance.",
              "No AQI is computed: hourly PM2.5 alone does not define a valid daily AQI category.",
              "No deep learning or hybrid model is part of this Phase 1 implementation."]
    Path(output, "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
