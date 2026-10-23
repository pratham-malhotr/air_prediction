"""Create report tables and figures from the completed real-data experiment."""

import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from air_prediction.config import load_config
from air_prediction.data import load_dataset, write_json
from air_prediction.features import build_features
from air_prediction.predict import predict_latest

ROOT = Path(__file__).resolve().parents[1]
DISPLAY = {"persistence": "Persistence", "ridge": "Ridge", "xgboost": "XGBoost", "seasonal_naive": "Seasonal naive"}


def write_latex_table(frame, destination):
    """Write the small report tables without pandas' optional Jinja2 dependency."""
    escapes = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$",
               "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}"}

    def cell(value):
        if isinstance(value, float):
            return f"{value:.3f}"
        return "".join(escapes.get(character, character) for character in str(value))

    columns = "l" + "r" * (len(frame.columns) - 1)
    lines = [r"\begin{tabular}{" + columns + "}", r"\toprule",
             " & ".join(cell(value) for value in frame.columns) + r" \\", r"\midrule"]
    lines.extend(" & ".join(cell(value) for value in row) + r" \\" for row in frame.itertuples(index=False, name=None))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    output = ROOT / "artifacts/phase1"
    manifest = json.loads((output / "run_manifest.json").read_text())
    if manifest["is_synthetic"] or manifest["status"] != "complete":
        raise ValueError("A completed real-data experiment is required")
    tables = output / "report_tables"
    figures = output / "figures"
    tracked = ROOT / "docs/results"
    tables.mkdir(exist_ok=True)
    tracked.mkdir(exist_ok=True)
    folds = pd.read_csv(output / "fold_metrics.csv")
    holdout = folds.loc[folds.phase == "holdout"]
    quarterly = holdout.pivot(index="fold", columns="model", values="rmse")
    quarterly = quarterly[["persistence", "ridge", "xgboost", "seasonal_naive"]]
    quarterly.index = [f"Q{i}" for i in range(1, 5)]
    quarterly = quarterly.rename(columns=DISPLAY)
    write_latex_table(quarterly.rename_axis("Quarter").reset_index(), tables / "quarterly_rmse.tex")
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, color in (("Persistence", "#64748b"), ("Ridge", "#08916b"), ("XGBoost", "#2563eb")):
        ax.plot(quarterly.index, quarterly[name], marker="o", color=color, label=name)
    ax.set(title="2014 quarterly RMSE on common forecast hours", ylabel="RMSE (µg/m³)", xlabel="Target quarter")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "quarterly_rmse.png", dpi=160)
    plt.close(fig)

    ablation = pd.read_csv(output / "validation_ablation.csv")
    labels = {"lags_only": "Lags only", "temporal": "+ rolling/calendar", "full": "+ weather"}
    table = ablation[["feature_set", "n", "mae", "rmse"]].rename(columns={"feature_set": "Features", "n": "Hours", "mae": "MAE", "rmse": "RMSE"})
    table["Features"] = table.Features.map(labels)
    write_latex_table(table, tables / "ablation.tex")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(ablation.feature_set.map(labels), ablation.rmse, color=["#64748b", "#08916b", "#2563eb"])
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set(title="XGBoost feature comparison — 2013 validation only", ylabel="RMSE (µg/m³)", ylim=(0, ablation.rmse.max() * 1.17))
    fig.tight_layout()
    fig.savefig(figures / "validation_ablation.png", dpi=160)
    plt.close(fig)

    regimes = pd.read_csv(output / "errors_by_pollution_regime.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, color in (("persistence", "#64748b"), ("ridge", "#08916b"), ("xgboost", "#2563eb")):
        part = regimes.loc[regimes.model == label].set_index("pollution_regime")
        axes[0].scatter(DISPLAY[label], part.loc["below_training_p90", "rmse"], color=color, s=70)
        axes[1].scatter(DISPLAY[label], part.loc["above_training_p90", "rmse"], color=color, s=70)
    axes[0].set(title="Below training p90", ylabel="RMSE (µg/m³)")
    axes[1].set(title="At/above 218 µg/m³", ylabel="RMSE (µg/m³)")
    fig.suptitle("2014 errors by observed pollution level")
    fig.tight_layout()
    fig.savefig(figures / "pollution_regime_errors.png", dpi=160)
    plt.close(fig)

    # Keep a compact, versionable record alongside the narrative; full predictions stay in artifacts.
    for name in ("metrics.csv", "fold_metrics.csv", "validation_candidates.csv", "validation_ablation.csv", "errors_by_pollution_regime.csv", "residual_autocorrelation.csv", "selection.json", "run_manifest.json"):
        shutil.copyfile(output / name, tracked / name)
    shutil.copyfile(output / "preparation/fold_coverage.csv", tracked / "fold_coverage.csv")
    for name in ("model_comparison.png", "quarterly_rmse.png", "validation_ablation.png", "pollution_regime_errors.png", "forecast_last_week.png"):
        shutil.copyfile(figures / name, tracked / name)
    for name in ("timeline_missingness.png", "distribution_seasonality.png", "correlation_autocorrelation.png"):
        shutil.copyfile(output / "eda" / name, tracked / name)
    data_path = ROOT / manifest["dataset_audit"]["path"]
    write_json(output / "next_historical_forecasts.json", {
        "purpose": "Unscored historical demonstration; not a present-day live forecast",
        "forecasts": [predict_latest(output / f"{name}.joblib", data_path) for name in ("ridge", "xgboost")],
    })
    predictions = pd.read_csv(output / "predictions.csv", parse_dates=["origin_time", "target_time"])
    held = predictions.loc[predictions.phase == "holdout"]
    worst = held.loc[held.model == "xgboost"].copy()
    worst["absolute_error"] = (worst.prediction - worst.actual).abs()
    worst.nlargest(10, "absolute_error")[["origin_time", "target_time", "actual", "prediction", "absolute_error"]].to_csv(output / "largest_xgboost_errors.csv", index=False)
    shutil.copyfile(output / "largest_xgboost_errors.csv", tracked / "largest_xgboost_errors.csv")
    first = held.loc[held.target_time == held.target_time.min()]
    origin = first.origin_time.iloc[0]
    data = build_features(load_dataset(data_path).frame, load_config(ROOT / "configs/phase1.toml"))
    audit = json.loads((output / "fold_audit.json").read_text())
    first_fit = next(fold for fold in audit if fold["phase"] == "holdout")
    write_json(output / "worked_forecast_example.json", {
        "purpose": "First common 2014 target, chosen by time rather than accuracy",
        "origin_time": str(origin), "target_time": str(first.target_time.iloc[0]),
        "actual_pm25": float(first.actual.iloc[0]),
        "selected_features": data.X.loc[origin, ["pm25_lag_0", "pm25_lag_1", "pm25_lag_24", "pm25_mean_3", "pm25_observed_fraction_24", "wind_direction"]].to_dict(),
        "latest_allowed_training_label": first_fit["fit_time"],
        "predictions": {row.model: {"prediction": float(row.prediction), "signed_error": float(row.prediction-row.actual)} for row in first.itertuples()},
    })
    # Draw article figures at their printed width so axis text remains readable.
    training = load_dataset(data_path).frame.loc[lambda frame: (frame.index >= pd.Timestamp(manifest["config"]["train_start"])) & (frame.index < pd.Timestamp(manifest["config"]["validation_start"]))]
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(3.4, 2.15))
        ax.hist(training["pm2.5"].dropna(), bins=45, color="#2563eb", alpha=0.85)
        ax.axvline(manifest["training_p90_pm25"], linestyle="--", color="#b45309", linewidth=1, label="Training p90 = 218")
        ax.set(xlabel="PM2.5 (µg/m³)", ylabel="Observed training hours")
        ax.legend()
        fig.tight_layout(pad=0.5)
        fig.savefig(figures / "article_distribution.pdf")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        for offset, name, color in ((-0.24, "persistence", "#64748b"), (0, "ridge", "#08916b"), (0.24, "xgboost", "#2563eb")):
            part = regimes.loc[regimes.model == name].set_index("pollution_regime")
            values = [part.loc["below_training_p90", "rmse"], part.loc["above_training_p90", "rmse"]]
            ax.bar([offset, 1+offset], values, width=0.23, label=DISPLAY[name], color=color)
        ax.set_xticks([0, 1], ["Below 218", "At/above 218"])
        ax.set(ylabel="RMSE (µg/m³)", xlabel="Observed target PM2.5 (µg/m³)", ylim=(0, 47))
        ax.legend(loc="upper center", ncol=3, handlelength=1, columnspacing=0.8, frameon=False)
        fig.tight_layout(pad=0.5)
        fig.savefig(figures / "article_peak_errors.pdf")
        plt.close(fig)
    print(f"Saved measured report tables/figures and a compact results record to {tracked}")


if __name__ == "__main__":
    main()
