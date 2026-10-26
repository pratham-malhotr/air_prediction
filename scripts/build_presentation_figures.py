"""Publication-size graphs drawn only from the completed Phase 1 experiment.

Exports 300 dpi PNGs for the deck and standalone vector PDFs for every graph.
The JSON manifest records source scopes and numerical checks for later review.
"""

from collections import defaultdict
import csv
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/phase1"
OUT = ROOT / "reports/presentation/figures"
NAVY, GREEN, BLUE, GRAY, ORANGE = "#102633", "#087E64", "#3779AA", "#73838D", "#DC813F"
COLORS = {"persistence": GRAY, "ridge": GREEN, "xgboost": BLUE, "seasonal_naive": "#B5BDC1"}
NAMES = {"persistence": "Persistence", "ridge": "Ridge", "xgboost": "XGBoost", "seasonal_naive": "Daily seasonal naive"}


def read(name):
    with (RUN / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#C9D4D6")
    ax.tick_params(colors="#526774", length=0, pad=6)
    if grid:
        ax.grid(axis=grid, color="#E8EEEC", linewidth=0.7)
    ax.set_axisbelow(True)


def build_figures():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RUN / "run_manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["is_synthetic"]:
        raise ValueError("Presentation graphs require the completed real-data run")
    raw_path = ROOT / manifest["dataset_audit"]["path"]
    if sha256(raw_path.read_bytes()).hexdigest() != manifest["dataset_audit"]["sha256"]:
        raise ValueError("The raw archive no longer matches the measured experiment")
    with raw_path.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    training = [float(r["pm2.5"]) for r in raw if "2010" <= r["year"] <= "2012" and r["pm2.5"] not in {"", "NA"}]
    years = defaultdict(lambda: [0, 0])
    for r in raw:
        years[r["year"]][0] += 1
        years[r["year"]][1] += r["pm2.5"] in {"", "NA"}
    metrics = {(r["phase"], r["model"]): r for r in read("metrics.csv")}
    folds = read("fold_metrics.csv")
    regimes = {(r["model"], r["pollution_regime"]): r for r in read("errors_by_pollution_regime.csv")}
    predictions = [r for r in read("predictions.csv") if r["phase"] == "holdout"]
    by_model = {name: sorted([r for r in predictions if r["model"] == name], key=lambda r: r["target_time"])
                for name in COLORS}
    observed = {r["target_time"]: float(r["actual"]) for r in by_model["xgboost"]}
    if len(observed) != 8531 or any({r["target_time"] for r in rows} != set(observed) for rows in by_model.values()):
        raise ValueError("The graphs must use identical scored target hours")
    diagnostics = {}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
                         "text.color": NAVY, "axes.labelcolor": "#526774", "pdf.fonttype": 42,
                         "axes.titleweight": "bold", "lines.linewidth": 2}):
        fig, ax = plt.subplots(figsize=(6.3, 2.85), layout="constrained")
        ax.hist(training, bins=np.arange(0, 1001, 25), color=GREEN, alpha=0.9, edgecolor="white", linewidth=0.5)
        ax.axvline(218, color=ORANGE, linestyle="--", linewidth=1.8, label="Training p90 = 218")
        ax.set(xlabel="Training PM2.5 (µg/m³)", ylabel="Observed hours", xlim=(0, 1000))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.legend(frameon=False, loc="upper right")
        style(ax); save(fig, "training_distribution")

        fig, ax = plt.subplots(figsize=(5.6, 2.85), layout="constrained")
        labels = sorted(years)
        rates = [100*years[y][1]/years[y][0] for y in labels]
        bars = ax.bar(labels, rates, width=0.58, color=[GREEN]*3+[BLUE]*2)
        ax.bar_label(bars, labels=[f"{x:.2f}%" for x in rates], padding=5, fontsize=12, color=NAVY)
        ax.set(ylabel="Missing PM2.5 readings (%)", ylim=(0, 10.4), xlabel="Year · full-archive quality audit")
        style(ax); save(fig, "missingness_by_year")

        acf = read("eda/autocorrelation.csv")
        fig, ax = plt.subplots(figsize=(4.0, 3.15), layout="constrained")
        values = [float(r["autocorrelation"]) for r in acf]
        ax.bar(range(len(acf)), values, width=0.65, color=[GREEN]+[BLUE]*7+[GRAY])
        ax.axhline(0, color="#ADBBBF", linewidth=0.8)
        ax.annotate(f"{values[0]:.3f}", (0, values[0]), xytext=(0, 6), textcoords="offset points", ha="center", weight="bold", color=GREEN)
        ax.annotate(f"{values[-1]:.3f}", (8, 0), xytext=(0, -21), textcoords="offset points", ha="center", color=GRAY, fontsize=10)
        ax.set_xticks(range(len(acf)), [r["lag_hours"] for r in acf])
        ax.set(xlabel="Historical lag (hours)", ylabel="Autocorrelation", ylim=(-0.13, 1.1))
        style(ax); save(fig, "training_autocorrelation")

        hourly = read("eda/profile_hour.csv")
        fig, ax = plt.subplots(figsize=(4.0, 3.15), layout="constrained")
        xx = [int(r["origin_time"]) for r in hourly]
        yy = [float(r["mean"]) for r in hourly]
        ax.plot(xx, yy, color=GREEN, marker="o", markersize=3.8)
        ax.fill_between(xx, yy, 75, color=GREEN, alpha=0.08)
        ax.annotate("01:00 peak", (1, yy[1]), xytext=(6, 115), arrowprops={"arrowstyle": "-", "color": GRAY}, fontsize=10)
        ax.set(xlabel="Hour of day (Beijing local clock)", ylabel="Mean PM2.5 (µg/m³)", ylim=(75, 120), xlim=(0, 23), xticks=[0, 6, 12, 18, 23])
        style(ax); save(fig, "training_hourly_profile")

        correlation = read("eda/correlations.csv")
        pmrow = next(r for r in correlation if r[""] == "pm2.5")
        names = [("Iws", "Wind speed"), ("DEWP", "Dew point"), ("PRES", "Pressure"), ("Ir", "Rain hours"), ("Is", "Snow hours"), ("TEMP", "Temperature")]
        fig, ax = plt.subplots(figsize=(4.0, 3.15), layout="constrained")
        values = [float(pmrow[key]) for key, _ in names]
        bars = ax.barh([label for _, label in names], values, color=[BLUE if x<0 else GREEN for x in values], height=0.6)
        ax.bar_label(bars, fmt="%+.3f", padding=4, fontsize=10)
        ax.axvline(0, color="#ADBBBF", linewidth=0.9)
        ax.invert_yaxis()
        ax.set(xlabel="Pearson correlation with PM2.5", xlim=(-0.41, 0.40), xticks=[-0.3, 0, 0.3])
        style(ax, "x"); save(fig, "training_weather_correlations")

        ablation = read("validation_ablation.csv")
        fig, ax = plt.subplots(figsize=(6.0, 2.45), layout="constrained")
        x = np.arange(3); y = [float(r["rmse"]) for r in ablation]
        ax.plot(x, y, color=BLUE, marker="o", markersize=8, linewidth=2)
        for xx, yy in zip(x, y):
            ax.annotate(f"{yy:.3f}", (xx, yy), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=12, weight="bold", color=BLUE)
        for name, color in [("persistence", GRAY), ("ridge", GREEN)]:
            value=float(metrics[("validation", name)]["rmse"])
            ax.axhline(value, color=color, linestyle="--", linewidth=1.3, label=f"{NAMES[name]}: {value:.3f}")
        ax.set_xticks(x, ["Lags only", "+ trailing / calendar", "+ weather"])
        ax.set(ylabel="Validation RMSE (µg/m³)", ylim=(22.4, 28.8), xlim=(-0.25, 2.25))
        ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=10)
        style(ax); save(fig, "validation_ablation")

        order = ["ridge", "xgboost", "persistence", "seasonal_naive"]
        fig, ax = plt.subplots(figsize=(6.5, 3.1), layout="constrained")
        vals=[float(metrics[("holdout", key)]["rmse"]) for key in order]
        bars=ax.barh([NAMES[key] for key in order], vals, color=[COLORS[key] for key in order], height=0.56)
        ax.bar_label(bars, fmt="%.3f", padding=7, fontsize=12, weight="bold", color=NAVY)
        ax.invert_yaxis(); ax.set(xlabel="2014 pooled RMSE (µg/m³) · lower is better", xlim=(0, 116))
        style(ax, "x"); save(fig, "holdout_rmse")

        fig, ax = plt.subplots(figsize=(6.0, 3.2), layout="constrained")
        for name in ["persistence", "ridge", "xgboost"]:
            part=sorted([r for r in folds if r["phase"]=="holdout" and r["model"]==name], key=lambda r:r["fold"])
            ax.plot([1,2,3,4], [float(r["rmse"]) for r in part], marker="o", markersize=6, color=COLORS[name], label=NAMES[name])
        ax.set(xticks=[1,2,3,4], xticklabels=["Q1", "Q2", "Q3", "Q4"], ylim=(12,30), yticks=[12,16,20,24,28], xlabel="Target quarter, 2014", ylabel="RMSE (µg/m³)")
        ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=10)
        style(ax); save(fig, "quarterly_rmse")

        fig, ax = plt.subplots(figsize=(6.0, 3.2), layout="constrained")
        for i,name in enumerate(["persistence", "ridge", "xgboost"]):
            values=[float(regimes[(name, key)]["rmse"]) for key in ["below_training_p90", "above_training_p90"]]
            bars=ax.bar(np.arange(2)+(i-1)*0.24, values, width=0.23, color=COLORS[name], label=NAMES[name])
            ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
        ax.set(xticks=[0,1], xticklabels=["Below 218\n7,666 hours", "At / above 218\n865 hours"], ylabel="RMSE (µg/m³)", ylim=(0,49))
        ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=10)
        style(ax); save(fig, "pollution_regime_errors")

        start=datetime(2014,12,25)
        clock=[start+timedelta(hours=i) for i in range(168)]
        fig, ax = plt.subplots(figsize=(12.3, 2.35), layout="constrained")
        ax.plot(clock, [observed.get(str(t), np.nan) for t in clock], color=NAVY, label="Observed", linewidth=1.9)
        for name, ls in [("ridge", "--"),("xgboost", "-")]:
            table={r["target_time"]:float(r["prediction"]) for r in by_model[name]}
            ax.plot(clock, [table.get(str(t), np.nan) for t in clock], color=COLORS[name], label=NAMES[name], linewidth=1.2, linestyle=ls, alpha=0.9)
        ax.set(ylabel="PM2.5 (µg/m³)", xlim=(clock[0],clock[-1]), ylim=(0,520))
        ax.xaxis.set_major_locator(mdates.DayLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.legend(frameon=False, ncol=3, loc="upper right")
        style(ax); save(fig, "forecast_last_week")

        fig, ax = plt.subplots(figsize=(6.0, 2.0), layout="constrained")
        for name in ["ridge", "xgboost"]:
            residual=np.array([float(r["prediction"])-float(r["actual"]) for r in by_model[name]])
            counts, edges=np.histogram(residual, bins=np.linspace(-80,80,41))
            ax.stairs(100*counts/len(residual), edges, color=COLORS[name], linewidth=1.5, label=NAMES[name], fill=True, alpha=0.16)
            ax.stairs(100*counts/len(residual), edges, color=COLORS[name], linewidth=1.4)
            diagnostics[name]={"residual_view_outside_80":int((np.abs(residual)>80).sum()),"n":len(residual)}
        ax.axvline(0, color=GRAY, linestyle="--", linewidth=1)
        ax.set(xlabel="Forecast − observed (µg/m³) · ±80 view", ylabel="Scored hours (%)", xlim=(-80,80))
        ax.text(0.02, 0.97,
                f"Outside view: Ridge {diagnostics['ridge']['residual_view_outside_80']}; XGBoost {diagnostics['xgboost']['residual_view_outside_80']}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=GRAY)
        ax.legend(frameon=False, loc="upper right", fontsize=10)
        style(ax); save(fig, "residual_distribution")

        worst=max(by_model["xgboost"], key=lambda r: abs(float(r["prediction"])-float(r["actual"])))
        spike=worst["target_time"]
        vals=[float(worst["actual"])]+[next(float(r["prediction"]) for r in by_model[name] if r["target_time"]==spike) for name in ["persistence", "ridge", "xgboost"]]
        fig, ax = plt.subplots(figsize=(6.0, 2.0), layout="constrained")
        bars=ax.barh(["Observed", "Persistence", "Ridge", "XGBoost"], vals, color=[ORANGE,GRAY,GREEN,BLUE], height=0.6)
        ax.bar_label(bars, fmt="%.1f", padding=5, fontsize=10.5)
        ax.invert_yaxis(); ax.set(xlabel="PM2.5 (µg/m³)", xlim=(0,675))
        style(ax,"x"); save(fig,"largest_spike_miss")

        importance=read("feature_importance.csv")
        top=importance[:5]
        labels={"numeric__pm25_lag_0":"Latest PM2.5", "numeric__pm25_mean_3":"3-hour mean",
                "numeric__pm25_observed_fraction_24":"24-hour support", "categorical__wind_direction_NW":"NW wind indicator",
                "numeric__pm25_observed_fraction_12":"12-hour support"}
        values=[100*float(r["gain_importance"]) for r in top]
        fig, ax=plt.subplots(figsize=(5.3,2.45),layout="constrained")
        bars=ax.barh([labels.get(r["feature"],r["feature"].replace("numeric__", "")) for r in top],values,color=[GREEN,BLUE,GRAY,GRAY,GRAY],height=0.6)
        ax.bar_label(bars,fmt="%.1f%%",padding=5,fontsize=10.5)
        ax.invert_yaxis(); ax.set(xlabel="Normalized gain importance (%)",xlim=(0,62))
        style(ax,"x"); save(fig,"deployment_feature_importance")

    audit={"source":"Completed Phase 1 real-data run", "dataset_sha256":manifest["dataset_audit"]["sha256"],
           "eda_scope":"2010–2012 only; missingness-by-year is a descriptive archive audit", "training_observations":len(training),
           "holdout_common_targets":len(observed), "missingness_by_year":{y:{"slots":v[0],"missing":v[1],"percent":100*v[1]/v[0]} for y,v in sorted(years.items())},
           "residual_plot":diagnostics, "largest_xgboost_miss":{"target_time":spike,"actual":float(worst["actual"]),"prediction":float(worst["prediction"])},
           "importance_scope":"Final deployment refit; not causal attribution or holdout feature selection", "figures":{}}
    for path in sorted(OUT.glob("*.png")):
        audit["figures"][path.stem]={"png_sha256":sha256(path.read_bytes()).hexdigest(),"vector_pdf":path.with_suffix(".pdf").name}
    (OUT/"figure_manifest.json").write_text(json.dumps(audit,indent=2)+"\n")
    return audit


if __name__=="__main__":
    result=build_figures()
    print(f"Built {len(result['figures'])} graphs from {result['holdout_common_targets']:,} shared holdout targets")
