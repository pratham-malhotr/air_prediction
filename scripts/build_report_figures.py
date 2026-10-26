"""Draw print-sized report figures from saved Phase 1 observations/predictions."""
from pathlib import Path
import csv
from datetime import datetime, timedelta
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / "artifacts/phase1"
    out = ROOT / "reports/ieee/figures"
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42}):
        with (run / "eda/autocorrelation.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        # Column labels are intentionally read from the saved EDA output.
        lookup = {int(row["lag_hours"]): float(row["autocorrelation"]) for row in rows}
        lags = [1, 24, 168]
        fig, ax = plt.subplots(figsize=(3.4, 1.95))
        bars = ax.bar(["1 hour", "24 hours", "168 hours"], [lookup[lag] for lag in lags],
                      color=["#087e64", "#3779aa", "#81929c"], width=0.55)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.axhline(0, color="#a2b3bc", linewidth=0.6)
        ax.set(ylabel="Autocorrelation", xlabel="Historical lag", ylim=(-0.13, 1.13))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.6)
        fig.savefig(out / "temporal_dependence.pdf")
        plt.close(fig)

        with (run / "predictions.csv").open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["phase"] == "holdout"
                    and row["target_time"] >= "2014-12-25"]
        start = datetime(2014, 12, 25)
        clock = [start + timedelta(hours=i) for i in range(168)]
        by_model = {name: {} for name in ("ridge", "xgboost")}
        actual = {}
        for row in rows:
            time = datetime.fromisoformat(row["target_time"])
            actual[time] = float(row["actual"])
            if row["model"] in by_model:
                by_model[row["model"]][time] = float(row["prediction"])
        fig, ax = plt.subplots(figsize=(7, 1.95))
        ax.plot(clock, [actual.get(t, math.nan) for t in clock], color="#102633", linewidth=1.2, label="Observed")
        for name, color, label in (("ridge", "#087e64", "Ridge"), ("xgboost", "#3779aa", "XGBoost")):
            ax.plot(clock, [by_model[name].get(t, math.nan) for t in clock], color=color, linewidth=0.85, alpha=0.85, label=label)
        ax.set(ylabel="PM2.5 (µg/m³)", xlabel="Target date, December 2014 (Beijing local clock)")
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.legend(loc="upper right", ncol=3, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        fig.tight_layout(pad=0.6)
        fig.savefig(out / "forecast_trace.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
