"""Model selection on 2013; frozen selection followed by rolling 2014 evaluation."""

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd

from .data import load_dataset, write_json
from .eda import run_eda
from .features import build_features
from .metrics import score_predictions
from .models import make_model
from .splits import fold_masks, make_folds


def backtest(data, config, phase, name, parameters=None, include_baselines=True):
    tables, audits = [], []
    for fold in make_folds(config, phase):
        train, test, audit = fold_masks(data, fold, config)
        model = make_model(name, data.X, config, parameters)
        model.fit(data.X.loc[train], data.y.loc[train])
        prediction = np.maximum(model.predict(data.X.loc[test]), 0.0)
        table = pd.DataFrame({
            "origin_time": data.X.index[test], "target_time": data.target_time[test].values,
            "actual": data.y[test].values, "prediction": prediction,
            "model": name, "fold": fold.name, "phase": phase,
        })
        tables.append(table)
        if include_baselines:
            for baseline in data.baselines:
                other = table.copy()
                other["model"] = baseline
                other["prediction"] = data.baselines.loc[test, baseline].values
                tables.append(other)
        audits.append(audit)
    return pd.concat(tables, ignore_index=True), audits


def select_parameters(data, config, names):
    records, selected, prediction_tables, fold_audits = [], {}, [], []
    for name in names:
        candidates = config.xgboost_candidates if name == "xgboost" else ({"alpha": 10.0},)
        best = None
        for number, parameters in enumerate(candidates, 1):
            predictions, audit = backtest(data, config, "validation", name, parameters if name == "xgboost" else None)
            scores = score_predictions(predictions)
            row = scores.loc[scores.model == name].iloc[0]
            records.append({"model": name, "candidate": number, "parameters": json.dumps(parameters, sort_keys=True),
                            "validation_rmse": row.rmse, "validation_mae": row.mae, "n": int(row.n)})
            if best is None or row.rmse < best[0]:
                best = (row.rmse, dict(parameters), predictions)
            fold_audits = audit
        selected[name] = best[1]
        prediction_tables.append(best[2])
    validation = pd.concat(prediction_tables, ignore_index=True).drop_duplicates(["origin_time", "model"])
    return selected, pd.DataFrame(records), validation, fold_audits


def _versions():
    output = {"python": sys.version, "platform": platform.platform()}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "matplotlib", "joblib"):
        try:
            output[name] = version(name)
        except PackageNotFoundError:
            output[name] = None
    return output


def _write_latex_table(scores, destination):
    holdout = scores.loc[scores.phase == "holdout"]
    lines = ["% Generated from this run's real predictions; source is recorded in run_manifest.json.",
             r"\begin{tabular}{lrrr}", r"\hline", r"Model & MAE & RMSE & $R^2$ \\", r"\hline"]
    for row in holdout.itertuples():
        r2 = f"{row.r2:.3f}" if pd.notna(row.r2) else "--"
        lines.append(f"{row.model.replace('_', ' ')} & {row.mae:.3f} & {row.rmse:.3f} & {r2} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    destination.write_text("\n".join(lines) + "\n")


def run_experiment(csv_path, config, output, names=("xgboost", "ridge"), ablations=False):
    if not names or len(set(names)) != len(names) or any(name not in {"xgboost", "ridge"} for name in names):
        raise ValueError("Choose one or both of xgboost and ridge, without duplicates")
    output = Path(output)
    if (output / "run_manifest.json").exists():
        raise FileExistsError(f"Run already exists: {output}. Use a new output directory to preserve its results.")
    dataset = load_dataset(csv_path)
    if dataset.source == "synthetic" and output.name == "phase1":
        raise ValueError("Synthetic data must use a separate demo directory, never artifacts/phase1")
    frame = dataset.frame
    if frame.index.min() > pd.Timestamp(config.train_start):
        raise ValueError("Input starts after the configured training start")
    if frame.index.max() < pd.Timestamp(config.holdout_end) - pd.Timedelta(hours=1):
        raise ValueError("Input does not cover the full configured holdout; do not report a partial period as complete")
    data = build_features(frame, config)
    # Check model availability before costly EDA or partial result files.
    for name in names:
        make_model(name, data.X, config, config.xgboost_candidates[0] if name == "xgboost" else None)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "RUNNING.json", {"source": dataset.source, "started_utc": datetime.now(timezone.utc).isoformat()})
    run_eda(dataset, output / "eda", config.validation_start)
    selected, candidates, validation, validation_audit = select_parameters(data, config, names)
    candidates.to_csv(output / "validation_candidates.csv", index=False)
    validation.to_csv(output / "validation_predictions.csv", index=False)
    # Persist the decision before opening final evaluation results.
    write_json(output / "selection.json", {"criterion": "pooled validation RMSE", "selected": selected, "holdout_used_for_selection": False})
    tables, holdout_audit = [], []
    for name in names:
        predictions, holdout_audit = backtest(data, config, "holdout", name, selected[name] if name == "xgboost" else None)
        tables.append(predictions)
    holdout = pd.concat(tables, ignore_index=True).drop_duplicates(["origin_time", "model"])
    predictions = pd.concat([validation, holdout], ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    scores = score_predictions(predictions)
    fold_scores = score_predictions(predictions, ("phase", "fold", "model"))
    scores.to_csv(output / "metrics.csv", index=False)
    fold_scores.to_csv(output / "fold_metrics.csv", index=False)
    write_json(output / "fold_audit.json", validation_audit + holdout_audit)
    # Error strata use the same held-out observations for each model.
    diagnostics = holdout.copy()
    diagnostics["month"] = pd.to_datetime(diagnostics.target_time).dt.month
    diagnostics["hour"] = pd.to_datetime(diagnostics.target_time).dt.hour
    for grouping in ("month", "hour"):
        score_predictions(diagnostics, ("model", grouping)).to_csv(output / f"errors_by_{grouping}.csv", index=False)
    threshold = float(frame.loc[frame.index < pd.Timestamp(config.validation_start), "pm2.5"].quantile(0.9))
    diagnostics["pollution_regime"] = np.where(diagnostics.actual >= threshold, "above_training_p90", "below_training_p90")
    score_predictions(diagnostics, ("model", "pollution_regime")).to_csv(output / "errors_by_pollution_regime.csv", index=False)
    # Deployment refit uses all currently observed labels, after evaluation is saved.
    eligible = data.warm & data.y.notna() & (data.X.index >= pd.Timestamp(config.train_start)) & (data.target_time <= frame.index.max())
    for name in names:
        final_model = make_model(name, data.X, config, selected[name] if name == "xgboost" else None)
        final_model.fit(data.X.loc[eligible], data.y.loc[eligible])
        joblib.dump({"pipeline": final_model, "config": config, "model_name": name,
                     "columns": list(data.X), "fit_through": str(frame.index.max()), "data_source": dataset.source}, output / f"{name}.joblib")
        if name == "xgboost":
            final_model.named_steps["regressor"].save_model(output / "xgboost.json")
            importance = pd.DataFrame({"feature": final_model.named_steps["preprocess"].get_feature_names_out(),
                                       "gain_importance": final_model.named_steps["regressor"].feature_importances_})
            importance.sort_values("gain_importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    if ablations and "xgboost" in names:
        rows = []
        for feature_set in ("lags_only", "temporal", "full"):
            subset = build_features(frame, config, feature_set)
            ablation_predictions, _ = backtest(subset, config, "validation", "xgboost", selected["xgboost"], include_baselines=False)
            metric = score_predictions(ablation_predictions).iloc[0].to_dict()
            rows.append({"feature_set": feature_set, **metric})
        pd.DataFrame(rows).to_csv(output / "validation_ablation.csv", index=False)
    _write_latex_table(scores, output / "results_table.tex")
    from .results import plot_results, write_summary
    plot_results(predictions, scores, output)
    manifest = {
        "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source": dataset.source, "is_synthetic": dataset.source == "synthetic", "dataset_audit": dataset.audit,
        "config": config.to_dict(), "models": list(names), "selected_parameters": selected,
        "versions": _versions(), "feature_count": len(data.X.columns), "training_p90_pm25": threshold,
        "protocol": "Point forecast at t+h, observed history through t, quarterly expanding refit by default, hourly updated covariates.",
        "prediction_policy": "Regression outputs clipped to zero for nonnegative concentrations, consistently in validation/test/inference.",
        "deployment_models": "Refitted on all available observed labels after evaluation; never used to produce backtest scores.",
    }
    write_json(output / "run_manifest.json", manifest)
    write_summary(scores, fold_scores, manifest, output)
    (output / "RUNNING.json").unlink(missing_ok=True)
    return {"metrics": scores, "fold_metrics": fold_scores, "predictions": predictions, "manifest": manifest, "output": output}
