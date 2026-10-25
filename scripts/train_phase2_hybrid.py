"""Phase 2 Training & Evaluation: Deep Learning (BiLSTM) & Hybrid Residual Model.

Trains and backtests:
1. Standalone Bi-LSTM with Temporal Attention
2. Hybrid Residual Model (Bi-LSTM Base + Tabular Residual XGBoost)
3. Evaluates on 2014 holdout period matching Phase 1 protocol
4. Saves hybrid.joblib and bilstm.joblib into artifacts/phase1/
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from air_prediction.config import load_config
from air_prediction.data import FILENAME, load_dataset
from air_prediction.deep_learning import DLConfig, DeepLearningPipeline, HybridResidualModel
from air_prediction.features import build_features, common_evaluation_mask
from air_prediction.metrics import score_predictions
from air_prediction.models import make_model
from air_prediction.splits import fold_masks, make_folds


def train_and_evaluate_phase2(csv_path="data/raw/" + FILENAME, config_path="configs/phase1.toml", artifacts_dir="artifacts/phase1"):
    artifacts_dir = Path(artifacts_dir)
    config = load_config(config_path)
    dataset = load_dataset(csv_path)
    frame = dataset.frame
    data = build_features(frame, config)

    print("=== Phase 2: Deep Learning & Hybrid Model Backtest ===")
    holdout_folds = make_folds(config, "holdout")
    
    # 1. Backtest Bi-LSTM and Hybrid on 2014 quarterly expanding windows
    bilstm_tables = []
    hybrid_tables = []

    for fold in holdout_folds:
        print(f"Evaluating fold: {fold.name} ({fold.start} to {fold.end})...")
        train_mask, test_mask, _ = fold_masks(data, fold, config)

        # Tabular XGBoost pipeline for residual fitting
        # Use tree_method='exact' and n_jobs=1 to avoid OpenMP runtime collision with PyTorch on macOS
        selected_xgb_params = {
            "n_estimators": 100, "max_depth": 4, "learning_rate": 0.05,
            "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_lambda": 5.0, "tree_method": "exact",
        }
        xgb_pipe = make_model("xgboost", data.X, config, selected_xgb_params)

        # Bi-LSTM pipeline: pass full historical frame up to test start to keep 24-hr sliding window intact
        dl_cfg = DLConfig(seq_len=24, hidden_dim=48, num_layers=2, dropout=0.2, epochs=4, batch_size=128, architecture="bilstm")
        bilstm_pipe = DeepLearningPipeline(dl_cfg)
        
        # Fit Bi-LSTM using continuous sequence up to end of fold train
        bilstm_pipe.fit(data.X, data.y, valid_mask=train_mask)
        dl_all_preds = bilstm_pipe.predict(data.X)
        dl_test_preds = dl_all_preds[test_mask]

        bilstm_table = pd.DataFrame({
            "origin_time": data.X.index[test_mask],
            "target_time": data.target_time[test_mask].values,
            "actual": data.y[test_mask].values,
            "prediction": dl_test_preds,
            "model": "bilstm",
            "fold": fold.name,
            "phase": "holdout",
        })
        bilstm_tables.append(bilstm_table)

        # Hybrid Residual Fitting
        hybrid_model = HybridResidualModel(dl_pipeline=bilstm_pipe, tabular_model=xgb_pipe)
        # Compute residuals on training fold
        X_train = data.X.loc[train_mask]
        y_train = data.y.loc[train_mask]
        dl_train_preds = pd.Series(dl_all_preds[train_mask], index=X_train.index)
        valid_train = dl_train_preds.notna() & y_train.notna()
        residuals = y_train.loc[valid_train] - dl_train_preds.loc[valid_train]
        xgb_pipe.fit(X_train.loc[valid_train], residuals)
        hybrid_model.is_fitted = True

        res_test = xgb_pipe.predict(data.X.loc[test_mask])
        hybrid_test_preds = np.clip(dl_test_preds + res_test, 0.0, None)
        hybrid_table = pd.DataFrame({
            "origin_time": data.X.index[test_mask],
            "target_time": data.target_time[test_mask].values,
            "actual": data.y[test_mask].values,
            "prediction": hybrid_test_preds,
            "model": "hybrid",
            "fold": fold.name,
            "phase": "holdout",
        })
        hybrid_tables.append(hybrid_table)

    all_bilstm = pd.concat(bilstm_tables, ignore_index=True)
    all_hybrid = pd.concat(hybrid_tables, ignore_index=True)

    # Filter to common evaluation set
    common_mask = common_evaluation_mask(data)
    held = all_hybrid.origin_time.isin(data.X.index[common_mask])
    
    score_bilstm = score_predictions(all_bilstm[held])
    score_hybrid = score_predictions(all_hybrid[held])
    
    print("\nPhase 2 2014 Holdout Scores:")
    print(score_bilstm.to_string(index=False))
    print(score_hybrid.to_string(index=False))

    # 2. Update and append to metrics.csv & fold_metrics.csv
    phase1_metrics = pd.read_csv(artifacts_dir / "metrics.csv")
    phase1_fold_metrics = pd.read_csv(artifacts_dir / "fold_metrics.csv")

    # Combine metrics without duplicates
    updated_metrics = pd.concat([
        phase1_metrics.query("model not in ['bilstm', 'hybrid']"),
        score_bilstm,
        score_hybrid,
    ], ignore_index=True)
    updated_metrics.to_csv(artifacts_dir / "metrics.csv", index=False)

    fold_bilstm = score_predictions(all_bilstm[held], ("phase", "fold", "model"))
    fold_hybrid = score_predictions(all_hybrid[held], ("phase", "fold", "model"))
    updated_folds = pd.concat([
        phase1_fold_metrics.query("model not in ['bilstm', 'hybrid']"),
        fold_bilstm,
        fold_hybrid,
    ], ignore_index=True)
    updated_folds.to_csv(artifacts_dir / "fold_metrics.csv", index=False)

    # 3. Final Deployment Refit on all available data
    print("\nRefitting final deployment models on full dataset through 2014-12-31...")
    eligible = data.warm & data.y.notna() & (data.X.index >= pd.Timestamp(config.train_start)) & (data.target_time <= frame.index.max())
    
    final_dl = DeepLearningPipeline(DLConfig(seq_len=24, hidden_dim=48, num_layers=2, dropout=0.2, epochs=5, batch_size=128, architecture="bilstm"))
    final_dl.fit(data.X, data.y, valid_mask=eligible)

    final_xgb = make_model("xgboost", data.X, config, selected_xgb_params)
    final_hybrid = HybridResidualModel(dl_pipeline=final_dl, tabular_model=final_xgb)

    dl_full_all = final_dl.predict(data.X)
    X_eligible = data.X.loc[eligible]
    y_eligible = data.y.loc[eligible]
    dl_full_preds = pd.Series(dl_full_all[eligible], index=X_eligible.index)
    valid_full = dl_full_preds.notna() & y_eligible.notna()
    residuals_full = y_eligible.loc[valid_full] - dl_full_preds.loc[valid_full]
    final_xgb.fit(X_eligible.loc[valid_full], residuals_full)
    final_hybrid.is_fitted = True

    # 4. Save artifacts
    joblib.dump({
        "pipeline": final_dl,
        "config": config,
        "model_name": "bilstm",
        "columns": list(data.X),
        "fit_through": str(frame.index.max()),
        "data_source": dataset.source,
    }, artifacts_dir / "bilstm.joblib")

    joblib.dump({
        "pipeline": final_hybrid,
        "config": config,
        "model_name": "hybrid",
        "columns": list(data.X),
        "fit_through": str(frame.index.max()),
        "data_source": dataset.source,
    }, artifacts_dir / "hybrid.joblib")

    print(f"Saved hybrid.joblib and bilstm.joblib to {artifacts_dir}")


if __name__ == "__main__":
    train_and_evaluate_phase2()
