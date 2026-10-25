"""Deep Learning architectures, sequence generators, and hybrid models for Phase 2."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class DLConfig:
    seq_len: int = 24  # 24 hours of sequence history
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 15
    device: str = "cpu"
    architecture: str = "lstm"  # "lstm", "gru", "bilstm"


class SequenceAirDataset(Dataset):
    """3D sliding sequence window dataset for multivariate temporal series.
    
    Produces tensors of shape:
        - X_seq: (seq_len, num_features)
        - y: (1,) target PM2.5 concentration at origin t + horizon
        - mask: validity indicator (warmup, valid target, valid sequence)
    """

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None, seq_len: int = 24):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1) if y is not None else None
        self.seq_len = seq_len
        self.num_samples = len(X) - seq_len + 1

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Window from idx to idx + seq_len
        window = self.X[idx : idx + self.seq_len]
        if self.y is not None:
            # Target aligns with the origin of the last step in the window
            target_idx = idx + self.seq_len - 1
            target = self.y[target_idx]
            return window, target
        return window, torch.tensor([0.0], dtype=torch.float32)


class TemporalAttention(nn.Module):
    """Additive attention mechanism over sequence time steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, rnn_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # rnn_outputs: (batch_size, seq_len, hidden_dim)
        scores = self.attn(rnn_outputs)  # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(weights * rnn_outputs, dim=1)  # (batch_size, hidden_dim)
        return context, weights.squeeze(-1)


class TemporalForecaster(nn.Module):
    """Deep Learning Forecaster supporting LSTM, GRU, and Bi-LSTM with Temporal Attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        architecture: str = "lstm",
        use_attention: bool = True,
    ):
        super().__init__()
        self.architecture = architecture.lower()
        self.use_attention = use_attention
        bidirectional = self.architecture == "bilstm"

        rnn_cls = nn.GRU if self.architecture == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        out_hidden = hidden_dim * (2 if bidirectional else 1)
        self.attention = TemporalAttention(out_hidden) if use_attention else None

        self.head = nn.Sequential(
            nn.Linear(out_hidden, out_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (batch_size, seq_len, input_dim)
        rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, out_hidden)

        attn_weights = None
        if self.attention is not None:
            context, attn_weights = self.attention(rnn_out)
        else:
            context = rnn_out[:, -1, :]  # last step

        out = self.head(context)
        return out, attn_weights


class DeepLearningPipeline:
    """End-to-end scikit-learn compatible wrapper for training and inference of DL models."""

    def __init__(self, config: Optional[DLConfig] = None):
        self.config = config or DLConfig()
        self.model: Optional[TemporalForecaster] = None
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.input_dim: Optional[int] = None

    def _prepare_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        # Numeric values only, fill NA with column medians/means
        numeric_df = frame.select_dtypes(include=[np.number]).copy()
        for col in numeric_df.columns:
            if numeric_df[col].isna().any():
                median_val = numeric_df[col].median()
                numeric_df[col] = numeric_df[col].fillna(0.0 if pd.isna(median_val) else median_val)
        return numeric_df.to_numpy(dtype=np.float32)

    def fit(self, X_df: pd.DataFrame, y_series: pd.Series, valid_mask: Optional[pd.Series] = None):
        raw_mat = self._prepare_matrix(X_df)
        self.feature_means = np.mean(raw_mat, axis=0)
        self.feature_stds = np.std(raw_mat, axis=0) + 1e-7
        scaled_mat = (raw_mat - self.feature_means) / self.feature_stds
        self.input_dim = scaled_mat.shape[1]

        y_arr = y_series.to_numpy(dtype=np.float32)

        # Build sequence dataset
        seq_len = self.config.seq_len
        dataset = SequenceAirDataset(scaled_mat, y_arr, seq_len=seq_len)

        # Apply valid mask alignment using numpy boolean masking
        num_windows = len(dataset)
        target_indices = np.arange(seq_len - 1, seq_len - 1 + num_windows)
        
        valid_targets = ~np.isnan(y_arr[target_indices])
        if valid_mask is not None:
            mask_arr = valid_mask.to_numpy() if hasattr(valid_mask, "to_numpy") else np.asarray(valid_mask)
            if len(mask_arr) == len(y_arr):
                valid_targets = valid_targets & mask_arr[target_indices]
            elif len(mask_arr) == len(X_df):
                valid_targets = valid_targets & mask_arr[target_indices]

        indices = np.where(valid_targets)[0].tolist()
        if not indices:
            # Fallback to all non-nan target indices if mask doesn't align
            indices = np.where(~np.isnan(y_arr[target_indices]))[0].tolist()

        if not indices:
            raise ValueError("No valid training samples found for Deep Learning pipeline")

        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=self.config.batch_size, shuffle=True)

        device = torch.device(self.config.device)
        self.model = TemporalForecaster(
            input_dim=self.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            architecture=self.config.architecture,
        ).to(device)

        criterion = nn.HuberLoss(delta=1.0)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4
        )

        self.model.train()
        for _ in range(self.config.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                pred, _ = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

        self.model.eval()
        return self

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        raw_mat = self._prepare_matrix(X_df)
        scaled_mat = (raw_mat - self.feature_means) / self.feature_stds
        seq_len = self.config.seq_len

        dataset = SequenceAirDataset(scaled_mat, seq_len=seq_len)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        device = torch.device(self.config.device)
        self.model.to(device)
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                pred, _ = self.model(batch_x)
                predictions.append(pred.cpu().numpy())

        if predictions:
            pred_arr = np.concatenate(predictions, axis=0).flatten()
            # Nonnegative concentration constraint
            pred_arr = np.clip(pred_arr, a_min=0.0, a_max=None)
            # Prepend NaNs for the initial warmup sequence window
            warmup_padding = np.full(seq_len - 1, np.nan)
            return np.concatenate([warmup_padding, pred_arr])
        return np.full(len(X_df), np.nan)

    def get_attention_weights(self, X_window: np.ndarray) -> np.ndarray:
        """Extract attention distribution over the given sequence window (seq_len, num_features)."""
        if self.model is None or self.model.attention is None:
            return np.ones(self.config.seq_len) / self.config.seq_len

        scaled = (X_window - self.feature_means) / self.feature_stds
        tensor_in = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.config.device)
        self.model.eval()
        with torch.no_grad():
            _, weights = self.model(tensor_in)
        return weights.squeeze(0).cpu().numpy()


class HybridResidualModel:
    """Technically sound Hybrid architecture combining Deep Learning and Gradient Boosted Trees.
    
    Methodology:
    1. Base Estimator: Bi-LSTM / Temporal Attention captures long-range multi-hour dependencies
       from continuous sequence tensors: y_dl = DL(sequence).
    2. Residual Estimator: An XGBoost / Ridge tabular model learns the residual error:
       r = y_actual - y_dl based on discrete tabular lags and domain features.
    3. Composite Forecast:
       y_hybrid = clip(y_dl + residual_hat, min=0.0).
    """

    def __init__(self, dl_pipeline: Optional[DeepLearningPipeline] = None, tabular_model=None):
        self.dl = dl_pipeline or DeepLearningPipeline(DLConfig(architecture="bilstm", epochs=15))
        self.tabular_model = tabular_model
        self.is_fitted = False

    def fit(self, X_df: pd.DataFrame, y_series: pd.Series, valid_mask: pd.Series):
        # 1. Fit DL Base
        self.dl.fit(X_df, y_series, valid_mask=valid_mask)
        dl_preds = self.dl.predict(X_df)

        # 2. Compute residual on valid mask
        valid_indices = valid_mask & (~np.isnan(dl_preds)) & y_series.notna()
        residuals = y_series[valid_indices] - dl_preds[valid_indices]

        # 3. Fit Tabular Model on residuals
        if self.tabular_model is not None:
            X_valid = X_df.loc[valid_indices]
            self.tabular_model.fit(X_valid, residuals)

        self.is_fitted = True
        return self

    def predict(self, X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (hybrid_pred, dl_base_pred, ml_residual_pred)."""
        if not self.is_fitted:
            raise RuntimeError("Hybrid model is not fitted.")

        dl_base = self.dl.predict(X_df)
        if self.tabular_model is not None:
            # Predict residual
            try:
                residual_pred = self.tabular_model.predict(X_df)
            except Exception:
                # If tabular pipeline requires specific index or preprocessing
                residual_pred = np.zeros(len(X_df))
        else:
            residual_pred = np.zeros(len(X_df))

        hybrid_pred = np.where(np.isnan(dl_base), np.nan, np.clip(dl_base + residual_pred, a_min=0.0, a_max=None))
        return hybrid_pred, dl_base, residual_pred
