"""Deep Learning architectures, sequence generators, and hybrid models for Phase 2."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    # Use the same CPU thread policy for direct CLI inference as for training
    # and the dashboard launcher. Multiple workers can crash mixed numerical
    # runtimes on macOS, including while loading serialized tensor storages.
    torch.set_num_threads(1)
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for deep learning and hybrid models. "
        "Install with `pip install -e '.[dl]'` or `pip install torch`."
    ) from exc


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
    random_seed: int = 42
    use_attention: bool = True
    target_mode: str = "level"  # "delta" learns a correction to current PM2.5
    quantiles: Optional[Tuple[float, ...]] = None  # None for standard point forecast, or (0.1, 0.5, 0.9)

    def __post_init__(self):
        for name in ("seq_len", "hidden_dim", "num_layers", "batch_size", "epochs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if self.architecture not in {"lstm", "gru", "bilstm"}:
            raise ValueError("architecture must be lstm, gru, or bilstm")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(self.use_attention, bool):
            raise ValueError("use_attention must be boolean")
        if self.target_mode not in {"level", "delta"}:
            raise ValueError("target_mode must be level or delta")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool) or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        if self.quantiles is not None:
            if not isinstance(self.quantiles, (tuple, list)) or not self.quantiles:
                raise ValueError("quantiles must be a non-empty sequence of floats strictly between 0 and 1")
            for q in self.quantiles:
                if not isinstance(q, (int, float)) or not 0 < q < 1:
                    raise ValueError("All quantiles must be strictly between 0 and 1")
            self.quantiles = tuple(float(q) for q in self.quantiles)


class SequenceAirDataset(Dataset):
    """3D sliding sequence window dataset for multivariate temporal series.
    
    Produces tensors of shape:
        - X_seq: (seq_len, num_features)
        - y: (1,) target PM2.5 concentration at origin t + horizon
    """

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None, seq_len: int = 24):
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 1:
            raise ValueError("seq_len must be a positive integer")
        if np.ndim(X) != 2 or (y is not None and (np.ndim(y) != 1 or len(y) != len(X))):
            raise ValueError("Expected a 2D feature matrix and an aligned 1D target")
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


class PinballLoss(nn.Module):
    """Pinball (quantile) loss for probabilistic multi-quantile regression.

    L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """

    def __init__(self, quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)):
        super().__init__()
        if not quantiles or any(not 0 < q < 1 for q in quantiles):
            raise ValueError("All quantiles must be strictly between 0 and 1")
        self.quantiles = tuple(float(q) for q in quantiles)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)
        if targets.shape[-1] == 1 and preds.shape[-1] == len(self.quantiles):
            targets = targets.expand_as(preds)
        errors = targets - preds
        losses = []
        for i, q in enumerate(self.quantiles):
            err = errors[..., i]
            loss_q = torch.maximum((q - 1.0) * err, q * err)
            losses.append(loss_q)
        return torch.stack(losses, dim=-1).mean()


def empirical_coverage(actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Calculate the fraction of observed values that fall inside [lower, upper]."""
    actuals = np.asarray(actuals, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = np.isfinite(actuals) & np.isfinite(lower) & np.isfinite(upper)
    if not valid.any():
        return float("nan")
    covered = (actuals[valid] >= lower[valid]) & (actuals[valid] <= upper[valid])
    return float(np.mean(covered))


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
        output_dim: int = 1,
    ):
        super().__init__()
        self.architecture = architecture.lower()
        if self.architecture not in {"lstm", "gru", "bilstm"}:
            raise ValueError("architecture must be lstm, gru, or bilstm")
        if not isinstance(output_dim, int) or isinstance(output_dim, bool) or output_dim < 1:
            raise ValueError("output_dim must be a positive integer")
        self.use_attention = use_attention
        self.output_dim = output_dim
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
            nn.Linear(out_hidden // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (batch_size, seq_len, input_dim)
        rnn_out, hidden = self.rnn(x)  # (batch_size, seq_len, out_hidden)

        attn_weights = None
        if self.attention is not None:
            context, attn_weights = self.attention(rnn_out)
        else:
            state = hidden[0] if isinstance(hidden, tuple) else hidden
            context = (torch.cat((state[-2], state[-1]), dim=1)
                       if self.architecture == "bilstm" else state[-1])

        out = self.head(context)
        return out, attn_weights


def validate_hourly_frame(frame):
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
        raise ValueError("Sequences require a nonempty DatetimeIndex")
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    if not frame.columns.is_unique or not frame.index.equals(expected):
        raise ValueError("Sequences require unique columns and a sorted, unique, regular hourly index")


def aligned_mask(X, y, valid_mask):
    validate_hourly_frame(X)
    if not isinstance(y, pd.Series) or not y.index.equals(X.index):
        raise ValueError("Targets must have exactly the same index as features")
    if valid_mask is None:
        mask = np.ones(len(X), dtype=bool)
    else:
        if isinstance(valid_mask, pd.Series) and not valid_mask.index.equals(X.index):
            raise ValueError("Training mask index must match features")
        mask = np.asarray(valid_mask)
        if mask.shape != (len(X),) or mask.dtype != np.bool_:
            raise ValueError("Training mask must be an aligned boolean vector")
    return mask & np.isfinite(y.to_numpy(dtype=float))


class DeepLearningPipeline:
    """Hourly sequence regressor with preprocessing fitted on training windows only.

    Inference never learns fill values or category levels from the prediction batch.
    Numeric columns and categorical wind direction are retained in a frozen schema.
    """

    def __init__(self, config: Optional[DLConfig] = None):
        self.config = config or DLConfig()
        self.model = None

    def _prepare_matrix(self, frame):
        missing = set(self.columns_) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing fitted feature columns: {sorted(missing)}")
        numeric = frame[self.numeric_columns_].to_numpy(dtype=np.float32)
        if np.isinf(numeric).any():
            raise ValueError("Feature values must be finite or missing")
        numeric = np.where(np.isnan(numeric), self.feature_medians_, numeric)
        parts = [numeric]
        for column, levels in self.category_levels_.items():
            values = frame[column].fillna("MISSING").astype(str).to_numpy()
            parts.append((values[:, None] == np.asarray(levels)[None, :]).astype(np.float32))
        return np.concatenate(parts, axis=1)

    def _target_baseline(self, frame):
        if self.config.target_mode == "level":
            return np.zeros(len(frame), dtype=np.float32)
        if "pm25_lag_0" not in self.numeric_columns_:
            raise ValueError("Delta targets require the current PM2.5 feature pm25_lag_0")
        fill = self.feature_medians_[self.numeric_columns_.index("pm25_lag_0")]
        return frame["pm25_lag_0"].fillna(fill).to_numpy(dtype=np.float32)

    def fit(self, X_df, y_series, valid_mask=None):
        mask = aligned_mask(X_df, y_series, valid_mask)
        seq_len = self.config.seq_len
        mask[:seq_len - 1] = False
        endpoints = np.flatnonzero(mask)
        if not len(endpoints):
            raise ValueError("No valid training sequences; the supplied mask is never relaxed")
        # Union of the actual training windows, including their historical context.
        changes = np.zeros(len(X_df) + 1, dtype=int)
        np.add.at(changes, endpoints - seq_len + 1, 1)
        np.add.at(changes, endpoints + 1, -1)
        input_mask = np.cumsum(changes[:-1]) > 0
        self.columns_ = list(X_df.columns)
        self.numeric_columns_ = list(X_df.select_dtypes(include="number").columns)
        train_frame = X_df.loc[input_mask]
        self.feature_medians_ = (train_frame[self.numeric_columns_].median()
                                .fillna(0.0).to_numpy(dtype=np.float32))
        self.category_levels_ = {
            column: sorted(train_frame[column].fillna("MISSING").astype(str).unique())
            for column in self.columns_ if column not in self.numeric_columns_
        }
        raw = self._prepare_matrix(X_df)
        self.feature_means = raw[input_mask].mean(axis=0)
        std = raw[input_mask].std(axis=0)
        self.feature_stds = np.where(std > 1e-7, std, 1.0)
        scaled = (raw - self.feature_means) / self.feature_stds
        self.input_dim = scaled.shape[1]
        self.preprocessing_fit_start_ = str(X_df.index[input_mask].min())
        self.preprocessing_fit_end_ = str(X_df.index[input_mask].max())
        self.training_origins_ = X_df.index[endpoints]
        self.training_rows_ = len(endpoints)
        # Standardize targets so Huber's delta and gradients have a meaningful scale.
        targets = y_series.to_numpy(dtype=np.float32) - self._target_baseline(X_df)
        self.target_mean_ = float(targets[endpoints].mean())
        self.target_std_ = max(float(targets[endpoints].std()), 1e-7)
        normalized_y = (targets - self.target_mean_) / self.target_std_
        dataset = SequenceAirDataset(scaled, normalized_y, seq_len)
        subset = torch.utils.data.Subset(dataset, (endpoints - seq_len + 1).tolist())
        generator = torch.Generator().manual_seed(self.config.random_seed)
        loader = DataLoader(subset, batch_size=self.config.batch_size, shuffle=True, generator=generator)
        device = torch.device(self.config.device)
        torch.manual_seed(self.config.random_seed)
        output_dim = len(self.config.quantiles) if self.config.quantiles else 1
        self.model = TemporalForecaster(
            input_dim=self.input_dim, hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers, dropout=self.config.dropout,
            architecture=self.config.architecture,
            use_attention=self.config.use_attention,
            output_dim=output_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
        criterion = PinballLoss(self.config.quantiles) if self.config.quantiles else nn.HuberLoss(delta=1.0)
        self.loss_history_ = []
        for _ in range(self.config.epochs):
            self.model.train()
            total = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                prediction, _ = self.model(batch_x)
                loss = criterion(prediction, batch_y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite neural training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total += loss.item() * len(batch_x)
            self.loss_history_.append(total / len(subset))
        self.model.eval()
        return self

    def predict(self, X_df):
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        validate_hourly_frame(X_df)
        raw = self._prepare_matrix(X_df)
        scaled = (raw - self.feature_means) / self.feature_stds
        dataset = SequenceAirDataset(scaled, seq_len=self.config.seq_len)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        self.model.to(self.config.device).eval()
        predictions = []
        with torch.no_grad():
            for batch_x, _ in loader:
                prediction, _ = self.model(batch_x.to(self.config.device))
                predictions.append(prediction.cpu().numpy())
        if self.config.quantiles:
            output = np.full((len(X_df), len(self.config.quantiles)), np.nan)
            if predictions:
                stacked = np.vstack(predictions)
                baseline = self._target_baseline(X_df)[self.config.seq_len - 1:, None]
                values = stacked * self.target_std_ + self.target_mean_ + baseline
                if not np.isfinite(values).all():
                    raise RuntimeError("Nonfinite neural prediction")
                output[self.config.seq_len - 1:, :] = np.maximum(values, 0.0)
            return output
        else:
            output = np.full(len(X_df), np.nan)
            if predictions:
                values = (np.concatenate([p.ravel() for p in predictions]) * self.target_std_ + self.target_mean_
                          + self._target_baseline(X_df)[self.config.seq_len - 1:])
                if not np.isfinite(values).all():
                    raise RuntimeError("Nonfinite neural prediction")
                output[self.config.seq_len - 1:] = np.maximum(values, 0.0)
            return output

    def predict_quantiles(self, X_df):
        """Predict multi-quantile intervals as a DataFrame with one column per quantile."""
        if not self.config.quantiles:
            raise ValueError("Quantiles are not configured for this pipeline")
        preds = self.predict(X_df)
        columns = [f"q{int(round(q * 100)):02d}" for q in self.config.quantiles]
        return pd.DataFrame(preds, index=X_df.index, columns=columns)

    def get_attention_weights(self, X_window):
        """Explain one complete window using the same preprocessing as prediction."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        if self.model.attention is None:
            raise ValueError("Attention is disabled for this model")
        if not isinstance(X_window, pd.DataFrame) or len(X_window) != self.config.seq_len:
            raise ValueError("Attention requires one DataFrame window with seq_len rows")
        validate_hourly_frame(X_window)
        matrix = (self._prepare_matrix(X_window) - self.feature_means) / self.feature_stds
        tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(self.config.device)
        self.model.to(self.config.device).eval()
        with torch.no_grad():
            _, weights = self.model(tensor)
        return weights.squeeze(0).cpu().numpy()


class HybridResidualModel:
    """Bi-LSTM plus corrections learned on a later, disjoint calibration block.

    The base is frozen before calibration begins. Its labels must be observable
    at the first calibration origin. It is NOT refitted on calibration labels,
    including in the saved deployment model, so the residual target stays honest.
    """

    def __init__(self, dl_pipeline=None, tabular_model=None, calibration_fraction=0.2,
                 correction_weight=1.0):
        if not 0 < calibration_fraction < 1:
            raise ValueError("calibration_fraction must be between zero and one")
        self.dl = dl_pipeline or DeepLearningPipeline(DLConfig(architecture="bilstm"))
        self.tabular_model = tabular_model
        self.calibration_fraction = calibration_fraction
        if not np.isfinite(correction_weight) or not 0 <= correction_weight <= 1:
            raise ValueError("correction_weight must be finite and in [0, 1]")
        self.correction_weight = correction_weight
        self.is_fitted = False

    def fit(self, X_df, y_series, valid_mask, target_time=None):
        self.is_fitted = False
        if self.tabular_model is None:
            raise ValueError("A residual estimator is required")
        valid = aligned_mask(X_df, y_series, valid_mask)
        valid[:self.dl.config.seq_len - 1] = False
        positions = np.flatnonzero(valid)
        calibration_rows = max(1, int(np.ceil(len(positions) * self.calibration_fraction)))
        if len(positions) - calibration_rows < 2:
            raise ValueError("Insufficient training rows for separate base and calibration blocks")
        calibration_start = X_df.index[positions[-calibration_rows]]
        if target_time is None:
            target_time = pd.Series(X_df.index + pd.Timedelta(hours=1), index=X_df.index)
        if not isinstance(target_time, pd.Series) or not target_time.index.equals(X_df.index):
            raise ValueError("Target times must have exactly the same index as features")
        if target_time.isna().any() or (target_time <= X_df.index).any():
            raise ValueError("Target times must follow their forecast origins")
        base_mask = valid & (X_df.index < calibration_start) & (target_time <= calibration_start).to_numpy()
        calibration_mask = valid & (X_df.index >= calibration_start)
        if base_mask.sum() < 2:
            raise ValueError("Insufficient base training labels before calibration")
        self.dl.fit(X_df, y_series, valid_mask=base_mask)
        # Predict only through calibration end; keep the hourly input grid intact.
        stop = positions[-1] + 1
        base = self.dl.predict(X_df.iloc[:stop])
        calibration_positions = np.flatnonzero(calibration_mask)
        residuals = y_series.iloc[calibration_positions] - base[calibration_positions]
        self.tabular_model.fit(X_df.iloc[calibration_positions], residuals)
        self.fit_audit_ = {
            "base_training_rows": int(base_mask.sum()),
            "base_last_origin": str(X_df.index[base_mask].max()),
            "base_latest_target": str(target_time.loc[base_mask].max()),
            "calibration_start": str(calibration_start),
            "calibration_end": str(X_df.index[calibration_mask].max()),
            "calibration_rows": int(calibration_mask.sum()),
            "calibration_latest_target": str(target_time.loc[calibration_mask].max()),
            "preprocessing_fit_end": self.dl.preprocessing_fit_end_,
            "calibration_fraction": self.calibration_fraction,
            "residual_training": "chronological out-of-sample calibration; base frozen",
        }
        self.is_fitted = True
        return self

    def predict(self, X_df):
        """Return (hybrid concentration, base concentration, signed correction)."""
        if not self.is_fitted:
            raise RuntimeError("Hybrid model is not fitted")
        base = self.dl.predict(X_df)
        # A failed correction model must surface as an error, not an apparent success.
        residual = np.asarray(self.tabular_model.predict(X_df), dtype=float)
        if residual.shape != base.shape or not np.isfinite(residual).all():
            raise RuntimeError("Residual model returned invalid predictions")
        residual = residual * getattr(self, "correction_weight", 1.0)
        return np.maximum(base + residual, 0.0), base, residual
