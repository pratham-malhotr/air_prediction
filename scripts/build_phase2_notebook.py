"""Build a self-contained Phase 2 Colab notebook with deep learning models, training, loss curves, and evaluation."""

import argparse
import base64
import contextlib
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
import textwrap
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
PHASE2_TESTS = (
    'test_forecasting.py', 'test_data_and_integration.py',
    'test_phase2.py', 'test_phase2_research.py',
)


def source_bundle():
    paths = [ROOT / name for name in (
        'pyproject.toml', 'requirements.txt', 'requirements-phase1.txt', 'requirements-phase2.txt', 'LICENSE',
        'configs/phase1.toml', 'configs/phase2_research.toml', 'configs/demo.toml',
        'data/README.md', 'docs/methodology.md', 'docs/phase2_research.md', 'tests/conftest.py',
        *(f'tests/{name}' for name in PHASE2_TESTS if (ROOT / f'tests/{name}').exists()),
    )]
    paths.extend(sorted((ROOT / 'src/air_prediction').glob('*.py')))
    contents = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in paths}
    stream = io.BytesIO()
    with ZipFile(stream, 'w', compression=ZIP_DEFLATED) as archive:
        for name, payload in sorted(contents.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, payload)
    return stream.getvalue(), {name: sha256(payload).hexdigest() for name, payload in contents.items()}


def build():
    payload, hashes = source_bundle()
    cells = []

    def md(value):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(value).strip().splitlines(keepends=True)})

    def code(value, images=()):
        cells.append({'cell_type': 'code', 'metadata': {'aircast_images': list(images)} if images else {},
                      'source': (textwrap.dedent(value).strip() + '\n').splitlines(keepends=True),
                      'execution_count': None, 'outputs': []})

    md(r'''
    # Aircast Phase 2 — Deep Learning and Calibrated Hybrid PM2.5 Forecasting

    Run every cell in order in Google Colab or your local Jupyter environment.

    ### Overview of Phase 2 Deep Learning Architecture:
    * **Recurrent Sequence Models**: Evaluates **LSTM** and **Bi-LSTM** architectures for temporal time-series representation.
    * **Temporal Attention**: Additive attention mechanism across past sliding sequence windows ($L=24$ hours) to identify critical weather transitions.
    * **Delta vs. Level Formulation**: Models target level $y_{t+1}$ directly or learns a residual delta correction $\Delta y = y_{t+1} - y_t$ against the latest reading.
    * **Calibrated Hybrid Forecaster**: Pairs a frozen classical base model (XGBoost) with a deep neural residual correction block.
    * **Rigorous Validation Protocol**: Hyperparameter selection on 2013 validation folds, paired attention/directionality ablations across 3 predefined random seeds (42, 137, 2026), and backtests on 2014 holdout.
    ''')

    code('''
    import base64, io, json, os, sys, zipfile
    from hashlib import sha256
    from pathlib import Path
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    if (Path.cwd() / 'src/air_prediction').is_dir():
        ROOT = Path.cwd()
    elif (Path.cwd().parent / 'src/air_prediction').is_dir():
        ROOT = Path.cwd().parent
    else:
        ROOT = Path('/content/air_prediction') if IN_COLAB else Path.cwd() / 'air_prediction'
    ''')

    boot = f'''
SOURCE_BUNDLE = {base64.b64encode(payload).decode()!r}
SOURCE_BUNDLE_SHA256 = {sha256(payload).hexdigest()!r}
SOURCE_FILES = {hashes!r}
snapshot = base64.b64decode(SOURCE_BUNDLE)
assert sha256(snapshot).hexdigest() == SOURCE_BUNDLE_SHA256
ROOT.mkdir(parents=True, exist_ok=True)
if not (ROOT / 'src/air_prediction').is_dir():
    with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
        for name in SOURCE_FILES.keys():
            if name in archive.namelist():
                content = archive.read(name)
                target = (ROOT / name).resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
for name, expected in SOURCE_FILES.items():
    target = ROOT / name
    if target.exists() and name.startswith(('src/', 'configs/')):
        assert sha256(target.read_bytes()).hexdigest() == expected, 'Source differs: ' + name
os.chdir(ROOT)
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
sys.path.insert(0, str(ROOT / 'src'))
print('Aircast Phase 2 environment initialized.')
print('Source snapshot verified:', SOURCE_BUNDLE_SHA256[:16] + '...')
print('Runtime:', sys.version.split()[0], '| Colab:', IN_COLAB)
'''
    cells[-1]['source'].extend(boot.splitlines(keepends=True))

    md('''
    ## 1. Environment & Deep Learning Dependencies

    Installs PyTorch, XGBoost, scikit-learn, pandas, numpy, and matplotlib. Detects hardware acceleration (GPU/MPS/CPU).
    ''')

    code('''
    import subprocess
    INSTALL_DEPENDENCIES = IN_COLAB
    if INSTALL_DEPENDENCIES:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-phase2.txt'], check=True)
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    try:
        from IPython.display import display, Image as NotebookImage
    except ImportError:
        def display(val): print(val)
        NotebookImage = None

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f'PyTorch Version : {torch.__version__}')
    print(f'Computation Dev : {device}')
    print(f'CUDA Available  : {torch.cuda.is_available()}')
    ''')

    md('''
    ## 2. Dataset Loading & Integrity Audit

    Loads the historical Beijing hourly PM2.5 dataset (UCI Machine Learning Repository). If running in a fresh Colab environment, it automatically downloads and verifies the exact 43,824-hour archive against its cryptographic SHA-256 signature.
    ''')

    code('''
    from air_prediction.config import load_config
    from air_prediction.data import FILENAME, download_dataset, load_dataset
    from air_prediction.phase2_research import load_research_settings

    CONFIG = load_config(ROOT / 'configs/phase1.toml')
    SETTINGS = load_research_settings(ROOT / 'configs/phase2_research.toml')
    DATA = ROOT / 'data/raw' / FILENAME
    if not DATA.exists():
        download_dataset(DATA)
    dataset = load_dataset(DATA)

    print('Dataset Audit:')
    print(f"  Rows total       : {dataset.audit['hourly_rows']:,}")
    print(f"  Missing PM2.5    : {dataset.audit['missing_by_column']['pm2.5']} ({dataset.audit['missing_by_column']['pm2.5']/dataset.audit['hourly_rows']*100:.2f}%)")
    print(f"  Date range       : {dataset.audit['start']} to {dataset.audit['end']}")
    print(f"  SHA-256 checksum : {dataset.audit['sha256']}")
    ''')

    md(r'''
    ## 3. Sliding Sequence Windows for Deep Learning

    Standard feedforward models use single tabular feature vectors. In recurrent deep learning, we map the multivariate hourly series into 3D tensors:
    - Input sequence $X \in \mathbb{R}^{B \times L \times D}$, where $B$ is batch size, $L=24$ hours of historical context, and $D$ is feature dimensionality (meteorology + lag indicators + cyclics).
    - Target $y \in \mathbb{R}^{B \times 1}$, representing the concentration at origin $t + 1$.
    ''')

    code('''
    from air_prediction.deep_learning import DLConfig, DeepLearningPipeline, SequenceAirDataset, TemporalForecaster
    from air_prediction.features import build_features

    features_df = build_features(dataset.frame)
    dl_pipeline = DeepLearningPipeline(DLConfig(seq_len=24, hidden_dim=32, num_layers=1, architecture='bilstm'))
    train_mask = (features_df.index >= pd.Timestamp(CONFIG.train_start)) & (features_df.index < pd.Timestamp(CONFIG.validation_start))
    dl_pipeline.fit(features_df, features_df['target_1h'], train_mask=train_mask)

    print(f"Fitted feature columns : {len(dl_pipeline.columns_)} features")
    print(f"Numeric features       : {len(dl_pipeline.numeric_columns_)}")
    print(f"Categorical features   : {list(dl_pipeline.category_levels_.keys())}")
    ''')

    md(r'''
    ## 4. Model Architecture & Attention Mechanism

    We construct `TemporalForecaster`, which supports LSTM, GRU, and Bidirectional LSTM backbones.
    When `use_attention=True`, an additive attention layer computes a dynamic weighting across all $L=24$ sequence time steps:
    $$\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^L \exp(e_k)}, \quad e_t = v^T \tanh(W h_t)$$
    ''')

    code('''
    input_dim = len(dl_pipeline.columns_)
    lstm_model = TemporalForecaster(input_dim=input_dim, hidden_dim=32, num_layers=1, architecture='lstm', use_attention=True)
    bilstm_model = TemporalForecaster(input_dim=input_dim, hidden_dim=32, num_layers=1, architecture='bilstm', use_attention=False)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Architecture 1: LSTM with Temporal Attention")
    print(lstm_model)
    print(f"Total Trainable Parameters: {count_parameters(lstm_model):,}\\n")

    print("Architecture 2: Bi-LSTM (Bidirectional)")
    print(bilstm_model)
    print(f"Total Trainable Parameters: {count_parameters(bilstm_model):,}")
    ''')

    md('''
    ## 5. Live Deep Learning Training & Epoch Loss Curves

    Here we train a representative neural sequence model using the PyTorch Adam optimizer, MSE loss, and delta target mode. We log train and validation loss at each epoch to generate learning curves.
    ''')

    code('''
    from torch.utils.data import DataLoader

    # Prepare train and validation subsets
    val_mask = (features_df.index >= pd.Timestamp(CONFIG.validation_start)) & (features_df.index < pd.Timestamp(CONFIG.holdout_start))
    X_train_mat = dl_pipeline._prepare_matrix(features_df[train_mask])
    y_train = features_df.loc[train_mask, 'target_1h'].to_numpy(dtype=np.float32)

    X_val_mat = dl_pipeline._prepare_matrix(features_df[val_mask])
    y_val = features_df.loc[val_mask, 'target_1h'].to_numpy(dtype=np.float32)

    train_dataset = SequenceAirDataset(X_train_mat, y_train, seq_len=24)
    val_dataset = SequenceAirDataset(X_val_mat, y_val, seq_len=24)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    train_forecaster = TemporalForecaster(input_dim=X_train_mat.shape[1], hidden_dim=32, architecture='bilstm', use_attention=True)
    optimizer = torch.optim.Adam(train_forecaster.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 6
    train_losses, val_losses = [], []
    print(f"Training Bi-LSTM with Temporal Attention for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_forecaster.train()
        total_loss, total_count = 0.0, 0
        for X_batch, y_batch in train_loader:
            mask = torch.isfinite(y_batch)
            if not mask.any():
                continue
            optimizer.zero_grad()
            pred, _ = train_forecaster(X_batch)
            loss = criterion(pred[mask], y_batch[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            total_count += mask.sum().item()
        
        train_rmse = np.sqrt(total_loss / max(1, total_count))
        train_losses.append(train_rmse)

        # Validation phase
        train_forecaster.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                vmask = torch.isfinite(y_b)
                if not vmask.any():
                    continue
                p, _ = train_forecaster(X_b)
                vloss = criterion(p[vmask], y_b[vmask])
                val_loss += vloss.item() * vmask.sum().item()
                val_count += vmask.sum().item()
        val_rmse = np.sqrt(val_loss / max(1, val_count))
        val_losses.append(val_rmse)
        print(f"Epoch {epoch}/{epochs} - Train RMSE: {train_rmse:.3f} µg/m³ | Val RMSE: {val_rmse:.3f} µg/m³")

    # Plot Loss Curves
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), train_losses, 'o-', label='Training Loss (RMSE)', color='#2563eb')
    plt.plot(range(1, epochs + 1), val_losses, 's--', label='Validation Loss (RMSE)', color='#dc2626')
    plt.title('Deep Learning Training Progression: Bi-LSTM with Attention', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (µg/m³)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    ''')

    md('''
    ## 6. Attention Weights & Model Interpretability

    Deep learning models are often viewed as black boxes. By examining the attention weights $\\alpha_t$, we can see which past hours the model focused on when issuing its forecast.
    ''')

    code('''
    train_forecaster.eval()
    sample_seq = X_val_mat[:24]
    sample_tensor = torch.tensor(sample_seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        _, attn_weights = train_forecaster(sample_tensor)

    if attn_weights is not None:
        weights = attn_weights.squeeze(0).numpy()
        hours_back = [f"t-{23-i}h" for i in range(24)]

        plt.figure(figsize=(10, 3.5))
        bars = plt.bar(hours_back, weights, color='#3b82f6', edgecolor='#1d4ed8', alpha=0.85)
        plt.title('Learned Temporal Attention Across 24-Hour Sequence Window', fontsize=12, fontweight='bold')
        plt.xlabel('Historical Observation Horizon')
        plt.ylabel('Normalized Attention Weight')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
        print(f"Peak Attention Horizon: {hours_back[int(np.argmax(weights))]} (Weight: {weights.max():.4f})")
    ''')

    md('''
    ## 7. Comparative Holdout Evaluation: Deep Learning vs. Baselines

    We evaluate all 7 models across the complete, identical **8,531 hours in the 2014 holdout period**:
    - **Persistence baseline**
    - **Daily seasonal naive**
    - **Ridge regression** (linear reference)
    - **XGBoost** (classical champion)
    - **LSTM** (unidirectional neural model)
    - **Bi-LSTM** (bidirectional neural model)
    - **Calibrated Hybrid** (XGBoost + deep residual correction)
    ''')

    code('''
    RUN = ROOT / 'artifacts/phase2_research'
    if (RUN / 'metrics.csv').exists():
        metrics_df = pd.read_csv(RUN / 'metrics.csv')
        holdout_metrics = metrics_df[metrics_df['phase'] == 'holdout'].sort_values('rmse').reset_index(drop=True)
        print("2014 Holdout Evaluation Summary (Sorted by RMSE):")
        display(holdout_metrics[['model', 'n', 'mae', 'rmse', 'r2', 'bias']])

        # Bar chart comparison
        plt.figure(figsize=(10, 4.5))
        models = holdout_metrics['model'].str.upper()
        rmse_vals = holdout_metrics['rmse']
        mae_vals = holdout_metrics['mae']
        x = np.arange(len(models))
        width = 0.35

        plt.bar(x - width/2, rmse_vals, width, label='RMSE (µg/m³)', color='#1e3a8a')
        plt.bar(x + width/2, mae_vals, width, label='MAE (µg/m³)', color='#60a5fa')
        plt.axhline(rmse_vals.iloc[0], color='#ef4444', linestyle='--', alpha=0.7, label=f'Best RMSE ({rmse_vals.iloc[0]:.2f})')
        plt.title('Holdout Comparison: Classical vs. Deep Learning Models (2014)', fontsize=12, fontweight='bold')
        plt.xticks(x, models, rotation=25, ha='right')
        plt.ylabel('Error (µg/m³)')
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()
    ''')

    md(r'''
    ## 8. Pollution Regime Breakdown & Extreme Smog Events

    Error behavior differs markedly across pollution regimes. Models that excel during low/moderate pollution often struggle during extreme inversion spikes ($\ge$ 90th percentile, 218 µg/m³).
    ''')

    code('''
    if (RUN / 'errors_by_pollution_regime.csv').exists():
        regimes_df = pd.read_csv(RUN / 'errors_by_pollution_regime.csv')
        above_p90 = regimes_df[regimes_df['pollution_regime'] == 'above_training_p90'].sort_values('rmse')
        print("High Pollution Regime Performance (PM2.5 >= 218 µg/m³):")
        display(above_p90[['model', 'n', 'mae', 'rmse', 'r2', 'bias']])

        # Visualization
        plt.figure(figsize=(9, 4))
        plt.barh(above_p90['model'].str.upper(), above_p90['rmse'], color='#b91c1c')
        plt.title('RMSE during Hazardous Smog Hours (>= 90th Percentile: 218 µg/m³)', fontsize=12, fontweight='bold')
        plt.xlabel('RMSE (µg/m³)')
        plt.grid(axis='x', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
    ''')

    md('''
    ## 9. Ablation Studies & Seed Stability Analysis

    We review the ablation results testing:
    1. **Attention vs. No-Attention**
    2. **Unidirectional vs. Bidirectional LSTM**
    3. **Random seed stability across seeds 42, 137, and 2026**
    ''')

    code('''
    for fname in ('ablation_metrics.csv', 'seed_stability.csv'):
        if (RUN / fname).exists():
            print(f"\\n--- {fname} ---")
            display(pd.read_csv(RUN / fname))
    ''')

    md('''
    ## 10. Saved Research Artifacts & Report Figures

    Inspect the final published high-resolution scientific evaluation figures generated by the project pipeline.
    ''')

    code('''
    FIGURES = ROOT / 'docs/results/phase2_research'
    for fig_name in ('model_comparison.png', 'attention_ablation.png', 'pollution_regimes.png'):
        fig_path = FIGURES / fig_name
        if fig_path.exists() and NotebookImage is not None:
            print(f"Displaying: {fig_name}")
            display(NotebookImage(filename=str(fig_path)))
    ''')

    md('''
    ---
    **Citation & Attribution:**
    - Dataset: Chen, S. (2015), *Beijing PM2.5*, UCI Machine Learning Repository, DOI [10.24432/C5JS49](https://doi.org/10.24432/C5JS49), CC BY 4.0.
    - Codebase & Full Documentation: [Aircast Project Repository](https://github.com/pratham-malhotr/air_prediction)
    ''')

    for index, cell in enumerate(cells):
        cell['id'] = f'phase2-{index:02d}'

    return {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': sys.version.split()[0]},
                         'aircast': {'source_bundle_sha256': sha256(payload).hexdigest(), 'source_files': hashes}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'Air_Prediction_Phase2_Colab.ipynb')
    args = parser.parse_args()
    notebook = build()
    args.output.write_text(json.dumps(notebook, indent=1) + '\n')
    print('Generated Phase 2 Colab notebook:', args.output)


if __name__ == '__main__':
    main()
