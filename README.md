# 🌫️ Air Prediction (CSAI Project 19)

[![CI](https://github.com/pratham-malhotr/air_prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/pratham-malhotr/air_prediction/actions)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular, reproducible machine learning pipeline for hourly particulate matter ($PM_{2.5}$) forecasting using the UCI Beijing PM2.5 Data Set (UCI 381).

---

## 📌 Project Overview

This repository provides an end-to-end framework for time-series air quality prediction:
- **Data Integrity & Provenance:** Automated dataset downloading with SHA-256 checksum verification and provenance metadata tracking.
- **Leak-Free Time-Series Splits:** Strict calendar backtesting folds with expanding training horizons and label-availability boundaries.
- **Feature Engineering:** Calendar cyclical transformations, lag features, rolling statistics, and meteorological interaction signals.
- **Modeling & Ablation:** Ridge Regression baseline and gradient-boosted trees (XGBoost), accompanied by feature ablation analysis.
- **Synthetic Integration Demo:** Built-in self-contained end-to-end demo using synthetic data for testing and validation.

---

## 📂 Repository Structure

```text
air_prediction/
├── configs/                  # Experiment configuration files (TOML)
│   ├── demo.toml             # Configuration for synthetic integration demo
│   └── phase1.toml           # Phase 1 benchmark configuration
├── data/
│   └── raw/                  # Downloaded raw dataset (.gitkeep)
├── artifacts/                # Model checkpoints, metrics, and plots (.gitkeep)
├── src/
│   └── air_prediction/       # Core package source code
│       ├── cli.py            # CLI entry points
│       ├── config.py         # TOML config loader & dataclasses
│       ├── data.py           # Ingestion, validation, and SHA-256 checks
│       ├── demo.py           # Synthetic data generation for testing
│       ├── eda.py            # Exploratory data analysis routines
│       ├── experiment.py     # End-to-end training and evaluation pipeline
│       ├── features.py       # Feature transformations and rolling metrics
│       ├── metrics.py        # Evaluation metrics (RMSE, MAE, R², etc.)
│       ├── models.py         # Ridge and XGBoost model interfaces
│       ├── predict.py        # Inference pipeline on saved models
│       ├── results.py        # Metrics persistence & formatting
│       └── splits.py         # Time-series cross-validation folds
├── tests/                    # Unit and integration test suite
│   ├── conftest.py
│   ├── test_data_and_integration.py
│   └── test_forecasting.py
├── pyproject.toml            # Packaging and build specifications
└── requirements.txt          # Python dependencies
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/pratham-malhotr/air_prediction.git
cd air_prediction

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package and dependencies
pip install -r requirements.txt
pip install -e .
```

For development and test dependencies:
```bash
pip install -e ".[dev]"
```

---

## 💻 CLI Usage

The package exposes a command-line interface via `air-prediction`:

### 1. Download Dataset with Provenance Tracking
Downloads the UCI 381 dataset into `data/raw/` and records SHA-256 verification metadata:
```bash
air-prediction download
```

### 2. Exploratory Data Analysis (EDA)
Generates distribution plots, missing value summaries, and correlation matrices:
```bash
air-prediction eda --data data/raw/PRSA_data_2010.1.1-2014.12.31.csv --config configs/phase1.toml --output artifacts/phase1/eda
```

### 3. Run Experiments (Train & Evaluate)
Trains Ridge and XGBoost models across time-series folds and logs performance metrics:
```bash
air-prediction train --data data/raw/PRSA_data_2010.1.1-2014.12.31.csv --config configs/phase1.toml --models xgboost ridge
```

To run feature ablation studies:
```bash
air-prediction train --ablations
```

### 4. Run the Synthetic Integration Demo
Executes a quick, isolated end-to-end training and evaluation run using synthetic data:
```bash
air-prediction demo --output artifacts/demo
```

### 5. Generate Predictions
Generates forecasts using a trained model artifact:
```bash
air-prediction predict --model artifacts/phase1/model_xgboost.joblib --data data/raw/PRSA_data_2010.1.1-2014.12.31.csv
```

---

## 🧪 Testing

Run unit and integration tests with `pytest`:

```bash
pytest
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
