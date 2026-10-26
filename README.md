# Air Quality Prediction

CSAI Project 19 · Phase 1: Machine Learning

[![Live Demo](https://img.shields.io/badge/Live%20Demo-aircast--pm25.vercel.app-000000?style=for-the-badge&logo=vercel)](https://aircast-pm25.vercel.app/#forecast)

> 🌐 **Live Interactive App & Analysis:** [**aircast-pm25.vercel.app/#forecast**](https://aircast-pm25.vercel.app/#forecast)  
> Visit the live web application to test real-time 1-hour-ahead PM2.5 forecasting, compare XGBoost with baseline models, and explore interactive error analysis and historical visualizations.

For our Phase 1 project, we are working on a simple question: **can past pollution readings and weather conditions help us predict PM2.5 one hour ahead?** We use XGBoost for the main model and compare it with two simple forecasts and a linear model. The comparison matters as much as the model itself: we want to see what the extra features and model complexity add over using the latest reading.

## What we predict

Our target is PM2.5 concentration in µg/m³. A forecast issued after the 10:00 reading predicts the concentration at 11:00. It can use the 10:00 pollution and weather readings and anything earlier, but not the weather measured at 11:00.

This is why our features include a lag of zero: it refers to the observation available when we make the forecast. The target is still one hour in the future. All times follow the Beijing local clock.

## Our dataset

We use the [UCI Beijing PM2.5 dataset](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data). UCI lists 43,824 hourly records covering January 2010 to December 2014. The file is `PRSA_data_2010.1.1-2014.12.31.csv`.

Along with PM2.5, it contains temperature, dew point, pressure, wind direction, accumulated wind speed, and snow/rain duration. Missing readings appear as `NA`. PM2.5 and weather were recorded at different locations—the US Embassy in Beijing and Beijing Capital International Airport—so that mismatch is one limitation of the data.

The data loader checks the date fields, duplicate timestamps, and numeric values before sorting the records. It keeps a slot for every hour, including hours with missing measurements. Otherwise, shifting by one row could accidentally mean more than one hour. We leave missing concentrations unfilled and exclude the row identifier from the model.

The supplied CSV matches all 43,824 expected hourly timestamps, with no duplicates or absent timestamp rows. It has 2,067 missing PM2.5 readings (4.72%) and complete weather fields. We preserve those gaps rather than invent pollution measurements.

Dataset citation: Chen, S. (2015), *Beijing PM2.5*, UCI Machine Learning Repository, DOI [10.24432/C5JS49](https://doi.org/10.24432/C5JS49), CC BY 4.0.

## How we approach the problem

### Start with the training data

Our EDA covers missing readings, the concentration distribution, pollution peaks, hourly/monthly patterns, weather correlations, and autocorrelation. We keep it within the initial training period so our feature decisions do not depend on the final test year.

We retain valid high-pollution readings. Removing every statistical outlier would also remove events that the model needs to handle. The analysis saves the plots, numerical tables, and a readable `EDA_NOTES.md` explaining the calculated observations and their connection to the features. [Our analysis](docs/analysis.md) brings the measured EDA findings, reasoning, and forecast comparison together.

In the actual 2010–2012 data, mean PM2.5 is 97.82 µg/m³, the median is 73.00, and the one-hour autocorrelation is 0.961. Recent history is therefore an important starting point. The training data also contains long missing stretches and a substantial upper tail; both matter when we interpret the forecasts.

### Turn history into features

| Feature group | What we include | Why |
|---|---|---|
| PM2.5 lags | 0, 1, 2, 3, 6, 12, 24, 48, 72, and 168 hours | Recent changes, daily history, and the previous week |
| Trailing statistics | Mean, standard deviation, and observed fraction over 3, 6, 12, 24, 72, and 168 hours | Local level, variability, and how much data supports each window |
| Calendar | Cyclic hour, weekday, and month encodings; weekend flag | Time patterns known before the forecast is made |
| Weather | Numeric measurements at 0, 1, and 24 hours; encoded wind direction | Current conditions and recent environmental history |
| Missingness | PM2.5 lag and current-weather flags | Distinguish an absent measurement from a real low value |

All rolling windows end at the forecast origin. We allow 168 hours of history before scoring forecasts. Encoding and any learned preprocessing are fitted separately on each training fold.

### Compare against useful references

- **Persistence** predicts the latest PM2.5 reading. At a short horizon, this is an important reference.
- **Daily seasonal naive** uses the observation from the same target hour on the previous day. For target `t+1`, that is `y(t−23)`.
- **Ridge regression** checks how far a regularized linear relationship can take us. It uses training-only median imputation and scaling, with `alpha=10`.
- **XGBoost** models nonlinear relationships between the historical features and the next concentration. It handles missing numeric inputs directly.

We compare three XGBoost settings using validation RMSE. The settings vary tree depth, number of trees, learning rate, sampling, and regularization. Learned predictions are clipped at zero because concentration cannot be negative.

## How we evaluate the forecasts

We keep the data in time order:

| Period | Use |
|---|---|
| 2010–2012 | Initial training and EDA |
| 2013 | Four expanding quarterly folds for parameter selection |
| 2014 | Four quarterly backtests with the selected settings held fixed |

At each quarterly boundary, the model is refitted using labels already available at the first forecast origin. Within the quarter, we update the observed history each hour while keeping the fitted model fixed. Later quarters can use earlier observed quarters, following the same retraining rule.

Every method is scored on the same hours, with observed targets and available inputs for both naive forecasts. We report how many hours are included and why others are excluded. Missing targets are never treated as ground truth.

Our main metrics are **MAE**, **RMSE**, and **R²**, both overall and by quarter. MAE describes absolute error, while RMSE gives more weight to large misses. R² can be negative. We also inspect bias, errors during high-pollution hours, and residual autocorrelation. These checks help us understand where a model struggles, even if its overall score looks reasonable.

## What the measured results show

All methods were compared on the same **8,531 hours in 2014**, covering 97.39% of the expected target hours. MAE and RMSE are in µg/m³.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence | 11.926 | 22.033 | 0.9442 |
| Daily seasonal naive | 67.296 | 98.754 | −0.1205 |
| Ridge | **11.825** | **21.063** | **0.9490** |
| XGBoost | 11.833 | 21.088 | 0.9489 |

XGBoost improves RMSE over persistence by **4.29%**. Ridge improves it by **4.41%** and is slightly better overall. Their ranking changes across quarters, and the small pooled difference does not establish statistical significance.

The high-pollution subset is harder: XGBoost has RMSE 39.959 for concentrations at or above the training 90th percentile of 218 µg/m³, compared with 38.557 for persistence. Its overall improvement therefore does not mean it handles the worst pollution hours better.

On 2013 validation dates, adding trailing/calendar features and then weather reduces XGBoost RMSE from 27.513 to 27.092 to 26.772. Full-feature XGBoost still trails persistence and Ridge on pooled validation RMSE. We retained those results and the original experiment settings.

[Read the complete analysis, plots, and reasoning](docs/analysis.md).

## Interactive forecast workspace

Aircast is our frontend for entering new observations or uploading an hourly CSV and running the saved models. It combines the selected history with the current readings, predicts PM2.5 one hour ahead, and shows the result beside an interactive history chart. We can switch between XGBoost and Ridge, export forecasts, inspect the measured model comparisons, and open the project report. The forecast screen also shows history completeness, past model errors and input-range warnings.

The sample mode uses the end of the Beijing archive and clearly labels editable example inputs as historical scenarios. Uploaded data uses its own last recorded hour. The app keeps missingness visible and checks that enough history is available before predicting. It uses the original Python pipelines and leaves the Phase 1 experiment settings and results unchanged.

[Hosted dashboard](https://aircast-pm25.vercel.app) · [Frontend guide](docs/frontend.md) · [Vercel deployment guide](docs/deployment.md).

## Project files and outputs

The Python package in `src/air_prediction/` contains data handling, EDA, features, chronological splits, models, evaluation, and saved-model prediction. Experiment settings live in `configs/phase1.toml`, and `tests/` checks the timing and data-handling logic.

Outputs are saved under `artifacts/phase1/`: EDA notes and figures, validation scores, selected settings, timestamped predictions, quarterly/pooled metrics, coverage, error diagnostics, and saved models. Each completed run records the input hash, source hash, versions, and configuration. Models refitted for later prediction are kept separate from the fits used for the backtests.

The preparation folder includes the cleaned hourly series, missing-run dates, a dictionary of the 70 features before wind encoding, and a worked forecast-row preview. [Compact measured outputs](docs/results/) are kept with the project documentation.

The implementation uses Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, joblib, and optionally PyTorch for Phase 2 deep learning and hybrid residual models.

### Installation and Quick Start

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install package with development, web dashboard, and optional deep learning extras
pip install -e '.[dev,web]'       # Classical ML + Web Dashboard
pip install -e '.[dev,web,dl]'    # Includes PyTorch for Bi-LSTM & Hybrid models

# 3. Run the complete test suite (74 tests)
pytest -v

# 4. Start the interactive Aircast dashboard locally
bash scripts/start_frontend.sh
```

Further reading within the project:

- [System Architecture](docs/ARCHITECTURE.md)
- [Testing & QA Guide](docs/TESTING.md)
- [Known Issues & Technical Limits](docs/KNOWN_ISSUES.md)
- [Changelog](CHANGELOG.md)
- [Measured analysis and findings](docs/analysis.md)
- [Methodology and equations](docs/methodology.md)
- [Literature review](docs/literature_review.md)
- [Phase 1 checklist](docs/phase1_checklist.md)
- [Validation notes](docs/verification.md)
- [IEEE report PDF](reports/ieee/main.pdf) and [presentation PDF](reports/presentation/slides.pdf)
- [Viva notes](docs/viva_guide.md)

## 🌐 Live Interactive Demo

The project is deployed and accessible online:
🔗 **[aircast-pm25.vercel.app/#forecast](https://aircast-pm25.vercel.app/#forecast)**

Visitors can use this dashboard to:
- **Inspect Live Predictions:** Generate and examine 1-hour-ahead PM2.5 forecasts using the trained model weights.
- **Compare Models:** Evaluate performance side-by-side across XGBoost, Ridge, Persistence, and Daily Seasonal Naive.
- **Interactive Analysis:** Explore pollution regime breakdowns, residual diagnostics, and error distributions directly in the browser.

## Current status and limits

The real-data cleaning, EDA, feature construction, four-model backtests, validation-only feature comparison, diagnostics, saved-model forecasts, and Phase 2 deep learning & hybrid models have been completed. The four-page IEEE report and 15-slide presentation use these measured findings and have been compiled and visually reviewed. Aircast adds an interactive interface around the saved models. All 74 tests pass across data handling, classical ML forecasting, PyTorch sequence & hybrid models, dashboard service, and deployment packaging. Synthetic data remains limited to implementation tests. Vercel reports the production deployment Ready.

Our conclusions are limited to this historical Beijing series. They do not establish accuracy in another city or today, and weather correlations or feature importance do not establish causation. Missing baseline inputs also affect which hours we can compare. Changes made after reviewing these holdout results require a fresh evaluation or an explicit exploratory label.

This phase covers classical Machine Learning. We have left AQI mapping out because an hourly concentration forecast alone does not provide the averaging procedure required by an AQI standard.

The project code uses the [MIT license](LICENSE). The dataset has its own CC BY 4.0 attribution requirements.
