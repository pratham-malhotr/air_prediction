# Phase 1 literature review

These sources motivate the current experiment. Their published findings are not results from this repository. The current project does not claim a new forecasting architecture.

| Source | Method or contribution | Connection to this project | Limitation for our comparison |
|---|---|---|---|
| Liang et al. (2015), *Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating* | Statistical analysis of Beijing pollution and meteorological effects | Motivates retaining weather covariates and evaluating seasonal effects | This is pollution/weather adjustment and observational analysis; it is not our XGBoost one-hour forecasting benchmark |
| Chen and Guestrin (2016), *XGBoost: A Scalable Tree Boosting System* | Regularized, efficient tree boosting with sparse/missing-value support | Required nonlinear regression model; tree depth and regularization address complexity | General learning-system evidence does not establish Beijing forecast accuracy; temporal validation is still necessary |
| Hyndman and Athanasopoulos (2021), *Forecasting: Principles and Practice*, third edition | Forecasting baselines, point-error measures, and time-series cross-validation | Justifies persistence/seasonal naive and evaluation in time order | General forecasting guidance must be adapted to missing hourly environmental records and label availability |
| Hoerl and Kennard (1970), *Ridge Regression: Biased Estimation for Nonorthogonal Problems* | L2-regularized linear regression for correlated predictors | A transparent linear comparator for highly correlated lags | A linear mapping may miss nonlinear pollutant/weather effects; that is an empirical question here |

## Gap and testable question

At a one-hour horizon, recent PM2.5 is likely a strong predictor. Does an environmental feature model improve MAE and RMSE over persistence and daily seasonal naive on **the same observed 2014 targets**, after selecting settings only on earlier dates? Do any improvements hold across quarters and high-pollution hours? The project will answer those questions only after a real-data run. Persistence remains a meaningful outcome even if it wins.

Manual lags and rolling windows represent selected temporal scales; their limitations can motivate Phase 2 later. Phase 1 does not evaluate LSTM or assert that sequence models will improve on this baseline.

## Reading and citation links

1. Liang et al., Proceedings of the Royal Society A 471, 20150257 (2015). DOI: [10.1098/rspa.2015.0257](https://doi.org/10.1098/rspa.2015.0257). [Author-hosted paper](https://songxichen.com/Uploads/Files/Publication/Liang-etal-Chen-PRSA-PM2.5.pdf).
2. Chen, T., and Guestrin, C., KDD (2016), pp. 785–794. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754); DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
3. Hyndman, R. J., and Athanasopoulos, G., *Forecasting: Principles and Practice*, 3rd ed., OTexts (2021). [Forecaster's toolbox](https://otexts.com/fpp3/toolbox.html), especially simple methods, accuracy, and time-series cross-validation.
4. Hoerl, A. E., and Kennard, R. W., Technometrics 12(1), pp. 55–67 (1970). DOI: [10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634).
5. Dataset citation: Chen, S. (2015), *Beijing PM2.5*, UCI Machine Learning Repository. DOI: [10.24432/C5JS49](https://doi.org/10.24432/C5JS49). The dataset's 2010–2014 coverage should not be conflated with every period discussed in the introductory paper.

The sources here are a focused starting review, not a claim of an exhaustive systematic search. Team members should read the originals and record any additional empirical air-quality forecasting papers, their horizons, splits, and metrics before comparing reported numbers.
