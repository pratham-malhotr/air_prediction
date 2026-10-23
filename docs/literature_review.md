# Phase 1 literature review

We use these readings to connect the forecasting problem to the methods in our project. The Beijing pollution study helps explain why weather deserves attention, the forecasting text guides our baselines and time splits, and the model papers explain XGBoost and Ridge.

| Source | Method or contribution | Connection to this project | Limitation for our comparison |
|---|---|---|---|
| Liang et al. (2015), *Assessing Beijing's PM2.5 pollution: severity, weather impact, APEC and winter heating* | Statistical analysis of Beijing pollution and meteorological effects | Motivates retaining weather covariates and evaluating seasonal effects | This is pollution/weather adjustment and observational analysis; it is not our XGBoost one-hour forecasting benchmark |
| Chen and Guestrin (2016), *XGBoost: A Scalable Tree Boosting System* | Regularized, efficient tree boosting with sparse/missing-value support | Required nonlinear regression model; tree depth and regularization address complexity | General learning-system evidence does not establish Beijing forecast accuracy; temporal validation is still necessary |
| Hyndman and Athanasopoulos (2021), *Forecasting: Principles and Practice*, third edition | Forecasting baselines, point-error measures, and time-series cross-validation | Justifies persistence/seasonal naive and evaluation in time order | General forecasting guidance must be adapted to missing hourly environmental records and label availability |
| Hoerl and Kennard (1970), *Ridge Regression: Biased Estimation for Nonorthogonal Problems* | L2-regularized linear regression for correlated predictors | A transparent linear comparator for highly correlated lags | A linear mapping may miss nonlinear pollutant/weather effects; that is an empirical question here |

## Gap and testable question

Our main question is whether the environmental features improve MAE and RMSE over persistence and daily seasonal naive on **the same observed 2014 targets**, after choosing settings only from earlier dates. We also examine whether the comparison changes across quarters or high-pollution hours. The latest reading is already a strong one-hour forecast: in our measured backtests, XGBoost improves pooled RMSE by 4.29% over persistence, while Ridge is slightly better overall. XGBoost's high-pollution RMSE remains worse than persistence. [The complete findings](analysis.md) show why all three comparisons matter.

Our lags and windows represent selected hourly, daily, and weekly scales. They do not cover every possible temporal relationship. For this phase, we keep the focus on understanding and evaluating the classical ML setup.

## Reading and citation links

1. Liang et al., Proceedings of the Royal Society A 471, 20150257 (2015). DOI: [10.1098/rspa.2015.0257](https://doi.org/10.1098/rspa.2015.0257). [Author-hosted paper](https://songxichen.com/Uploads/Files/Publication/Liang-etal-Chen-PRSA-PM2.5.pdf).
2. Chen, T., and Guestrin, C., KDD (2016), pp. 785–794. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754); DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
3. Hyndman, R. J., and Athanasopoulos, G., *Forecasting: Principles and Practice*, 3rd ed., OTexts (2021). [Forecaster's toolbox](https://otexts.com/fpp3/toolbox.html), especially simple methods, accuracy, and time-series cross-validation.
4. Hoerl, A. E., and Kennard, R. W., Technometrics 12(1), pp. 55–67 (1970). DOI: [10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634).
5. Dataset citation: Chen, S. (2015), *Beijing PM2.5*, UCI Machine Learning Repository. DOI: [10.24432/C5JS49](https://doi.org/10.24432/C5JS49). The dataset's 2010–2014 coverage should not be conflated with every period discussed in the introductory paper.

When adding another forecasting study, we need to record its target, horizon, time split, and error measures. A published score from another horizon or evaluation period is useful context, but it is not a direct comparison with our experiment.
