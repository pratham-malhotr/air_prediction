# Our presentation outline

The presentation should tell one connected story: what we predict, what the data allows us to use, why we chose these features and models, and what the measured comparison shows.

1. Start with one forecast example: after observing hour t, we predict PM2.5 at t+1. State the units and what information is available.
2. Introduce the UCI dataset and the difference between the pollution and weather observation locations.
3. Use the training EDA to explain missingness, the concentration range, time patterns, and the history/weather relationships. Link each plot to a modeling decision.
4. Trace one row through the lag and rolling features. Explain lag zero and the t−23 seasonal reference.
5. Explain why we compare persistence, daily seasonal naive, Ridge, and XGBoost. Give the main idea behind regularization and boosting.
6. Show the chronological split and quarterly retraining rule. Explain why every training label has to exist before the forecast is issued.
7. Present the same-target MAE, RMSE, and R² alongside 97.39% coverage and quarterly variation. XGBoost improves pooled RMSE over persistence by 4.29%; Ridge is slightly better overall.
8. Explain the difficult cases: XGBoost's high-pollution RMSE is 39.959 versus persistence 38.557. Its first validation quarter is also weaker, so the 2013 and 2014 comparisons should both be discussed.
9. Show a saved-model forecast and the files needed to reproduce the experiment. The archive's next-hour forecast is historical, not live air quality.
10. Finish with the measured answer to our question and the limits of that answer.

The report and slides should use the same dates, model settings, and results. We should all be able to explain the full pipeline and walk through a concrete prediction during the viva.
