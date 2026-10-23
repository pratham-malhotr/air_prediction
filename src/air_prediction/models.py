"""Fold-local encoders and imputation; XGBoost is never silently substituted."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_model(name, X, config, parameters=None):
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [column for column in X if column not in numeric]
    if name == "ridge":
        transform = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ])
        estimator = Ridge(alpha=10.0)
    elif name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "XGBoost is unavailable. Install requirements.txt (on macOS, libomp may also be required) "
                "or run Air_Prediction_Phase1_Colab.ipynb. --models ridge is a separate diagnostic run, not an XGBoost result."
            ) from exc
        transform = "passthrough"  # Native missing-value splits; no global imputation.
        estimator = XGBRegressor(
            objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
            random_state=config.random_seed, n_jobs=config.threads, verbosity=0, **(parameters or {}),
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    preprocess = ColumnTransformer([
        ("numeric", transform, numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], remainder="drop", sparse_threshold=0)
    return Pipeline([("preprocess", preprocess), ("regressor", estimator)])
