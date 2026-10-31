"""Small, comparable regression model registry."""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from ds_starter.preprocessing import GroupwiseNumericImputer, build_preprocessor


def build_model(config: dict, name: str) -> Pipeline:
    """Create a fresh estimator, including every training-dependent transform."""
    seed = config.get("random_seed", 42)
    settings = config.get("models", {})
    if name == "baseline":
        estimator = DummyRegressor(strategy=settings.get("baseline_strategy", "median"))
    elif name == "ols":
        estimator = LinearRegression()
    elif name == "ridge":
        estimator = Ridge(alpha=settings.get("ridge_alpha", 10.0))
    elif name == "random_forest":
        params = {"n_estimators": 160, "min_samples_leaf": 3, "random_state": seed, "n_jobs": -1}
        params.update(settings.get("random_forest", {}))
        estimator = RandomForestRegressor(**params)
    elif name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Use: pip install -e '.[boosting]' to enable xgboost") from exc
        params = {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05,
                  "random_state": seed, "n_jobs": 2}
        params.update(settings.get("xgboost", {}))
        estimator = XGBRegressor(**params)
    else:
        raise ValueError(f"Unsupported model: {name}")

    linear = name in {"ols", "ridge"}
    if linear and settings.get("linear_target_transform") == "log1p":
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
    imputation = config.get("group_imputation", {})
    return Pipeline([
        ("group_imputer", GroupwiseNumericImputer(
            rules=imputation.get("rules", {}),
            strategy=imputation.get("strategy", "median"),
        )),
        ("preprocess", build_preprocessor(config, linear=linear)),
        ("estimator", estimator),
    ])
