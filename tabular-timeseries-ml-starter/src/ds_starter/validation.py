"""Cross-validation, metrics, and model selection."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from ds_starter.models import build_model


def make_cv(config: dict):
    opts = config["cv"]
    strategy = opts["strategy"]
    n = opts.get("n_splits", 3)
    if strategy == "time":
        return TimeSeriesSplit(n_splits=n, test_size=opts.get("test_size"), gap=opts.get("gap", 0))
    if strategy == "group":
        return GroupKFold(n_splits=n)
    return KFold(n_splits=n, shuffle=opts.get("shuffle", True), random_state=config.get("random_seed", 42) if opts.get("shuffle", True) else None)


def regression_metrics(actual, predicted) -> dict:
    """Metrics on original target scale; missing targets must be filtered first."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if len(y_true) == 0 or not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metrics need nonempty finite targets and predictions")
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def cross_validate_models(X: pd.DataFrame, y: pd.Series, config: dict, groups=None):
    """Fit preprocessors independently inside each training fold."""
    cv = make_cv(config)
    if config["cv"]["strategy"] == "group":
        if groups is None:
            raise ValueError("Group cross-validation requires groups")
        splits = cv.split(X, y, groups=groups)
    else:
        splits = cv.split(X, y)

    rows = []
    for fold, (train_positions, val_positions) in enumerate(splits, start=1):
        train_x, val_x = X.iloc[train_positions], X.iloc[val_positions]
        train_y, val_y = y.iloc[train_positions], y.iloc[val_positions]
        # Do not impute unknown response values.
        valid_train, valid_val = train_y.notna(), val_y.notna()
        if valid_train.sum() < 2 or valid_val.sum() < 2:
            raise ValueError(f"Fold {fold} has too few observed targets to score")
        for name in config["models"]["names"]:
            if name in {"ols", "ridge"} and config["models"].get("linear_target_transform") == "log1p":
                if (train_y[valid_train] < 0).any():
                    raise ValueError("log1p targets require nonnegative training responses")
            estimator = build_model(config, name)
            estimator.fit(train_x.loc[valid_train], train_y.loc[valid_train])
            prediction = estimator.predict(val_x.loc[valid_val])
            scores = regression_metrics(val_y.loc[valid_val], prediction)
            rows.append({"fold": fold, "model": name, "n_train": int(valid_train.sum()),
                         "n_valid": int(valid_val.sum()), **scores})
    results = pd.DataFrame(rows)
    summary = (results.groupby("model")[["RMSE", "MAE", "R2"]]
               .agg(["mean", "std"]))
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    metric = config.get("selection_metric", "RMSE")
    if metric not in {"RMSE", "MAE", "R2"}:
        raise ValueError("selection_metric must be RMSE, MAE or R2")
    summary = summary.sort_values(metric + "_mean", ascending=(metric != "R2"))
    return results, summary
