"""End-to-end, config-driven regression experiment."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ds_starter.cleaning import apply_fixed_rules
from ds_starter.config import load_config
from ds_starter.data import build_target, development_holdout_indices, load_table
from ds_starter.eda import (
    data_quality_report,
    save_prediction_diagnostics,
    save_target_histogram,
    save_timeseries_plot,
    top_correlations,
)
from ds_starter.features import create_features, model_inputs
from ds_starter.models import build_model
from ds_starter.validation import cross_validate_models, regression_metrics


def _prepare(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    return create_features(apply_fixed_rules(frame, config.get("cleaning")), config)


def _external_predictions(model, config: dict, output_directory: Path) -> None:
    external_file = config["data"].get("predict_path")
    if not external_file:
        return
    raw = load_table(external_file, config["data"].get("time_column"))
    transformed = _prepare(raw, config)
    features = model_inputs(transformed, config)
    predictions = model.predict(features)
    ids = [col for col in config["data"].get("id_columns", []) if col in raw]
    export = raw[ids].copy()
    export["prediction"] = predictions
    export.to_csv(output_directory / "external_predictions.csv", index=False)


def run_experiment(config_path: str | Path) -> dict:
    """EDA -> CV -> refit -> ONE final holdout evaluation -> export."""
    config = load_config(config_path)
    report_dir = Path(config.get("report_dir", f"reports/{config['name']}"))
    model_dir = Path(config.get("model_dir", f"models/{config['name']}"))
    figure_dir = report_dir / "figures"
    for path in (report_dir, model_dir, figure_dir):
        path.mkdir(parents=True, exist_ok=True)

    raw = load_table(config["data"]["train_path"], config["data"].get("time_column"))
    train_ids, holdout_ids = development_holdout_indices(raw, config)
    # Predeclared cleaning rules and past-only feature transforms are deterministic.
    # Learned imputation, encoding, scaling and model fitting happen only within folds.
    transformed = _prepare(raw, config)
    y = build_target(transformed, config)
    X = model_inputs(transformed, config)
    X_dev, X_holdout = X.iloc[train_ids], X.iloc[holdout_ids]
    y_dev, y_holdout = y.iloc[train_ids], y.iloc[holdout_ids]

    # No holdout statistics in EDA or data-driven cleaning decisions.
    dev_frame = transformed.iloc[train_ids]
    pd.Series({
        "development_rows": len(dev_frame),
        "development_columns": len(dev_frame.columns),
        "duplicate_rows": int(dev_frame.duplicated().sum()),
        "observed_targets": int(y_dev.notna().sum()),
        "missing_targets": int(y_dev.isna().sum()),
        "holdout_rows_reserved": len(holdout_ids),
    }, name="value").to_csv(report_dir / "development_overview.csv", header=True)
    data_quality_report(dev_frame).to_csv(report_dir / "development_data_quality.csv")
    predictors = config["features"].get("numeric", [])
    top_correlations(dev_frame[predictors], top_n=20).to_csv(
        report_dir / "development_top_correlations.csv", index=False
    )
    save_target_histogram(y_dev.rename(config["target_column"]), figure_dir / "development_target.png")

    group_name = config["data"].get("group_column")
    groups = dev_frame[group_name].astype("string").fillna("__missing_group__") if group_name else None
    cv_rows, summary = cross_validate_models(X_dev, y_dev, config, groups=groups)
    cv_rows.to_csv(report_dir / "cv_folds.csv", index=False)
    summary.to_csv(report_dir / "cv_summary.csv")
    chosen_name = str(summary.index[0])
    trained = build_model(config, chosen_name)
    valid_dev = y_dev.notna()
    if valid_dev.sum() < 2:
        raise ValueError("Too few development rows with known targets")
    trained.fit(X_dev.loc[valid_dev], y_dev.loc[valid_dev])
    all_predictions = pd.Series(trained.predict(X_holdout), index=X_holdout.index)
    keep = y_holdout.notna()
    scores = {}
    if keep.any():
        actual = y_holdout.loc[keep]
        prediction = all_predictions.loc[keep]
        scores["selected_model"] = regression_metrics(actual, prediction)
        baseline_strategy = config["models"].get("baseline_strategy", "median")
        baseline_value = getattr(y_dev.loc[valid_dev], baseline_strategy)()
        baseline_prediction = np.repeat(baseline_value, len(actual))
        scores[f"development_{baseline_strategy}_baseline"] = regression_metrics(actual, baseline_prediction)
        save_prediction_diagnostics(actual, prediction, figure_dir)
        if config["split"]["strategy"] == "time":
            stamp_col = config["data"]["time_column"]
            save_timeseries_plot(raw.iloc[holdout_ids].loc[keep, stamp_col].to_numpy(),
                                 actual.to_numpy(), prediction.to_numpy(),
                                 figure_dir / "holdout_daily.png")
    else:
        scores["note"] = "No labeled holdout rows; predictions exported without metrics"

    summary_scores = pd.DataFrame(scores).T
    summary_scores.to_csv(report_dir / "holdout_scores.csv")
    ids = [col for col in config["data"].get("id_columns", []) if col in raw]
    export = raw.iloc[holdout_ids][ids].copy()
    export["actual"] = y_holdout.to_numpy()
    export["prediction"] = all_predictions.to_numpy()
    export.to_csv(report_dir / "holdout_predictions.csv", index=False)

    # Post-selection model inspection on a capped holdout subsample. Do not use
    # these diagnostics to change a model without creating a fresh holdout.
    if keep.sum() >= 2 and config.get("permutation_importance", {}).get("enabled", True):
        cap = config.get("permutation_importance", {}).get("max_rows", 500)
        observed_ids = X_holdout.index[keep]
        if len(observed_ids) > cap:
            rng = np.random.default_rng(config.get("random_seed", 42))
            observed_ids = rng.choice(observed_ids.to_numpy(), size=cap, replace=False)
        result = permutation_importance(
            trained, X_holdout.loc[observed_ids], y_holdout.loc[observed_ids],
            scoring="neg_root_mean_squared_error", n_repeats=3,
            random_state=config.get("random_seed", 42), n_jobs=1,
        )
        importance = pd.DataFrame({
            "feature": X_holdout.columns,
            "mean_RMSE_increase": result.importances_mean,
            "std": result.importances_std,
        }).sort_values("mean_RMSE_increase", ascending=False)
        importance.to_csv(report_dir / "post_holdout_permutation_importance.csv", index=False)

    joblib.dump({"model": trained, "config": config}, model_dir / "trained_pipeline.joblib")
    _external_predictions(trained, config, report_dir)
    return {"chosen_model": chosen_name, "cv_summary": summary,
            "holdout_scores": summary_scores, "report_dir": str(report_dir),
            "model_dir": str(model_dir)}


def predict_from_bundle(model_path: str | Path, data_path: str | Path,
                        output_path: str | Path) -> pd.DataFrame:
    """Load only trusted local joblib files, then score new input CSV."""
    bundle = joblib.load(model_path)
    model, config = bundle["model"], bundle["config"]
    frame = load_table(data_path, config["data"].get("time_column"))
    features = model_inputs(_prepare(frame, config), config)
    ids = [col for col in config["data"].get("id_columns", []) if col in frame]
    output = frame[ids].copy()
    output["prediction"] = model.predict(features)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output
