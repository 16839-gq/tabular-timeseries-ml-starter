"""Read and validate simple YAML experiments."""

from pathlib import Path

import yaml


VALID_SPLITS = {"random", "group", "time"}
VALID_MODELS = {"baseline", "ols", "ridge", "random_forest", "xgboost"}


def load_config(path: str | Path) -> dict:
    """Load project-root-relative YAML settings, rejecting unsupported modes."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    for required in ("name", "data", "target_column", "features", "split", "cv", "models"):
        if required not in config:
            raise ValueError(f"Missing config section: {required}")

    if config["split"].get("strategy") not in VALID_SPLITS:
        raise ValueError("split.strategy must be random, group or time")
    if config["cv"].get("strategy") not in VALID_SPLITS:
        raise ValueError("cv.strategy must be random, group or time")
    for name in config["models"].get("names", []):
        if name not in VALID_MODELS:
            raise ValueError(f"Unknown model: {name}")
    if not config["models"].get("names"):
        raise ValueError("Specify at least one model")
    features = config["features"]
    numeric = features.get("numeric", [])
    categorical = features.get("categorical", [])
    if set(numeric) & set(categorical):
        raise ValueError("Numerical and categorical feature lists must not overlap")
    forbidden = {config["target_column"], *config.get("target_sum_columns", [])}
    if forbidden & (set(numeric) | set(categorical)):
        raise ValueError("Target and target component columns cannot be predictors")
    for feature, key_sets in config.get("group_imputation", {}).get("rules", {}).items():
        if feature not in numeric:
            raise ValueError("Group imputation supports only listed numeric predictors")
        if any(key in forbidden for keys in key_sets for key in keys):
            raise ValueError("Group imputation keys cannot use target columns")
    for lag_source in features.get("lags", {}):
        if lag_source in forbidden:
            raise ValueError("Cannot construct lagged target features in this starter")
    for rule in features.get("ratios", []):
        if {rule["numerator"], rule["denominator"]} & forbidden:
            raise ValueError("Derived ratios cannot use target columns")
    fraction = config["split"].get("test_size", 0.2)
    if not 0 < fraction < 1:
        raise ValueError("split.test_size must be a fraction between zero and one")
    if config["cv"].get("n_splits", 3) < 2:
        raise ValueError("cv.n_splits must be at least two")
    if (config["split"]["strategy"] == "time" or config["cv"]["strategy"] == "time") and not config["data"].get("time_column"):
        raise ValueError("Time splits require data.time_column")
    if (config["split"]["strategy"] == "group" or config["cv"]["strategy"] == "group") and not config["data"].get("group_column"):
        raise ValueError("Group splits require data.group_column")
    return config
