"""Pure, domain-agnostic feature engineering helpers."""

import numpy as np
import pandas as pd


def create_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add ratios, past-covariate lags, and cyclic time features without fitting."""
    out = frame.copy()
    feature_spec = config["features"]

    for rule in feature_spec.get("ratios", []):
        denominator = pd.to_numeric(out[rule["denominator"]], errors="coerce")
        numerator = pd.to_numeric(out[rule["numerator"]], errors="coerce")
        out[rule["name"]] = numerator.div(denominator.where(denominator.ne(0)))
        out[rule["name"]] = out[rule["name"]].replace([np.inf, -np.inf], np.nan)

    # The caller sorts by timestamp before creating lags. These are only previous-row
    # measurements, not previous target values. A missing hour still means a lagged row.
    for col, n_lags in feature_spec.get("lags", {}).items():
        if not isinstance(n_lags, int) or n_lags <= 0:
            raise ValueError("Lag count must be a positive integer")
        out[f"{col}_lag{n_lags}"] = out[col].shift(n_lags)

    if feature_spec.get("calendar", False):
        time_col = config["data"].get("time_column")
        if not time_col:
            raise ValueError("calendar features require data.time_column")
        date = pd.to_datetime(out[time_col], utc=True)
        hour = date.dt.hour.to_numpy()
        leap_days = np.where(date.dt.is_leap_year, 366, 365)
        year_fraction = (date.dt.dayofyear.to_numpy() - 1) / leap_days
        out["__month"] = date.dt.month.to_numpy()
        out["__hour"] = hour
        for harmonic in (1, 2):
            out[f"hour_sin{harmonic}"] = np.sin(2 * np.pi * harmonic * hour / 24)
            out[f"hour_cos{harmonic}"] = np.cos(2 * np.pi * harmonic * hour / 24)
        out["year_sin"] = np.sin(2 * np.pi * year_fraction)
        out["year_cos"] = np.cos(2 * np.pi * year_fraction)
    return out


def model_input_columns(config: dict) -> list[str]:
    """Predictor columns + temporary group keys used only during imputation."""
    spec = config["features"]
    columns = [*spec.get("numeric", []), *spec.get("categorical", [])]
    for rule_list in config.get("group_imputation", {}).get("rules", {}).values():
        for keys in rule_list:
            columns.extend(keys)
    return list(dict.fromkeys(columns))


def model_inputs(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    columns = model_input_columns(config)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model input columns: {sorted(missing)}")
    result = frame.loc[:, columns].copy()
    for col in config["features"].get("numeric", []):
        result[col] = pd.to_numeric(result[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return result
