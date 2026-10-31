"""Preprocessing building blocks: units, missingness, lags and target safety."""

import numpy as np
import pandas as pd
import pytest

from ds_starter.cleaning import apply_fixed_rules
from ds_starter.config import load_config
from ds_starter.data import build_target
from ds_starter.features import create_features, model_input_columns, model_inputs


def test_fixed_rules_do_not_modify_input():
    raw = pd.DataFrame({"x": [3.0, -1.0, 1000.0], "group": [" A ", "B", "C"]})
    cleaned = apply_fixed_rules(raw, {"sentinels": {"x": [-1]},
                                      "numeric_bounds": {"x": {"max": 30}}})
    assert cleaned["x"].isna().sum() == 2
    assert cleaned.loc[0, "group"] == "A"
    assert raw.loc[1, "x"] == -1


def test_ratios_and_lags_cannot_generate_infinite_values():
    times = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame({"time": times, "x": [4, 6, 8], "y": [2, 0, 4]})
    spec = {"data": {"time_column": "time"}, "features": {
        "ratios": [{"name": "ratio", "numerator": "x", "denominator": "y"}],
        "lags": {"x": 1}, "calendar": True,
        "numeric": ["ratio", "x_lag1", "hour_sin1"], "categorical": [],
    }}
    out = create_features(frame, spec)
    assert out["ratio"].iloc[0] == 2
    assert np.isnan(out["ratio"].iloc[1])
    assert pd.isna(out["x_lag1"].iloc[0])
    assert out["x_lag1"].iloc[1] == 4
    assert np.isfinite(out["hour_cos1"]).all()
    assert model_input_columns(spec) == ["ratio", "x_lag1", "hour_sin1"]
    assert list(model_inputs(out, spec)) == ["ratio", "x_lag1", "hour_sin1"]


def test_component_target_missing_if_any_source_missing():
    frame = pd.DataFrame({"a": [1, 2], "b": [3, np.nan]})
    y = build_target(frame, {"target_column": "sum", "target_sum_columns": ["a", "b"]})
    assert y.iloc[0] == 4
    assert pd.isna(y.iloc[1])


def test_config_forbids_target_leakage(tmp_path):
    path = tmp_path / "bad.yml"
    path.write_text("""
name: test
data: {train_path: example.csv}
target_column: price
features: {numeric: [price], categorical: []}
split: {strategy: random, test_size: 0.2}
cv: {strategy: random, n_splits: 3}
models: {names: [ridge]}
""")
    with pytest.raises(ValueError, match="cannot be predictors"):
        load_config(path)
