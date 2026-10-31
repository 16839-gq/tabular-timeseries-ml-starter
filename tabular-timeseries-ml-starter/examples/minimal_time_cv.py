"""Minimal chronological regression template: python examples/minimal_time_cv.py."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
time = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
signal = np.sin(2 * np.pi * time.hour / 24) + rng.normal(0, 0.2, len(time))
frame = pd.DataFrame({"time": time, "signal": signal})
frame["signal_lag1"] = frame["signal"].shift(1)  # previous measurement only
frame["hour_sin"] = np.sin(2 * np.pi * frame["time"].dt.hour / 24)
frame["target"] = 3 * frame["signal"] + frame["hour_sin"] + rng.normal(0, 0.5, len(frame))
frame = frame.dropna().reset_index(drop=True)

X = frame[["signal_lag1", "hour_sin"]]
y = frame["target"]
for fold, (train, valid) in enumerate(TimeSeriesSplit(n_splits=3, test_size=100).split(X), 1):
    assert train.max() < valid.min()
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10))])
    model.fit(X.iloc[train], y.iloc[train])
    pred = model.predict(X.iloc[valid])
    rmse = np.sqrt(mean_squared_error(y.iloc[valid], pred))
    print(f"Fold {fold} | chronological RMSE: {rmse:.3f}")
