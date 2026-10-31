"""Minimal grouped regression template: python examples/minimal_group_cv.py."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

rng = np.random.default_rng(42)
n = 120
frame = pd.DataFrame({
    "group": np.repeat(np.arange(12), 10).astype(str),
    "size": rng.normal(10, 2, n),
    "region": rng.choice(["A", "B", "C"], n),
})
frame["target"] = 300 + 25 * frame["size"] + rng.normal(0, 30, n)
frame.loc[rng.choice(n, 8, replace=False), "size"] = np.nan

X = frame[["size", "region"]]
y = frame["target"]
cv = GroupKFold(n_splits=3)
for fold, (train, valid) in enumerate(cv.split(X, y, groups=frame["group"]), 1):
    process = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), ["size"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["region"]),
    ])
    model = Pipeline([("preprocess", process),
                      ("model", RandomForestRegressor(n_estimators=60, random_state=42))])
    model.fit(X.iloc[train], y.iloc[train])  # preprocessing fitted inside this fold
    prediction = model.predict(X.iloc[valid])
    print(f"Fold {fold} | MAE: {mean_absolute_error(y.iloc[valid], prediction):.2f}")
