import numpy as np
import pandas as pd

from ds_starter.data import development_holdout_indices
from ds_starter.validation import make_cv, regression_metrics


def test_group_holdout_has_no_overlap():
    df = pd.DataFrame({"group": [f"G{i // 3}" for i in range(30)]})
    config = {"random_seed": 42, "data": {"group_column": "group"},
              "split": {"strategy": "group", "test_size": 0.2}}
    tr, te = development_holdout_indices(df, config)
    assert set(df.iloc[tr].group).isdisjoint(set(df.iloc[te].group))


def test_time_split_and_cv_never_train_on_future():
    df = pd.DataFrame({"time": pd.date_range("2023-01-01", periods=80, freq="h")})
    config = {"split": {"strategy": "time", "test_size": 0.2},
              "cv": {"strategy": "time", "n_splits": 3, "test_size": 12, "gap": 2}}
    dev, holdout = development_holdout_indices(df, config)
    assert dev.max() < holdout.min()
    for train, valid in make_cv(config).split(df.iloc[dev]):
        assert train.max() + 2 < valid.min()


def test_metrics_are_on_original_scale():
    scores = regression_metrics(np.array([10.0, 20.0]), np.array([12.0, 18.0]))
    assert np.isclose(scores["RMSE"], 2.0)
    assert np.isclose(scores["MAE"], 2.0)
