"""Data loading, split discipline, and target handling."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def load_table(path: str | Path, time_column: str | None = None) -> pd.DataFrame:
    """Load CSV; parse timestamps as UTC and enforce chronological ordering if supplied."""
    frame = pd.read_csv(path)
    if time_column:
        if time_column not in frame:
            raise ValueError(f"Missing time column: {time_column}")
        frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="raise")
        if frame[time_column].isna().any():
            raise ValueError("Time column cannot contain missing timestamps")
        frame = frame.sort_values(time_column, kind="stable").reset_index(drop=True)
    return frame


def build_target(frame: pd.DataFrame, config: dict) -> pd.Series:
    """Sum components only when all are observed, otherwise use a target column."""
    sources = config.get("target_sum_columns", [])
    if sources:
        missing = set(sources) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing target component columns: {sorted(missing)}")
        components = frame[sources].apply(pd.to_numeric, errors="coerce")
        return components.sum(axis=1, min_count=len(sources)).rename(config["target_column"])
    return pd.to_numeric(frame[config["target_column"]], errors="coerce")


def development_holdout_indices(frame: pd.DataFrame, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Split before fitting transformations or looking at holdout outcomes."""
    strategy = config["split"]["strategy"]
    fraction = config["split"].get("test_size", 0.2)
    seed = config.get("random_seed", 42)
    positions = np.arange(len(frame))
    if strategy == "time":
        n_test = max(1, int(np.ceil(len(positions) * fraction)))
        if n_test >= len(positions):
            raise ValueError("Not enough rows for a chronological holdout")
        return positions[:-n_test], positions[-n_test:]
    if strategy == "group":
        group = frame[config["data"]["group_column"]].astype("string").fillna("__missing_group__")
        cv = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        return next(cv.split(frame, groups=group))
    return train_test_split(positions, test_size=fraction, random_state=seed, shuffle=True)
