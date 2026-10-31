"""Generic EDA summaries; call on development data only."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 (select noninteractive backend before pyplot)
import numpy as np
import pandas as pd


def data_quality_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Inspect missingness, unique counts, numeric bounds, and duplicate rates."""
    rows = []
    for column in frame.columns:
        s = frame[column]
        numerical = pd.to_numeric(s, errors="coerce") if pd.api.types.is_numeric_dtype(s) else None
        rows.append({
            "column": column,
            "dtype": str(s.dtype),
            "missing_n": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 2),
            "unique_n": int(s.nunique(dropna=True)),
            "min": float(numerical.min()) if numerical is not None and numerical.notna().any() else np.nan,
            "max": float(numerical.max()) if numerical is not None and numerical.notna().any() else np.nan,
        })
    return pd.DataFrame(rows).set_index("column")


def top_correlations(frame: pd.DataFrame, *, exclude: list[str] | None = None, top_n: int = 15) -> pd.DataFrame:
    """Strongest absolute correlations between DISTINCT numeric predictors."""
    numeric = frame.drop(columns=exclude or [], errors="ignore").select_dtypes(include="number")
    corr = numeric.corr()
    rows = []
    cols = corr.columns
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            value = corr.loc[left, right]
            if pd.notna(value):
                rows.append({"feature_1": left, "feature_2": right, "correlation": float(value)})
    return (pd.DataFrame(rows, columns=["feature_1", "feature_2", "correlation"])
            .assign(abs_correlation=lambda x: x["correlation"].abs())
            .sort_values("abs_correlation", ascending=False).head(top_n))


def save_target_histogram(target: pd.Series, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(target.dropna(), bins=25)
    ax.set(title="Development target distribution", xlabel=target.name, ylabel="Count")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_prediction_diagnostics(actual, predicted, directory: str | Path) -> None:
    directory = Path(directory)
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(predicted, actual - predicted, s=12, alpha=0.5)
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.set(xlabel="Predicted", ylabel="Residual (actual - predicted)", title="Holdout residuals")
    fig.tight_layout()
    fig.savefig(directory / "holdout_residuals.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(actual, predicted, s=12, alpha=0.5)
    lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set(xlabel="Actual", ylabel="Predicted", title="Holdout actual vs predicted")
    fig.tight_layout()
    fig.savefig(directory / "holdout_actual_vs_predicted.png", dpi=130)
    plt.close(fig)


def save_timeseries_plot(times, actual, predicted, path: str | Path) -> None:
    frame = pd.DataFrame({"time": pd.to_datetime(times, utc=True), "actual": actual,
                          "predicted": predicted}).set_index("time")
    # Both curves must use the same valid target timestamps.
    daily = frame.resample("D").mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    daily.plot(ax=ax, title="Daily averages on matched, observed holdout timestamps")
    ax.set_ylabel("Mean of hourly observations (not energy)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
