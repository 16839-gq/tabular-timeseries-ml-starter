# Copy-friendly EDA recipes (development data only)

The functions in `ds_starter.eda` produce basic summaries. The following snippets cover the most transferable patterns from the two notebooks, not their dataset-specific conclusions.

## Quality, outliers, and correlations

```python
import numpy as np
from ds_starter.eda import data_quality_report, top_correlations

quality = data_quality_report(development_frame)
print(quality.sort_values("missing_pct", ascending=False).head(15))

numeric = development_frame.select_dtypes(include="number")
q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
iqr = q3 - q1
flags = numeric.lt(q1 - 1.5 * iqr) | numeric.gt(q3 + 1.5 * iqr)
print(flags.sum().sort_values(ascending=False).head())
# IQR = diagnostic flag, NOT automatic reason to delete rows.

print(top_correlations(development_frame, exclude=["target"], top_n=10))
```

## Grouped comparisons and composition checks

```python
print(development_frame.groupby("category")["target"].agg(["count", "mean", "median"]))
print(development_frame.groupby(["region", "category"])["target"].median())

# Contrast overall vs within-category patterns to avoid composition bias.
```

## Time-series patterns (UTC)

```python
s = development_frame.set_index("time")["signal"]
daily = s.resample("D").mean()
trend = daily.rolling(180, min_periods=90).mean()
monthly = s.groupby(s.index.month).mean()
hourly = s.groupby(s.index.hour).mean()

trend.plot(title="Rolling trend")
monthly.plot(title="Seasonal profile: month")
hourly.plot(title="Seasonal profile: UTC hour")

# 'D' daily mean of hourly power observations is NOT daily energy.
```

## Month-based deseasonalization (descriptive)

```python
s = daily.dropna()
month_mean = s.groupby(s.index.month).transform("mean")
deseasonalized = s - month_mean + s.mean()
print({"original_skew": s.skew(), "deseasonalized_skew": deseasonalized.skew()})
# A more symmetric distribution does not prove Gaussianity.
```

For model validation, compute group statistics or seasonal adjustment parameters only on a training fold, then transform held-out rows with those fixed estimates.
