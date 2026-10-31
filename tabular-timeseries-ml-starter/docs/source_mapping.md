# Source notebook → reusable pattern

The two input sources were:

- `boat_cleaned(1).ipynb`: boat listing data cleaning, price EDA, model comparison, and unpriced listing estimates.
- `Electricity_clean_organized(1).ipynb`: hourly electricity data cleaning, source-generation EDA, total Thermal target, time-series validation, and held-out evaluation.

**This page describes observed notebook structure.** The reusable framework itself is a new implementation; recommendations for corrections are identified separately below.

| Stage | Boat notebook | Electricity notebook | Generalized module |
|---|---|---|---|
| Load and inspect | Separate labeled/unlabeled CSVs; duplicates, nulls, numeric categories | Read timestamped hourly CSV; reserve latest 20% | `data.py`, `eda.py` |
| Cleaning | Parse mixed units and engine text; fix Year and categories; inspect IQR outliers | Mask negatives/fixed high cutoffs, inspect spike windows, drop constant columns | `cleaning.py` fixed rules + documented domain adapter work |
| EDA | Numeric correlations, price distribution, expensive decile, geography/hull type, construction year | Source trends, monthly/hourly cycles, solar distribution, pumped storage | `eda.py`, `docs/eda_recipes.md` |
| Target | Boat `Listing Price (USD)` | Sum retained `Thermal_*` columns; only complete rows have known total | `data.build_target()` |
| Feature engineering | Hull/beam and engine/displacement ratios; selected categorical fields | Same-hour solar/wind, lag-1 hydro/storage, hourly and annual sine/cosine | `features.py` |
| Imputation | Make/Variant then Hull type then overall median; missing category Unknown | Month×hour / month / hour means with overall mean fallback | `preprocessing.GroupwiseNumericImputer` |
| Categorical/scaling | OneHotEncoder; StandardScaler for linear models, not RF | Mostly numeric; StandardScaler for OLS/Ridge, not trees | `preprocessing.build_preprocessor()` |
| CV | KFold and GroupKFold for unseen makes | Expanding TimeSeriesSplit | `validation.py` |
| Models | Median, OLS, Ridge, RF; XGB declared but unused in its final CV loop | OLS, Ridge, RF, XGB | `models.py` with optional XGBoost |
| Selection | Summary metrics, but final RF choice is manually assigned | Select lowest average validation RMSE | `validation.py`, `runner.py` |
| Final results | Predicted prices for ten unlabeled boats; RF importances | Final refit, held-out RMSE/MAE/R², daily actual vs predicted | `runner.py`, `eda.py` |

## What the starter deliberately changes

1. **Leakage protection for boat imputation.** In the original notebook, group medians and global medians were computed on all labeled training rows *before* KFold/GroupKFold. These estimates can incorporate information from validation rows. The starter fits grouped imputation *inside* each fold.
2. **Separate model selection from holdout.** The boat notebook manually sets the final model to Random Forest after reporting metrics; the electricity notebook automatically selects lowest validation RMSE. The starter uses the configured metric consistently and evaluates the final holdout only once.
3. **One consistent preprocessing pipeline per model.** Encoding, imputation, and scaling are stored inside the fitted estimator; inference reuses the same training-derived transform.
4. **Robust feature importance.** Rather than relying solely on tree impurity-based importances, the starter reports original-column permutation importances after evaluation (descriptive, not a tuning signal).
5. **Domain rules remain domain-specific.** The starter does not silently infer whether `-1` means missing, whether `5` is year 2005, whether a generation spike is implausible, or whether a timestamp is truly evenly spaced.

## Important scope distinctions

- The electricity notebook explicitly uses contemporaneous Solar/Wind. That is same-hour estimation / nowcasting. It is **not** an advance forecast unless predictors are reliably available ahead of the target hour.
- The boat notebook contains domain-specific transformations for strings such as lengths in feet/metres and horsepower. Reproduce those only in your own domain adapter after reviewing the data and units.
- A target is always kept outside the feature list; a summed target is observed only if all required components are observed.
- The repository's examples are **synthetic** and do not reproduce notebook-specific performance numbers or imply the original data can be distributed.
