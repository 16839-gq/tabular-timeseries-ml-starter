# Validation and data leakage checklist

## How to choose a split

| Deployment question | Holdout | Cross-validation |
|---|---|---|
| Predict independent rows from the same population | Random | KFold |
| Predict unseen makers/customers/accounts/subjects | Group-based | GroupKFold |
| Predict future observations | Latest-period chronological holdout | Expanding TimeSeriesSplit |

`GroupKFold` keeps each group wholly within one fold, while `GroupShuffleSplit` makes an independent group-based holdout. If the same group appears in both fitting and scoring, the result does **not** measure new-group generalization.

`TimeSeriesSplit` uses earlier data for training and later data for validation. An optional `gap` excludes recent training rows before the validation block, which can be necessary for feature arrival delays, lookahead, labels, or embargoes. `gap=0` is *not* safe for every financial forecasting problem.

## Safe order of operations

1. Reserve a final holdout appropriate to the deployment setting.
2. Inspect *development data* and document domain anomalies; freeze any fixed cleaning rules.
3. For each CV fold, learn grouped-imputation means/medians, fallback statistics, categorical vocabularies, and scaling **only** from that fold's training data.
4. Score each fold on rows with **observed** targets; never infer missing targets for evaluation.
5. Select model/hyperparameters from **validation** scores, then refit on all observed development labels.
6. Evaluate the final holdout once; write predictions and diagnostics. Do not inspect final holdout performance and then re-select a model without new validation data.

The starter keeps grouped imputation, `SimpleImputer`, `OneHotEncoder`, and `StandardScaler` inside a scikit-learn `Pipeline`, so the fit/transform boundary is enforced for every fold.

## Checks for source-specific problems

**Tabular**
- Are category labels standardized before fitting the encoder?
- Is the evaluation meant to cover *new groups* or *new records from existing groups*?
- Are ratios protected from division by zero? (Yes, in `features.py`.)
- Are categorical levels/feature names consistent between train and inference? (Yes, fitted `Pipeline`.)
- Are units and invalid placeholders explicitly documented?

**Time series**
- Are rows equally spaced? The starter *sorts* timestamps but does not fill gaps. Validate granularity before training; if needed reindex and record truly missing hours.
- Are lag features based on the **previous measurement**, not merely the previous available row after dropping missing timestamps? The starter shifts rows, so review gaps and reindex if needed.
- Is a contemporaneous covariate actually available at prediction time?
- Could the group/month/hour mean change in future? It is fitted only to each past training fold.
- Does a target component have missing values? A summed target is missing if **any component** is missing.
- Do you need feature availability delays, an embargo, or a rolling rather than expanding evaluation?

## Error metrics

- `MAE` = mean absolute error; less sensitive to occasional large misses.
- `RMSE` = square root of mean squared error; emphasizes large misses.
- `R²` = 1 − residual sum of squares / total sum of squares; can be negative out of sample.
- Compare against a simple mean or median baseline. For highly autocorrelated time series, consider adding a last-value or seasonal-naive baseline when operationally valid.

When `log1p` is used for OLS/Ridge, inverse-transform predictions before evaluating MAE/RMSE in original target units. Exponentiating a conditional log prediction does **not** guarantee an unbiased estimate of the conditional level mean.

## Model files

`joblib`/pickle-style serialization can execute arbitrary code when loaded. Load only from trusted sources, and use compatible Python / scikit-learn versions. `.joblib` and real data are ignored by Git.
