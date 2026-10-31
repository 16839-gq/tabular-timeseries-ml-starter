# Adapt a third dataset in about 10 steps

1. **Understand the question.** What is the target, at what time is the prediction made, and what information is available then?
2. **Inspect raw columns.** Open a CSV sample and make a data dictionary (meaning, units, missing sentinels, expected range).
3. **Choose evaluation strategy first.** Random, grouped, or chronological. If group generalization matters, use group holdout and GroupKFold.
4. **Separate the final holdout.** Do not use it to select thresholds or models.
5. **Clean fixed representations.** For example, parse `"2.1 m"` to `2.1`, normalize category spelling and record domain-supported limits. Use dedicated adapter code when nonstandard strings cannot be cleaned with the YAML rules.
6. **List allowed features.** Add ratios and previous-covariate lags using `features.ratios` and `features.lags`; include derived names in `features.numeric`.
7. **Specify imputation hierarchy.** `group_imputation.rules` maps numeric features to ordered fallback keys. For example, `power: [[make, model], [category]]`. The final global training statistic is automatic.
8. **Choose the simplest credible models.** Include the `baseline`, OLS/Ridge, and a tree. Enable XGBoost only if useful and installed.
9. **Run, diagnose, record.** Check `cv_folds.csv`, `cv_summary.csv`, `development_data_quality.csv`, and held-out residual plots. R² can be negative.
10. **Predict safely.** Reuse the saved fitted pipeline with the same schema. Check unknown categories, missing lags, and train/inference consistency.

## Example: feature configuration

```yaml
features:
  numeric: [length, width, area_ratio]
  categorical: [region, category]
  ratios:
    - {name: area_ratio, numerator: length, denominator: width}
group_imputation:
  strategy: median
  rules:
    length: [[category, region], [category]]
```

The group key(s) are passed through imputation but not automatically included as model predictors. Include them in `features.categorical` if they should also inform predictions. If no grouped rule exists for a feature, the numeric `SimpleImputer` supplies its training-fold median.

## Common exceptions that need a custom adapter

- Mixed or compound units (`"1152 ft²107 m²"`, feet + metres, two engine powers).
- Real timestamps with gaps, duplicate hours, daylight saving transitions, or mixed time zones.
- Sequential forward-fill constrained to past observations and known maximum observation age.
- Rolling features, per-entity lags, multiple forecast horizons, target leakage via joins.
- Domain-specific outlier detection requiring human inspection.
- Imbalanced classifications, survival analysis, financial transaction costs, non-regression targets.

This is an intentionally small **regression** starter, not a full forecasting platform or a general-purpose missing-data engine.
