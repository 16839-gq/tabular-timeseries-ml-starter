# External research: code templates, tools, documentation, and search queries

These are **external** sources, not material extracted from the two notebooks. We researched established project templates and official documentation to choose a minimal, conventional structure. We link to original authors; their code has not been copied wholesale into this repository.

| Resource | Use in this repository | Authoritative link |
|---|---|---|
| Cookiecutter Data Science (CCDS v2) | Inspired `data/`, `src/`, `notebooks/`, `reports/`, `models/`, documentation separation | https://cookiecutter-data-science.drivendata.org/ |
| Cookiecutter Data Science source | Full-sized alternative starter for future projects | https://github.com/drivendataorg/cookiecutter-data-science |
| scikit-learn Pipeline / ColumnTransformer | Fit preprocessing and models together, with separate numeric/categorical branches | https://scikit-learn.org/stable/modules/compose.html |
| scikit-learn common pitfalls | Avoid preprocessing leakage across CV / holdout boundaries | https://scikit-learn.org/stable/common_pitfalls.html |
| scikit-learn GroupKFold | Measure generalization to unseen groups | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html |
| scikit-learn TimeSeriesSplit | Expanding-window validation, optionally with a gap | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html |
| scikit-learn OneHotEncoder | Unknown-category handling and grouping rare categories | https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html |
| scikit-learn model inspection | Permutation importance as a post-selection diagnostic | https://scikit-learn.org/stable/modules/permutation_importance.html |
| scikit-learn model persistence | Warnings about `pickle`/`joblib`, compatibility and model security | https://scikit-learn.org/stable/model_persistence.html |
| pytest | Fast unit and integration tests | https://docs.pytest.org/en/stable/getting-started.html |
| Ruff | Linting and formatting through `pyproject.toml` | https://docs.astral.sh/ruff/ |
| GitHub Actions for Python | Automated checks on push and pull request | https://docs.github.com/en/actions/tutorials/build-and-test-code/python |
| uv | Optional environment/dependency workflow (`uv sync`, `uv run pytest`) | https://docs.astral.sh/uv/ |
| JupyterLab | Lightweight interactive EDA and experiment reports | https://jupyterlab.readthedocs.io/en/stable/ |
| XGBoost | Optional boosted-tree model, installed separately | https://xgboost.readthedocs.io/en/stable/python/python_api.html |

## Useful search queries when you have only Google / simple coding help

Search these exact phrases to find reusable documentation and small examples. Replace the bracketed names with your columns:

```text
scikit learn ColumnTransformer SimpleImputer OneHotEncoder Pipeline regression example
sklearn GroupKFold cross validation groups fit predict example
sklearn TimeSeriesSplit n_splits test_size gap regression example
pandas groupby transform median fillna multiple columns example
pandas groupby month hour mean fill missing values example
pandas shift lag1 time series dataframe example
pandas datetime hour sine cosine features example
sklearn TransformedTargetRegressor log1p expm1 example
sklearn permutation_importance pipeline original columns example
pytest test pandas sklearn pipeline leakage group imputation
GitHub Actions python pytest ruff pyproject starter
Cookiecutter Data Science v2 README src notebooks data structure
```

## When to use which tool

- **pandas**: CSV, datetime, missingness, group statistics and tabular cleaning.
- **NumPy**: numerical transforms, cycles, vectorized calculations.
- **scikit-learn**: fitting, CV, imputers, encoders, pipelines, metrics, RF/Ridge/OLS.
- **Matplotlib / optional seaborn**: simple EDA / residual charts.
- **SciPy** (optional): Q–Q plots, distribution diagnostics.
- **XGBoost** (optional): stronger nonlinear model to compare with Random Forest.
- **pytest + Ruff + GitHub Actions**: portable software quality checks.
- **JupyterLab**: quick exploratory analysis; reusable implementations belong in `src/`.
- **uv** (optional): alternative to pip/venv for Python environment and package workflow.

**Important:** this catalog contains *general documentation and search starting points*. The two source notebooks determine the project's workflow and examples; the external sources inform engineering best practices and tooling. Different published templates have their own licenses.
