# Tabular & Time-Series ML Starter

**A reusable, GitHub-ready Python project for exploratory data analysis and regression.** Extracted from the shared high-level workflow of two practice notebooks (boat listing prices and hourly electricity generation), and strengthened with standard Python project practices.

**No real datasets or original notebook outputs are bundled.** Instead, two synthetic examples exercise the entire workflow. Use the source modules independently in notebooks or run experiments with YAML.

## Shared workflow

```text
Load data and reserve the holdout
         |
Development-only quality checks and EDA
         |
Fixed cleaning + feature engineering
         |
CV folds (random / grouped / chronological)
         |
Fit imputation, encoding and scaling INSIDE each training fold
         |
Median/mean baseline | OLS | Ridge | Random Forest | optional XGBoost
         |
Compare validation MAE / RMSE / R² and select model
         |
Refit using development data -> evaluate holdout once
         |
Save scores, plots, feature importance, trained model, predictions
```

### What is shared vs. what changes?

| Part | Universal implementation | Configurable special cases |
|---|---|---|
| Data inspection | Missingness, types, duplicates, correlations, distributions | Units, invalid markers, spikes, manual anomaly review |
| Holdout | Never use for preprocessing or model selection | Random, unseen groups, or latest time period |
| Imputation | Training-fold-only statistics + global fallback | `make -> category` medians, `month + hour` means |
| Features | Numeric ratios, covariate lags, cyclic calendar features | Which features are known at prediction time |
| Encoding | One-hot with infrequent-category handling | Unseen manufacturers, optional unknown-category behavior |
| Models | Baseline, OLS, Ridge, Random Forest | Optional XGBoost; log target for nonnegative linear regression |
| Evaluation | Per-fold RMSE/MAE/R², held-out scores, residuals | Group generalization versus chronological generalization |

The raw notebooks are **not** represented as finished, reproducible production pipelines. The repository makes their common patterns reusable and addresses the most important validation pitfalls. See [Notebook-to-code mapping](docs/source_mapping.md).

## Quickstart (Python 3.11+)

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python scripts/make_demo_data.py

# Two complete, runnable workflows
python -m ds_starter.cli run --config configs/tabular_demo.yml
python -m ds_starter.cli run --config configs/timeseries_demo.yml

# Unit tests
python -m pytest -q
```

Optional installations:

```bash
python -m pip install -e '.[all]'  # Jupyter, seaborn, scipy, XGBoost, pytest, Ruff
```

Or `make demo`, `make test`, and `make lint` if `make` is available. To launch Jupyter, install `.[notebooks]` and run `jupyter lab`.

The CLI prints the chosen model, cross-validation summary, and final holdout metrics. It writes (per experiment):

- `reports/<name>/development_overview.csv`, `development_data_quality.csv`, `development_top_correlations.csv`
- `reports/<name>/cv_folds.csv`, `cv_summary.csv`, `holdout_scores.csv`
- `reports/<name>/holdout_predictions.csv`, optional `external_predictions.csv`
- `reports/<name>/post_holdout_permutation_importance.csv`, and plots in `figures/`
- `models/<name>/trained_pipeline.joblib`

You can reload a pipeline on an unlabeled CSV (only trust your own `.joblib` files):

```bash
python -m ds_starter.cli predict \
  --model models/tabular_demo/trained_pipeline.joblib \
  --input data/raw/tabular_unlabeled_demo.csv \
  --output data/processed/predictions.csv
```

## Project structure

```text
.
├── .github/                    # CI, issue and pull request templates
├── configs/                    # Two runnable demos; two real-dataset adapter EXAMPLES
├── data/raw/                   # User data (gitignored)
├── data/processed/             # Generated data (gitignored)
├── docs/
│   ├── source_mapping.md       # Precisely what came from the source notebooks
│   ├── validation_and_leakage.md
│   ├── adapting_a_dataset.md
│   ├── eda_recipes.md
│   └── resource_catalog.md     # Researched external tools, templates, docs, and search terms
├── examples/                   # Small, standalone code recipes
├── models/                     # Saved pipelines (gitignored)
├── notebooks/                  # Minimal exploratory notebook template
├── reports/                    # Metrics, images (gitignored)
├── scripts/make_demo_data.py   # Reproducible synthetic test datasets
├── src/ds_starter/
│   ├── cli.py                  # run / predict
│   ├── config.py               # YAML validation
│   ├── data.py                 # CSV loading, splits, target construction
│   ├── cleaning.py             # Domain-reviewed fixed rules
│   ├── features.py             # Ratios, lagged covariates, calendars
│   ├── preprocessing.py        # Fold-only grouped imputation, encoding, scaling
│   ├── models.py               # Model registry
│   ├── validation.py           # CV comparison and metrics
│   ├── eda.py                  # Quality summaries, simple plots
│   └── runner.py               # End-to-end orchestration and saving
├── tests/                      # Unit and smoke tests
├── pyproject.toml              # Dependencies, lint, test, package metadata
├── Makefile
└── LICENSE
```

## Adapting to a new dataset

1. Put the CSV in `data/raw/` **without committing it**. Identify the target and available predictors.
2. Choose the split from the use case: `random`, `group` for unseen entities, or `time` for future periods. Use group-based holdout if assessing group generalization.
3. Copy a YAML in `configs/`, changing the column names, `data.*`, feature lists, fixed cleaning rules, imputation groups, and model settings. File paths are resolved relative to your working directory (run from repository root).
4. Run EDA on the development partition. Inspect units, missing-value conventions, invalid periods, and whether a time lag is genuinely observed at inference.
5. Run the experiment. **Do not revise the model based on holdout results**, unless you create a separate untouched holdout afterwards.

For real raw boat records, first convert mixed-unit strings into numeric columns; for hourly records, verify an equally spaced time grid and timezone before creating lag features. The example adapters deliberately **do not** implement unverified, dataset-specific cleaning for you. See [Adapting your data](docs/adapting_a_dataset.md).

## Practical guardrails

- Group imputation, numeric imputation, categorical encoding, and scaling are fitted **inside every CV training fold** via scikit-learn `Pipeline`.
- Holdout rows are separated before any exploratory inspection or fitted preprocessing. Domain-fixed cleaning rules are applied consistently to train and holdout.
- Unknown target values are never imputed. The target component sum is valid only when **every** required component is present.
- A `lag1` feature means the **previous row**. Time-series projects must verify hourly continuity and missing timestamps. Lags are not automatically grouped, shifted by duration, or forecast-safe.
- The `TimeSeriesSplit` option assumes equally spaced observations for comparable fold horizons. Adjust `gap` when observation delay or feature availability requires it.
- Permutation importance is generated only **after** choosing the model; it measures dependence on inputs, not causal impact. It must not be used to tune a model on the final holdout.
- `log1p` transformation is supported only for nonnegative linear-model targets; metrics are reported on the original scale.
- Any external `predict` run must provide the same input columns. For lagged features, include sufficient historical covariate context; otherwise the initial lag is missing and gets imputed.
- Use `joblib.load` only on trusted files; serialized Python models may execute code during loading and can be version-sensitive.

## Tests and style

```bash
python -m pytest -q
ruff check .
ruff format .  # optional, applies consistent formatting
```

GitHub Actions runs these checks on Python 3.11–3.13 and executes both synthetic demos. It does **not** access your private datasets.

## Publishing on GitHub

This folder is a local Git repository once initialized. To publish from your own account, create an **empty** GitHub repository (do not add a README or license there), then run from this directory:

```bash
git init -b main
git add .
git commit -m "Initial reusable ML starter"
git remote add origin https://github.com/YOUR_USERNAME/tabular-timeseries-ml-starter.git
git push -u origin main
```

Alternatively, use `gh repo create tabular-timeseries-ml-starter --private --source=. --remote=origin --push` after authenticating with GitHub CLI. Replace `--private` with `--public` only after checking that no confidential data or code is included. GitHub publication requires your credentials/permission and is not performed by this downloadable project.

## Sources and attribution

- **User notebooks:** structural ideas, variable categories, and workflow differences; only high-level patterns were abstracted. Original raw data, plots, and notebooks are not republished.
- **External resources:** documentation and general patterns from scikit-learn, Cookiecutter Data Science, pytest, Ruff, and GitHub Actions; see [Research and code-template catalog](docs/resource_catalog.md).
- **New implementation:** lightweight reusable modules, synthetic datasets, test cases, CLI, config schema, and repository packaging.

The project code is MIT licensed. Third-party packages and templates retain their own licenses; links in the catalog do not imply a third-party code copy.
