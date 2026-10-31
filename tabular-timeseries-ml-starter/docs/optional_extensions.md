# Optional tools and alternative repository templates

The base starter intentionally depends only on a few mature Python libraries. The following were researched as **future additions**, not hidden dependencies. Avoid installing a large stack for a short, six-hour exploratory project.

| Need | Option | When worth adding | Official documentation |
|---|---|---|---|
| Full pipeline orchestration and structured data catalog | **Kedro** and its starter projects | Many repeated multi-step runs, many datasets, multiple contributors | https://docs.kedro.org/en/stable/create/starters/ |
| Dataset and model versioning | **DVC** | Large files that should not live in Git, remote shared storage | https://doc.dvc.org/user-guide/data-management/remote-storage |
| Validate tabular schema before training | **Pandera** | Complex feed contracts or recurring raw data issues | https://pandera.readthedocs.io/en/latest/dataframe_schemas.html |
| Experiment tracking and model versioning | **MLflow** | Many experiments, parameter searches, collaborators, auditability | https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html |
| Automated hyperparameter tuning | **Optuna** | The basic baselines are working and CV is trustworthy | https://optuna.readthedocs.io/en/stable/ |
| Richer full project template | **Cookiecutter Data Science** | Starting a larger multi-month workstream | https://cookiecutter-data-science.drivendata.org/ |

## A reasonable growth path

Start with this repository and one `configs/*.yml`. If the project grows:

1. Add a Pandera schema for column types, required fields, and valid ranges.
2. Add DVC only if you need reproducible access to large raw datasets. Keep secrets out of Git.
3. Add MLflow to record run settings and CV summaries.
4. Introduce Optuna with only *development CV* as the optimization objective; preserve an untouched final holdout.
5. Use Kedro or another orchestration tool only if the pipeline warrants it.

**Do not assume** these optional tools make a small dataset prediction more accurate. Correct split strategy, target definition, feature availability and leak-free preprocessing matter more.
