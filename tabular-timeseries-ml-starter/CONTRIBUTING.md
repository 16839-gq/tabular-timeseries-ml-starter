# Contributing

This is a compact, teaching-oriented regression starter. Before contributing, please read the validation guidance in `docs/validation_and_leakage.md`.

1. Fork the repository; use a feature branch and a clear pull request description.
2. Keep dataset-specific rules in adapters/configs, not the generic core.
3. Add or update tests, including a case with unseen groups or a future-only holdout where relevant.
4. Run `python -m pytest -q`, `ruff check .`, and optionally `ruff format .`.
5. Do not commit real raw data, private notebook outputs, secrets, or model artifacts. Use synthetic minimal examples for bug reports.

All contributions must be compatible with the MIT license, and credited third-party snippets must comply with their original licenses.
