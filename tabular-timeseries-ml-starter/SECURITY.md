# Security and privacy

- Keep sensitive or proprietary datasets in `data/raw/` or another ignored directory, not in public Git history.
- Review `git status` and `git diff --cached` before pushing to GitHub.
- Never commit API tokens, cloud credentials, or `.env` files.
- Load only **trusted** `.joblib` pipelines: pickle/joblib deserialization can execute code.
- Files created by the synthetic demo generator contain no actual user records.
- Domain data validation and restrictions are the responsibility of the project maintainer.

Please avoid opening public issues containing private data. Instead, prepare a synthetic, reproducible example.
