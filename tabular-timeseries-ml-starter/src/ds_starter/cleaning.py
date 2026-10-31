"""Fixed, domain-reviewed data rules (no statistics learned from holdout data)."""

import numpy as np
import pandas as pd


def apply_fixed_rules(frame: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    """Apply predeclared missing markers and numeric limits consistently.

    Thresholds must be chosen from domain knowledge or development data only.
    Do not use test statistics to pick a cutoff.
    """
    rules = rules or {}
    out = frame.copy()
    for col, missing_values in rules.get("sentinels", {}).items():
        if col in out:
            out[col] = out[col].replace(missing_values, np.nan)
    for col, bounds in rules.get("numeric_bounds", {}).items():
        if col not in out:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if "min" in bounds:
            out[col] = out[col].mask(out[col] < bounds["min"])
        if "max" in bounds:
            out[col] = out[col].mask(out[col] > bounds["max"])
    if rules.get("strip_strings", True):
        for col in out.select_dtypes(include=["object", "string"]).columns:
            out[col] = out[col].map(lambda x: " ".join(x.split()) if isinstance(x, str) else x)
            out[col] = out[col].replace("", np.nan)
    for col, aliases in rules.get("category_aliases", {}).items():
        if col in out:
            out[col] = out[col].replace(aliases)
    return out
