"""Train-fold-only grouped imputation and heterogeneous feature preprocessing."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


class GroupwiseNumericImputer(TransformerMixin, BaseEstimator):
    """Fill numeric values with learned subgroup means/medians and global fallback.

    rules: {"measurement": [["make", "variant"], ["hull_type"]]}
    Fit receives only this CV fold's training data. Unknown groups use the next
    group rule, then a global training statistic. It never fills targets.
    """

    def __init__(self, rules: dict | None = None, strategy: str = "median"):
        self.rules = rules
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y=None):
        if self.strategy not in {"mean", "median"}:
            raise ValueError("Groupwise strategy must be mean or median")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("GroupwiseNumericImputer requires a pandas DataFrame")
        self.statistics_ = {}
        self.global_ = {}
        for feature, key_sets in (self.rules or {}).items():
            if feature not in X:
                raise ValueError(f"Group-imputation feature not present: {feature}")
            value = pd.to_numeric(X[feature], errors="coerce")
            self.global_[feature] = getattr(value, self.strategy)()
            self.statistics_[feature] = []
            for keys in key_sets:
                if not keys or any(col not in X for col in keys):
                    raise ValueError(f"Unknown or empty group keys: {keys}")
                training = X[keys].copy()
                training[feature] = value
                stats = (
                    training.groupby(keys, dropna=False)[feature]
                    .agg(self.strategy)
                    .rename("__group_fill__")
                    .reset_index()
                )
                self.statistics_[feature].append((keys, stats))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "statistics_")
        out = X.copy()
        for feature, candidates in self.statistics_.items():
            current = pd.to_numeric(out[feature], errors="coerce")
            for keys, stats in candidates:
                # Merge uses row order, not the original index; this preserves
                # DataFrames with nonconsecutive fold indices.
                lookup = out[keys].merge(stats, on=keys, how="left", sort=False)
                fills = pd.Series(lookup["__group_fill__"].to_numpy(), index=out.index)
                current = current.fillna(fills)
            if pd.notna(self.global_[feature]):
                current = current.fillna(self.global_[feature])
            out[feature] = current
        return out


def build_preprocessor(config: dict, *, linear: bool) -> ColumnTransformer:
    """Scale numerics for linear models; never scale for trees."""
    numeric = config["features"].get("numeric", [])
    categorical = config["features"].get("categorical", [])
    transformers = []
    if numeric:
        steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if linear:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        min_frequency = config.get("encoding", {}).get("min_frequency", 5)
        unknown = config.get("encoding", {}).get("handle_unknown", "infrequent_if_exist")
        steps = [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(
                handle_unknown=unknown,
                min_frequency=min_frequency,
                sparse_output=False,
            )),
        ]
        transformers.append(("categorical", Pipeline(steps), categorical))
    if not transformers:
        raise ValueError("Specify numerical or categorical predictors")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
