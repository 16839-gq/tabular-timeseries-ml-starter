import numpy as np
import pandas as pd

from ds_starter.preprocessing import GroupwiseNumericImputer, build_preprocessor


def test_group_imputer_learns_only_training_statistics():
    train = pd.DataFrame({
        "make": ["A", "A", "B", "B"],
        "type": ["T1", "T1", "T2", "T2"],
        "size": [10.0, np.nan, 30.0, 50.0],
    }, index=[12, 13, 15, 17])
    valid = pd.DataFrame({
        "make": ["A", "NEW", "NEW"],
        "type": ["T1", "T2", "UNSEEN"],
        "size": [np.nan, np.nan, np.nan],
    }, index=[30, 48, 51])
    imputer = GroupwiseNumericImputer(
        rules={"size": [["make"], ["type"]]}, strategy="median"
    ).fit(train)
    out = imputer.transform(valid)
    assert list(out.index) == [30, 48, 51]
    assert out["size"].tolist() == [10, 40, 30]  # new make uses T2; unknown both global
    assert valid["size"].isna().all()  # no mutation


def test_empty_group_fallback_to_simple_imputer():
    cfg = {"features": {"numeric": ["x"], "categorical": ["kind"]}}
    preprocessing = build_preprocessor(cfg, linear=True)
    X = pd.DataFrame({"x": [1.0, np.nan, 3.0], "kind": ["A", None, "B"]})
    result = preprocessing.fit_transform(X)
    assert result.shape[0] == 3
    assert np.isfinite(result).all()
