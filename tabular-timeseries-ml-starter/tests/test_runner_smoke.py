from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ds_starter.runner import predict_from_bundle, run_experiment


def test_grouped_pipeline_end_to_end(tmp_path: Path):
    rng = np.random.default_rng(3)
    n = 72
    groups = np.repeat([f"G{i}" for i in range(12)], 6)
    x = rng.normal(10, 2, n)
    frame = pd.DataFrame({"id": range(n), "group": groups, "feature": x,
                          "category": np.where(np.arange(n) % 2 == 0, "A", "B"),
                          "target": 50 + 3 * x + rng.normal(0, 2, n)})
    frame.loc[[0, 5, 9], "feature"] = np.nan
    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    cfg = {
        "name": "smoke", "random_seed": 42,
        "data": {"train_path": str(source), "group_column": "group", "id_columns": ["id"]},
        "target_column": "target",
        "split": {"strategy": "group", "test_size": 0.2},
        "cv": {"strategy": "group", "n_splits": 3},
        "features": {"numeric": ["feature"], "categorical": ["category"]},
        "group_imputation": {"rules": {"feature": [["group"]]}},
        "models": {"names": ["baseline", "ridge"]},
        "permutation_importance": {"enabled": False},
        "report_dir": str(tmp_path / "reports"), "model_dir": str(tmp_path / "models"),
    }
    path = tmp_path / "cfg.yml"
    path.write_text(yaml.safe_dump(cfg))
    result = run_experiment(path)
    assert result["chosen_model"] in ("baseline", "ridge")
    assert (tmp_path / "reports" / "holdout_scores.csv").exists()
    assert (tmp_path / "reports" / "cv_folds.csv").exists()
    artifact = tmp_path / "models" / "trained_pipeline.joblib"
    output = predict_from_bundle(artifact, source, tmp_path / "predictions.csv")
    assert len(output) == n
    assert output.prediction.notna().all()
