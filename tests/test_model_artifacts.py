import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_model_metadata_matches_exported_forest():
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))
    model = np.load(ROOT / "models" / "model.npz", allow_pickle=False)

    n_trees = int(meta["n_trees"])
    assert n_trees > 0
    assert len(meta["feature_names"]) > 0

    for i in range(n_trees):
        assert f"cl_{i}" in model
        assert f"cr_{i}" in model
        assert f"feat_{i}" in model
        assert f"thr_{i}" in model
        assert f"val_{i}" in model
