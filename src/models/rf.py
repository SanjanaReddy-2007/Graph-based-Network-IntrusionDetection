"""Random Forest flow-level baseline (uses node features only, ignores the graph)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..train.task import Task


def fit_predict_rf(task: Task, cfg: dict, seed: int) -> dict:
    rcfg = cfg["rf"]
    Gtr = task.graphs["train"]
    itr = task.idx["train"]
    rng = np.random.default_rng(seed)
    cap = rcfg.get("max_train_samples")
    if cap and len(itr) > cap:
        itr = rng.choice(itr, size=cap, replace=False)
    Xtr, ytr = Gtr.x[itr], Gtr.y[itr]
    rf = RandomForestClassifier(
        n_estimators=rcfg["n_estimators"], max_depth=rcfg.get("max_depth"), min_samples_leaf=rcfg.get("min_samples_leaf", 1),
        class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
    )
    rf.fit(Xtr, ytr)

    def proba(part: str) -> np.ndarray:
        p = rf.predict_proba(task.graphs[part].x[task.idx[part]])
        return p[:, list(rf.classes_).index(1)] if 1 in rf.classes_ else np.zeros(len(p), dtype=np.float32)

    return {"prob_val": proba("val").astype(np.float32), "prob_test": proba("test").astype(np.float32),
            "n_params": int(sum(t.tree_.node_count for t in rf.estimators_)), "hidden": 0, "epochs": 0,
            "best_val_loss": float("nan"), "backend": "sklearn"}
