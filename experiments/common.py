"""Helpers shared by the experiment scripts."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.dataset import Dataset
from src.eval.metrics import compute_metrics, per_attack_recall
from src.eval.stats import paired_bootstrap_gap, seed_level_tests
from src.models.param_matching import mlp_param_count, sage_param_count
from src.train.task import Task
from src.train.trainer import run_model
from src.utils import get_logger

log = get_logger("experiments")
MODELS = ("rf", "mlp", "sage")
METRICS = ["macro_f1", "attack_recall", "fpr", "pr_auc", "mcc", "accuracy", "weighted_f1"]


def evaluate_models(ds: Dataset, task: Task, models: Sequence[str], cfg: dict, seed: int,
                    extra: Dict) -> Tuple[List[dict], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Train + evaluate each model on one task. Returns (rows, {model: (y_test, prob_test)})."""
    rows, preds = [], {}
    y_te = task.y("test")
    lab_te = ds.attack_label[task.global_ids("test")]
    for m in models:
        res = run_model(m, task, cfg, seed)
        met = compute_metrics(y_te, res["prob_test"])
        rec = {f"recall::{k}": v for k, v in per_attack_recall(lab_te, res["prob_test"]).items()}
        row = dict(extra, model=m, seed=seed, n_params=res["n_params"], hidden=res["hidden"], epochs=res["epochs"],
                   backend=res["backend"], n_test=len(y_te), test_attack_frac=float(y_te.mean()), **met, **rec)
        rows.append(row)
        preds[m] = (y_te, res["prob_test"])
        log.info("  [%s] %s seed=%d macroF1=%.4f PR-AUC=%.4f recall=%.4f FPR=%.4f (params=%d, epochs=%d)",
                 extra.get("protocol", extra.get("knob", "")), m, seed, met["macro_f1"], met["pr_auc"],
                 met["attack_recall"], met["fpr"], res["n_params"], res["epochs"])
    return rows, preds


def flatten_overlap(task: Task, extra: Dict, seed: int) -> dict:
    row = dict(extra, seed=seed)
    for k, v in task.info.get("overlap_ref", {}).items():
        row[f"ref_{k}"] = v
    for k, v in task.info.get("overlap_vis", {}).items():
        row[f"vis_{k}"] = v
    return row


def mean_std_table(runs: pd.DataFrame, group_cols: List[str], metrics: Sequence[str] = METRICS) -> pd.DataFrame:
    cols = [m for m in metrics if m in runs.columns]
    g = runs.groupby(group_cols)[cols]
    mean, std = g.mean(), g.std(ddof=1)
    out = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    out["n_seeds"] = runs.groupby(group_cols)["seed"].nunique()
    return out.reset_index()


def gap_stats(pa: Dict[int, Tuple[np.ndarray, np.ndarray]], pb: Dict[int, Tuple[np.ndarray, np.ndarray]],
              cfg: dict, metrics: Sequence[str] = ("macro_f1", "pr_auc", "attack_recall", "mcc")) -> dict:
    """Gap = metric(A) - metric(B), paired by seed. ``pa[seed] = (y_test, prob_test)``.

    Returns per metric: mean gap +- std over seeds, paired t / Wilcoxon p-values (seed level) and a
    two-stage paired bootstrap 95 % CI (seeds, then test nodes) for threshold-based metrics. PR-AUC gets a
    seed-level percentile CI only (it is not decomposable over resampled nodes cheaply)."""
    seeds = sorted(set(pa) & set(pb))
    st_cfg = cfg["stats"]
    ma = {s: compute_metrics(*pa[s]) for s in seeds}
    mb = {s: compute_metrics(*pb[s]) for s in seeds}
    out: Dict[str, float] = {}
    for met in metrics:
        gaps = np.array([ma[s][met] - mb[s][met] for s in seeds], dtype=float)
        for k, v in seed_level_tests(gaps).items():
            out[f"{met}_{k}"] = v
        if met in ("macro_f1", "mcc", "attack_recall", "fpr"):
            bs = paired_bootstrap_gap([pa[s][0] for s in seeds], [pa[s][1] for s in seeds], [pb[s][1] for s in seeds],
                                      metric=met, B=st_cfg["bootstrap_B"], ci=st_cfg["ci"])
            out[f"{met}_ci_lo"], out[f"{met}_ci_hi"] = bs["ci_lo"], bs["ci_hi"]
        elif len(gaps) >= 2:
            rng = np.random.default_rng(0)
            bm = [np.nanmean(rng.choice(gaps, len(gaps))) for _ in range(st_cfg["bootstrap_B"])]
            out[f"{met}_ci_lo"], out[f"{met}_ci_hi"] = float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))
    out["n_seeds"] = float(len(seeds))
    return out


def param_report(d: int, cfg: dict) -> dict:
    from src.models.param_matching import match_mlp_hidden
    h = cfg["model"]["hidden"]
    m, pm, ps = match_mlp_hidden(d, h, 2, cfg["model"].get("param_tol", 0.05))
    return {"n_features": d, "sage_hidden": h, "sage_params": ps, "mlp_hidden": m, "mlp_params": pm,
            "rel_diff": abs(pm - ps) / ps}
