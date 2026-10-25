"""Statistics for the GraphSAGE-MLP gap. With few seeds (default 5) the smallest two-sided Wilcoxon p-value
is 0.0625, so the Wilcoxon test can NEVER reach p<0.05 with 5 seeds: use >=6 seeds for it to be informative, and
rely on the bootstrap CI / paired t-test / effect sizes with caution."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as st

from .metrics import confusion, metrics_from_counts


def _metric(y: np.ndarray, pred: np.ndarray, metric: str) -> float:
    return metrics_from_counts(*confusion(y, pred))[metric]


def paired_bootstrap_gap(
    ys: Sequence[np.ndarray], pa: Sequence[np.ndarray], pb: Sequence[np.ndarray],
    metric: str = "macro_f1", thr: float = 0.5, B: int = 500, ci: float = 0.95, seed: int = 0,
) -> Dict[str, float]:
    """Two-stage paired bootstrap of gap = metric(A) - metric(B).

    Stage 1 resamples seeds (runs) with replacement, stage 2 resamples test nodes within each chosen run
    (same resampled rows for A and B -> paired). Returns the point estimate and percentile CI.
    """
    rng = np.random.default_rng(seed)
    S = len(ys)
    preds_a = [p >= thr for p in pa]
    preds_b = [p >= thr for p in pb]
    point = float(np.mean([_metric(y, a, metric) - _metric(y, b, metric) for y, a, b in zip(ys, preds_a, preds_b)]))
    boots = np.empty(B)
    for i in range(B):
        gaps = []
        for s in rng.integers(0, S, S):
            n = len(ys[s])
            rows = rng.integers(0, n, n)
            gaps.append(_metric(ys[s][rows], preds_a[s][rows], metric) - _metric(ys[s][rows], preds_b[s][rows], metric))
        boots[i] = np.mean(gaps)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"gap": point, "ci_lo": float(lo), "ci_hi": float(hi)}


def seed_level_tests(gaps: Sequence[float]) -> Dict[str, float]:
    """One-sample tests on per-seed paired gaps (H0: mean gap = 0)."""
    g = np.asarray(gaps, dtype=float)
    g = g[~np.isnan(g)]
    out = {"n_seeds": float(len(g)), "mean_gap": float(g.mean()) if len(g) else float("nan"),
           "std_gap": float(g.std(ddof=1)) if len(g) > 1 else float("nan")}
    if len(g) >= 3 and g.std(ddof=1) > 0:
        out["t_pvalue"] = float(st.ttest_1samp(g, 0.0).pvalue)
        out["cohen_dz"] = float(g.mean() / g.std(ddof=1))
    else:
        out["t_pvalue"] = out["cohen_dz"] = float("nan")
    try:
        out["wilcoxon_pvalue"] = float(st.wilcoxon(g).pvalue) if len(g) >= 3 and np.any(g != 0) else float("nan")
    except ValueError:
        out["wilcoxon_pvalue"] = float("nan")
    return out


def correlations(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": float(len(x)), "spearman": float("nan"), "spearman_p": float("nan"),
                "pearson": float("nan"), "pearson_p": float("nan")}
    sr = st.spearmanr(x, y)
    pr = st.pearsonr(x, y)
    return {"n": float(len(x)), "spearman": float(sr.statistic), "spearman_p": float(sr.pvalue),
            "pearson": float(pr.statistic), "pearson_p": float(pr.pvalue)}
