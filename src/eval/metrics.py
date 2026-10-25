"""Classification metrics (binary, attack = positive class). Accuracy is reported only for comparison with
prior work: on CICIDS2017 (>80 % benign) it is misleading, so macro-F1 / MCC / PR-AUC are primary."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score


def confusion(y: np.ndarray, pred: np.ndarray):
    y, pred = y.astype(bool), pred.astype(bool)
    tp = int((y & pred).sum()); tn = int((~y & ~pred).sum())
    fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
    return tn, fp, fn, tp


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def metrics_from_counts(tn, fp, fn, tp) -> Dict[str, float]:
    n = tn + fp + fn + tp
    f1_att, f1_ben = _f1(tp, fp, fn), _f1(tn, fn, fp)
    p_att = tp / (tp + fp) if tp + fp else 0.0
    num = tp * tn - fp * fn
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    n_att, n_ben = tp + fn, tn + fp
    return {
        "accuracy": (tp + tn) / n if n else 0.0,
        "macro_f1": (f1_att + f1_ben) / 2,
        "weighted_f1": (f1_att * n_att + f1_ben * n_ben) / n if n else 0.0,
        "attack_recall": tp / n_att if n_att else float("nan"),
        "attack_precision": p_att,
        "fpr": fp / n_ben if n_ben else float("nan"),
        "mcc": float(num / den) if den > 0 else 0.0,
    }


def compute_metrics(y: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    pred = prob >= thr
    out = metrics_from_counts(*confusion(y, pred))
    out["pr_auc"] = float(average_precision_score(y, prob)) if 0 < y.sum() < len(y) else float("nan")
    return out


def per_attack_recall(attack_label: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """Recall of the binary detector for each original attack label (BENIGN excluded)."""
    pred = prob >= thr
    out = {}
    for lab in np.unique(attack_label):
        if lab == "BENIGN":
            continue
        m = attack_label == lab
        out[str(lab)] = float(pred[m].mean())
    return out
