"""Feature transforms. The scaler is *always* fitted on training nodes only."""
from __future__ import annotations

from typing import Optional

import numpy as np


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """sign(x) * log1p(|x|): defined for negatives, compresses heavy tails. Stateless (no leakage)."""
    return (np.sign(x) * np.log1p(np.abs(x))).astype(np.float32)


class Scaler:
    """Standardiser fitted on an explicit index set; remembers which indices it saw (for leak checks)."""

    def __init__(self, clip: float = 10.0):
        self.clip = clip
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.fit_idx_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, idx: np.ndarray) -> "Scaler":
        self.fit_idx_ = np.asarray(idx).copy()
        sub = X[self.fit_idx_]
        self.mean_ = sub.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = sub.std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Scaler not fitted"
        out = (X - self.mean_) / self.std_
        return np.clip(out, -self.clip, self.clip).astype(np.float32)
