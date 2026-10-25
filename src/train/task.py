"""Containers handed to the trainers: a graph with features/labels and a train/val/test task."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import scipy.sparse as sp


@dataclass
class GraphData:
    """One graph. ``adj`` is row-normalised (mean aggregation); rows = target nodes."""

    x: np.ndarray            # (n, d) float32, already standardised with the TRAIN scaler
    y: np.ndarray            # (n,) int64 binary label
    adj: sp.csr_matrix       # (n, n)
    node_ids: np.ndarray     # global dataset row ids of the nodes (for leakage checks / attack labels)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return self.x.shape[0]

    @property
    def adj_t(self) -> sp.csr_matrix:
        if "adj_t" not in self._cache:
            self._cache["adj_t"] = self.adj.T.tocsr()
        return self._cache["adj_t"]

    @property
    def ax(self) -> np.ndarray:
        """Mean of neighbour features (constant, computed once)."""
        if "ax" not in self._cache:
            self._cache["ax"] = np.asarray(self.adj @ self.x, dtype=np.float32)
        return self._cache["ax"]


@dataclass
class Task:
    """graphs['train'|'val'|'test'] may be the *same object* (transductive) or distinct graphs (inductive).
    idx[...] are local indices (into the corresponding graph) of the nodes belonging to that split."""

    graphs: Dict[str, GraphData]
    idx: Dict[str, np.ndarray]
    info: Dict[str, object] = field(default_factory=dict)

    def y(self, part: str) -> np.ndarray:
        return self.graphs[part].y[self.idx[part]]

    def global_ids(self, part: str) -> np.ndarray:
        return self.graphs[part].node_ids[self.idx[part]]


def class_weights(y: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """Inverse-frequency class weights (mean-1 convention): w_c = n / (C * n_c)."""
    cnt = np.bincount(y, minlength=n_classes).astype(np.float64)
    cnt[cnt == 0] = 1.0
    return (len(y) / (n_classes * cnt)).astype(np.float32)
