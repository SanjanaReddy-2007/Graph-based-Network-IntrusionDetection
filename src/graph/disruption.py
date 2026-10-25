"""Topology disruption for the graph-structure ablation. Node features / labels are never touched
(except through relabelling for 'shuffle'), so model capacity and inputs are identical across variants."""
from __future__ import annotations

import numpy as np

from .construction import Arcs

TOPOLOGIES = ("original", "rewire", "random", "shuffle", "none")


def _dedup_no_loops(src: np.ndarray, dst: np.ndarray, n: int) -> Arcs:
    m = src != dst
    src, dst = src[m], dst[m]
    code = np.unique(src.astype(np.int64) * n + dst)
    return code // n, code % n


def disrupt(src: np.ndarray, dst: np.ndarray, n: int, method: str, seed: int) -> Arcs:
    """Return a disrupted copy of the arcs.

    original : unchanged
    rewire   : degree-preserving randomisation (configuration model): the destinations of all arcs are
               permuted, so every node keeps its out- and in-degree (up to removal of self-loops and
               duplicate arcs); edges no longer connect flows that truly share an entity.
    random   : same number of arcs, endpoints uniformly random (Erdos-Renyi like).
    shuffle  : keeps the exact topology but relabels nodes with a random permutation, i.e. every
               neighbourhood is attached to the wrong flow.
    none     : empty graph (SAGE reduces to self-weights only -> MLP-like).
    """
    rng = np.random.default_rng(seed)
    if method == "original":
        return src, dst
    if method == "none" or len(src) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if method == "rewire":
        return _dedup_no_loops(src, dst[rng.permutation(len(dst))], n)
    if method == "random":
        m = len(src)
        return _dedup_no_loops(rng.integers(0, n, m), rng.integers(0, n, m), n)
    if method == "shuffle":
        perm = rng.permutation(n)
        return perm[src], perm[dst]
    raise ValueError(f"unknown topology '{method}', choose from {TOPOLOGIES}")
