"""Controlled overlap variation: edge-level knob (host-level knob lives in data.splitting)."""
from __future__ import annotations

import numpy as np


def drop_cross_split_edges(src: np.ndarray, dst: np.ndarray, split_of_node: np.ndarray,
                           keep_frac: float, rng: np.random.Generator):
    """Keep every within-split edge and a random ``keep_frac`` of the cross-split edges.

    The coin is flipped per *undirected* pair so that (u,v) and (v,u) are kept or dropped together.
    """
    if keep_frac >= 1.0:
        return src, dst
    n = len(split_of_node)
    cross = split_of_node[src] != split_of_node[dst]
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    pair = lo.astype(np.int64) * n + hi
    uniq, inv = np.unique(pair, return_inverse=True)
    coin = rng.random(len(uniq)) < keep_frac
    keep = (~cross) | coin[inv]
    return src[keep], dst[keep]
