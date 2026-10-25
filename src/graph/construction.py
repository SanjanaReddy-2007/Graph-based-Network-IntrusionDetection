"""Flow-level graph construction.

Node = one flow (row of the chronologically sorted frame). Two flows are linked when they share an
*endpoint entity* (same Source IP, same Destination IP, or same (src,dst) pair) and start within
``window_seconds`` of each other. Per entity type each flow is linked to its ``per_type`` most recent
predecessors (vectorised sorted-window grouping, O(n * per_type), never O(n^2)); the union over
entity types is symmetrised, de-duplicated, and finally each node keeps its ``max_degree``
nearest-in-time neighbours.

Arcs are stored as (src, dst) arrays meaning "src is an in-neighbour of dst" (information flows
src -> dst). The cap is applied on in-arcs, so the cap can leave a few asymmetric pairs.
CICIDS2017 timestamps have *minute* resolution, hence delta-t values are multiples of 60 s.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from ..utils import get_logger, resolve_path, stable_hash

log = get_logger(__name__)
Arcs = Tuple[np.ndarray, np.ndarray]


def build_arcs(
    t: np.ndarray,
    src_ip: np.ndarray,
    dst_ip: np.ndarray,
    window: float = 300.0,
    per_type: int = 5,
    max_degree: int = 20,
    entity_types: Sequence[str] = ("src", "dst", "pair"),
    seed: int = 0,
) -> Arcs:
    """Build (src, dst) in-arcs for nodes already sorted by time ``t``. Local indices 0..n-1."""
    n = len(t)
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if n < 2:
        return empty
    assert (np.diff(t) >= 0).all(), "nodes must be sorted by time"
    src_ip = src_ip.astype(np.int64)
    dst_ip = dst_ip.astype(np.int64)
    n_ip = int(max(src_ip.max(), dst_ip.max())) + 1
    keys = []
    for et in entity_types:
        if et == "src":
            keys.append(src_ip)
        elif et == "dst":
            keys.append(dst_ip)
        elif et == "pair":
            keys.append(src_ip * n_ip + dst_ip)
        else:
            raise ValueError(f"unknown entity type {et}")
    us, vs = [], []
    for key in keys:
        order = np.argsort(key, kind="stable")  # stable => time-ordered inside every group
        ks, ts = key[order], t[order]
        for d in range(1, min(per_type, n - 1) + 1):
            ok = (ks[d:] == ks[:-d]) & ((ts[d:] - ts[:-d]) <= window)
            us.append(order[d:][ok])
            vs.append(order[:-d][ok])
    if not us:
        return empty
    u, v = np.concatenate(us), np.concatenate(vs)
    if len(u) == 0:
        return empty
    a = np.concatenate([u, v])
    b = np.concatenate([v, u])
    code = np.unique(a * n + b)  # symmetrise + de-duplicate
    a, b = code // n, code % n
    # degree cap on in-arcs: keep the max_degree nearest-in-time neighbours of every node
    dt = np.abs(t[a] - t[b])
    rnd = np.random.default_rng(seed).random(len(a))  # random tie-break (many equal dt: minute resolution)
    order = np.lexsort((rnd, dt, b))
    a, b = a[order], b[order]
    first = np.ones(len(b), dtype=bool)
    first[1:] = b[1:] != b[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(len(b)), 0))
    rank = np.arange(len(b)) - start
    keep = rank < max_degree
    return a[keep].astype(np.int64), b[keep].astype(np.int64)


def to_adj(n: int, src: np.ndarray, dst: np.ndarray, normalise: bool = True) -> sp.csr_matrix:
    """Row-normalised (mean-aggregation) adjacency: row i averages over the in-neighbours of node i."""
    if len(src) == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    if normalise:
        deg = np.bincount(dst, minlength=n).astype(np.float64)
        w = (1.0 / deg[dst]).astype(np.float32)
    else:
        w = np.ones(len(src), dtype=np.float32)
    return sp.csr_matrix((w, (dst, src)), shape=(n, n), dtype=np.float32)


def cached_arcs(ds, nodes: np.ndarray, gcfg: dict, cfg: dict) -> Arcs:
    """Graph over the (sorted) global node subset ``nodes``; arcs use local indices 0..len(nodes)-1."""
    nodes = np.asarray(nodes)
    assert (np.diff(nodes) > 0).all(), "node subset must be strictly increasing (== chronological)"
    cache_dir = resolve_path(cfg, "cache_dir")
    key = stable_hash(nodes.astype(np.int64), gcfg, ds.sig)
    f = cache_dir / f"arcs_{key}.npz"
    if f.exists():
        z = np.load(f)
        return z["src"], z["dst"]
    src, dst = build_arcs(
        ds.t[nodes], ds.src_id[nodes], ds.dst_id[nodes],
        window=gcfg["window_seconds"], per_type=gcfg["neighbors_per_type"],
        max_degree=gcfg["max_degree"], entity_types=gcfg["entity_types"], seed=0,
    )
    np.savez_compressed(f, src=src, dst=dst)
    return src, dst
