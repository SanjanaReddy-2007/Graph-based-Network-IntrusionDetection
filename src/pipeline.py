"""Glue: splits per protocol -> Task (graphs + features) with leakage checks and overlap metrics."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .data.dataset import Dataset
from .data.features import Scaler
from .data.splitting import host_grouped_split, stratified_random_split, temporal_split
from .eval import leakage as lk
from .graph.construction import cached_arcs, to_adj
from .graph.disruption import disrupt
from .overlap.metrics import compute_overlap
from .overlap.variation import drop_cross_split_edges
from .train.task import GraphData, Task
from .utils import get_logger

log = get_logger(__name__)
PARTS = ("train", "val", "test")
PROTOCOLS = {"P1": "transductive", "P2": "inductive", "P3": "inductive"}


def make_split(ds: Dataset, protocol: str, cfg: dict, seed: int) -> Dict[str, np.ndarray]:
    """P1 and P2 share the SAME stratified random node split (so only the graph handling differs)."""
    pc = cfg["protocols"]
    if protocol in ("P1", "P2"):
        split = stratified_random_split(ds.y, pc["split"], seed)
    elif protocol == "P3":
        tc = pc["temporal"]
        split = temporal_split(ds.day, ds.t, tc["train_days"], tc["test_days"], tc["val_fraction"])
        lk.assert_temporal_order(ds.t, split)
    else:
        raise ValueError(protocol)
    lk.assert_disjoint(split)
    return split


def build_task(
    ds: Dataset, split: Dict[str, np.ndarray], mode: str, cfg: dict, seed: int,
    topology: str = "original", cross_keep: Optional[float] = None, overlap: bool = True,
) -> Task:
    """mode='transductive': one graph over all split nodes, whole graph visible during message passing.
    mode='inductive'   : split first, then one graph per split built from that split's nodes only.
    Overlap metrics are computed on the *reference* graph (the full graph a transductive protocol would
    expose) and on the *visible* graph (the edges the model really uses)."""
    gcfg = cfg["graph"]
    lk.assert_disjoint(split)
    scaler = Scaler().fit(ds.X_log, split["train"])
    lk.assert_scaler_train_only(scaler, split["train"])
    X = scaler.transform(ds.X_log)
    y = ds.y

    U = np.unique(np.concatenate([split[p] for p in PARTS]))
    split_of_U = np.full(len(U), -1, dtype=np.int8)
    for code, p in enumerate(PARTS):
        split_of_U[np.searchsorted(U, split[p])] = code
    ref_s, ref_d = cached_arcs(ds, U, gcfg, cfg)
    rng = np.random.default_rng(seed + 1234)

    graphs: Dict[str, GraphData] = {}
    idx: Dict[str, np.ndarray] = {}
    if mode == "transductive":
        s, d = ref_s, ref_d
        if cross_keep is not None and cross_keep < 1.0:
            s, d = drop_cross_split_edges(s, d, split_of_U, cross_keep, rng)
        s, d = disrupt(s, d, len(U), topology, seed + 11)
        G = GraphData(X[U], y[U], to_adj(len(U), s, d), U)
        for p in PARTS:
            graphs[p] = G
            idx[p] = np.searchsorted(U, split[p])
        vis_s, vis_d = s, d
    elif mode == "inductive":
        if cross_keep is not None and cross_keep < 1.0:
            raise ValueError("cross_keep only applies to the transductive mode")
        vs, vd = [], []
        for code, p in enumerate(PARTS):
            nodes = split[p]
            s, d = cached_arcs(ds, nodes, gcfg, cfg)
            s, d = disrupt(s, d, len(nodes), topology, seed + 11 + code)
            lk.assert_graph_nodes_in_split(nodes, split[p], p)
            graphs[p] = GraphData(X[nodes], y[nodes], to_adj(len(nodes), s, d), nodes)
            idx[p] = np.arange(len(nodes))
            vs.append(np.searchsorted(U, nodes[s]))
            vd.append(np.searchsorted(U, nodes[d]))
        vis_s, vis_d = np.concatenate(vs), np.concatenate(vd)
        lk.assert_no_cross_split_arcs(vis_s, vis_d, split_of_U)
    else:
        raise ValueError(mode)

    task = Task(graphs=graphs, idx=idx, info={"mode": mode, "topology": topology, "cross_keep": cross_keep,
                                              "n_nodes": int(len(U)), "n_ref_arcs": int(len(ref_s))})
    if overlap:
        sub = dict(src_id=ds.src_id[U], dst_id=ds.dst_id[U], y=y[U], flow_key=ds.flow_key[U])
        ref, per_node = compute_overlap(ref_s, ref_d, split_of_U, two_hop_sample=cfg["overlap"].get("two_hop_sample", 0),
                                        seed=seed, **sub)
        vis, per_node_vis = compute_overlap(vis_s, vis_d, split_of_U, two_hop_sample=0, seed=seed, **sub)
        task.info["overlap_ref"] = ref
        task.info["overlap_vis"] = vis
        task.info["per_node_one_hop_ref"] = per_node
        task.info["per_node_one_hop_vis"] = per_node_vis
    return task
