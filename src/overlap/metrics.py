"""Train/test structural-overlap metrics (standalone, unit-tested).

Notation: nodes are flows. H(v) = {src_ip(v), dst_ip(v)} is the host set of flow v. S_tr / S_te are the
train / test node sets, H_tr = U_{v in S_tr} H(v), H_te likewise. A (reference) graph is given as arcs
(u -> v) = "u is an in-neighbour of v". ``split_of_node``: 0 train, 1 val, 2 test, -1 unassigned.

 host_jaccard        = |H_tr ∩ H_te| / |H_tr ∪ H_te|
 host_containment    = |H_tr ∩ H_te| / |H_te|              (share of test hosts also seen in training)
 test_seen_any/both  = fraction of test flows with >=1 / both hosts in H_tr
 seen_host_test_nbr  = fraction of test nodes with >=1 *test* neighbour u such that H(u) ∩ H_tr != ∅
 cross_edge_ratio    = |{arcs (u,v): split(u) != split(v)}| / |arcs|       (all assigned nodes)
 test_cross_edge_ratio = among arcs into test nodes, share whose source is not a test node
 one_hop_*           = per test node v with deg(v)>0:  |N(v) ∩ S_tr| / |N(v)|   (mean/median/quartiles)
 test_with_train_nbr = fraction of test nodes with >=1 training neighbour
 two_hop_train_frac  = same ratio over the 2-hop neighbourhood (v excluded), on a random sample
 edge_homophily      = fraction of arcs joining nodes with the same (binary) label
 flow5tuple_seen     = fraction of test flows whose CICFlowMeter 'Flow ID' (5-tuple) also occurs in train
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp

TRAIN, VAL, TEST = 0, 1, 2


def host_overlap(src_id: np.ndarray, dst_id: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray) -> Dict[str, float]:
    h_tr = np.unique(np.concatenate([src_id[train_mask], dst_id[train_mask]]))
    h_te = np.unique(np.concatenate([src_id[test_mask], dst_id[test_mask]]))
    inter = np.intersect1d(h_tr, h_te, assume_unique=True)
    union = np.union1d(h_tr, h_te)
    return {
        "host_jaccard": len(inter) / max(len(union), 1),
        "host_containment": len(inter) / max(len(h_te), 1),
        "n_train_hosts": float(len(h_tr)),
        "n_test_hosts": float(len(h_te)),
    }


def compute_overlap(
    arc_src: np.ndarray,
    arc_dst: np.ndarray,
    split_of_node: np.ndarray,
    src_id: np.ndarray,
    dst_id: np.ndarray,
    y: Optional[np.ndarray] = None,
    flow_key: Optional[np.ndarray] = None,
    two_hop_sample: int = 0,
    seed: int = 0,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Return (scalar metrics, per-test-node 1-hop train-neighbour fractions)."""
    n = len(split_of_node)
    split = split_of_node
    tr, te = split == TRAIN, split == TEST
    if te.sum() == 0 or tr.sum() == 0:
        raise ValueError("compute_overlap needs non-empty train and test sets")
    out: Dict[str, float] = {"n_train": float(tr.sum()), "n_test": float(te.sum()), "n_arcs": float(len(arc_src))}
    out.update(host_overlap(src_id, dst_id, tr, te))

    h_tr = np.unique(np.concatenate([src_id[tr], dst_id[tr]]))
    seen_s, seen_d = np.isin(src_id, h_tr), np.isin(dst_id, h_tr)
    seen_any, seen_both = seen_s | seen_d, seen_s & seen_d
    out["test_seen_any"] = float(seen_any[te].mean())
    out["test_seen_both"] = float(seen_both[te].mean())

    s_src, s_dst = split[arc_src], split[arc_dst]
    both_te = (s_src == TEST) & (s_dst == TEST)
    flag = np.bincount(arc_dst[both_te], weights=seen_any[arc_src[both_te]].astype(float), minlength=n) > 0
    out["seen_host_test_nbr"] = float(flag[te].mean())

    assigned = (s_src >= 0) & (s_dst >= 0)
    out["cross_edge_ratio"] = float((s_src[assigned] != s_dst[assigned]).mean()) if assigned.any() else float("nan")
    into_te = assigned & (s_dst == TEST)
    if into_te.any():
        out["test_cross_edge_ratio"] = float((s_src[into_te] != TEST).mean())
        out["train_nbr_arc_ratio"] = float((s_src[into_te] == TRAIN).mean())
    else:
        out["test_cross_edge_ratio"] = out["train_nbr_arc_ratio"] = float("nan")

    deg = np.bincount(arc_dst[into_te], minlength=n)
    trn = np.bincount(arc_dst[into_te & (s_src == TRAIN)], minlength=n)
    deg_te, trn_te = deg[te], trn[te]
    has = deg_te > 0
    frac = np.full(len(deg_te), np.nan)
    frac[has] = trn_te[has] / deg_te[has]
    if has.any():
        f = frac[has]
        out.update(one_hop_mean=float(f.mean()), one_hop_median=float(np.median(f)), one_hop_std=float(f.std()),
                   one_hop_q25=float(np.quantile(f, 0.25)), one_hop_q75=float(np.quantile(f, 0.75)))
    else:
        out.update(one_hop_mean=0.0, one_hop_median=0.0, one_hop_std=0.0, one_hop_q25=0.0, one_hop_q75=0.0)
    out["test_with_train_nbr"] = float((trn_te > 0).mean())
    out["test_isolated"] = float((~has).mean())
    out["mean_test_degree"] = float(deg_te.mean())

    if y is not None and assigned.any():
        out["edge_homophily"] = float((y[arc_src[assigned]] == y[arc_dst[assigned]]).mean())
        att = assigned & (y[arc_dst] == 1)
        out["attack_nbr_purity"] = float((y[arc_src[att]] == 1).mean()) if att.any() else float("nan")
    if flow_key is not None:
        out["flow5tuple_seen"] = float(np.isin(flow_key[te], flow_key[tr]).mean())

    if two_hop_sample and len(arc_src):
        rng = np.random.default_rng(seed)
        te_idx = np.where(te)[0]
        samp = rng.choice(te_idx, size=min(two_hop_sample, len(te_idx)), replace=False)
        A = sp.csr_matrix((np.ones(len(arc_src), dtype=np.float32), (arc_dst, arc_src)), shape=(n, n))
        N1 = A[samp]
        M = ((N1 + N1 @ A) > 0).tocoo()
        keep = M.col != samp[M.row]  # remove the node itself
        rows, cols = M.row[keep], M.col[keep]
        tot = np.bincount(rows, minlength=len(samp))
        trc = np.bincount(rows[split[cols] == TRAIN], minlength=len(samp))
        ok = tot > 0
        out["two_hop_train_frac"] = float((trc[ok] / tot[ok]).mean()) if ok.any() else 0.0
    return out, frac
