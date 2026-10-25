"""Node splits: stratified random (P1/P2), chronological (P3) and host-grouped (overlap sweep)."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from ..utils import get_logger

log = get_logger(__name__)
Split = Dict[str, np.ndarray]


def stratified_random_split(y: np.ndarray, fracs: Dict[str, float], seed: int) -> Split:
    """Stratified train/val/test split of node indices (each array sorted ascending = time order)."""
    n = len(y)
    idx = np.arange(n)
    rest = fracs["val"] + fracs["test"]
    tr, tmp = train_test_split(idx, test_size=rest, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=fracs["test"] / rest, stratify=y[tmp], random_state=seed)
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}


def temporal_split(day: np.ndarray, t: np.ndarray, train_days: Sequence[str], test_days: Sequence[str],
                   val_fraction: float) -> Split:
    """Train = earliest (1-val_fraction) of the train days, val = the latest slice, test = later days."""
    tr_all = np.where(np.isin(day, list(train_days)))[0]
    te = np.where(np.isin(day, list(test_days)))[0]
    if len(tr_all) == 0 or len(te) == 0:
        raise ValueError(f"Temporal split empty: train_days={train_days}, test_days={test_days}, "
                         f"days present={sorted(set(day))}")
    tr_all = tr_all[np.argsort(t[tr_all], kind="stable")]
    cut = int(round(len(tr_all) * (1.0 - val_fraction)))
    tr, va = tr_all[:cut], tr_all[cut:]
    if t[te].min() < t[va].max():
        raise ValueError("Test days must be strictly later than train days.")
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}


def host_grouped_split(
    src_id: np.ndarray,
    dst_id: np.ndarray,
    y: np.ndarray,
    target_overlap: float,
    seed: int,
    test_frac: float = 0.25,
    val_frac: float = 0.15,
    heavy_frac: float = 0.005,
    n_q_grid: int = 30,
) -> Dict[str, object]:
    """Split flows so that a controllable share of *test-side hosts* is also seen in training.

    Hosts (IPs, both endpoints of each flow) are partitioned into
      T  train-only hosts,  S  test-only hosts,  B  shared hosts (seen on both sides),
    with |B| / (|S|+|B|) = ``target_overlap`` among the *light* hosts. "Heavy" hosts
    (>= ``heavy_frac`` of all flows: the attacker, the victim server, DNS, ...) are always shared;
    CICIDS2017 has only a handful of attacker IPs, so disjoint attacker hosts would remove an entire
    class from one side. Because of heavy hosts, the achieved overlap can never reach 0 -- the
    *achieved* values are measured separately by ``overlap.metrics`` and are what we report.

    A flow is eligible for train iff none of its light endpoints is in S; eligible for test iff none is
    in T. Flows eligible for neither are dropped. Flows eligible for both go to test w.p. ``test_frac``.
    The fraction of test-side light hosts ``q`` is grid-searched so that the realised test share of the
    retained flows is close to ``test_frac``.
    """
    n = len(src_id)
    rng = np.random.default_rng(seed)
    n_ip = int(max(src_id.max(), dst_id.max())) + 1
    cnt = np.bincount(np.concatenate([src_id, dst_id]), minlength=n_ip)
    heavy = cnt >= heavy_frac * n
    light = np.where((cnt > 0) & (~heavy))[0]
    rng.shuffle(light)
    u = rng.random(n)  # fixed per-flow uniform draws -> assignment monotone in q
    # role code per host: 0 = T, 1 = S, 2 = B, 3 = heavy (shared)
    best = None
    for q in np.linspace(0.02, 1.0, n_q_grid):
        n_test_side = int(round(q * len(light)))
        n_shared = int(round(target_overlap * n_test_side))
        role = np.zeros(n_ip, dtype=np.int8)
        role[heavy] = 3
        role[light[:n_shared]] = 2
        role[light[n_shared:n_test_side]] = 1
        rs, rd = role[src_id], role[dst_id]
        elig_train = (rs != 1) & (rd != 1)
        elig_test = (rs != 0) & (rd != 0)
        to_test = elig_test & (~elig_train | (u < test_frac))
        to_train = elig_train & (~to_test)
        kept = to_test.sum() + to_train.sum()
        if kept == 0 or to_test.sum() == 0:
            continue
        share = to_test.sum() / kept
        score = abs(share - test_frac) + 0.25 * (n - kept) / n  # prefer little dropping
        if best is None or score < best[0]:
            best = (score, q, to_train.copy(), to_test.copy(), n_shared, n_test_side - n_shared)
    if best is None:
        raise ValueError("host_grouped_split: could not realise a non-empty test set.")
    _, q, to_train, to_test, n_shared, n_s = best
    tr_all = np.where(to_train)[0]
    te = np.where(to_test)[0]
    if y[te].min() == y[te].max() or y[tr_all].min() == y[tr_all].max():
        raise ValueError("host_grouped_split: a split contains a single class; try another seed / level.")
    # validation = random slice of the train side (hosts remain seen, same as in conventional practice)
    va_mask = rng.random(len(tr_all)) < val_frac
    return {
        "train": np.sort(tr_all[~va_mask]),
        "val": np.sort(tr_all[va_mask]),
        "test": np.sort(te),
        "dropped_frac": float(1 - (len(tr_all) + len(te)) / n),
        "q_test_side": float(q),
        "n_shared_light_hosts": int(n_shared),
        "n_test_only_hosts": int(n_s),
        "n_heavy_hosts": int(heavy.sum()),
    }
