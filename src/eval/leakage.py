"""Assertion-based leakage checks. They raise ``LeakageError`` (an AssertionError) so a run can never
silently report numbers from a leaky configuration."""
from __future__ import annotations

from typing import Dict

import numpy as np


class LeakageError(AssertionError):
    pass


def assert_disjoint(split: Dict[str, np.ndarray]) -> None:
    """No flow (node id) belongs to two splits."""
    names = ["train", "val", "test"]
    for i in range(3):
        for j in range(i + 1, 3):
            inter = np.intersect1d(split[names[i]], split[names[j]])
            if len(inter):
                raise LeakageError(f"{len(inter)} nodes shared between {names[i]} and {names[j]}")


def assert_scaler_train_only(scaler, train_idx: np.ndarray) -> None:
    if scaler.fit_idx_ is None or not np.array_equal(np.sort(scaler.fit_idx_), np.sort(train_idx)):
        raise LeakageError("Scaler was not fitted exactly on the training nodes")


def assert_no_cross_split_arcs(arc_src: np.ndarray, arc_dst: np.ndarray, split_of_node: np.ndarray) -> None:
    """Inductive protocols: every message-passing edge stays inside one split."""
    cross = split_of_node[arc_src] != split_of_node[arc_dst]
    if cross.any():
        raise LeakageError(f"{int(cross.sum())} cross-split edges visible to the model in an inductive protocol")


def assert_graph_nodes_in_split(node_ids: np.ndarray, split_ids: np.ndarray, name: str) -> None:
    if not np.isin(node_ids, split_ids).all():
        raise LeakageError(f"graph '{name}' contains nodes outside its split")


def assert_temporal_order(t: np.ndarray, split: Dict[str, np.ndarray]) -> None:
    """Chronological protocol: train < val < test in time (no future information)."""
    if not (t[split["train"]].max() <= t[split["val"]].min() + 1e-9):
        raise LeakageError("validation overlaps the training period")
    if not (t[split["val"]].max() <= t[split["test"]].min() + 1e-9):
        raise LeakageError("test period starts before validation ends")
