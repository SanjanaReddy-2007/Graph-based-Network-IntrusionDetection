"""Pure-numpy MLP / GraphSAGE with hand-written back-propagation (CPU fallback + test oracle).

Same architecture and training recipe as ``nn_torch``: Adam, class-weighted cross-entropy,
full-batch training, early stopping on the validation loss, dropout before the classifier head.
Mean aggregation uses the pre-normalised sparse adjacency ``G.adj`` (rows = target nodes).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..train.task import GraphData, Task, class_weights
from .param_matching import match_mlp_hidden


class NumpyNet:
    def __init__(self, kind: str, d: int, h: int, c: int, dropout: float, seed: int):
        assert kind in ("mlp", "sage")
        self.kind, self.dropout = kind, dropout
        rng = np.random.default_rng(seed)

        def glorot(i: int, o: int) -> np.ndarray:
            lim = np.sqrt(6.0 / (i + o))
            return rng.uniform(-lim, lim, (i, o)).astype(np.float32)

        P: Dict[str, np.ndarray] = {"Ws1": glorot(d, h), "b1": np.zeros(h, np.float32),
                                    "Ws2": glorot(h, h), "b2": np.zeros(h, np.float32),
                                    "Wo": glorot(h, c), "bo": np.zeros(c, np.float32)}
        if kind == "sage":
            P["Wn1"], P["Wn2"] = glorot(d, h), glorot(h, h)
        self.P = P

    @property
    def n_params(self) -> int:
        return int(sum(v.size for v in self.P.values()))

    def forward(self, G: GraphData, train: bool = False, rng: np.random.Generator | None = None):
        P, sage = self.P, self.kind == "sage"
        pre1 = G.x @ P["Ws1"] + P["b1"]
        if sage:
            pre1 = pre1 + G.ax @ P["Wn1"]
        H1 = np.maximum(pre1, 0)
        pre2 = H1 @ P["Ws2"] + P["b2"]
        AH1 = None
        if sage:
            AH1 = G.adj @ H1
            pre2 = pre2 + AH1 @ P["Wn2"]
        H2 = np.maximum(pre2, 0)
        mask = None
        H2d = H2
        if train and self.dropout > 0:
            mask = ((rng.random(H2.shape) >= self.dropout) / (1.0 - self.dropout)).astype(np.float32)
            H2d = H2 * mask
        Z = H2d @ P["Wo"] + P["bo"]
        return Z, (H1, AH1, H2, H2d, mask)

    def backward(self, G: GraphData, cache, dZ: np.ndarray) -> Dict[str, np.ndarray]:
        P, sage = self.P, self.kind == "sage"
        H1, AH1, H2, H2d, mask = cache
        g: Dict[str, np.ndarray] = {}
        g["Wo"], g["bo"] = H2d.T @ dZ, dZ.sum(0)
        dH2 = dZ @ P["Wo"].T
        if mask is not None:
            dH2 = dH2 * mask
        dpre2 = dH2 * (H2 > 0)
        g["Ws2"], g["b2"] = H1.T @ dpre2, dpre2.sum(0)
        dH1 = dpre2 @ P["Ws2"].T
        if sage:
            g["Wn2"] = AH1.T @ dpre2
            dH1 = dH1 + G.adj_t @ (dpre2 @ P["Wn2"].T)
        dpre1 = dH1 * (H1 > 0)
        g["Ws1"], g["b1"] = G.x.T @ dpre1, dpre1.sum(0)
        if sage:
            g["Wn1"] = G.ax.T @ dpre1
        return g


def weighted_ce(Z: np.ndarray, y: np.ndarray, idx: np.ndarray, w: np.ndarray, want_grad: bool):
    """Class-weighted mean cross-entropy over ``idx``. Returns (loss, dZ or None)."""
    Zt = Z[idx].astype(np.float64)
    Zt = Zt - Zt.max(1, keepdims=True)
    P = np.exp(Zt)
    P /= P.sum(1, keepdims=True)
    yt = y[idx]
    wi = w[yt]
    loss = float(-(wi * np.log(P[np.arange(len(idx)), yt] + 1e-12)).sum() / wi.sum())
    if not want_grad:
        return loss, None
    d = P.copy()
    d[np.arange(len(idx)), yt] -= 1.0
    d *= (wi / wi.sum())[:, None]
    dZ = np.zeros(Z.shape, dtype=np.float32)
    dZ[idx] = d
    return loss, dZ


class Adam:
    def __init__(self, params: Dict[str, np.ndarray], lr: float, wd: float, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.wd, self.b1, self.b2, self.eps, self.t = lr, wd, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k, p in params.items():
            g = grads[k] + self.wd * p
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(p.dtype)


def softmax_attack_prob(Z: np.ndarray) -> np.ndarray:
    Z = Z.astype(np.float64)
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z)
    return (P[:, 1] / P.sum(1)).astype(np.float32)


def fit_predict_numpy(kind: str, task: Task, cfg: dict, seed: int) -> dict:
    tcfg, mcfg = cfg["train"], cfg["model"]
    d = task.graphs["train"].x.shape[1]
    h = mcfg["hidden"]
    if kind == "mlp":
        h, _, _ = match_mlp_hidden(d, h, 2, mcfg.get("param_tol", 0.05))
    net = NumpyNet(kind, d, h, 2, mcfg["dropout"], seed)
    rng = np.random.default_rng(seed + 7)
    Gtr, Gva, Gte = task.graphs["train"], task.graphs["val"], task.graphs["test"]
    ytr, yva = Gtr.y, Gva.y
    itr, iva, ite = task.idx["train"], task.idx["val"], task.idx["test"]
    w = class_weights(ytr[itr], 2)
    opt = Adam(net.P, tcfg["lr"], tcfg["weight_decay"])
    best, best_P, wait, ep_run = np.inf, None, 0, 0
    for ep in range(tcfg["epochs"]):
        Z, cache = net.forward(Gtr, True, rng)
        _, dZ = weighted_ce(Z, ytr, itr, w, True)
        opt.step(net.P, net.backward(Gtr, cache, dZ))
        Zv, _ = net.forward(Gva)
        vloss, _ = weighted_ce(Zv, yva, iva, w, False)
        ep_run = ep + 1
        if vloss < best - 1e-5:
            best, best_P, wait = vloss, {k: v.copy() for k, v in net.P.items()}, 0
        else:
            wait += 1
            if wait >= tcfg["patience"]:
                break
    net.P = best_P
    Zv, _ = net.forward(Gva)
    Zt, _ = net.forward(Gte)
    return {"prob_val": softmax_attack_prob(Zv[iva]), "prob_test": softmax_attack_prob(Zt[ite]),
            "n_params": net.n_params, "hidden": h, "epochs": ep_run, "best_val_loss": best, "backend": "numpy"}
