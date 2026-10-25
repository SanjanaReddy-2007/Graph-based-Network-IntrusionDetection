"""PyTorch MLP / GraphSAGE (no PyG needed): mean aggregation = sparse matmul with the row-normalised adjacency.

Full-batch training on the (sub-sampled) flow graph. This is mathematically the full-neighbourhood
GraphSAGE-mean model (up to the per-node degree cap used in graph construction), without the
sampling noise of NeighborLoader, and it is the same recipe as the numpy back-end.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..train.task import GraphData, Task, class_weights
from .param_matching import match_mlp_hidden


def _torch():
    import torch  # local import: the numpy backend must work without torch

    return torch


def resolve_device(pref: str = "auto"):
    torch = _torch()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def build_net(kind: str, d: int, h: int, c: int, dropout: float):
    torch = _torch()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.s1 = nn.Linear(d, h)
            self.s2 = nn.Linear(h, h)
            if kind == "sage":  # separate neighbour weights (no bias): SAGEConv(mean) semantics
                self.n1 = nn.Linear(d, h, bias=False)
                self.n2 = nn.Linear(h, h, bias=False)
            self.out = nn.Linear(h, c)
            self.drop = nn.Dropout(dropout)

        def forward(self, g: Dict):
            if self.kind == "sage":
                h1 = torch.relu(self.s1(g["x"]) + self.n1(g["ax"]))
                h2 = torch.relu(self.s2(h1) + self.n2(torch.sparse.mm(g["A"], h1)))
            else:
                h1 = torch.relu(self.s1(g["x"]))
                h2 = torch.relu(self.s2(h1))
            return self.out(self.drop(h2))

    return Net()


def _to_torch_graph(G: GraphData, device) -> Dict:
    torch = _torch()
    coo = G.adj.tocoo()
    idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    A = torch.sparse_coo_tensor(idx, torch.from_numpy(coo.data.astype(np.float32)), size=G.adj.shape).coalesce().to(device)
    return {"x": torch.from_numpy(G.x).to(device), "ax": torch.from_numpy(G.ax).to(device), "A": A,
            "y": torch.from_numpy(G.y.astype(np.int64)).to(device)}


def fit_predict_torch(kind: str, task: Task, cfg: dict, seed: int) -> dict:
    torch = _torch()
    F = torch.nn.functional
    tcfg, mcfg = cfg["train"], cfg["model"]
    device = resolve_device(tcfg.get("device", "auto"))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    d = task.graphs["train"].x.shape[1]
    h = mcfg["hidden"]
    if kind == "mlp":
        h, _, _ = match_mlp_hidden(d, h, 2, mcfg.get("param_tol", 0.05))
    net = build_net(kind, d, h, 2, mcfg["dropout"]).to(device)
    n_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    cache: Dict[int, Dict] = {}
    def tg(G: GraphData) -> Dict:
        if id(G) not in cache:
            cache[id(G)] = _to_torch_graph(G, device)
        return cache[id(G)]

    gtr, gva, gte = tg(task.graphs["train"]), tg(task.graphs["val"]), tg(task.graphs["test"])
    itr = torch.from_numpy(task.idx["train"].astype(np.int64)).to(device)
    iva = torch.from_numpy(task.idx["val"].astype(np.int64)).to(device)
    ite = torch.from_numpy(task.idx["test"].astype(np.int64)).to(device)
    w = torch.from_numpy(class_weights(task.y("train"), 2)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    best, best_state, wait, ep_run = float("inf"), None, 0, 0
    for ep in range(tcfg["epochs"]):
        net.train()
        opt.zero_grad()
        loss = F.cross_entropy(net(gtr)[itr], gtr["y"][itr], weight=w)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vloss = F.cross_entropy(net(gva)[iva], gva["y"][iva], weight=w).item()
        ep_run = ep + 1
        if vloss < best - 1e-5:
            best, wait = vloss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= tcfg["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pv = torch.softmax(net(gva)[iva], dim=1)[:, 1].cpu().numpy()
        pt = torch.softmax(net(gte)[ite], dim=1)[:, 1].cpu().numpy()
    return {"prob_val": pv, "prob_test": pt, "n_params": n_params, "hidden": h, "epochs": ep_run,
            "best_val_loss": best, "backend": f"torch-{device.type}"}
