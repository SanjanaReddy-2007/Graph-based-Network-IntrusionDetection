"""Publication-style figures built from the CSV/NPZ logs in results/."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import resolve_path

from .common import log

COL = {"rf": "#8C8C8C", "mlp": "#DD8452", "sage": "#4C72B0"}
NAME = {"rf": "Random Forest", "mlp": "MLP (capacity-matched)", "sage": "GraphSAGE"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    log.info("saved %s", name)


def fig_protocols(out: Path):
    f = out / "protocols_summary.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, met, title in zip(axes, ["macro_f1", "pr_auc", "attack_recall"], ["Macro-F1", "PR-AUC", "Attack recall"]):
        protos = list(d.protocol.unique())
        w = 0.26
        for i, m in enumerate(["rf", "mlp", "sage"]):
            s = d[d.model == m].set_index("protocol").reindex(protos)
            ax.bar(np.arange(len(protos)) + (i - 1) * w, s[f"{met}_mean"], w, yerr=s[f"{met}_std"].fillna(0),
                   color=COL[m], label=NAME[m], capsize=2)
        ax.set_xticks(range(len(protos)))
        ax.set_xticklabels(protos)
        ax.set_title(title); ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Model performance under P1 (random/transductive), P2 (leakage-safe inductive), P3 (chronological); mean ± std over seeds", fontsize=9)
    _save(fig, out, "fig_protocols")


def fig_overlap_gap(out: Path):
    f = out / "sweep_gaps.csv"
    if not f.exists():
        return
    g = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    spec = {"host": ("achieved_ref_host_containment", "Achieved host overlap (share of test hosts seen in train)"),
            "edge": ("achieved_vis_one_hop_mean", "Achieved visible 1-hop train-neighbour fraction")}
    for ax, knob in zip(axes, ["host", "edge"]):
        s = g[g.knob == knob].sort_values(spec[knob][0])
        if s.empty:
            continue
        x = s[spec[knob][0]].to_numpy()
        ax.plot(x, s.macro_f1_mean_gap, "o-", color=COL["sage"], label="macro-F1 gap")
        ax.fill_between(x, s.macro_f1_ci_lo, s.macro_f1_ci_hi, color=COL["sage"], alpha=0.25, label="95% CI")
        if "pr_auc_mean_gap" in s:
            ax.plot(x, s.pr_auc_mean_gap, "s--", color=COL["mlp"], label="PR-AUC gap")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(spec[knob][1]); ax.set_ylabel("GraphSAGE − MLP")
        ax.set_title("Host-grouped splits" if knob == "host" else "Cross-split edge keep-fraction")
    axes[0].legend(fontsize=7)
    _save(fig, out, "fig_overlap_gap")


def fig_ablation(out: Path):
    f = out / "ablation_summary.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    protos = list(d.protocol.unique())
    models = list(d.model_topology.unique())
    order = ["mlp"] + [m for m in ["sage:original", "sage:rewire", "sage:random", "sage:shuffle", "sage:none"] if m in models]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    w = 0.8 / len(order)
    for i, m in enumerate(order):
        s = d[d.model_topology == m].set_index("protocol").reindex(protos)
        ax.bar(np.arange(len(protos)) + (i - len(order) / 2 + 0.5) * w, s.macro_f1_mean, w, yerr=s.macro_f1_std.fillna(0),
               label=m, color=cmap(i), capsize=2)
    ax.set_xticks(range(len(protos))); ax.set_xticklabels(protos)
    ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 1.02); ax.legend(fontsize=7, ncol=3)
    ax.set_title("Graph-structure ablation (same features, same capacity; only the topology changes)")
    _save(fig, out, "fig_ablation")


def fig_nbr_dist(out: Path):
    f = out / "protocols_nbr_overlap_dist.npz"
    if not f.exists():
        return
    z = np.load(f)
    keys = [k for k in z.files if k.endswith("_ref")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, suffix, title in zip(axes, ["_ref", "_vis"], ["Reference graph (what a transductive split exposes)",
                                                           "Graph actually visible to the model"]):
        data = [z[k.replace("_ref", suffix)] for k in keys]
        data = [d if len(d) else np.array([0.0]) for d in data]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(COL["sage"]); b.set_alpha(0.5)
        ax.set_xticks(range(1, len(keys) + 1)); ax.set_xticklabels([k.replace("_ref", "") for k in keys])
        ax.set_title(title, fontsize=8); ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("per test node: fraction of neighbours in train")
    _save(fig, out, "fig_nbr_overlap_dist")


def fig_recall_heatmap(out: Path):
    f = out / "protocols_runs.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    rc = [c for c in d.columns if c.startswith("recall::")]
    if not rc:
        return
    m = d.groupby(["protocol", "model"])[rc].mean().T
    m.index = [i.replace("recall::", "") for i in m.index]
    m = m.loc[m.notna().any(axis=1)]
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * m.shape[1], 0.5 + 0.32 * len(m)))
    im = ax.imshow(m.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels([f"{p}\n{mm}" for p, mm in m.columns], fontsize=7)
    ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index, fontsize=7)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.6 else "k", fontsize=6)
    fig.colorbar(im, ax=ax, label="recall (binary detector)")
    ax.set_title("Per-attack-type recall (blank = attack type absent from that protocol's test set)", fontsize=8)
    _save(fig, out, "fig_recall_heatmap")


def main(cfg: dict) -> None:
    out = resolve_path(cfg, "results_dir")
    for fn in (fig_protocols, fig_overlap_gap, fig_ablation, fig_nbr_dist, fig_recall_heatmap):
        try:
            fn(out)
        except Exception as e:  # keep going: one broken figure must not block the rest
            log.warning("figure %s failed: %s", fn.__name__, e)
