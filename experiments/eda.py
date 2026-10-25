"""Exploratory data analysis on the processed data: class balance per day, host statistics, graph statistics."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset import DAY_ORDER, load_dataset
from src.graph.construction import cached_arcs
from src.utils import resolve_path, save_json

from .common import log


def main(cfg: dict) -> None:
    ds = load_dataset(cfg)
    out = resolve_path(cfg, "results_dir")
    m = ds.meta
    counts = m.groupby(["day", "label"]).size().unstack(fill_value=0).reindex(DAY_ORDER).fillna(0).astype(int)
    counts.to_csv(out / "eda_class_counts_per_day.csv")

    nodes = np.arange(ds.n)
    src, dst = cached_arcs(ds, nodes, cfg["graph"], cfg)
    deg = np.bincount(dst, minlength=ds.n)
    y = ds.y
    stats = {
        "n_flows": int(ds.n), "n_features": len(ds.feature_cols), "n_ips": len(ds.ip_table),
        "attack_fraction": float(y.mean()), "n_arcs": int(len(src)),
        "mean_degree": float(deg.mean()), "median_degree": float(np.median(deg)), "isolated_fraction": float((deg == 0).mean()),
        "edge_homophily": float((y[src] == y[dst]).mean()) if len(src) else None,
        "attack_nbr_purity": float((y[src[y[dst] == 1]] == 1).mean()) if len(src) else None,
        "flows_per_day": m.day.value_counts().reindex(DAY_ORDER).fillna(0).astype(int).to_dict(),
        "distinct_flow_ids": int(m.flow_key.nunique()),
        "attack_hosts_top5": [(ds.ip_table[i], int(c)) for i, c in
                              pd.Series(ds.src_id[y == 1]).value_counts().head(5).items()],
    }
    save_json(stats, out / "eda_summary.json")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    counts.plot(kind="bar", stacked=True, ax=ax[0], legend=False, width=0.8)
    ax[0].set_yscale("log"); ax[0].set_ylabel("flows (log)"); ax[0].set_title("Flows per day and label (after sub-sampling)")
    ax[0].legend(fontsize=6, ncol=2, loc="upper right")
    ax[1].hist(np.minimum(deg, deg.max()), bins=np.arange(0, deg.max() + 2) - 0.5, color="#4C72B0")
    ax[1].set_xlabel("node degree"); ax[1].set_ylabel("flows"); ax[1].set_title("Flow-graph degree distribution")
    fig.tight_layout()
    fig.savefig(out / "fig_eda.png", dpi=200)
    plt.close(fig)
    log.info("EDA written to %s (%s)", out, {k: stats[k] for k in ("n_flows", "n_ips", "mean_degree", "edge_homophily")})
