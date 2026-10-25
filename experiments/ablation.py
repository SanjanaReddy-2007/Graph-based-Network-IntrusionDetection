"""Experiment 3: graph-structure ablation. Same node features, same model capacity; only the topology of
the graph given to GraphSAGE changes (original / degree-preserving rewire / random / shuffle / none)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset import load_dataset
from src.pipeline import PROTOCOLS, build_task, make_split
from src.utils import resolve_path

from .common import evaluate_models, gap_stats, log, mean_std_table


def main(cfg: dict) -> None:
    ds = load_dataset(cfg)
    out = resolve_path(cfg, "results_dir")
    seeds, topologies = cfg["seeds"], cfg["ablation"]["topologies"]
    runs, preds = [], {}
    for protocol in cfg["protocols"]["list"]:
        for seed in seeds:
            split = make_split(ds, protocol, cfg, seed)
            for topo in topologies:
                log.info("=== ablation %s topology=%s seed=%d ===", protocol, topo, seed)
                task = build_task(ds, split, PROTOCOLS[protocol], cfg, seed, topology=topo, overlap=False)
                use = ["mlp", "sage"] if topo == "original" else ["sage"]
                rows, p = evaluate_models(ds, task, use, cfg, seed, {"protocol": protocol, "topology": topo})
                for r in rows:
                    r["model_topology"] = f"{r['model']}:{topo}" if r["model"] == "sage" else "mlp"
                runs += rows
                for m, v in p.items():
                    preds.setdefault((protocol, "mlp" if m == "mlp" else topo), {}).setdefault(seed, v)
        pd.DataFrame(runs).to_csv(out / "ablation_runs.csv", index=False)
    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(out / "ablation_runs.csv", index=False)
    mean_std_table(runs_df, ["protocol", "model_topology"]).to_csv(out / "ablation_summary.csv", index=False)

    rows = []
    for protocol in cfg["protocols"]["list"]:
        orig = preds.get((protocol, "original"))
        if orig is None:
            continue
        pa = {s: v for s, v in orig.items()}
        for topo in topologies:
            if topo == "original":
                continue
            pb = preds.get((protocol, topo))
            if pb:
                rows.append({"protocol": protocol, "comparison": f"sage:original - sage:{topo}", **gap_stats(pa, pb, cfg)})
        mlp = preds.get((protocol, "mlp"))
        none = preds.get((protocol, "none"))
        if mlp and none:  # sanity check: GraphSAGE without edges should behave like the MLP
            rows.append({"protocol": protocol, "comparison": "sanity: sage:none - mlp", **gap_stats(none, mlp, cfg)})
    pd.DataFrame(rows).to_csv(out / "ablation_gaps.csv", index=False)
    log.info("Ablation finished -> %s", out)
