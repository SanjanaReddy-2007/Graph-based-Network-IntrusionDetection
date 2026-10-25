"""Experiment 1: RF vs capacity-matched MLP vs GraphSAGE under P1 (random/transductive), P2 (leakage-safe
inductive) and P3 (chronological). Also logs overlap metrics, parameter counts, per-attack recall."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset import load_dataset
from src.pipeline import PROTOCOLS, build_task, make_split
from src.utils import resolve_path, save_json

from .common import MODELS, evaluate_models, flatten_overlap, gap_stats, log, mean_std_table, param_report


def main(cfg: dict) -> None:
    ds = load_dataset(cfg)
    out = resolve_path(cfg, "results_dir")
    seeds, models = cfg["seeds"], cfg.get("models", list(MODELS))
    save_json(param_report(ds.X_log.shape[1], cfg), out / "param_counts.json")
    runs, ov, preds = [], [], {}
    dist = {}
    for protocol in cfg["protocols"]["list"]:
        for seed in seeds:
            log.info("=== %s seed=%d ===", protocol, seed)
            split = make_split(ds, protocol, cfg, seed)
            task = build_task(ds, split, PROTOCOLS[protocol], cfg, seed)
            rows, p = evaluate_models(ds, task, models, cfg, seed, {"protocol": protocol})
            for r in rows:
                r.update({"n_train": len(split["train"]), "n_val": len(split["val"])})
            runs += rows
            ov.append(flatten_overlap(task, {"protocol": protocol}, seed))
            preds.setdefault(protocol, {})[seed] = p
            if seed == seeds[0]:
                dist[f"{protocol}_ref"] = task.info["per_node_one_hop_ref"]
                dist[f"{protocol}_vis"] = task.info["per_node_one_hop_vis"]
        pd.DataFrame(runs).to_csv(out / "protocols_runs.csv", index=False)  # incremental save
    runs_df, ov_df = pd.DataFrame(runs), pd.DataFrame(ov)
    runs_df.to_csv(out / "protocols_runs.csv", index=False)
    ov_df.to_csv(out / "protocols_overlap.csv", index=False)
    np.savez_compressed(out / "protocols_nbr_overlap_dist.npz", **{k: v[~np.isnan(v)] for k, v in dist.items()})
    mean_std_table(runs_df, ["protocol", "model"]).to_csv(out / "protocols_summary.csv", index=False)

    gaps = []
    for protocol, byseed in preds.items():
        for a, b in (("sage", "mlp"), ("sage", "rf"), ("mlp", "rf")):
            if a in models and b in models:
                pa = {s: byseed[s][a] for s in byseed}
                pb = {s: byseed[s][b] for s in byseed}
                gaps.append({"protocol": protocol, "A": a, "B": b, **gap_stats(pa, pb, cfg)})
    pd.DataFrame(gaps).to_csv(out / "protocols_gaps.csv", index=False)
    log.info("Protocols experiment finished -> %s", out)
