"""Experiment 2: controlled overlap variation.
 knob 'host': host-grouped splits (share of test hosts also seen in train ~100/75/50/25/0 %), inductive graphs.
 knob 'edge': conventional random split (P1), keep a fraction p of cross-split edges.
Reports ACHIEVED overlap metrics, performance of RF / MLP / GraphSAGE and the GraphSAGE-MLP gap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset import load_dataset
from src.data.splitting import host_grouped_split, stratified_random_split
from src.eval.metrics import compute_metrics
from src.eval.stats import correlations
from src.pipeline import build_task
from src.utils import resolve_path

from .common import MODELS, evaluate_models, flatten_overlap, gap_stats, log, mean_std_table

OVERLAP_X = ["ref_host_containment", "ref_one_hop_mean", "ref_test_cross_edge_ratio", "vis_one_hop_mean",
             "vis_test_cross_edge_ratio"]


def main(cfg: dict) -> None:
    ds = load_dataset(cfg)
    out = resolve_path(cfg, "results_dir")
    oc = cfg["overlap"]
    seeds, models = cfg["seeds"], cfg.get("models", list(MODELS))
    runs, ov, preds = [], [], {}
    for knob, levels in (("host", oc["host_levels"]), ("edge", oc["edge_levels"])):
        for seed in seeds:
            fixed = {}  # edge knob: rf/mlp do not use edges -> train once per seed, reuse
            for level in levels:
                log.info("=== knob=%s level=%.2f seed=%d ===", knob, level, seed)
                extra = {"knob": knob, "level": level}
                if knob == "host":
                    try:
                        sp = host_grouped_split(ds.src_id, ds.dst_id, ds.y, level, seed, oc["test_frac"],
                                                oc.get("val_frac", 0.15), oc["heavy_host_frac"])
                    except ValueError as e:
                        log.warning("skip host level %.2f seed %d: %s", level, seed, e)
                        continue
                    split = {k: sp[k] for k in ("train", "val", "test")}
                    extra.update(dropped_frac=sp["dropped_frac"], n_heavy_hosts=sp["n_heavy_hosts"])
                    task = build_task(ds, split, "inductive", cfg, seed)
                    use = models
                else:
                    split = stratified_random_split(ds.y, cfg["protocols"]["split"], seed)
                    task = build_task(ds, split, "transductive", cfg, seed, cross_keep=level)
                    use = [m for m in models if m == "sage" or m not in fixed]
                rows, p = evaluate_models(ds, task, use, cfg, seed, extra)
                if knob == "edge":
                    for r, m in zip(rows, use):
                        if m != "sage":
                            fixed[m] = (r, p[m])
                    for m, (r, pm) in fixed.items():
                        if m not in use:
                            rows.append(dict(r, level=level))
                            p[m] = pm
                runs += rows
                ov.append(flatten_overlap(task, extra, seed))
                preds.setdefault((knob, level), {})[seed] = p
            pd.DataFrame(runs).to_csv(out / "sweep_runs.csv", index=False)
    runs_df, ov_df = pd.DataFrame(runs), pd.DataFrame(ov)
    runs_df.to_csv(out / "sweep_runs.csv", index=False)
    ov_df.to_csv(out / "sweep_overlap.csv", index=False)
    mean_std_table(runs_df, ["knob", "level", "model"]).to_csv(out / "sweep_summary.csv", index=False)

    # ---- gap per (knob, level) with CIs + achieved overlap ----
    gap_rows, points = [], []
    for (knob, level), byseed in preds.items():
        pa = {s: byseed[s]["sage"] for s in byseed if "sage" in byseed[s]}
        pb = {s: byseed[s]["mlp"] for s in byseed if "mlp" in byseed[s]}
        if len(pa) < 1:
            continue
        g = gap_stats(pa, pb, cfg)
        o = ov_df[(ov_df.knob == knob) & (ov_df.level == level)]
        gap_rows.append({"knob": knob, "level": level, **g,
                         **{f"achieved_{c}": float(o[c].mean()) for c in OVERLAP_X if c in o}})
        for s in pa:
            ma, mb = compute_metrics(*pa[s]), compute_metrics(*pb[s])
            orow = o[o.seed == s].iloc[0]
            points.append({"knob": knob, "level": level, "seed": s,
                           "gap_macro_f1": ma["macro_f1"] - mb["macro_f1"], "gap_pr_auc": ma["pr_auc"] - mb["pr_auc"],
                           "gap_attack_recall": ma["attack_recall"] - mb["attack_recall"],
                           **{c: orow[c] for c in OVERLAP_X if c in orow}})
    pd.DataFrame(gap_rows).sort_values(["knob", "level"]).to_csv(out / "sweep_gaps.csv", index=False)
    pts = pd.DataFrame(points)
    pts.to_csv(out / "sweep_points.csv", index=False)

    corr = []
    for scope, sub in [("host", pts[pts.knob == "host"]), ("edge", pts[pts.knob == "edge"]), ("all", pts)]:
        for xcol in OVERLAP_X + ["level"]:
            if xcol not in sub:
                continue
            for ycol in ("gap_macro_f1", "gap_pr_auc"):
                corr.append({"scope": scope, "overlap_measure": xcol, "gap_metric": ycol,
                             **correlations(sub[xcol].to_numpy(), sub[ycol].to_numpy())})
    pd.DataFrame(corr).to_csv(out / "sweep_correlations.csv", index=False)
    log.info("Overlap sweep finished -> %s", out)
