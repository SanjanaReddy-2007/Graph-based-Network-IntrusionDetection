# Quantifying Structural Overlap in Graph-Based Network Intrusion Detection

*Separating relational information from evaluation artifacts* - Computer Networks Lab [BCE308P]
Are Sanjana Reddy (24BDS0187), Chelintha Lingam T (24BCE2098)

Do GNN-based NIDS beat flow-level classifiers because of **genuine relational information**, or because random
train/test splits create **structural overlap** (shared hosts, shared neighbourhoods, cross-split edges)?
This repository measures that overlap and tests two hypotheses neutrally:

* **H1** - the GraphSAGE-MLP gap shrinks as train/test overlap is reduced (artifact explanation).
* **H2** - GraphSAGE keeps its advantage despite reduced overlap (genuine relational value).

Models: Random Forest, capacity-matched MLP (parameter count within ±5 % of GraphSAGE), GraphSAGE.
Protocols: **P1** random/transductive, **P2** leakage-safe inductive, **P3** chronological.

---

## 1. Getting the data

Use **`GeneratedLabelledFlows.zip`** from <https://cicresearch.ca/CICDataset/CIC-IDS-2017/> (CSVs folder).
(`MachineLearningCSV.zip` has no IP addresses / timestamps and **cannot** be used.)

```
unzip GeneratedLabelledFlows.zip  ->  copy the 8 CSV files into  data/raw/
python tools/check_dataset.py          # verifies headers (Flow ID, Source IP, Destination IP, Timestamp, Label ...)
```
Nested folders are fine (`data/raw/TrafficLabelling/*.csv`). The files are not valid UTF-8; the loader handles
`cp1252` / `latin-1` automatically.

## 2. Installation

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch            # recommended (GPU or CPU build, https://pytorch.org); no PyTorch-Geometric needed
```
Without PyTorch the project falls back to a pure-numpy GraphSAGE/MLP (identical architecture and recipe, but
CPU-only and slower; fine for `configs/quick.yaml`). Google Colab: upload the folder, `pip install -r requirements.txt`, select a GPU runtime.

## 3. Running

```bash
python run.py --experiment prepare        # clean + sub-sample + cache  (data/processed/)
python run.py --experiment eda
python run.py --experiment protocols      # RF / MLP / GraphSAGE x P1,P2,P3 x seeds
python run.py --experiment overlap_sweep  # host knob + edge knob, achieved overlap vs gap
python run.py --experiment ablation       # topology ablation (+ sanity check)
python run.py --experiment figures
python run.py --experiment report         # results/report.md + report.docx (tables from the logs)
python run.py --experiment slides         # results/slides.pptx
python run.py --experiment all            # everything, in order
```
First time, try the fast configuration (3 seeds, smaller graph):
`python run.py --experiment all --config configs/quick.yaml`.
Any config value can be overridden: `--set seeds=[0,1,2,3,4,5,6,7,8,9] --set train.epochs=100`.

**Smoke test (no dataset needed):** `python run.py --experiment smoke` generates a small *synthetic* file set in the
CICIDS format, runs the whole pipeline with the numpy back-end and writes to `results_smoke/`. It only proves the code
runs; **those numbers are not results.** Unit tests: `python -m unittest discover -s tests -v`.

**Expected runtime** (rough estimates, not measured on the real data): `prepare` a few minutes (2.8 M rows are read once);
with a GPU the default config needs on the order of tens of minutes to a few hours for all experiments; on CPU-only machines use
`quick.yaml` and expect longer. Cost scales with nodes x epochs x seeds; reduce `subsample.*`, `seeds`, `train.epochs` first.

## 4. What the code does

| Step | Implementation |
|---|---|
| Cleaning | strip headers, drop NaN/inf/duplicate rows, repair mis-decoded labels, **AM/PM timestamp repair** (hours < 8 -> +12 h), drop duplicate/constant columns, signed-log1p features, **train-only standardisation** per protocol |
| Features | all numeric CICFlowMeter features; **excluded**: Flow ID, IPs, ports, timestamp |
| Labels | binary (benign vs attack) is the primary task; original labels are kept for per-attack recall; Heartbleed/Infiltration/SQLi = "Rare" group |
| Graph | node = flow; edges = shared src IP / dst IP / (src,dst) within `window_seconds`; per type the `neighbors_per_type` most recent predecessors; symmetrised; each node keeps its `max_degree` nearest-in-time neighbours. Vectorised, O(n·k) |
| P1 | stratified random split; one graph over all flows; whole graph visible in message passing |
| P2 | *same* random split, but split first; train/val/test graphs built from their own nodes only |
| P3 | train/val = earlier days (val = last 15 % of that period), test = later days; graphs per split |
| Leakage checks | disjoint node sets; no cross-split edges in P2/P3; train<val<test in time; scaler fitted exactly on train nodes (`src/eval/leakage.py`) |
| Capacity match | `src/models/param_matching.py`: MLP width solved so params are within ±5 % of GraphSAGE; both counts logged (`results/param_counts.json`, `n_params` column) |
| Training | Adam, class-weighted cross-entropy, dropout, early stopping on validation loss, same for MLP and GraphSAGE |
| Overlap sweep | **host knob** (share of test hosts seen in train ~100/75/50/25/0 %) and **edge knob** (keep fraction *p* of cross-split edges) |
| Ablation | GraphSAGE on original / degree-preserving *rewire* / *random* / *shuffle* / *none* topology, same features & capacity; `sage:none` vs MLP is the sanity check |
| Statistics | mean ± std over seeds; two-stage paired bootstrap 95 % CI (seeds -> test nodes); paired t-test and Wilcoxon over seeds; Spearman/Pearson of overlap vs gap |

### Overlap metrics (formulas)
`H(v) = {src_ip, dst_ip}` of flow *v*; `H_tr`, `H_te` = host sets of train / test flows; `N(v)` = neighbours of *v*.

* host Jaccard = |H_tr ∩ H_te| / |H_tr ∪ H_te| ; host containment = |H_tr ∩ H_te| / |H_te|
* seen-host test neighbour = fraction of test nodes with ≥1 *test* neighbour *u* such that H(u) ∩ H_tr ≠ ∅
* cross-split edge ratio = #edges joining different splits / #edges ; test variant = share of arcs into test nodes whose source is not a test node
* 1-hop neighbourhood overlap(v) = |N(v) ∩ S_tr| / |N(v)| for each test node (mean, median, quartiles, full distribution saved)
* 2-hop overlap = same over the 2-hop neighbourhood (sampled test nodes) ; edge homophily = share of edges joining equal-label nodes
* 5-tuple recurrence = share of test flows whose CICFlowMeter *Flow ID* also occurs in train
* Each metric is reported on the **reference graph** (`ref_*`, what a transductive split exposes) and on the **visible graph** (`vis_*`, the edges the model really uses; 0 for P2/P3 by construction).

## 5. Outputs (`results/`)

CSV: `protocols_runs / _summary / _gaps / _overlap`, `sweep_runs / _summary / _gaps / _points / _correlations / _overlap`,
`ablation_runs / _summary / _gaps`, `eda_*`, `param_counts.json`, `table_*.csv`.
Figures (PNG + PDF): `fig_protocols`, `fig_overlap_gap` (CI bands), `fig_ablation`, `fig_nbr_overlap_dist`, `fig_recall_heatmap`, `fig_eda`.
Documents: `report.md`, `report.docx` (IEEE-style section structure; all numbers read from the CSVs; the Discussion bullet list is a
rule-based summary that you must review and rewrite), `slides.pptx`. Cleaning decisions: `data/processed/cleaning_report.json`.

## 6. Design decisions and deviations (please read)

* **GraphSAGE is implemented in plain PyTorch (full-batch sparse mean aggregation), not PyG `NeighborLoader`.** This removes the PyG installation
  problem and makes the model exact; scalability comes from the degree cap and the sub-sampling options. A numpy twin with hand-written
  gradients (checked by finite differences in the tests) is used when PyTorch is missing.
* **Node = flow.** The optional host-level alternative (nodes = IPs, edge classification) is *not* implemented.
* **Binary task only for training.** Multiclass labels are used for the per-attack-type recall breakdown and the "Rare" grouping, but no multiclass classifier is trained.
* **Host knob cannot reach 0 % overlap.** CICIDS2017 has very few attacker/victim IPs; hosts touching ≥0.5 % of flows (`overlap.heavy_host_frac`) are always shared, otherwise whole
  attack classes would vanish from train or test. Targets are therefore only targets - **achieved** overlap is measured and reported.
* **P3 mixes drift with novel attacks:** Thursday/Friday contain attack types absent from Monday-Wednesday. Change `protocols.temporal` if you want another day split.
* **Statistics with 5 seeds:** the smallest possible two-sided Wilcoxon p-value is 0.0625, so it can never be "significant" with 5 seeds. Use ≥6 (preferably 10) seeds for that test, and look at the CIs and effect sizes.
* **Sub-sampling** (`subsample.*`) thins neighbourhoods; document the setting you use and, if resources allow, repeat with larger caps.
* **MLP vs GraphSAGE without edges** are not identical (different width, SAGE keeps unused neighbour weights), so `sage:none - mlp` is expected to be *small*, not exactly 0.

## 7. Repository layout

```
run.py                     single CLI entry point
configs/                   default.yaml, quick.yaml, smoke.yaml (YAML; `inherit:` supported)
src/data/                  loading, cleaning, subsample, features(scaler), splitting, dataset (prepare/load)
src/graph/                 construction (vectorised), disruption (rewire/random/shuffle/none)
src/overlap/               metrics (unit-tested), variation (edge knob)
src/models/                rf, nn_torch, nn_numpy, param_matching
src/train/                 task containers, trainer dispatch
src/eval/                  metrics, stats, leakage checks
src/pipeline.py            split -> Task (graphs, scaler, leakage checks, overlap metrics)
experiments/               protocols, overlap_sweep, ablation, eda, figures, report(+text), slides
tools/                     check_dataset.py, make_synthetic_cicids.py (smoke-test data only)
notebooks/                 01_EDA_and_results.ipynb
tests/                     unit tests + synthetic end-to-end smoke test
```

## 8. Troubleshooting

* `Processed data not found` -> run `--experiment prepare` first.
* `missing columns [...] looks like MachineLearningCSV` -> wrong zip; use GeneratedLabelledFlows.
* Out of memory -> lower `subsample.benign_cap_per_day` / `attack_cap_per_label_day`, `graph.max_degree`, or use `--config configs/quick.yaml`.
* A host-level level is skipped with a warning ("single class") -> the split could not keep both classes on each side; try other seeds or a lower `overlap.heavy_host_frac`.
* Ideas for the real data: if the timestamp layout differs from what the parser auto-detects, check `cleaning_report.json` (`dropped_bad_timestamp`).

## 9. References
See `experiments/report_text.py` (references [1]-[8] are from the project proposal; [9]-[11] were added: GraphSAGE, CICIDS2017, and the CICIDS2017 flaws study).
