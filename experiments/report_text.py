"""Static narrative for the report (everything that does not depend on measured results).
Result-dependent text is generated from results/*.csv in report.py. Nothing here states a result."""

TITLE = ("Quantifying Structural Overlap in Graph-Based Network Intrusion Detection: "
         "Separating Relational Information from Evaluation Artifacts")
AUTHORS = "Are Sanjana Reddy (24BDS0187), Chelintha Lingam T (24BCE2098) - Computer Networks Lab [BCE308P]"

ABSTRACT = (
    "Graph Neural Network (GNN)-based Network Intrusion Detection Systems (NIDS) are frequently reported to outperform "
    "conventional flow-level classifiers by exploiting relational information among communicating hosts. However, conventional "
    "graph construction and train/test splitting can introduce structural overlap between training and test data, potentially "
    "inflating the apparent advantage of graph-based models. Using CICIDS2017, we compare a Random Forest, a capacity-matched "
    "Multi-Layer Perceptron (MLP) and GraphSAGE under conventional random (transductive), leakage-safe inductive and chronological "
    "evaluation protocols. We measure train/test host and neighbourhood overlap with explicit metrics, vary the overlap in a "
    "controlled way, and run a graph-structure ablation that keeps node features and model capacity fixed. We test two hypotheses "
    "neutrally: that the GraphSAGE-MLP gap shrinks as overlap is reduced (H1), or that GraphSAGE keeps its advantage despite "
    "reduced overlap (H2)."
)

INTRO = [
    "Network intrusion detection systems classify traffic as benign or malicious. Flow-level classifiers look at one flow's "
    "statistics (duration, packet counts, inter-arrival times, flag counts) in isolation. Graph-based detectors additionally let "
    "each flow aggregate information from related flows - flows that share a source host, a destination host or a host pair - "
    "and are often reported to outperform flow-level models.",
    "Two different explanations are possible for such a gain. The first is genuine relational information: attacks such as port "
    "scans, DoS floods and brute-force logins are inherently collective, and many similar flows from the same source in a short "
    "window are evidence in themselves. The second is an evaluation artifact: if training and test flows are drawn at random "
    "from one graph and the whole graph is visible during message passing, a test flow's neighbours are mostly training flows "
    "(often from the same hosts and the same attack campaign), so the model benefits from structural familiarity that a deployed "
    "system would not have.",
    "This project quantifies that overlap and tests, without presupposing the answer, how much of the GraphSAGE-over-MLP advantage "
    "survives when overlap is reduced. Contributions: (i) a set of overlap metrics (host, neighbourhood, cross-split edge); "
    "(ii) three evaluation protocols with assertion-based leakage checks; (iii) controlled overlap variation at host level and "
    "edge level; (iv) a graph-structure ablation with capacity-matched baselines; (v) an open, reproducible code base.",
]

RELATED_INTRO = (
    "Table 1 summarises how closely related work addresses the evaluation issues studied here (taken from the project "
    "proposal; the last row is this work). To our knowledge none of the listed works measures train/test node and neighbourhood "
    "overlap explicitly and relates it to the GNN advantage over a capacity-matched baseline."
)
RELATED_HEAD = ["Paper", "Dataset", "Random split", "Inductive eval.", "Temporal eval.", "Capacity-controlled baseline",
                "Leakage analysis", "Node/neighbourhood overlap analysis"]
RELATED_ROWS = [
    ["[1] Dey & Quaium", "Edge-IIoTSet", "Yes", "Yes", "Yes", "No", "Yes (leakage gap quantified)", "No* (view overlap, not entity overlap)"],
    ["[2] Friji et al.", "CICIDS2017 + ToN-IoT", "Yes", "Partial", "No", "No", "Yes", "No"],
    ["[3] Van Langendonck et al.", "3 NetFlow datasets", "Unverified", "Yes", "Yes", "Yes", "No", "No"],
    ["[4] Lusa et al.", "NF-UNSW-NB15-v3", "No (rejected)", "Yes", "Yes", "No", "No", "No"],
    ["[5] Deng et al.", "UNSW-NB15 + CICIDS2017", "Contrast only", "No", "Yes", "No", "No", "No"],
    ["[6] Moczkodan & Ragab", "CICIDS2017", "Yes", "Yes", "Yes", "Yes", "Yes", "No"],
    ["[7] Sun et al.", "Synthetic + CICIDS2017-derived", "Yes", "No", "No", "Yes", "No", "No"],
    ["This project", "CICIDS2017", "Yes", "Yes", "Yes", "Yes (MLP)", "Yes", "Yes"],
]
REFERENCES = [
    "[1] R. Dey and F. M. Quaium, \"Leakage-Safe Multi-View Graph Construction for Robust Intrusion Detection in Edge-IIoT Networks,\" in Proc. NSysS, 2025.",
    "[2] H. Friji, A. Olivereau, and M. Sarkiss, \"Efficient Network Representation for GNN-based Intrusion Detection,\" arXiv:2310.05956, 2023.",
    "[3] L. Van Langendonck, I. Castell-Uroz, and P. Barlet-Ros, \"PPT-GNN: A Practical Pre-Trained Spatio-Temporal Graph Neural Network for Network Security,\" arXiv:2406.13365, 2024.",
    "[4] R. Lusa, D. Pintar, and M. Vranic, \"TE-G-SAGE: Explainable Edge-Aware Graph Neural Networks for Network Intrusion Detection,\" Modelling, vol. 6, no. 4, Art. 165, 2025, doi: 10.3390/modelling6040165.",
    "[5] M. Deng, S. Fan, Y. Kan, and C. Sun, \"An Intrusion Detection Model Based on Equalization Loss and Spatio-Temporal Feature Extraction,\" Electronics, vol. 15, no. 3, Art. 646, 2026, doi: 10.3390/electronics15030646.",
    "[6] Z. Moczkodan and H. Ragab, \"Do Transformers Actually Help Intrusion Detection? A Temporal Sequence Evaluation on CIC-IDS2017,\" arXiv:2606.11098, 2026.",
    "[7] Z. Sun, A. M. H. Teixeira, and S. Toor, \"GNN-IDS: Graph Neural Network based Intrusion Detection System,\" in Proc. ARES, 2024.",
    "[8] M. A. Jahin, S. Soudeep, M. F. Mridha, R. Kabir, and M. R. Islam, \"CAGN-GAT Fusion: A Hybrid Contrastive Attentive Graph Neural Network for Network Intrusion Detection,\" arXiv:2503.00961, 2025.",
    "[9] W. L. Hamilton, R. Ying, and J. Leskovec, \"Inductive Representation Learning on Large Graphs,\" in Proc. NeurIPS, 2017.",
    "[10] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, \"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,\" in Proc. ICISSP, 2018.",
    "[11] G. Engelen, V. Rimmer, and W. Joosen, \"Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study,\" in IEEE S&P Workshops, 2021.",
]

NETWORK_ANGLE = [
    ("Flows, hosts and IP semantics",
     "A flow (as exported by CICFlowMeter) is a unidirectional-pair aggregate of packets sharing a 5-tuple (source IP, source port, "
     "destination IP, destination port, protocol) until a timeout or FIN. The identity of the endpoints is therefore carried by the "
     "two IP addresses; ports are largely ephemeral on the client side. We use IP addresses only to build the graph and to compute "
     "overlap metrics. They are never fed to the models, and ports are also excluded, because both would let a classifier memorise "
     "the lab topology (e.g. the attacker 172.16.0.1 and the victim 192.168.10.50) instead of learning traffic behaviour."),
    ("Why shared endpoints create relational signal",
     "Many attacks are collective behaviours. A port scan produces many short flows from one source to many ports or hosts; a DoS or "
     "DDoS flood produces a very large number of near-identical flows towards one destination; FTP/SSH brute force produces repeated "
     "login attempts between the same host pair. Individually each flow may look unremarkable, but flows that share the source, the "
     "destination or the pair within a short window are mutually informative. A GNN with mean aggregation can exploit exactly this "
     "by averaging the features of neighbouring flows, which is a legitimate and useful relational signal."),
    ("Where overlap becomes an artifact",
     "The same mechanism turns into an evaluation artifact when a test flow's neighbours are training flows. With a random split on one "
     "graph, a test flow of a DoS burst has training flows of the same burst as neighbours, and the label of the burst is effectively "
     "shared through the graph and the features. A deployed IDS sees live traffic whose hosts, sessions and bursts have not been seen "
     "during training. The inductive and chronological protocols emulate this: the model is trained on one graph and applied to a "
     "separate, later graph."),
    ("Deployment realism",
     "In deployment (i) flows arrive in time order, so neighbours can only be past flows; (ii) the model must be applicable to hosts and "
     "to attack campaigns that were not in the training capture; (iii) the topology changes (DHCP leases, NAT, new servers). "
     "Host-level overlap therefore measures how much the evaluation relies on 'known' machines, and neighbourhood overlap measures how much "
     "message passing relies on 'known' flows. A benchmark that reports GNN gains without such measurements cannot tell the operator "
     "which of these conditions the reported gain will survive."),
]

METHOD = [
    ("Dataset and preprocessing",
     "We use the CICIDS2017 'GeneratedLabelledFlows' CSVs [10], which contain Flow ID, source/destination IP and port, protocol and timestamp "
     "in addition to the 78 CICFlowMeter features. (The 'MachineLearningCSV' version has no IP addresses and cannot be used for graph "
     "construction.) Steps: header whitespace stripped; rows with missing label/IP/timestamp removed; inf/NaN feature rows removed; exact duplicate "
     "rows removed; mis-decoded labels normalised (e.g. the en-dash in 'Web Attack - Brute Force'); duplicate and constant columns dropped; timestamps "
     "parsed and the known AM/PM ambiguity of the 12-hour clock repaired (hours before 08:00 are afternoon) [11]; flows sorted chronologically. "
     "Identifier fields (IPs, ports, Flow ID, timestamp) are excluded from the model inputs. Features pass through a signed log1p transform "
     "(defined for the negative values present in some columns) and are standardised with a scaler fitted on the training nodes of each "
     "protocol only. The primary task is binary (benign vs attack); the original labels are kept for the per-attack-type recall analysis, "
     "with Heartbleed, Infiltration and SQL injection reported as the 'Rare' group. Benign and attack flows are capped per day / per label "
     "(configurable) so that experiments run on a laptop or a free Colab GPU."),
    ("Graph construction",
     "Each flow is a node with its pre-processed features. Two flows are connected if they share the source IP, the destination IP or the "
     "(source, destination) pair and start within W seconds. Per entity type, every flow is linked to its most recent predecessors "
     "(sorted-window grouping, O(n k)); the union is symmetrised and each node keeps its k nearest-in-time neighbours. CICIDS2017 timestamps have "
     "minute resolution. The graph is built either once over all flows (transductive) or separately from each split's nodes (inductive)."),
    ("Models",
     "(1) Random Forest on flow features. (2) MLP with three linear layers, whose hidden width is solved so that its trainable-parameter "
     "count is within 5 % of GraphSAGE. (3) GraphSAGE [9] with two mean-aggregation layers with separate self and neighbour weights followed by "
     "a linear classifier, trained full-batch on the sparse adjacency. All neural models use Adam, a class-weighted cross-entropy, dropout, "
     "early stopping on the validation loss and the same input features; every configuration is repeated over several seeds."),
    ("Evaluation protocols",
     "P1 (conventional): stratified random node split on the full graph; the entire graph is visible during message passing. "
     "P2 (leakage-safe inductive): the same random node split, but the split is made first and training, validation and test graphs are built from their own nodes only, "
     "so no cross-split edge exists. P3 (chronological): train/validation on the earlier days, test on later days, with separate graphs. "
     "Assertions guarantee disjoint node sets, absence of cross-split edges in P2/P3, temporal ordering and train-only scaler fitting."),
    ("Overlap metrics",
     "With H(v) the host set {src, dst} of flow v, H_tr and H_te the host sets of train and test flows, and N(v) the neighbours of v: "
     "host Jaccard = |H_tr ∩ H_te| / |H_tr ∪ H_te|; host containment = |H_tr ∩ H_te| / |H_te|; the fraction of test nodes that have a test neighbour whose host is seen in training; "
     "cross-split edge ratio = |{edges joining different splits}| / |edges| (and its test-node variant); 1-hop neighbourhood overlap(v) = |N(v) ∩ S_tr| / |N(v)| for each test node "
     "(mean and distribution); an optional 2-hop version; edge label homophily; and the fraction of test flows whose CICFlowMeter 5-tuple Flow ID also occurs in training. "
     "Each metric is computed on the reference graph (what a transductive split exposes) and on the visible graph (what the model actually uses)."),
    ("Controlled overlap variation",
     "Host knob: IPs are partitioned into train-only, test-only and shared hosts so that a target share of test hosts is also seen in training "
     "(100/75/50/25/0 %), flows with endpoints in conflicting groups are dropped, and graphs are built per split. Hosts touching more than 0.5 % of all flows "
     "(attacker, servers) are always shared, because CICIDS2017 has only a few attacker IPs and fully disjoint hosts would remove entire classes from one side; therefore the "
     "achieved overlap cannot reach zero and is always measured and reported. Edge knob: under P1 only a fraction p of the cross-split edges is kept (p = 1, .75, .5, .25, 0)."),
    ("Graph-structure ablation",
     "GraphSAGE is run with the original topology and with disrupted topologies while features, labels and capacity are fixed: degree-preserving rewiring "
     "(configuration-model randomisation of arc destinations), uniformly random edges with the same arc count, a node-relabelling shuffle (same topology attached to wrong flows) and no edges. "
     "GraphSAGE without edges is a sanity check: it should behave like the MLP."),
    ("Metrics and statistics",
     "Primary metrics: macro-F1, attack recall, false-positive rate, PR-AUC and MCC; accuracy and weighted-F1 are reported only for comparison with prior work, because on CICIDS2017 "
     "(mostly benign traffic) accuracy is misleading. Per-attack-type recall is reported. The GraphSAGE-MLP gap is paired by seed and split; we report mean ± std over seeds, "
     "a two-stage paired bootstrap 95 % CI (seeds, then test nodes), a paired t-test and a Wilcoxon signed-rank test across seeds. With five seeds the smallest attainable two-sided "
     "Wilcoxon p-value is 0.0625, so significance claims are made cautiously."),
]

THREATS = [
    "CICIDS2017 label and feature flaws are documented [11]: mislabelled flows, TCP-termination artefacts, and duplicated/timestamp-ambiguous records. We repair some (AM/PM, duplicates) but not all.",
    "IP-based entities: the testbed has very few attacker IPs (mostly one NATed address), so host-disjoint splits are partly infeasible; the host knob keeps heavy hosts shared and reports achieved overlap. IPs are lab-specific and do not generalise to other networks.",
    "Single dataset: conclusions may not transfer to other datasets or to real enterprise traffic.",
    "Sub-sampling: capping flows per day/label thins neighbourhoods and changes the density of the graph; results on the sub-sampled graph may differ from the full dataset.",
    "Edge-construction choices (entity types, window W, per-type neighbours, degree cap k, minute-resolution timestamps) define what 'relational structure' means; other constructions (e.g. host-level graphs, edge classification) may behave differently.",
    "Temporal protocol: Thursday-Friday contain attack types absent from Monday-Wednesday, so P3 conflates temporal drift with novel-attack generalisation.",
    "Few seeds: five seeds limit statistical power; confidence intervals should be read together with the effect sizes.",
    "Full-batch GraphSAGE (no neighbour sampling) is used for exactness and simplicity; it uses the same aggregation as sampled GraphSAGE but not its sampling noise.",
]
