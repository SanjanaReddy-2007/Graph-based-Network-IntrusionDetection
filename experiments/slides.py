"""Lab-presentation deck (python-pptx). Numbers come from results/*.csv; missing results are marked, never invented."""
from __future__ import annotations

from pathlib import Path

from src.utils import resolve_path

from . import report_text as T
from .common import log
from .report import tables, verdict


def main(cfg: dict) -> None:
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except ImportError:
        log.warning("python-pptx not installed: skipping slides (pip install python-pptx)")
        return
    out = resolve_path(cfg, "results_dir")
    tb, vd = tables(out), verdict(out)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
    blank, title_only = prs.slide_layouts[6], prs.slide_layouts[5]

    def add(title, bullets=None, image=None, table=None, size=18):
        s = prs.slides.add_slide(title_only)
        s.shapes.title.text = title
        s.shapes.title.text_frame.paragraphs[0].font.size = Pt(30)
        top = Inches(1.5)
        if bullets:
            w = Inches(5.6) if image else Inches(12.3)
            tb_ = s.shapes.add_textbox(Inches(0.5), top, w, Inches(5.5)).text_frame
            tb_.word_wrap = True
            for i, b in enumerate(bullets):
                p = tb_.paragraphs[0] if i == 0 else tb_.add_paragraph()
                p.text = "• " + b
                p.font.size = Pt(size)
                p.space_after = Pt(8)
        if image and (out / image).exists():
            x = Inches(6.3) if bullets else Inches(0.8)
            s.shapes.add_picture(str(out / image), x, top, width=Inches(6.7) if bullets else Inches(11.5))
        elif image:
            s.shapes.add_textbox(Inches(6.3), top, Inches(6), Inches(1)).text_frame.text = "[figure not generated yet]"
        if table is not None:
            rows, cols = len(table) + 1, len(table.columns)
            sh = s.shapes.add_table(rows, cols, Inches(0.5), Inches(1.6), Inches(12.3), Inches(0.4) * rows).table
            for j, c in enumerate(table.columns):
                sh.cell(0, j).text = str(c)
            for i, (_, r) in enumerate(table.iterrows(), start=1):
                for j, v in enumerate(r.values):
                    sh.cell(i, j).text = str(v)
            for i in range(rows):
                for j in range(cols):
                    for p in sh.cell(i, j).text_frame.paragraphs:
                        p.font.size = Pt(10)
        return s

    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "Quantifying Structural Overlap in Graph-Based NIDS"
    s.placeholders[1].text = "Separating relational information from evaluation artifacts\n" + T.AUTHORS
    add("Motivation", ["GNN-based NIDS are often reported to beat flow-level classifiers",
                       "Random splits on one graph let test flows see training flows as neighbours",
                       "Is the GNN advantage genuine relational information - or structural overlap?"])
    add("Research question and hypotheses", ["H1: the GraphSAGE-MLP gap shrinks as train/test structural overlap shrinks (artifact)",
                                              "H2: GraphSAGE keeps its advantage despite reduced overlap (genuine relational value)",
                                              "Tested neutrally; mixed or null results are reported as such"])
    add("Related work gap", ["Leakage-aware and temporal evaluations exist ([1],[3],[6])",
                              "None measures train/test node and neighbourhood overlap explicitly",
                              "None relates overlap to the gap against a capacity-matched baseline"])
    add("Network view: flows, hosts, shared endpoints", ["Node = flow (CICFlowMeter 5-tuple aggregate); IPs only build the graph",
                                                          "Scans, DoS floods, brute force = collective behaviour over shared src / dst / pair",
                                                          "Deployment: unseen hosts, unseen campaigns, time-ordered traffic"])
    add("Method: graph, models, capacity", [f"Edges: shared src/dst/pair within W={cfg['graph']['window_seconds']}s, degree cap {cfg['graph']['max_degree']}",
                                             "Models: Random Forest, MLP (params matched to +-5%), GraphSAGE (mean, 2 layers)",
                                             "Same features, loss, optimiser, early stopping; multiple seeds"])
    add("Evaluation protocols", ["P1 random / transductive: whole graph visible",
                                  "P2 inductive: split first, graphs built per split, no cross-split edges",
                                  "P3 chronological: earlier days train, later days test",
                                  "Assertions: disjoint nodes, no cross edges, time order, train-only scaler"])
    add("Overlap metrics and controlled variation", ["Host Jaccard / containment; seen-host test neighbours",
                                                      "Cross-split edge ratio; 1-hop and 2-hop neighbourhood overlap; 5-tuple recurrence",
                                                      "Host knob (100-0% shared hosts) and edge knob (keep p of cross edges)",
                                                      "Achieved - not target - overlap is reported"], size=17)
    add("Results: three protocols", image="fig_protocols.png", table=None)
    pg = tb.get("protocol_gaps")
    if pg is not None:
        add("Results: GraphSAGE vs MLP gap", table=pg[pg["A - B"] == "sage - mlp"])
    add("Results: gap vs overlap", image="fig_overlap_gap.png")
    add("Results: topology ablation", image="fig_ablation.png")
    add("Discussion (auto-summary of logged statistics)", vd[-6:], size=14)
    add("Threats to validity and conclusion", T.THREATS[:5] + ["Framework: overlap-aware, controlled benchmarking of GNN-IDS"], size=14)
    prs.save(out / "slides.pptx")
    log.info("Slides written: %s", out / "slides.pptx")
