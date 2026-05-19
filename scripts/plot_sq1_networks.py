"""Produce two graph-style network visualisations for the SQ1 presentation.

1. ``network_example.png`` — for the *problem* slide.
   Single small directed graph from the average-control L2/3 IT pseudobulk,
   showing what a per-cell GRN looks like as a picture.  TFs and targets
   colored distinctly, edge widths = weight.  Just enough complexity to
   illustrate the structure, not so much that it's unreadable.

2. ``network_ad_vs_control.png`` — for the *results* slide.
   Two side-by-side panels (control mean vs AD mean) over the same
   focal gene set (the 3 L2/3 IT q<0.05 hits' TFs + their candidate
   targets).  Edge widths = weight, the 3 hit edges in red.  Differences
   in edge thickness make the "wiring weakens in AD" story visual.

Both plots use the per-cell wScReNI networks we already inferred on the
cluster, aggregated to control-mean and ad-mean matrices.  No new
inference, just pseudobulk + draw.
"""

from __future__ import annotations

import logging
import os
import sys

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from screni.data.combine import combine_wscreni_networks
from screni.data.differential import pseudobulk_per_donor
from screni.data.loading_seaad import (
    add_condition_column,
    add_copathology_columns,
    select_eligible_donors,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATASET = "seaad_paired"
SUB_TAG = "sq1"
CELL_TYPE_FOR_EXAMPLE = "L2/3 IT"
ADNC_LABELS = ["Not AD", "Low", "Intermediate", "High"]

DATA_DIR = os.path.join(REPO_ROOT, "data", "processed", "seaad")
RNA_SUB_PATH = os.path.join(DATA_DIR, f"{DATASET}_rna_{SUB_TAG}.h5ad")
RNA_FULL_PATH = os.path.join(DATA_DIR, f"{DATASET}_rna.h5ad")
TRIPLETS_PATH = os.path.join(DATA_DIR, f"{DATASET}_{SUB_TAG}_triplets.csv")
OUT_DIR = os.path.join(REPO_ROOT, "src", "screni", "data", "output")
NETWORKS_DIR = os.path.join(OUT_DIR, f"networks_{DATASET}")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

NEEDED_OBS = [
    "Donor ID", "Overall AD neuropathological Change",
    "LATE", "Highest Lewy Body Disease", "Age at Death", "Sex",
    "Continuous Pseudo-progression Score",
]

# The 3 L2/3 IT q<0.05 hits (we want to highlight these specifically)
HIT_EDGES = [
    ("ZNF581", "DAB1"),
    ("LEF1",   "HEG1"),
    ("E2F2",   "ADCY4"),
]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def _pseudobulks_for_ct(rna_sub, networks, ct_pretty):
    mask = (
        (rna_sub.obs["cell_type"].astype(str) == ct_pretty)
        & rna_sub.obs["condition"].notna()
    )
    rna_ct = rna_sub[mask].copy()
    donor_meta = select_eligible_donors(
        rna_ct, cell_type=ct_pretty, min_cells_per_donor=1,
    ).set_index("donor_id")
    cell_to_donor = rna_ct.obs["Donor ID"].astype(str).to_dict()
    ct_networks = type(networks)(
        {c: networks[c] for c in rna_ct.obs_names if c in networks},
        gene_names=networks.gene_names,
    )
    pb = pseudobulk_per_donor(
        networks=ct_networks,
        cell_to_donor=cell_to_donor,
        donor_metadata=donor_meta,
        aggregator="mean",
    )
    return pb, donor_meta


def _condition_avg(pb, condition: str) -> np.ndarray:
    """Average per-donor pseudobulk matrices across donors with given condition."""
    donor_ids = [d for d in pb.weights if str(pb.metadata.loc[d, "condition"]) == condition]
    if not donor_ids:
        raise ValueError(f"no donors with condition={condition!r}")
    stack = np.stack([pb.weights[d] for d in donor_ids], axis=0)
    return stack.mean(axis=0)


def _build_focal_gene_set(triplets: pd.DataFrame, focal_tfs: list[str]) -> tuple[list[str], list[str]]:
    """Pick the focal TFs plus their candidate targets from Phase 3 triplets.
    Returns (tfs, targets) as lists.
    """
    sub = triplets[triplets["TF"].isin(focal_tfs)]
    targets = sorted(sub["target_gene"].unique().tolist())
    tfs = [t for t in focal_tfs if t in sub["TF"].unique()]
    return tfs, targets


def _build_graph(mat: np.ndarray, gene_to_idx: dict, tfs: list[str], targets: list[str],
                 top_k_per_tf: int = 6) -> nx.DiGraph:
    """Build a DiGraph: each TF has up to top_k_per_tf strongest outgoing edges
    among its triplet targets."""
    G = nx.DiGraph()
    for t in tfs:
        G.add_node(t, kind="TF")
    for tgt in targets:
        if tgt in tfs:
            continue  # self-listed: keep as TF
        G.add_node(tgt, kind="target")

    for tf in tfs:
        if tf not in gene_to_idx:
            continue
        j = gene_to_idx[tf]
        # edges from this TF onto target genes in our set
        candidate_targets = [t for t in targets if t in gene_to_idx and t != tf]
        weights = [(t, mat[gene_to_idx[t], j]) for t in candidate_targets]
        # rank by absolute weight, take top_k
        weights.sort(key=lambda kv: abs(kv[1]), reverse=True)
        for tgt, w in weights[:top_k_per_tf]:
            G.add_edge(tf, tgt, weight=w)
    return G


def _draw_grn(G: nx.DiGraph, ax: plt.Axes, title: str, layout: dict | None = None,
              hit_edges: set | None = None, max_lw: float = 5.0):
    if layout is None:
        layout = nx.kamada_kawai_layout(G)

    # Node colors
    node_colors = ["#e8a020" if G.nodes[n].get("kind") == "TF" else "#88aacc"
                   for n in G.nodes]
    node_sizes = [620 if G.nodes[n].get("kind") == "TF" else 380 for n in G.nodes]
    nx.draw_networkx_nodes(G, layout, ax=ax, node_color=node_colors,
                            node_size=node_sizes, edgecolors="white", linewidths=1.2)

    # Edge widths scaled to weight magnitude
    weights = [abs(d["weight"]) for _, _, d in G.edges(data=True)]
    if weights:
        wmax = max(weights) if max(weights) > 0 else 1.0
        widths = [0.4 + max_lw * (w / wmax) for w in weights]
    else:
        widths = []

    hit_edges = hit_edges or set()
    edge_colors = []
    for u, v, d in G.edges(data=True):
        if (u, v) in hit_edges:
            edge_colors.append("#d63838")    # red for hit edges
        elif d.get("weight", 0) < 0:
            edge_colors.append("#5577aa")    # blue for negative
        else:
            edge_colors.append("#666666")    # grey for positive

    nx.draw_networkx_edges(
        G, layout, ax=ax, width=widths, edge_color=edge_colors,
        alpha=0.7, arrows=True, arrowsize=12, connectionstyle="arc3,rad=0.08",
        min_target_margin=10,
    )
    # Labels
    nx.draw_networkx_labels(G, layout, ax=ax, font_size=9,
                             font_color="black")
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    logger.info("loading rna_sub ...")
    rna_sub = ad.read_h5ad(RNA_SUB_PATH)
    missing = [c for c in NEEDED_OBS if c not in rna_sub.obs.columns]
    if missing:
        logger.info(f"  backfilling: {missing}")
        rna_full = ad.read_h5ad(RNA_FULL_PATH, backed="r")
        full_obs = rna_full.obs.loc[rna_sub.obs_names, missing].copy()
        rna_full.file.close()
        for c in missing:
            rna_sub.obs[c] = full_obs[c].values
    add_condition_column(rna_sub)
    add_copathology_columns(rna_sub)

    logger.info("loading per-cell networks (slow, ~10 min) ...")
    networks = combine_wscreni_networks(
        cell_names=rna_sub.obs_names.tolist(),
        network_dir=NETWORKS_DIR,
    )
    gene_to_idx = {g: i for i, g in enumerate(networks.gene_names)}
    logger.info(f"  loaded {len(networks)} networks, {len(gene_to_idx)} genes")

    logger.info(f"pseudobulk for {CELL_TYPE_FOR_EXAMPLE} ...")
    pb, _meta = _pseudobulks_for_ct(rna_sub, networks, CELL_TYPE_FOR_EXAMPLE)
    mat_ctrl = _condition_avg(pb, "control")
    mat_ad   = _condition_avg(pb, "ad")
    logger.info(f"  control avg: {mat_ctrl.shape}, ad avg: {mat_ad.shape}")

    triplets = pd.read_csv(TRIPLETS_PATH)

    # ---- Subnetwork: 3 L2/3 IT hit TFs + their candidate triplet targets ----
    focal_tfs = [tf for tf, _ in HIT_EDGES]
    tfs, targets = _build_focal_gene_set(triplets, focal_tfs)
    logger.info(f"focal subnetwork: {len(tfs)} TFs + {len(targets)} targets")

    # ---- Figure 1: 'example GRN' for the problem slide ---------------------
    # Single panel showing the average-control network around the 3 hit TFs.
    # No condition contrast — just "this is what a per-donor GRN looks like."
    G_ctrl_example = _build_graph(mat_ctrl, gene_to_idx, tfs, targets, top_k_per_tf=6)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    layout = nx.kamada_kawai_layout(G_ctrl_example)
    _draw_grn(G_ctrl_example, ax,
              title=f"Example: a per-donor regulatory subnetwork ({CELL_TYPE_FOR_EXAMPLE}, control donors)\n"
                    f"3 transcription factors (orange) and their top targets (blue) — edge widths = inferred regulatory strength",
              layout=layout)
    # legend
    leg = [
        mpatches.Patch(color="#e8a020", label="Transcription factor (TF)"),
        mpatches.Patch(color="#88aacc", label="Target gene"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=10, frameon=False)
    out_path = os.path.join(PLOTS_DIR, "network_example.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")

    # ---- Figure 2: control vs AD side-by-side -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    # use same layout for both panels so eye can compare topology directly
    G_ctrl = _build_graph(mat_ctrl, gene_to_idx, tfs, targets, top_k_per_tf=6)
    G_ad   = _build_graph(mat_ad,   gene_to_idx, tfs, targets, top_k_per_tf=6)
    # build a combined graph for layout so node positions match across panels
    G_all = nx.DiGraph()
    for n in set(G_ctrl.nodes) | set(G_ad.nodes):
        kind = G_ctrl.nodes.get(n, {}).get("kind") or G_ad.nodes.get(n, {}).get("kind", "target")
        G_all.add_node(n, kind=kind)
    common_layout = nx.kamada_kawai_layout(G_all)

    hit_set = {(tf, tgt) for tf, tgt in HIT_EDGES}
    _draw_grn(G_ctrl, axes[0], title="Average control donor",
              layout=common_layout, hit_edges=hit_set)
    _draw_grn(G_ad,   axes[1], title="Average AD donor",
              layout=common_layout, hit_edges=hit_set)

    leg = [
        mpatches.Patch(color="#e8a020", label="TF"),
        mpatches.Patch(color="#88aacc", label="Target"),
        mpatches.Patch(color="#d63838", label="q<0.05 differential edge"),
    ]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=3, frameon=False, fontsize=11)
    fig.suptitle(
        f"Regulatory subnetwork around the 3 L2/3 IT q<0.05 differential edges\n"
        f"(thicker arrows = stronger inferred regulation; red = differential hit)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(PLOTS_DIR, "network_ad_vs_control.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
