"""Clean, focused plots for the SQ1 midterm presentation.

Produces three readable PNGs in src/screni/data/output/plots/:

* ``network_example_v2.png``
    Single-TF fan-out subnetwork.  One TF + ~5 of its top inferred targets,
    drawn as a star.  Use on the *what is a GRN* introduction slide.

* ``network_ad_vs_control_v2.png``
    Two side-by-side panels, *bipartite* layout (TFs on left, targets on
    right), showing only the 3 L2/3 IT q<0.05 hit edges + a handful of
    additional context edges per TF.  Less than ~12 edges per panel so
    nothing is cluttered.  Use on the *results* slide.

* ``cohort_table_adnc.png``
    Donor metadata table with rows colored by ADNC severity (Not AD ->
    Low -> Intermediate -> High instead of binarised control/AD).
    Shows the severity gradient at a glance.

Reuses the per-cell wScReNI networks already on disk.
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
CELL_TYPE = "L2/3 IT"

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

# 3 L2/3 IT q<0.05 hits
HIT_EDGES = [
    ("ZNF581", "DAB1"),
    ("LEF1",   "HEG1"),
    ("E2F2",   "ADCY4"),
]

# Color scheme for the 4 ADNC severity levels (healthy → severe gradient).
ADNC_ORDER = ["Not AD", "Low", "Intermediate", "High"]
ADNC_COLORS = {
    "Not AD":       "#c8e6c9",   # pale green
    "Low":          "#fff59d",   # pale yellow
    "Intermediate": "#ffcc80",   # light orange
    "High":         "#ef9a9a",   # light red
}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "figure.dpi": 130,
    "savefig.dpi": 170,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_tagged_rna():
    rna_sub = ad.read_h5ad(RNA_SUB_PATH)
    missing = [c for c in NEEDED_OBS if c not in rna_sub.obs.columns]
    if missing:
        logger.info(f"  backfilling obs from full RNA: {missing}")
        rna_full = ad.read_h5ad(RNA_FULL_PATH, backed="r")
        full_obs = rna_full.obs.loc[rna_sub.obs_names, missing].copy()
        rna_full.file.close()
        for c in missing:
            rna_sub.obs[c] = full_obs[c].values
    add_condition_column(rna_sub)
    add_copathology_columns(rna_sub)
    return rna_sub


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
    donor_ids = [d for d in pb.weights if str(pb.metadata.loc[d, "condition"]) == condition]
    if not donor_ids:
        raise ValueError(f"no donors with condition={condition!r}")
    stack = np.stack([pb.weights[d] for d in donor_ids], axis=0)
    return stack.mean(axis=0)


# ---------------------------------------------------------------------------
# Plot 1: single-TF fan-out (intro slide)
# ---------------------------------------------------------------------------


def plot_single_tf_fanout(mat: np.ndarray, gene_to_idx: dict, triplets: pd.DataFrame,
                          tfs: list[str], k_per_tf: int = 8, save_path: str = ""):
    """Multi-TF fan-out: several TFs, each with their top-K candidate targets.
    Produces a richer-looking network for the intro slide.
    """
    G = nx.DiGraph()
    tf_targets: dict[str, list[tuple[str, float]]] = {}
    for tf in tfs:
        if tf not in gene_to_idx:
            continue
        candidates = triplets[triplets["TF"] == tf]["target_gene"].unique().tolist()
        j = gene_to_idx[tf]
        weights = [(t, mat[gene_to_idx[t], j]) for t in candidates if t in gene_to_idx and t != tf]
        weights.sort(key=lambda kv: abs(kv[1]), reverse=True)
        tf_targets[tf] = weights[:k_per_tf]

    # Build graph: keep TFs and all their picked targets
    for tf, picks in tf_targets.items():
        G.add_node(tf, kind="TF")
        for t, w in picks:
            G.add_node(t, kind="target")
            G.add_edge(tf, t, weight=w)

    # Layout: TFs in a circle in the center, their targets fanning out
    # behind them (each TF's targets clustered on the side away from the
    # other TFs).
    n_tfs = len(tf_targets)
    pos = {}
    tf_list = list(tf_targets.keys())
    radius_tf = 0.45
    for i, tf in enumerate(tf_list):
        angle = 2 * np.pi * i / max(n_tfs, 1) + np.pi / 2
        pos[tf] = (radius_tf * np.cos(angle), radius_tf * np.sin(angle))

    radius_tgt = 1.6
    for i, tf in enumerate(tf_list):
        picks = tf_targets[tf]
        n = len(picks)
        base_angle = 2 * np.pi * i / max(n_tfs, 1) + np.pi / 2
        spread = np.pi / max(n_tfs, 1) * 1.6
        for k, (t, _) in enumerate(picks):
            theta = base_angle + spread * ((k / max(n - 1, 1)) - 0.5)
            pos[t] = (radius_tgt * np.cos(theta), radius_tgt * np.sin(theta))

    fig, ax = plt.subplots(figsize=(14, 12))

    # nodes
    tf_nodes = [n for n in G.nodes if G.nodes[n].get("kind") == "TF"]
    tgt_nodes = [n for n in G.nodes if G.nodes[n].get("kind") == "target"]
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=tf_nodes, node_color="#e8a020",
        node_shape="D", node_size=4500, edgecolors="white", linewidths=2.5,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=tgt_nodes, node_color="#88aacc",
        node_size=2800, edgecolors="white", linewidths=2.5,
    )

    # edges scaled by abs(weight)
    all_w = [abs(d["weight"]) for _, _, d in G.edges(data=True)]
    wmax = max(all_w) if all_w else 1.0
    widths = [2.0 + 9.0 * abs(d["weight"]) / wmax for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(
        G, pos, ax=ax, width=widths, edge_color="#555555",
        arrows=True, arrowsize=28, alpha=0.7,
        min_target_margin=28, node_size=2800,
    )

    # labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=14, font_weight="bold")

    n_tgt = sum(len(v) for v in tf_targets.values())
    ax.set_title(
        f"Example: gene regulatory subnetwork  ({len(tf_targets)} TFs, {n_tgt} target connections)\n"
        f"Orange diamonds = transcription factors;  blue circles = target genes;  "
        f"thicker arrows = stronger inferred regulation",
        fontsize=13,
    )
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    # legend below the plot
    legend = [
        mpatches.Patch(color="#e8a020", label="Transcription factor (regulator)"),
        mpatches.Patch(color="#88aacc", label="Target gene"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.05),
              ncol=2, fontsize=11, frameon=False)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"wrote {save_path}")


# ---------------------------------------------------------------------------
# Plot 2: bipartite 3-hit subnetwork, control vs AD
# ---------------------------------------------------------------------------


def plot_ad_vs_control_bipartite(mat_ctrl, mat_ad, gene_to_idx, triplets: pd.DataFrame,
                                  save_path: str, k_context_per_tf: int = 6):
    """Two side-by-side bipartite panels:
    3 hit TFs (left column) and their hit targets + extra context targets
    (right column).  The 3 hit edges are drawn red; the rest grey.  This
    gives the network real density without changing the headline story.
    """
    hit_tfs = [tf for tf, _ in HIT_EDGES]
    hit_set = set(HIT_EDGES)

    # Pick the targets to display: the 3 hit targets + up to k_context_per_tf
    # additional targets per TF, ranked by abs(weight) in the control matrix.
    target_set: list[str] = [t for _, t in HIT_EDGES]
    for tf in hit_tfs:
        if tf not in gene_to_idx:
            continue
        j = gene_to_idx[tf]
        candidates = (triplets[triplets["TF"] == tf]["target_gene"].unique().tolist())
        candidates = [c for c in candidates if c in gene_to_idx and c != tf]
        candidates.sort(key=lambda c: abs(mat_ctrl[gene_to_idx[c], j]), reverse=True)
        for c in candidates:
            if c in target_set:
                continue
            if (tf, c) in hit_set:
                continue
            target_set.append(c)
            if sum(1 for t in target_set
                   if (tf, t) not in hit_set
                   and abs(mat_ctrl[gene_to_idx[t], j]) > 0) > k_context_per_tf * len(hit_tfs):
                break

    # Deduplicate but keep order
    seen = set()
    targets = [t for t in target_set if not (t in seen or seen.add(t))]

    def build(mat):
        G = nx.DiGraph()
        for tf in hit_tfs:
            G.add_node(tf, kind="TF")
        for tgt in targets:
            G.add_node(tgt, kind="target")
        # Hit edges
        for tf, tgt in HIT_EDGES:
            if tf in gene_to_idx and tgt in gene_to_idx:
                w = mat[gene_to_idx[tgt], gene_to_idx[tf]]
                G.add_edge(tf, tgt, weight=w, hit=True)
        # Context edges: for each TF, draw to each target in the display
        # set whose triplet pairing exists (i.e. it's a candidate edge).
        triplet_pairs = set(zip(triplets["TF"], triplets["target_gene"]))
        for tf in hit_tfs:
            if tf not in gene_to_idx:
                continue
            j = gene_to_idx[tf]
            for tgt in targets:
                if (tf, tgt) in hit_set:
                    continue
                if (tf, tgt) not in triplet_pairs:
                    continue
                if tgt not in gene_to_idx:
                    continue
                w = mat[gene_to_idx[tgt], j]
                G.add_edge(tf, tgt, weight=w, hit=False)
        return G

    G_ctrl = build(mat_ctrl)
    G_ad = build(mat_ad)

    # Bipartite layout: TFs on left at x=0, targets on right at x=1.
    # Order targets so each one is closest to the TF that connects most
    # strongly — minimises edge crossings.
    pos = {}
    for i, tf in enumerate(hit_tfs):
        pos[tf] = (0.0, (len(hit_tfs) - 1 - i) * (len(targets) / max(len(hit_tfs), 1)))

    # Assign each target to a primary TF (the one with strongest control weight)
    target_primary_tf = {}
    for tgt in targets:
        best_tf, best_w = None, -1.0
        for tf in hit_tfs:
            if tf not in gene_to_idx or tgt not in gene_to_idx:
                continue
            w = abs(mat_ctrl[gene_to_idx[tgt], gene_to_idx[tf]])
            if w > best_w:
                best_w, best_tf = w, tf
        target_primary_tf[tgt] = best_tf

    # Group targets by their primary TF, place each group near that TF
    targets_by_tf = {tf: [] for tf in hit_tfs}
    for tgt in targets:
        targets_by_tf[target_primary_tf[tgt]].append(tgt)
    y_cursor = 0.0
    for tf in hit_tfs:
        for tgt in targets_by_tf[tf]:
            pos[tgt] = (1.6, y_cursor)
            y_cursor += 1.0

    # figure size scales with target count for readability
    fig_h = max(8.0, 0.45 * len(targets))
    fig, axes = plt.subplots(1, 2, figsize=(18, fig_h))
    # global edge-weight max so both panels share scale
    shared_wmax = max(
        max((abs(d["weight"]) for *_, d in G_ctrl.edges(data=True)), default=0.0),
        max((abs(d["weight"]) for *_, d in G_ad.edges(data=True)), default=0.0),
    ) or 1.0
    for ax, G, title in [
        (axes[0], G_ctrl, "Average control donor"),
        (axes[1], G_ad,   "Average AD donor"),
    ]:
        # TF nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=hit_tfs, node_color="#e8a020",
            node_shape="D", node_size=5400, edgecolors="white", linewidths=2.5,
        )
        # Target nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=targets, node_color="#88aacc",
            node_size=2400, edgecolors="white", linewidths=2.0,
        )

        # Draw context edges first (grey, thinner), then hit edges (red, thicker)
        ctx_edges = [(u, v, d) for u, v, d in G.edges(data=True) if not d.get("hit")]
        hit_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("hit")]

        if ctx_edges:
            widths = [0.5 + 5.0 * abs(d["weight"]) / shared_wmax for *_, d in ctx_edges]
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=[(u, v) for u, v, _ in ctx_edges],
                width=widths, edge_color="#9a9a9a", arrows=True,
                arrowsize=18, alpha=0.55,
                min_target_margin=20, node_size=2400,
            )
        if hit_edges:
            widths = [3.5 + 12.0 * abs(d["weight"]) / shared_wmax for *_, d in hit_edges]
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=[(u, v) for u, v, _ in hit_edges],
                width=widths, edge_color="#d63838", arrows=True,
                arrowsize=32, alpha=0.95,
                min_target_margin=22, node_size=2400,
            )

        # Labels — TFs bold and larger
        nx.draw_networkx_labels(G, pos, ax=ax, labels={t: t for t in hit_tfs},
                                 font_size=16, font_weight="bold")
        nx.draw_networkx_labels(G, pos, ax=ax, labels={t: t for t in targets},
                                 font_size=11)

        # Weight annotations only on the 3 hit edges (don't clutter context)
        for u, v, d in hit_edges:
            x_mid = (pos[u][0] + pos[v][0]) / 2
            y_mid = (pos[u][1] + pos[v][1]) / 2
            ax.text(x_mid, y_mid + 0.25, f"w={d['weight']:.1e}",
                    ha="center", fontsize=10, color="#b71c1c",
                    bbox=dict(facecolor="white", edgecolor="#d63838", alpha=0.9, pad=2))

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlim(-0.4, 2.0)
        ax.set_ylim(-1.0, len(targets) + 0.5)
        ax.set_aspect("equal")
        ax.set_axis_off()

    legend = [
        mpatches.Patch(color="#e8a020", label="Transcription factor"),
        mpatches.Patch(color="#88aacc", label="Target gene"),
        mpatches.Patch(color="#d63838", label="q<0.05 differential edge"),
        mpatches.Patch(color="#9a9a9a", label="other inferred edges (context)"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=4, frameon=False, fontsize=11)
    fig.suptitle(
        f"Regulatory subnetwork around the 3 L2/3 IT q<0.05 hits — control vs AD\n"
        f"red edges = significant differential edges; grey = other candidate regulatory edges from Phase 3",
        fontsize=13, y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"wrote {save_path}")


# ---------------------------------------------------------------------------
# Plot 3: cohort table colored by ADNC severity (4 levels)
# ---------------------------------------------------------------------------


def plot_cohort_table_adnc(rna_sub, save_path: str):
    """Donor metadata table, one row per donor, colored by ADNC severity.

    Pulls the 4-level ADNC value directly from obs (not the binarized
    condition column) so the gradient is visible at a glance.
    """
    # one row per donor across both target cell types; we use the union
    rows = []
    target_types = ["Microglia-PVM", "L2/3 IT"]
    obs = rna_sub.obs
    for d, grp in obs.groupby("Donor ID", observed=True):
        first = grp.iloc[0]
        adnc = str(first["Overall AD neuropathological Change"])
        if adnc not in ADNC_ORDER:
            continue
        n_micro = int((grp["cell_type"] == "Microglia-PVM").sum())
        n_l23   = int((grp["cell_type"] == "L2/3 IT").sum())
        rows.append({
            "donor_id": d,
            "ADNC": adnc,
            "age": float(first["Age at Death"]) if pd.notna(first["Age at Death"]) else np.nan,
            "sex": str(first["Sex"]),
            "n_Microglia": n_micro,
            "n_L2/3 IT": n_l23,
            "LATE": "Y" if bool(first.get("LATE_present", False)) else "—",
            "Lewy": "Y" if bool(first.get("LBD_present", False)) else "—",
        })
    df = pd.DataFrame(rows)
    # order by ADNC severity, then donor_id
    df["_sev"] = df["ADNC"].map(lambda x: ADNC_ORDER.index(x) if x in ADNC_ORDER else 99)
    df = df.sort_values(["_sev", "donor_id"]).drop(columns=["_sev"]).reset_index(drop=True)
    df_display = df.copy()
    df_display["age"] = df_display["age"].round(0).astype("Int64")

    n_rows = len(df_display)
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * n_rows + 1.3))
    ax.axis("off")

    cols = df_display.columns.tolist()
    tbl = ax.table(
        cellText=df_display.values, colLabels=cols,
        loc="center", cellLoc="center", colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)

    # color rows by ADNC
    for i, row in df_display.iterrows():
        color = ADNC_COLORS[row["ADNC"]]
        for j in range(len(cols)):
            tbl[(i + 1, j)].set_facecolor(color)
    # header style
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#444444")
        tbl[(0, j)].set_text_props(weight="bold", color="white")

    # legend below
    legend_handles = [
        mpatches.Patch(color=ADNC_COLORS[lvl], label=lvl) for lvl in ADNC_ORDER
    ]
    ax.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=10, frameon=False,
        title="ADNC severity (Overall AD neuropathological Change)",
    )

    ax.set_title(
        f"SeaAD cohort — {n_rows} donors, sorted by Alzheimer's severity",
        fontsize=12, pad=10,
    )
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"wrote {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    logger.info("loading rna_sub ...")
    rna_sub = _load_tagged_rna()

    # cohort table doesn't need network data — do it first
    plot_cohort_table_adnc(
        rna_sub, os.path.join(PLOTS_DIR, "cohort_table_adnc.png")
    )

    # Cache control/AD pseudobulk means + gene index so future styling
    # iterations skip the 10-min network reload.
    cache_path = os.path.join(OUT_DIR, f"{DATASET}_{SUB_TAG}_pb_means.npz")
    if os.path.exists(cache_path):
        logger.info(f"loading cached pseudobulk means from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        mat_ctrl = cache["mat_ctrl"]
        mat_ad = cache["mat_ad"]
        gene_names = list(cache["gene_names"])
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    else:
        logger.info("loading per-cell networks (slow, ~10 min) ...")
        networks = combine_wscreni_networks(
            cell_names=rna_sub.obs_names.tolist(),
            network_dir=NETWORKS_DIR,
        )
        gene_to_idx = {g: i for i, g in enumerate(networks.gene_names)}

        logger.info(f"pseudobulk for {CELL_TYPE} ...")
        pb, _meta = _pseudobulks_for_ct(rna_sub, networks, CELL_TYPE)
        mat_ctrl = _condition_avg(pb, "control")
        mat_ad = _condition_avg(pb, "ad")
        np.savez(
            cache_path,
            mat_ctrl=mat_ctrl, mat_ad=mat_ad,
            gene_names=np.array(list(networks.gene_names), dtype=object),
        )
        logger.info(f"  cached -> {cache_path}")

    triplets = pd.read_csv(TRIPLETS_PATH)

    # Plot 1: multi-TF fan-out using control matrix.  Use the 3 hit TFs
    # plus 8 of their top inferred targets each for a richer network look.
    plot_single_tf_fanout(
        mat=mat_ctrl, gene_to_idx=gene_to_idx, triplets=triplets,
        tfs=[tf for tf, _ in HIT_EDGES],
        k_per_tf=8,
        save_path=os.path.join(PLOTS_DIR, "network_example_v2.png"),
    )

    # Plot 2: bipartite control vs AD with 3 hit edges (red) plus ~6
    # context edges per TF (grey) to other candidate triplet targets.
    plot_ad_vs_control_bipartite(
        mat_ctrl=mat_ctrl, mat_ad=mat_ad, gene_to_idx=gene_to_idx,
        triplets=triplets, k_context_per_tf=6,
        save_path=os.path.join(PLOTS_DIR, "network_ad_vs_control_v2.png"),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
