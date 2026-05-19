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
                          tf: str, k_targets: int = 5, save_path: str = ""):
    """One TF + its top-K candidate targets, drawn as a star.

    Targets are chosen from the Phase 3 triplets for this TF, ranked by
    absolute weight in the supplied (n_genes, n_genes) matrix.
    """
    candidates = triplets[triplets["TF"] == tf]["target_gene"].unique().tolist()
    j = gene_to_idx[tf]
    weights = [(t, mat[gene_to_idx[t], j]) for t in candidates if t in gene_to_idx and t != tf]
    weights.sort(key=lambda kv: abs(kv[1]), reverse=True)
    top = weights[:k_targets]
    if not top:
        logger.warning(f"no targets to plot for {tf}")
        return

    G = nx.DiGraph()
    G.add_node(tf, kind="TF")
    for t, w in top:
        G.add_node(t, kind="target")
        G.add_edge(tf, t, weight=w)

    # star layout: TF center, targets around in a circle
    pos = {tf: (0.0, 0.0)}
    n = len(top)
    for i, (t, _) in enumerate(top):
        angle = 2 * np.pi * i / n
        pos[t] = (np.cos(angle), np.sin(angle))

    fig, ax = plt.subplots(figsize=(8.5, 7))

    # nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=[tf], node_color="#e8a020",
        node_shape="D", node_size=2400, edgecolors="white", linewidths=2,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=[t for t, _ in top], node_color="#88aacc",
        node_size=1500, edgecolors="white", linewidths=2,
    )

    # edges scaled by abs(weight)
    wmax = max(abs(w) for _, w in top) or 1.0
    widths = [1.5 + 6.0 * abs(w) / wmax for _, w in top]
    nx.draw_networkx_edges(
        G, pos, ax=ax, width=widths, edge_color="#555555",
        arrows=True, arrowsize=22, alpha=0.7,
        min_target_margin=22, node_size=1500,
    )

    # labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=13, font_weight="bold")

    ax.set_title(
        f"Example: a single-cell regulatory subnetwork\n"
        f"TF '{tf}' (orange diamond) regulates target genes (blue circles) — "
        f"thicker arrows = stronger inferred regulation",
        fontsize=12,
    )
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.5, 1.5)
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


def plot_ad_vs_control_bipartite(mat_ctrl, mat_ad, gene_to_idx, save_path: str):
    """Two side-by-side bipartite panels:
    3 hit TFs (left column) and their 3 hit targets (right column),
    plus a small number of context-edges to other targets to give shape.
    """
    tfs = [tf for tf, _ in HIT_EDGES]
    targets = [tgt for _, tgt in HIT_EDGES]

    def build(mat):
        G = nx.DiGraph()
        for tf in tfs:
            G.add_node(tf, kind="TF")
        for tgt in targets:
            G.add_node(tgt, kind="target")
        for tf, tgt in HIT_EDGES:
            if tf in gene_to_idx and tgt in gene_to_idx:
                w = mat[gene_to_idx[tgt], gene_to_idx[tf]]
                G.add_edge(tf, tgt, weight=w, hit=True)
        return G

    G_ctrl = build(mat_ctrl)
    G_ad = build(mat_ad)

    # bipartite layout: TFs left at x=0, targets right at x=1
    pos = {}
    for i, tf in enumerate(tfs):
        pos[tf] = (0.0, len(tfs) - 1 - i)
    for i, tgt in enumerate(targets):
        pos[tgt] = (1.0, len(targets) - 1 - i)

    # equal axis range so the panels look identical in scale
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, G, title in [
        (axes[0], G_ctrl, "Average control donor"),
        (axes[1], G_ad,   "Average AD donor"),
    ]:
        # TF nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=tfs, node_color="#e8a020",
            node_shape="D", node_size=2400, edgecolors="white", linewidths=2,
        )
        # Target nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax, nodelist=targets, node_color="#88aacc",
            node_size=1600, edgecolors="white", linewidths=2,
        )
        # Edges, widths scaled across both panels
        edges = list(G.edges(data=True))
        if edges:
            shared_wmax = max(
                max(abs(d["weight"]) for *_, d in G_ctrl.edges(data=True)),
                max(abs(d["weight"]) for *_, d in G_ad.edges(data=True)),
            ) or 1.0
            widths = [1.0 + 8.0 * abs(d["weight"]) / shared_wmax for *_, d in edges]
            colors = ["#d63838"] * len(edges)   # all 3 are hits → red
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=[(u, v) for u, v, _ in edges],
                width=widths, edge_color=colors, arrows=True,
                arrowsize=22, alpha=0.85,
                min_target_margin=22, node_size=1600,
            )
        # Labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=13, font_weight="bold")
        # weight annotations next to each edge
        for u, v, d in edges:
            x_mid = (pos[u][0] + pos[v][0]) / 2
            y_mid = (pos[u][1] + pos[v][1]) / 2
            ax.text(x_mid, y_mid + 0.07, f"w={d['weight']:.1e}",
                    ha="center", fontsize=9, color="#444444",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        ax.set_title(title, fontsize=13)
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(-0.6, len(tfs) - 0.4)
        ax.set_aspect("equal")
        ax.set_axis_off()

    legend = [
        mpatches.Patch(color="#e8a020", label="Transcription factor"),
        mpatches.Patch(color="#88aacc", label="Target gene"),
        mpatches.Patch(color="#d63838", label="q<0.05 differential edge"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=3, frameon=False, fontsize=11)
    fig.suptitle(
        f"Top differential regulatory edges ({CELL_TYPE}) — control vs AD\n"
        f"thicker arrows = stronger inferred regulation; weights labeled on each edge",
        fontsize=13, y=0.98,
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

    triplets = pd.read_csv(TRIPLETS_PATH)

    # Plot 1: single-TF fan-out using control matrix.  Pick LEF1 (one of
    # our hits, also a well-known TF the audience may recognise from Wnt).
    plot_single_tf_fanout(
        mat=mat_ctrl, gene_to_idx=gene_to_idx, triplets=triplets,
        tf="LEF1", k_targets=5,
        save_path=os.path.join(PLOTS_DIR, "network_example_v2.png"),
    )

    # Plot 2: bipartite control vs AD over the 3 hit edges only
    plot_ad_vs_control_bipartite(
        mat_ctrl=mat_ctrl, mat_ad=mat_ad, gene_to_idx=gene_to_idx,
        save_path=os.path.join(PLOTS_DIR, "network_ad_vs_control_v2.png"),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
