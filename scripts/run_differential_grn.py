"""SQ1 differential GRN analysis on the wScReNI inference output.

Script counterpart to ``src/screni/data/run_differential_grn.ipynb`` for
cluster execution.  Loops over both target cell types in a single run so
we produce the full SQ1 deliverable (per-cell-type ranked TF->target
table + Wilcoxon sensitivity) without re-loading networks twice.

Pipeline:

1. Load the subsampled RNA + backfill obs metadata from the full RNA
   file if the SeaAD ``_sq1`` h5ad doesn't already carry it.
2. Load per-cell wScReNI networks from disk via combine_wscreni_networks.
3. For each target cell type (Microglia-PVM, L2/3 IT):
   a. Filter to that cell type.
   b. Build per-donor metadata (one row per donor).
   c. Pseudobulk: average per-cell network matrices per donor.
   d. Load Phase 3 candidate edges.
   e. Run differential_edges with OLS (primary) and Wilcoxon (sensitivity).
   f. Write the ranked CSV(s).

Inference output is reused across both cell types — the network files
on disk contain all 2552 cells; we filter per cell type at pseudobulk
time, not at inference time.
"""

from __future__ import annotations

import logging
import os
import sys

import anndata as ad
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from screni.data.combine import combine_wscreni_networks
from screni.data.differential import (
    differential_edges,
    ols_with_covariates,
    pseudobulk_per_donor,
    wilcoxon_unadjusted,
)
from screni.data.loading_seaad import (
    add_condition_column,
    add_copathology_columns,
    select_eligible_donors,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---- Configuration ----------------------------------------------------------

DATASET = "seaad_paired"
SUB_TAG = "sq1"
TARGET_CELL_TYPES = ["Microglia-PVM", "L2/3 IT"]

DATA_DIR = os.path.join(REPO_ROOT, "data", "processed", "seaad")
RNA_FULL_PATH = os.path.join(DATA_DIR, f"{DATASET}_rna.h5ad")
RNA_SUB_PATH = os.path.join(DATA_DIR, f"{DATASET}_rna_{SUB_TAG}.h5ad")
TRIPLETS_PATH = os.path.join(DATA_DIR, f"{DATASET}_{SUB_TAG}_triplets.csv")

OUT_DIR = os.path.join(REPO_ROOT, "src", "screni", "data", "output")
NETWORKS_DIR = os.path.join(OUT_DIR, f"networks_{DATASET}")

# obs columns the differential regression needs.  If the _sq1.h5ad already
# carries them (it does, per the prep we ran), the backfill is a no-op.
NEEDED_OBS = [
    "Donor ID", "Overall AD neuropathological Change",
    "LATE", "Highest Lewy Body Disease", "Age at Death", "Sex",
]


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace("-", "_")


def main() -> int:
    logger.info("=" * 70)
    logger.info(f"SQ1 differential GRN | DATASET={DATASET} | SUB_TAG={SUB_TAG}")
    logger.info(f"target cell types : {TARGET_CELL_TYPES}")
    logger.info("=" * 70)

    for label, path in [
        ("RNA full", RNA_FULL_PATH), ("RNA sub", RNA_SUB_PATH),
        ("triplets", TRIPLETS_PATH), ("networks dir", NETWORKS_DIR),
    ]:
        ok = "OK" if os.path.exists(path) else "MISSING"
        logger.info(f"  {ok:7s} {label}: {path}")
        if not os.path.exists(path):
            logger.error(f"Required input missing: {path}")
            return 2
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Step 1: load subsampled RNA + backfill obs --------------------------
    logger.info("\n=== Step 1: load subsampled RNA + tag condition/co-pathology ===")
    rna_sub = ad.read_h5ad(RNA_SUB_PATH)
    logger.info(f"  rna_sub: {rna_sub.shape}")
    logger.info(f"  obs cols available: {len(rna_sub.obs.columns)}")

    missing = [c for c in NEEDED_OBS if c not in rna_sub.obs.columns]
    if missing:
        logger.info(f"  backfilling obs columns from full RNA: {missing}")
        rna_full = ad.read_h5ad(RNA_FULL_PATH, backed="r")
        full_obs = rna_full.obs.loc[rna_sub.obs_names, missing].copy()
        rna_full.file.close()
        for c in missing:
            rna_sub.obs[c] = full_obs[c].values

    add_condition_column(rna_sub)
    add_copathology_columns(rna_sub)
    logger.info(
        f"  condition counts: {rna_sub.obs['condition'].value_counts().to_dict()}"
    )

    # ---- Step 2: load all per-cell wScReNI networks (once) ------------------
    logger.info("\n=== Step 2: load per-cell wScReNI networks ===")
    networks = combine_wscreni_networks(
        cell_names=rna_sub.obs_names.tolist(),
        network_dir=NETWORKS_DIR,
    )
    logger.info(f"  loaded {len(networks)} per-cell networks")
    if networks.gene_names:
        logger.info(f"  gene_names: {len(networks.gene_names)} entries")

    # ---- Step 3: load Phase 3 candidate edges -------------------------------
    triplets = pd.read_csv(TRIPLETS_PATH)
    unique_edges = triplets[["TF", "target_gene"]].drop_duplicates()
    logger.info(
        f"\n  triplets: {triplets.shape}; unique TF->target pairs: {len(unique_edges)}"
    )

    # ---- Step 4: per-cell-type differential analysis ------------------------
    for ct in TARGET_CELL_TYPES:
        logger.info("\n" + "=" * 70)
        logger.info(f"Cell type: {ct}")
        logger.info("=" * 70)

        # 4a. filter cells
        mask = (
            (rna_sub.obs["cell_type"].astype(str) == ct)
            & rna_sub.obs["condition"].notna()
        )
        rna_ct = rna_sub[mask].copy()
        if rna_ct.n_obs == 0:
            logger.warning(f"  no cells matched cell_type={ct!r}; skipping")
            continue
        logger.info(
            f"  cells: {rna_ct.n_obs}; donors: {rna_ct.obs['Donor ID'].nunique()}; "
            f"condition: {rna_ct.obs['condition'].value_counts().to_dict()}"
        )

        # 4b. per-donor metadata table
        donor_meta = select_eligible_donors(
            rna_ct, cell_type=ct, min_cells_per_donor=1,
            require_condition=True,
        ).set_index("donor_id")
        logger.info(
            f"\n  donor metadata ({len(donor_meta)} donors, "
            f"control={(donor_meta['condition']=='control').sum()}, "
            f"ad={(donor_meta['condition']=='ad').sum()}):"
        )
        logger.info("\n" + donor_meta.to_string())

        # 4c. pseudobulk per donor on the filtered cell subset
        cells_in_ct = list(rna_ct.obs_names)
        networks_ct = type(networks)(
            {c: networks[c] for c in cells_in_ct if c in networks},
            gene_names=networks.gene_names,
        )
        cell_to_donor = rna_ct.obs["Donor ID"].astype(str).to_dict()
        pseudobulks = pseudobulk_per_donor(
            networks=networks_ct,
            cell_to_donor=cell_to_donor,
            donor_metadata=donor_meta,
            aggregator="mean",
        )

        # 4d. OLS-with-covariates differential test (primary)
        logger.info(f"\n  running OLS-with-covariates differential test ...")
        results_ols = differential_edges(
            pseudobulks=pseudobulks,
            edges=triplets,
            test=ols_with_covariates,
            cell_type=ct,
        )
        ols_csv = os.path.join(
            OUT_DIR, f"differential_edges_{_safe(ct)}.csv"
        )
        results_ols.to_csv(ols_csv, index=False)
        logger.info(f"  wrote OLS results -> {ols_csv}")
        logger.info(f"\n  top-10 by q-value:\n{results_ols.head(10).to_string()}")

        # 4e. Wilcoxon sensitivity check (no covariate adjustment)
        logger.info(f"\n  running Wilcoxon sensitivity test ...")
        results_wx = differential_edges(
            pseudobulks=pseudobulks,
            edges=triplets,
            test=wilcoxon_unadjusted,
            cell_type=ct,
        )
        wx_csv = os.path.join(
            OUT_DIR, f"differential_edges_{_safe(ct)}_wilcoxon.csv"
        )
        results_wx.to_csv(wx_csv, index=False)
        logger.info(f"  wrote Wilcoxon results -> {wx_csv}")

        # 4f. overlap diagnostic
        top_ols = set(zip(results_ols.head(50)["TF"], results_ols.head(50)["target"]))
        top_wx = set(zip(results_wx.head(50)["TF"], results_wx.head(50)["target"]))
        overlap = top_ols & top_wx
        logger.info(
            f"\n  overlap of top-50 (OLS ∩ Wilcoxon): {len(overlap)} / 50"
        )

    logger.info("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
