"""SQ3 step 1: extract per-cell incoming-regulatory-activity from the wScReNI networks.

For each cell we reduce its (target x regulator) weight matrix to a per-gene *incoming
activity* vector = the row sum (total regulatory weight each target gene receives in that
cell). Stacked across cells this gives a (n_cells x n_genes) matrix that is the input to the
cross-cell co-regulation networks used for module preservation (SQ3).

Why incoming activity: two genes are "co-regulated" when their regulatory input rises and
falls together across cells — the GRN analogue of co-expression. A cross-DONOR prototype
(n=27) over this signal gave coherent, AD-relevant modules (APOE/IRF8, CEBPD/CSF1, DAB1),
unlike the per-arm mean matrix. The per-arm split needs cells (7 control donors is too few
to correlate), hence this per-cell extraction.

Orientation note: the loaded wScReNI matrix is [target(row), regulator(col)]
(see scripts/compute_all_coefficients.py:202), so incoming activity = sum over axis=1.

Run on the cluster (reads the 47 GB of per-cell network files via combine_wscreni_networks).
Output cache is tiny (~20 MB) and reused by scripts/sq3_module_preservation_run.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import anndata as ad
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from screni.data.combine import combine_wscreni_networks
from screni.data.loading_seaad import add_condition_column, add_copathology_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATASET = "seaad_paired"
SUB_TAG = "sq1"
CELL_TYPES = [("Microglia-PVM", "Microglia_PVM"), ("L2/3 IT", "L2_3_IT")]

DATA_DIR = os.path.join(REPO_ROOT, "data", "processed", "seaad")
RNA_SUB_PATH = os.path.join(DATA_DIR, f"{DATASET}_rna_{SUB_TAG}.h5ad")
RNA_FULL_PATH = os.path.join(DATA_DIR, f"{DATASET}_rna.h5ad")

OUT_DIR = os.path.join(REPO_ROOT, "src", "screni", "data", "output")
NETWORKS_DIR = os.path.join(OUT_DIR, f"networks_{DATASET}")

NEEDED_OBS = [
    "Donor ID", "Overall AD neuropathological Change",
    "LATE", "Highest Lewy Body Disease", "Age at Death", "Sex",
    "Continuous Pseudo-progression Score",
]


def _load_tagged_rna() -> ad.AnnData:
    rna_sub = ad.read_h5ad(RNA_SUB_PATH)
    missing = [c for c in NEEDED_OBS if c not in rna_sub.obs.columns]
    if missing:
        rna_full = ad.read_h5ad(RNA_FULL_PATH, backed="r")
        full_obs = rna_full.obs.loc[rna_sub.obs_names, missing].copy()
        rna_full.file.close()
        for c in missing:
            rna_sub.obs[c] = full_obs[c].values
    add_condition_column(rna_sub)
    add_copathology_columns(rna_sub)
    return rna_sub


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=0,
                        help="if >0, load only this many cells (fast end-to-end check)")
    args = parser.parse_args()

    logger.info("SQ3 incoming-activity extraction")
    rna_sub = _load_tagged_rna()
    logger.info(f"rna_sub: {rna_sub.shape}")

    smoke_n = args.smoke
    cell_names = rna_sub.obs_names.tolist()
    if smoke_n > 0:
        cell_names = cell_names[:smoke_n]
        rna_sub = rna_sub[cell_names].copy()
        logger.info(f"SMOKE mode: limiting to {smoke_n} cells")

    logger.info("loading per-cell wScReNI networks (slow, ~10 min) ...")
    networks = combine_wscreni_networks(
        cell_names=cell_names,
        network_dir=NETWORKS_DIR,
    )
    genes = list(networks.gene_names)
    n_genes = len(genes)
    logger.info(f"  {len(networks)} networks, {n_genes} genes")

    for ct_pretty, ct_safe in CELL_TYPES:
        mask = (
            (rna_sub.obs["cell_type"].astype(str) == ct_pretty)
            & rna_sub.obs["condition"].notna()
        )
        cells = [c for c in rna_sub.obs_names[mask.values] if c in networks]
        logger.info(f"\n{ct_pretty}: {len(cells)} cells")

        # per-cell incoming activity = row sum (sum over regulators) of [target, regulator] matrix
        activity = np.zeros((len(cells), n_genes), dtype=np.float32)
        for i, c in enumerate(cells):
            m = np.asarray(networks[c], dtype=np.float32)
            activity[i] = m.sum(axis=1)
            if (i + 1) % 200 == 0:
                logger.info(f"  {i + 1}/{len(cells)} cells")

        obs = rna_sub.obs.loc[cells]
        cache = os.path.join(OUT_DIR, f"sq3_incoming_activity_{ct_safe}.npz")
        np.savez(
            cache,
            activity=activity,                                      # (n_cells, n_genes)
            gene_names=np.array(genes, dtype=object),
            cell_names=np.array(cells, dtype=object),
            donor_ids=obs["Donor ID"].astype(str).values,
            condition=obs["condition"].astype(str).values,
            adnc_ordinal=obs["adnc_ordinal"].astype(float).values,
        )
        n_ctrl = int((obs["condition"].astype(str) == "control").sum())
        n_ad = int((obs["condition"].astype(str) == "ad").sum())
        logger.info(f"  wrote {cache}  activity={activity.shape}  control={n_ctrl} ad={n_ad}")

    logger.info("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
