"""Per-edge OLS with ALL coefficient stats retained.

The differential pipeline so far only saves stats for the AD_severity
coefficient (the one we care about for SQ1).  This script reruns the
same OLS but keeps stats for ALL predictors:

    weight ~ adnc_ordinal + age + sex_Male + LATE_present + LBD_present

For each cell type it produces:

* ``output/coefficients_all_<celltype>.csv`` — one row per (TF, target)
  edge, columns for each predictor's coef, stderr, t-stat, p-value,
  q-value, plus the per-condition mean weights.

* ``output/per_edge_per_donor_<celltype>.npz`` — small cache
  ``(n_edges, n_donors)`` array of per-donor edge weights, plus the
  donor ids + the (TF, target) index.  This lets the plotting script
  draw scatter/box plots without re-loading the 47 GB of network files.

Run on the cluster (needs the full per-cell networks if no cache yet).
Subsequent re-runs are fast because the per-edge cache is small.
"""

from __future__ import annotations

import logging
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from screni.data.combine import combine_wscreni_networks
from screni.data.differential import benjamini_hochberg, pseudobulk_per_donor
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
CELL_TYPES = [("Microglia-PVM", "Microglia_PVM"), ("L2/3 IT", "L2_3_IT")]

DATA_DIR = os.path.join(REPO_ROOT, "data", "processed", "seaad")
RNA_SUB_PATH = os.path.join(DATA_DIR, f"{DATASET}_rna_{SUB_TAG}.h5ad")
RNA_FULL_PATH = os.path.join(DATA_DIR, f"{DATASET}_rna.h5ad")
TRIPLETS_PATH = os.path.join(DATA_DIR, f"{DATASET}_{SUB_TAG}_triplets.csv")

OUT_DIR = os.path.join(REPO_ROOT, "src", "screni", "data", "output")
NETWORKS_DIR = os.path.join(OUT_DIR, f"networks_{DATASET}")

NEEDED_OBS = [
    "Donor ID", "Overall AD neuropathological Change",
    "LATE", "Highest Lewy Body Disease", "Age at Death", "Sex",
    "Continuous Pseudo-progression Score",
]

# Predictors in the OLS, IN ORDER.  Position 0 is the intercept (added by
# sm.add_constant).  The remaining 5 are the regressors we fit explicitly.
PREDICTORS = ["adnc_ordinal", "age", "sex_Male", "LATE_present", "LBD_present"]


# ---- Helpers ----------------------------------------------------------------


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


def _build_design_matrix(donor_metadata: pd.DataFrame, donor_ids: list[str]) -> np.ndarray:
    """Return design matrix (n_donors, 6) with constant + 5 predictors,
    in the order in PREDICTORS.
    """
    meta = donor_metadata.loc[donor_ids]
    cols = {
        "adnc_ordinal": meta["adnc_ordinal"].astype(float).values,
        "age": meta["age"].astype(float).values,
        "sex_Male": (meta["sex"].astype(str) == "Male").astype(float).values,
        "LATE_present": meta["LATE_present"].astype(float).values,
        "LBD_present": meta["LBD_present"].astype(float).values,
    }
    X = np.column_stack([cols[p] for p in PREDICTORS])
    X = sm.add_constant(X, has_constant="add")   # (n_donors, 6)
    return X


def _fit_one(edge_values: np.ndarray, X: np.ndarray) -> dict:
    """Fit OLS and return all 6 coefficients' stats."""
    n_predictors = X.shape[1]
    try:
        fit = sm.OLS(edge_values, X).fit()
        return {
            "coefs":   np.asarray(fit.params),
            "stderrs": np.asarray(fit.bse),
            "tvalues": np.asarray(fit.tvalues),
            "pvalues": np.asarray(fit.pvalues),
        }
    except Exception as e:
        logger.debug(f"OLS failed: {e}")
        return {
            "coefs":   np.full(n_predictors, np.nan),
            "stderrs": np.full(n_predictors, np.nan),
            "tvalues": np.full(n_predictors, np.nan),
            "pvalues": np.full(n_predictors, np.nan),
        }


# ---- Main pipeline ----------------------------------------------------------


def main() -> int:
    logger.info("=" * 70)
    logger.info("All-coefficient OLS analysis")
    logger.info("=" * 70)

    rna_sub = _load_tagged_rna()
    logger.info(f"rna_sub: {rna_sub.shape}")

    logger.info("loading per-cell wScReNI networks (slow, ~10 min) ...")
    networks = combine_wscreni_networks(
        cell_names=rna_sub.obs_names.tolist(),
        network_dir=NETWORKS_DIR,
    )
    gene_to_idx = {g: i for i, g in enumerate(networks.gene_names)}
    logger.info(f"  loaded {len(networks)} networks, {len(gene_to_idx)} genes")

    triplets = pd.read_csv(TRIPLETS_PATH)
    unique_edges = (
        triplets[["TF", "target_gene"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    logger.info(f"unique candidate edges: {len(unique_edges)}")

    for ct_pretty, ct_safe in CELL_TYPES:
        logger.info("\n" + "=" * 70)
        logger.info(f"Cell type: {ct_pretty}")
        logger.info("=" * 70)

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
        donor_ids = list(pb.weights.keys())
        n_donors = len(donor_ids)
        logger.info(f"  {n_donors} donors, building design matrix ...")
        X = _build_design_matrix(pb.metadata, donor_ids)

        # Stack the per-donor pseudobulk matrices into one (n_donors, G, G)
        # tensor so we can extract per-edge weight vectors quickly.
        weights_tensor = np.stack([pb.weights[d] for d in donor_ids], axis=0)
        logger.info(f"  pseudobulk tensor: {weights_tensor.shape}")

        # Fit OLS for every candidate edge
        rows = []
        per_edge_per_donor = []
        is_ad = pb.metadata.loc[donor_ids, "condition"].astype(str).values == "ad"
        for ei, (_, edge) in enumerate(unique_edges.iterrows()):
            tf = edge["TF"]
            target = edge["target_gene"]
            if tf not in gene_to_idx or target not in gene_to_idx:
                continue
            i_tgt = gene_to_idx[target]
            i_tf = gene_to_idx[tf]
            # wScReNI weight matrix: [target, regulator]
            w = weights_tensor[:, i_tgt, i_tf]
            stats = _fit_one(w, X)

            ctrl = w[~is_ad]
            ad_arr = w[is_ad]
            mean_ctrl = float(np.mean(ctrl)) if len(ctrl) else np.nan
            mean_ad = float(np.mean(ad_arr)) if len(ad_arr) else np.nan
            eps = 1e-12
            log2fc = (
                float(
                    np.log2(max(abs(mean_ad), eps) / max(abs(mean_ctrl), eps))
                    * np.sign(mean_ad - mean_ctrl)
                )
                if not np.isnan(mean_ad) and not np.isnan(mean_ctrl)
                else np.nan
            )

            row = {
                "TF": tf,
                "target": target,
                "mean_control": mean_ctrl,
                "mean_ad": mean_ad,
                "log2FC": log2fc,
                "n_donors": n_donors,
            }
            # Position 0 = intercept; positions 1..5 = our predictors
            for pi, pname in enumerate(PREDICTORS):
                idx = pi + 1
                row[f"{pname}_coef"] = float(stats["coefs"][idx])
                row[f"{pname}_stderr"] = float(stats["stderrs"][idx])
                row[f"{pname}_t"] = float(stats["tvalues"][idx])
                row[f"{pname}_p"] = float(stats["pvalues"][idx])
            rows.append(row)
            per_edge_per_donor.append(w)

            if (ei + 1) % 5000 == 0:
                logger.info(f"  fit {ei + 1}/{len(unique_edges)} edges ...")

        df = pd.DataFrame(rows)
        # BH-FDR PER PREDICTOR across all tested edges
        for pname in PREDICTORS:
            df[f"{pname}_q"] = benjamini_hochberg(df[f"{pname}_p"].values)

        out_csv = os.path.join(OUT_DIR, f"coefficients_all_{ct_safe}.csv")
        df.to_csv(out_csv, index=False)
        logger.info(f"  wrote {out_csv} ({len(df)} edges)")

        # Save per-edge per-donor matrix cache for downstream plotting
        per_edge_arr = np.stack(per_edge_per_donor, axis=0)   # (n_edges, n_donors)
        cache_path = os.path.join(OUT_DIR, f"per_edge_per_donor_{ct_safe}.npz")
        np.savez(
            cache_path,
            edge_weights=per_edge_arr,
            donor_ids=np.array(donor_ids, dtype=object),
            tfs=df["TF"].values,
            targets=df["target"].values,
            adnc_ordinal=X[:, 1],
            age=X[:, 2],
            sex_Male=X[:, 3],
            LATE_present=X[:, 4],
            LBD_present=X[:, 5],
            condition=pb.metadata.loc[donor_ids, "condition"].astype(str).values,
        )
        logger.info(f"  wrote cache {cache_path} ({per_edge_arr.shape})")

        # Quick per-predictor summary
        logger.info("  per-predictor q<0.05 hit counts:")
        for pname in PREDICTORS:
            n_sig = int((df[f"{pname}_q"] < 0.05).sum())
            best_q = float(df[f"{pname}_q"].min())
            logger.info(f"    {pname:15s}: {n_sig:5d} q<0.05  (best q = {best_q:.4f})")

    logger.info("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
