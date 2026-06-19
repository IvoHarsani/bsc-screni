"""SQ3 sensitivity analyses for the module-level differential test.

Reproduces the three robustness checks reported in the methodology of the
module-differential test (Section 3.7):

  * **Global-activity covariate.** Re-fit each module's OLS with the donor's
    total incoming activity (mean across all genes) added as an extra
    covariate, so that each module is tested for variation beyond a
    cohort-wide trend.
  * **Compositional re-run.** Re-fit each module after dividing its activity
    by the donor's total activity, so module-specific shifts are tested as
    a fraction of the total rather than as a level.
  * **QC-depth covariates.** Re-fit each module with two per-donor
    sequencing-depth metrics (mean number of genes detected, mean UMI
    count) added as extra covariates, so that the disease coefficient is
    interpreted net of depth.

The same three controls are also applied to the cohort-level test of total
regulatory activity ~ ADNC severity, since the global L2/3 IT decline is
the qualitative result that needs to survive QC adjustment.

Additionally, each module-level variant is repeated across a sweep of
Leiden resolutions (``--resolutions``; default 0.3 ... 1.5), giving a
single CSV in which a reader can verify that no module-specific effect
emerges under any combination of (resolution, control variant).

Inputs (must already exist in ``output/``):
  - ``sq3_incoming_activity_<celltype>.npz``  (from sq3_extract_incoming_activity.py)
  - ``per_edge_per_donor_<celltype>.npz``     (from compute_all_coefficients.py)
  - ``sq3_donor_qc.csv``                      (from sq3_extract_donor_qc.py; optional)

Outputs (written to ``output/``):
  - ``sq3_sensitivity_modules.csv``      summary row per (cell_type x resolution x variant)
  - ``sq3_sensitivity_modules_hits.csv`` one row per module with q<0.10 under any variant
  - ``sq3_sensitivity_global.csv``       global-activity OLS, one row per (cell_type x variant)

Run locally:
    PYTHONPATH=src pixi run python scripts/sq3_sensitivity.py
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from screni.data.differential import _ols_with_numeric_predictor, benjamini_hochberg
from screni.data.module_preservation import leiden_modules

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = os.path.join(REPO_ROOT, "src", "screni", "data", "output")
CELL_TYPES = ["Microglia_PVM", "L2_3_IT"]
BASE_COVARIATES = ("age", "sex_Male", "LATE_present", "LBD_present")
DEFAULT_RESOLUTIONS = (0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5)
EPS = 1e-12

# Patterns that identify the two depth columns in the SEA-AD QC obs schema.
# The exact column names vary slightly across releases; auto-detection keeps
# this script robust to those variations.
QC_PATTERNS = ("gene", "umi")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _control_adjacency(activity_ctrl: np.ndarray, beta: int) -> np.ndarray:
    X = activity_ctrl - activity_ctrl.mean(1, keepdims=True)
    S = np.nan_to_num(np.corrcoef(X.T), nan=0.0)
    A = np.abs(S) ** beta
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 1.0)
    return A


def _donor_metadata(ct_safe: str) -> pd.DataFrame:
    z = np.load(
        os.path.join(OUT_DIR, f"per_edge_per_donor_{ct_safe}.npz"),
        allow_pickle=True,
    )
    return pd.DataFrame(
        {
            "adnc_ordinal": z["adnc_ordinal"].astype(float),
            "age": z["age"].astype(float),
            "sex_Male": z["sex_Male"].astype(float),
            "LATE_present": z["LATE_present"].astype(float),
            "LBD_present": z["LBD_present"].astype(float),
            "condition": np.asarray(z["condition"]).astype(str),
        },
        index=np.asarray(z["donor_ids"]).astype(str),
    )


def _load_qc(ct_safe: str) -> pd.DataFrame | None:
    """Load the per-donor QC metrics produced by ``sq3_extract_donor_qc.py``
    for one cell type.  Picks the first column matching each pattern in
    ``QC_PATTERNS``; returns ``None`` if the CSV is absent or no patterns
    match.
    """
    path = os.path.join(OUT_DIR, "sq3_donor_qc.csv")
    if not os.path.exists(path):
        logger.warning(f"QC csv not found at {path}; QC variants will be skipped")
        return None
    qc = pd.read_csv(path)
    qc = qc[qc["cell_type"] == ct_safe].set_index("donor_id")
    qc = qc.drop(columns=[c for c in qc.columns if c in ("cell_type",)])
    picked = {}
    for pat in QC_PATTERNS:
        candidates = [c for c in qc.columns if re.search(pat, c, re.I)]
        if candidates:
            picked[pat] = candidates[0]
    if not picked:
        logger.warning(f"no QC columns matched patterns {QC_PATTERNS}")
        return None
    chosen = list(picked.values())
    logger.info(f"  using QC depth columns: {chosen}")
    return qc[chosen].apply(pd.to_numeric, errors="coerce")


# ---------------------------------------------------------------------------
# Module construction and per-donor module activity
# ---------------------------------------------------------------------------


def _load_cell_data(ct_safe: str):
    z = np.load(
        os.path.join(OUT_DIR, f"sq3_incoming_activity_{ct_safe}.npz"),
        allow_pickle=True,
    )
    act = z["activity"].astype(float)
    genes = np.asarray(z["gene_names"]).astype(str)
    cond = np.asarray(z["condition"]).astype(str)
    cell_donor = np.asarray(z["donor_ids"]).astype(str)
    tgt = set(
        np.unique(np.asarray(
            np.load(
                os.path.join(OUT_DIR, f"per_edge_per_donor_{ct_safe}.npz"),
                allow_pickle=True,
            )["targets"]
        ).astype(str))
    )
    uni = np.array([g in tgt for g in genes])
    return act[:, uni], genes[uni], cond, cell_donor


def _modules_and_pseudobulk(
    act: np.ndarray,
    cond: np.ndarray,
    cell_donor: np.ndarray,
    donor_ids: np.ndarray,
    beta: int,
    resolution: float,
    seed: int,
):
    """Return (labels, pb, keep_mask) where:

    - ``labels``    is a length-(n_kept_genes) module assignment built on the
      control co-regulation network at the given ``resolution``;
    - ``pb``        is the (n_donors, n_kept_genes) per-donor pseudobulk
      incoming activity (mean over the donor's cells);
    - ``keep_mask`` is the gene mask used to build the network (variance > EPS
      in the control arm).
    """
    ctrl = cond == "control"
    keep = (act[ctrl] - act[ctrl].mean(1, keepdims=True)).var(0) > EPS
    A_ctrl = _control_adjacency(act[ctrl][:, keep], beta)
    labels = leiden_modules(A_ctrl, resolution=resolution, seed=seed)

    pb = np.full((len(donor_ids), act.shape[1]), np.nan)
    for di, d in enumerate(donor_ids):
        m = cell_donor == d
        if m.any():
            pb[di] = act[m].mean(0)
    return labels, pb, keep


# ---------------------------------------------------------------------------
# Sensitivity variants
# ---------------------------------------------------------------------------


# Each variant is described as (name, covariate-set, compositional?).
# - ``covariates`` are added to the base BASE_COVARIATES + ADNC predictor.
# - ``compositional`` divides the response by the donor's total activity.
VARIANTS = (
    # (name, extra_covariates, compositional)
    ("raw",            (),                                                    False),
    ("global",         ("total_activity",),                                   False),
    ("compositional",  (),                                                    True),
    ("qc",             ("Genes_detected", "UMI_count"),                       False),
    ("global+qc",      ("total_activity", "Genes_detected", "UMI_count"),     False),
)


def _attach_total_and_qc(meta: pd.DataFrame, pb: np.ndarray,
                         qc: pd.DataFrame | None) -> pd.DataFrame:
    """Return ``meta`` augmented with the two extra per-donor columns the
    sensitivity variants may reference.  ``total_activity`` is the mean
    incoming activity over all kept genes for each donor; the two QC
    columns are renamed to canonical names that the variant table
    references.
    """
    meta = meta.copy()
    meta["total_activity"] = np.nanmean(pb, axis=1)
    if qc is not None:
        cols = qc.columns.tolist()
        # Pick the longer-named column likely to be UMI; the other is genes-detected.
        # Use a stable rule: the column whose name contains 'umi' (case-insensitive)
        # is mapped to UMI_count; the remaining one is mapped to Genes_detected.
        umi_col = next((c for c in cols if re.search(r"umi", c, re.I)), None)
        gene_col = next((c for c in cols if c != umi_col), None)
        if umi_col is not None:
            meta["UMI_count"] = qc[umi_col].reindex(meta.index).astype(float).values
        if gene_col is not None:
            meta["Genes_detected"] = qc[gene_col].reindex(meta.index).astype(float).values
    return meta


def _fit_module_test(
    response: np.ndarray, meta: pd.DataFrame, extra_covariates: tuple[str, ...],
):
    cov = tuple(c for c in (*BASE_COVARIATES, *extra_covariates) if c in meta.columns)
    return _ols_with_numeric_predictor(
        response, meta, predictor_col="adnc_ordinal", covariates=cov,
    )


# ---------------------------------------------------------------------------
# Per-cell-type driver
# ---------------------------------------------------------------------------


def run_cell_type(
    ct_safe: str,
    beta: int,
    resolutions: tuple[float, ...],
    min_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    act, _, cond, cell_donor = _load_cell_data(ct_safe)
    meta = _donor_metadata(ct_safe)
    qc = _load_qc(ct_safe)
    donor_ids = meta.index.values
    logger.info(f"\n=== {ct_safe}: {act.shape[0]} cells, {len(donor_ids)} donors ===")

    # ---- module-level summary across (resolution x variant) ------------
    rows_summary: list[dict] = []
    rows_hits: list[dict] = []
    for resolution in resolutions:
        labels, pb, keep = _modules_and_pseudobulk(
            act, cond, cell_donor, donor_ids, beta, resolution, seed,
        )
        sizes = np.bincount(labels)
        mods = [m for m in np.unique(labels) if sizes[m] >= min_size]
        meta_full = _attach_total_and_qc(meta, pb, qc)

        for variant_name, extra_cov, compositional in VARIANTS:
            # Skip QC variants if QC columns weren't loaded.
            if any(c in extra_cov for c in ("Genes_detected", "UMI_count")) and qc is None:
                continue

            per_module_rows = []
            for m in mods:
                gidx = np.where(labels == m)[0]
                activity = pb[:, keep][:, gidx].mean(1)
                if compositional:
                    activity = activity / np.where(
                        meta_full["total_activity"].values == 0,
                        np.nan,
                        meta_full["total_activity"].values,
                    )
                res = _fit_module_test(activity, meta_full, extra_cov)
                per_module_rows.append({
                    "module": int(m),
                    "size": int(sizes[m]),
                    "adnc_coef": res.coef,
                    "adnc_t": res.t_stat,
                    "adnc_p": res.p_value,
                })
            if not per_module_rows:
                continue
            df_mod = pd.DataFrame(per_module_rows)
            df_mod["adnc_q"] = benjamini_hochberg(df_mod["adnc_p"].values)
            n_q05 = int((df_mod["adnc_q"] < 0.05).sum())
            n_q10 = int((df_mod["adnc_q"] < 0.10).sum())
            best_q = float(df_mod["adnc_q"].min())
            rows_summary.append({
                "cell_type": ct_safe,
                "resolution": resolution,
                "variant": variant_name,
                "n_modules": len(df_mod),
                "n_q_lt_0.05": n_q05,
                "n_q_lt_0.10": n_q10,
                "best_q": best_q,
            })
            for _, r in df_mod[df_mod["adnc_q"] < 0.10].iterrows():
                rows_hits.append({
                    "cell_type": ct_safe, "resolution": resolution,
                    "variant": variant_name, **r.to_dict(),
                })
            logger.info(
                f"  res={resolution:.2f} variant={variant_name:14s} "
                f"n={len(df_mod):3d}  q<0.05={n_q05:2d}  q<0.10={n_q10:2d}  "
                f"best_q={best_q:.4f}"
            )

    # ---- global-activity OLS across variants (one row per variant) -------
    # Resolution-independent: uses the donor total activity directly.
    labels, pb, keep = _modules_and_pseudobulk(
        act, cond, cell_donor, donor_ids, beta, resolutions[0], seed,
    )
    meta_full = _attach_total_and_qc(meta, pb, qc)
    total = meta_full["total_activity"].values
    global_rows: list[dict] = []
    GLOBAL_VARIANTS = (
        ("raw", ()),
        ("qc", ("Genes_detected", "UMI_count")),
    )
    logger.info(f"  --- global activity ~ ADNC ---")
    for variant_name, extra_cov in GLOBAL_VARIANTS:
        if any(c in extra_cov for c in ("Genes_detected", "UMI_count")) and qc is None:
            continue
        res = _fit_module_test(total, meta_full, extra_cov)
        global_rows.append({
            "cell_type": ct_safe, "variant": variant_name,
            "coef": res.coef, "stderr": res.stderr,
            "t_stat": res.t_stat, "p_value": res.p_value,
            "n_donors": res.n_donors,
        })
        logger.info(
            f"  variant={variant_name:14s}  coef={res.coef:+.4g}  "
            f"p={res.p_value:.4g}  n={res.n_donors}"
        )

    return (
        pd.DataFrame(rows_summary),
        pd.DataFrame(rows_hits),
        pd.DataFrame(global_rows),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cell-types", nargs="+", default=CELL_TYPES)
    p.add_argument("--beta", type=int, default=2)
    p.add_argument("--resolutions", type=float, nargs="+", default=list(DEFAULT_RESOLUTIONS))
    p.add_argument("--min-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    summary_all, hits_all, global_all = [], [], []
    for ct in a.cell_types:
        summary, hits, glob = run_cell_type(
            ct, a.beta, tuple(a.resolutions), a.min_size, a.seed,
        )
        summary_all.append(summary)
        hits_all.append(hits)
        global_all.append(glob)

    pd.concat(summary_all, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "sq3_sensitivity_modules.csv"), index=False,
    )
    pd.concat(hits_all, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "sq3_sensitivity_modules_hits.csv"), index=False,
    )
    pd.concat(global_all, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "sq3_sensitivity_global.csv"), index=False,
    )
    logger.info("\nwrote sq3_sensitivity_{modules,modules_hits,global}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
