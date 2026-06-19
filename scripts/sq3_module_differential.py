"""SQ3 (positive-result instrument): donor-level module-differential analysis.

Module preservation (sq3_module_preservation_run.py) showed module-level co-regulation
*architecture* is conserved in AD (control->AD ~ control->control). This script asks the
complementary, more sensitive question: does each module's overall regulatory *level* shift
with AD severity?

  1. Define co-regulation modules on the CONTROL network (cross-cell, beta, Leiden).
  2. Per (module x donor): mean incoming regulatory activity of the module's genes, pseudobulked
     per donor (donor = correct statistical unit, as in SQ1).
  3. Per module, OLS  activity ~ adnc_ordinal + age + sex_Male + LATE_present + LBD_present
     across the 27 donors (SQ1's ordinal-severity test). BH-FDR over the modules.
  4. Functional label per module (g:Profiler GO:BP), and cross-cell-type comparison.

Only a handful of modules x 2 cell types => negligible FDR burden => power for subtle effects
that the 35,590-edge SQ1 test and the structural preservation test could not reach.

Run locally:  PYTHONPATH=src pixi run python scripts/sq3_module_differential.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from screni.data.differential import _ols_with_numeric_predictor, benjamini_hochberg
from screni.data.module_preservation import leiden_modules

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = os.path.join(REPO_ROOT, "src", "screni", "data", "output")
CELL_TYPES = ["Microglia_PVM", "L2_3_IT"]
COVARIATES = ("age", "sex_Male", "LATE_present", "LBD_present")
EPS = 1e-12


def control_adjacency(activity_ctrl: np.ndarray, beta: int):
    X = activity_ctrl - activity_ctrl.mean(1, keepdims=True)
    S = np.nan_to_num(np.corrcoef(X.T), nan=0.0)
    A = np.abs(S) ** beta
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 1.0)
    return A


def donor_metadata(ct_safe: str) -> pd.DataFrame:
    """Per-donor design-matrix columns (from the SQ1 per-edge cache, which carries them)."""
    z = np.load(os.path.join(OUT_DIR, f"per_edge_per_donor_{ct_safe}.npz"), allow_pickle=True)
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


def run_cell_type(ct: str, beta: int, resolution: float, min_size: int, seed: int):
    z = np.load(os.path.join(OUT_DIR, f"sq3_incoming_activity_{ct}.npz"), allow_pickle=True)
    act = z["activity"].astype(float)
    genes = np.asarray(z["gene_names"]).astype(str)
    cond = np.asarray(z["condition"]).astype(str)
    cell_donor = np.asarray(z["donor_ids"]).astype(str)

    tgt = set(np.unique(np.asarray(
        np.load(os.path.join(OUT_DIR, f"per_edge_per_donor_{ct}.npz"), allow_pickle=True)["targets"]
    ).astype(str)))
    uni = np.array([g in tgt for g in genes])
    act, genes = act[:, uni], genes[uni]

    meta = donor_metadata(ct)
    donors = meta.index.values
    logger.info(f"\n=== {ct}: {act.shape[0]} cells, {len(donors)} donors, {uni.sum()} target genes ===")

    # --- modules on the CONTROL co-regulation network ---
    ctrl_cells = cond == "control"
    keep = (act[ctrl_cells] - act[ctrl_cells].mean(1, keepdims=True)).var(0) > EPS
    A_ctrl = control_adjacency(act[ctrl_cells][:, keep], beta)
    labels = leiden_modules(A_ctrl, resolution=resolution, seed=seed)
    genes_k = genes[keep]
    # seed stability of the module assignment
    ari = adjusted_rand_score(labels, leiden_modules(A_ctrl, resolution=resolution, seed=seed + 1))
    sizes = np.bincount(labels)
    mods = [m for m in np.unique(labels) if sizes[m] >= min_size]
    logger.info(f"  {len(mods)} modules (size>={min_size}); sizes={sorted(sizes[sizes>=min_size].tolist(), reverse=True)}; seed-ARI={ari:.2f}")

    # --- per-donor pseudobulk incoming activity over module genes ---
    pb = np.full((len(donors), act.shape[1]), np.nan)
    for di, d in enumerate(donors):
        m = cell_donor == d
        if m.any():
            pb[di] = act[m].mean(0)

    # --- per-module OLS (ordinal ADNC severity + covariates) ---
    rows = []
    for m in mods:
        gidx = np.where(labels == m)[0]
        donor_module_activity = pb[:, keep][:, gidx].mean(1)   # (n_donors,)
        res = _ols_with_numeric_predictor(
            donor_module_activity, meta, predictor_col="adnc_ordinal", covariates=COVARIATES
        )
        rows.append({
            "module": int(m),
            "size": int(sizes[m]),
            "adnc_coef": res.coef,
            "adnc_t": res.t_stat,
            "adnc_p": res.p_value,
            "direction": "down" if res.coef < 0 else "up",
            "genes": ";".join(genes_k[gidx]),
        })
    df = pd.DataFrame(rows)
    df["adnc_q"] = benjamini_hochberg(df["adnc_p"].values)
    df["cell_type"] = ct
    df = df.sort_values("adnc_q").reset_index(drop=True)

    n_sig = int((df["adnc_q"] < 0.05).sum())
    logger.info(f"  q<0.05 modules: {n_sig}/{len(df)}  (best q={df['adnc_q'].min():.4f})")
    for _, r in df[df["adnc_q"] < 0.10].iterrows():
        logger.info(f"    module {r['module']} (n={r['size']}, {r['direction']}): "
                    f"coef={r['adnc_coef']:.3g} p={r['adnc_p']:.2e} q={r['adnc_q']:.3f}")

    out = os.path.join(OUT_DIR, f"sq3_module_differential_{ct}.csv")
    df.to_csv(out, index=False)
    logger.info(f"  wrote {out}")
    return df


def enrich(df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Attach a top GO:BP term to each module via g:Profiler (guarded; needs internet)."""
    try:
        from gprofiler import GProfiler
        gp = GProfiler(return_dataframe=True)
    except Exception as e:
        logger.warning(f"enrichment skipped ({e})")
        df["top_GO"] = ""
        return df
    terms = []
    for _, r in df.iterrows():
        g = r["genes"].split(";")
        try:
            res = gp.profile(organism="hsapiens", query=g, sources=["GO:BP"],
                             significance_threshold_method="fdr", user_threshold=0.05)
            terms.append(res.sort_values("p_value")["name"].iloc[0] if len(res) else "")
        except Exception:
            terms.append("")
    df["top_GO"] = terms
    return df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cell-types", nargs="+", default=CELL_TYPES)
    p.add_argument("--beta", type=int, default=2)
    p.add_argument("--resolution", type=float, default=1.5)
    p.add_argument("--min-size", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-enrich", action="store_true")
    a = p.parse_args()

    results = {}
    for ct in a.cell_types:
        df = run_cell_type(ct, a.beta, a.resolution, a.min_size, a.seed)
        if not a.no_enrich:
            df = enrich(df)
            df.to_csv(os.path.join(OUT_DIR, f"sq3_module_differential_{ct}.csv"), index=False)
        results[ct] = df

    logger.info("\n=== cross-cell-type (q<0.05 modules) ===")
    for ct, df in results.items():
        sig = df[df["adnc_q"] < 0.05]
        logger.info(f"  {ct}: {len(sig)} significant modules")
        for _, r in sig.iterrows():
            go = r.get("top_GO", "")
            logger.info(f"    n={r['size']} {r['direction']} q={r['adnc_q']:.3f}  {go}")
    logger.info("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
