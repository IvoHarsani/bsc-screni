"""SQ3 step 2: module-preservation analysis on the cross-cell co-regulation networks.

Consumes the per-cell incoming-activity caches written by sq3_extract_incoming_activity.py.
For each cell type:

  1. Restrict to the candidate-edge target genes (the mechanistically-grounded universe,
     matching the cross-donor diagnostic that validated this substrate).
  2. Split cells by condition; build cross-CELL co-regulation adjacencies
     A_control (reference) and A_ad (test): per-cell-center the incoming activity, then
     correlate genes across cells, soft-threshold |corr|**beta.
  3. Define modules on A_control via Leiden (module def is decoupled from the WGCNA stats).
  4. module_preservation(A_control, A_ad, modules) -> Zsummary per module (disrupted < 2).
  5. Stability gate: rebuild control modules from a random half of the control DONORS' cells,
     report ARI vs the full-control modules (guards against donor-identity artifacts /
     pseudoreplication).

Then compare disrupted modules across the two cell types (SQ3 + the SQ4 cross-cell-type angle).

Small inputs -> runs locally:  pixi run test scripts/... won't work (it's a script); use
  PYTHONPATH=src pixi run python scripts/sq3_module_preservation_run.py
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

from screni.data.module_preservation import leiden_modules, module_preservation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = os.path.join(REPO_ROOT, "src", "screni", "data", "output")
CELL_TYPES = ["Microglia_PVM", "L2_3_IT"]
EPS = 1e-12


def build_adjacency(activity: np.ndarray, beta: int, cell_center: bool = True):
    """(n_cells, n_genes) incoming activity -> (n_genes, n_genes) co-regulation adjacency.

    Genes are adjacent when their incoming activity co-varies across cells. Per-cell centering
    removes the global per-cell 'overall regulatory intensity' axis (the analogue of the
    row-centering that rescued the mean-matrix prototype). Returns (A, keep_mask) over genes
    with non-zero across-cell variance.
    """
    X = activity.astype(float)
    if cell_center:
        X = X - X.mean(axis=1, keepdims=True)
    keep = X.var(axis=0) > EPS
    S = np.nan_to_num(np.corrcoef(X[:, keep].T), nan=0.0)
    A = np.abs(S) ** beta
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 1.0)
    return A, keep


def load_universe(ct_safe: str, gene_names: np.ndarray) -> np.ndarray:
    """Candidate-edge target genes for this cell type (mask over gene_names). Falls back to
    'all genes with incoming activity' if the per-edge cache is absent."""
    cache = os.path.join(OUT_DIR, f"per_edge_per_donor_{ct_safe}.npz")
    if os.path.exists(cache):
        targets = set(np.unique(np.asarray(np.load(cache, allow_pickle=True)["targets"]).astype(str)))
        mask = np.array([g in targets for g in gene_names])
        logger.info(f"  universe = {mask.sum()} candidate-edge target genes")
        return mask
    logger.warning(f"  no per-edge cache for {ct_safe}; universe = all genes")
    return np.ones(len(gene_names), dtype=bool)


def run_cell_type(ct_safe: str, beta: int, resolution: float, n_perm: int, seed: int) -> pd.DataFrame:
    cache = os.path.join(OUT_DIR, f"sq3_incoming_activity_{ct_safe}.npz")
    z = np.load(cache, allow_pickle=True)
    activity = z["activity"].astype(float)                 # (n_cells, n_genes)
    genes = np.asarray(z["gene_names"]).astype(str)
    cond = np.asarray(z["condition"]).astype(str)
    donors = np.asarray(z["donor_ids"]).astype(str)
    logger.info(f"\n=== {ct_safe}: {activity.shape[0]} cells "
                f"(control={int((cond=='control').sum())}, ad={int((cond=='ad').sum())}) ===")

    uni = load_universe(ct_safe, genes)
    activity = activity[:, uni]
    genes = genes[uni]

    ctrl = cond == "control"
    ad = cond == "ad"
    A_ctrl, keep_c = build_adjacency(activity[ctrl], beta)
    A_ad, keep_a = build_adjacency(activity[ad], beta)
    # restrict both networks to genes variable in BOTH arms (same node set / ordering)
    keep = keep_c & keep_a
    genes_k = genes[keep]
    A_ctrl, _ = build_adjacency(activity[ctrl][:, keep], beta)
    A_ad, _ = build_adjacency(activity[ad][:, keep], beta)
    logger.info(f"  network over {len(genes_k)} genes")

    labels = leiden_modules(A_ctrl, resolution=resolution, seed=seed)
    sizes = np.bincount(labels)
    logger.info(f"  {len(sizes)} control modules; sizes(top)={np.sort(sizes)[::-1][:8].tolist()}")

    res = module_preservation(A_ctrl, A_ad, labels, n_permutations=n_perm, seed=seed)
    df = res.to_frame()
    df["cell_type"] = ct_safe
    # attach gene members per module
    members = {m: ";".join(genes_k[labels == m]) for m in df["module"]}
    df["genes"] = df["module"].map(members)

    n_disrupted = int((df["Zsummary"] < 2).sum())
    logger.info(f"  Zsummary: {n_disrupted}/{len(df)} modules disrupted (<2); "
                f"range [{df['Zsummary'].min():.1f}, {df['Zsummary'].max():.1f}]")

    # --- stability gate: rebuild control modules from half the control donors ---
    rng = np.random.default_rng(seed)
    cdonors = np.unique(donors[ctrl])
    half = set(rng.choice(cdonors, size=max(1, len(cdonors) // 2), replace=False).tolist())
    half_mask = ctrl & np.array([d in half for d in donors])
    A_half, kh = build_adjacency(activity[half_mask][:, keep], beta)
    labels_half = leiden_modules(A_half, resolution=resolution, seed=seed)
    ari = adjusted_rand_score(labels, labels_half)
    logger.info(f"  STABILITY: ARI(full-control vs half-donors {sorted(half)}) = {ari:.3f}")
    df.attrs["stability_ari"] = ari

    out = os.path.join(OUT_DIR, f"sq3_preservation_{ct_safe}.csv")
    df.to_csv(out, index=False)
    logger.info(f"  wrote {out}")
    return df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cell-types", nargs="+", default=CELL_TYPES)
    p.add_argument("--beta", type=int, default=6)
    p.add_argument("--resolution", type=float, default=2.0)
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=12345)
    a = p.parse_args()

    results = {}
    for ct in a.cell_types:
        results[ct] = run_cell_type(ct, a.beta, a.resolution, a.n_perm, a.seed)

    # cross-cell-type comparison: disrupted gene overlap
    if len(results) == 2:
        cts = list(results)
        dis = {ct: set(";".join(results[ct].loc[results[ct]["Zsummary"] < 2, "genes"]).split(";"))
               - {""} for ct in cts}
        logger.info("\n=== cross-cell-type ===")
        for ct in cts:
            logger.info(f"  {ct}: {len(dis[ct])} genes in disrupted modules")
        both = dis[cts[0]] & dis[cts[1]]
        logger.info(f"  shared disrupted genes: {len(both)} -> {sorted(both)[:25]}")
    logger.info("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
