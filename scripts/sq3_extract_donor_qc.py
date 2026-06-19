"""SQ3 QC-confound check input: per-donor sequencing-depth / QC metrics.

The 'global L2/3 regulatory activity declines with AD' finding could be a data-quality artifact
(AD cells with lower library size -> fewer/weaker RF associations -> lower total importance).
This extracts per-(cell_type x donor) mean QC metrics (genes detected, UMI counts, etc.) so the
global-decline OLS can be re-run with QC as a covariate. Tiny + fast (reads obs only, no networks).
"""

from __future__ import annotations

import logging
import os
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(REPO_ROOT, "data", "processed", "seaad")
RNA_SUB = os.path.join(DATA_DIR, "seaad_paired_rna_sq1.h5ad")
RNA_FULL = os.path.join(DATA_DIR, "seaad_paired_rna.h5ad")
OUT_DIR = os.path.join(REPO_ROOT, "src", "screni", "data", "output")
CELL_TYPES = [("Microglia-PVM", "Microglia_PVM"), ("L2/3 IT", "L2_3_IT")]


def main() -> int:
    obs = ad.read_h5ad(RNA_SUB).obs.copy()
    need = [c for c in ["Donor ID", "cell_type"] if c not in obs.columns]
    qc_like = [c for c in obs.columns if re.search(r"gene|umi|count|read|detect|frac|mito|qc", c, re.I)]
    if need or not qc_like:
        full = ad.read_h5ad(RNA_FULL, backed="r")
        fobs = full.obs
        pull = need + [c for c in fobs.columns if re.search(r"gene|umi|count|read|detect", c, re.I)]
        pull = [c for c in dict.fromkeys(pull) if c in fobs.columns]
        add = fobs.loc[obs.index, pull].copy()
        full.file.close()
        for c in pull:
            if c not in obs.columns:
                obs[c] = add[c].values

    # numeric QC columns only
    qc_cols = [c for c in obs.columns
               if re.search(r"gene|umi|count|read|detect", c, re.I)
               and pd.api.types.is_numeric_dtype(obs[c])]
    logger.info(f"QC columns used: {qc_cols}")

    rows = []
    for ct_pretty, ct_safe in CELL_TYPES:
        sub = obs[obs["cell_type"].astype(str) == ct_pretty]
        g = sub.groupby(sub["Donor ID"].astype(str))[qc_cols].mean()
        g["cell_type"] = ct_safe
        g["donor_id"] = g.index
        rows.append(g.reset_index(drop=True))
        logger.info(f"{ct_pretty}: {len(g)} donors")

    out = os.path.join(OUT_DIR, "sq3_donor_qc.csv")
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    logger.info(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
