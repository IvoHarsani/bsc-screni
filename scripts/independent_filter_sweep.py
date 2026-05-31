"""Independent-filtering sensitivity analysis for SQ1 differential edges.

Address the supervisor's question: "are we missing hits because the FDR
correction over 35,590 edges is too punishing?"

Approach: re-rank using BH-FDR on a *subset* of edges chosen by an
independent filter — one that does NOT use the AD condition or the
AD_severity coefficient.  Specifically, drop edges with the lowest
*overall mean weight* (averaged across all 27 donors regardless of
condition).  This is the same "independent filtering" pattern DESeq2
uses by default for RNA-seq DE.

For each cell type and each test variant (ordinal, OLS, Wilcoxon, CPS),
we sweep the filter threshold over X% ∈ {0, 25, 50, 75, 90} of the
mean-weight distribution and report:

- # edges remaining after filter
- # edges with q<0.05 and q<0.10 after re-correcting
- Best q-value achieved
- For sufficiently strong filters: which specific edges become hits

Outputs:
- ``output/independent_filter_sweep.csv``       — summary table across sweeps
- ``output/independent_filter_hits.csv``        — every (TF, target, cell type,
                                                  test, X%) tuple that achieved
                                                  q<0.10 (so we can see who
                                                  rises through the filter)
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from screni.data.differential import benjamini_hochberg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = os.path.join(REPO_ROOT, "src", "screni", "data", "output")

CELL_TYPES = [("Microglia-PVM", "Microglia_PVM"), ("L2/3 IT", "L2_3_IT")]
TESTS = [
    ("ols",      "differential_edges_{ct}.csv",          "AD-vs-control OLS (binary)"),
    ("wilcoxon", "differential_edges_{ct}_wilcoxon.csv", "Wilcoxon (binary)"),
    ("ordinal",  "differential_edges_{ct}_ordinal.csv",  "Ordinal ADNC (0..3) OLS"),
    ("cps",      "differential_edges_{ct}_cps.csv",      "Continuous pseudo-progression OLS"),
]
DROP_PERCENTILES = [0, 25, 50, 75, 90]

# n_control / n_ad from the SQ1 cohort (we use these to weight per-condition
# means when reconstructing 'mean over all donors' — independent of condition
# label because the WEIGHTED average across both groups equals the overall
# mean across all donors).  See note below.
N_CONTROL = 7
N_AD = 20
N_TOTAL = N_CONTROL + N_AD


def _overall_mean(df: pd.DataFrame) -> np.ndarray:
    """Mean edge weight over all 27 donors, derived from per-condition means
    in the CSV.  This is independent of the condition label (it's the same
    number you'd get by averaging all 27 donors directly), so using it as a
    filter does not violate independent-filtering.
    """
    return (df["mean_control"] * N_CONTROL + df["mean_ad"] * N_AD) / N_TOTAL


def main() -> int:
    sweep_rows = []
    hit_rows = []

    for ct_pretty, ct_safe in CELL_TYPES:
        logger.info(f"\n=== {ct_pretty} ===")

        for tag, file_template, label in TESTS:
            path = os.path.join(OUT_DIR, file_template.format(ct=ct_safe))
            if not os.path.exists(path):
                logger.warning(f"  missing {path}; skipping")
                continue
            df = pd.read_csv(path)
            n_total = len(df)

            # Independent filter statistic: |overall mean weight| across the
            # 27 donors.  Edges with the smallest absolute weights are
            # dropped first.  This is condition-label-independent.
            abs_overall = np.abs(_overall_mean(df))
            df = df.assign(_abs_overall=abs_overall)

            for pct_drop in DROP_PERCENTILES:
                if pct_drop == 0:
                    keep_mask = np.ones(len(df), dtype=bool)
                else:
                    threshold = np.nanpercentile(abs_overall, pct_drop)
                    keep_mask = abs_overall >= threshold

                sub = df.loc[keep_mask].copy()
                n_kept = len(sub)
                if n_kept == 0:
                    continue

                # Recompute BH-FDR on the FILTERED subset of p-values.
                # The original q_value column in the CSV was computed on
                # the full 35,590-edge set; here we're explicitly redoing
                # it on the smaller universe.
                sub["q_value_filtered"] = benjamini_hochberg(sub["p_value"].values)

                n_q05 = int((sub["q_value_filtered"] < 0.05).sum())
                n_q10 = int((sub["q_value_filtered"] < 0.10).sum())
                best_q = float(sub["q_value_filtered"].min())

                sweep_rows.append({
                    "cell_type": ct_pretty,
                    "test": tag,
                    "test_label": label,
                    "pct_dropped": pct_drop,
                    "n_kept": n_kept,
                    "n_total": n_total,
                    "fraction_kept": n_kept / n_total,
                    "n_q_lt_0.05": n_q05,
                    "n_q_lt_0.10": n_q10,
                    "best_q_filtered": best_q,
                })

                # Record edges that survive q<0.10 in the FILTERED universe
                top = sub[sub["q_value_filtered"] < 0.10].sort_values("q_value_filtered")
                for _, row in top.iterrows():
                    hit_rows.append({
                        "cell_type": ct_pretty,
                        "test": tag,
                        "pct_dropped": pct_drop,
                        "n_kept": n_kept,
                        "TF": row["TF"],
                        "target": row["target"],
                        "p_value": row["p_value"],
                        "q_value_filtered": row["q_value_filtered"],
                        "q_value_original": row["q_value"],
                        "log2FC": row["log2FC"],
                        "mean_control": row["mean_control"],
                        "mean_ad": row["mean_ad"],
                    })

            logger.info(f"  {label}: {n_total} edges total")
            # Compact per-test summary line
            for pct in DROP_PERCENTILES:
                row = [r for r in sweep_rows
                       if r["cell_type"] == ct_pretty and r["test"] == tag
                       and r["pct_dropped"] == pct]
                if not row:
                    continue
                r = row[0]
                logger.info(
                    f"    drop {pct:>2d}%  → kept {r['n_kept']:>5d}  "
                    f"q<0.05: {r['n_q_lt_0.05']:>3d}  "
                    f"q<0.10: {r['n_q_lt_0.10']:>3d}  "
                    f"best q: {r['best_q_filtered']:.4f}"
                )

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = os.path.join(OUT_DIR, "independent_filter_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    logger.info(f"\nwrote {sweep_path}")

    hits_df = pd.DataFrame(hit_rows)
    hits_path = os.path.join(OUT_DIR, "independent_filter_hits.csv")
    hits_df.to_csv(hits_path, index=False)
    logger.info(f"wrote {hits_path}")

    # Print a Markdown-friendly summary table to the log
    logger.info("\n=== Summary table ===")
    pivot = sweep_df.pivot_table(
        index=["cell_type", "test"],
        columns="pct_dropped",
        values="n_q_lt_0.05",
        aggfunc="first",
    )
    logger.info("\nq<0.05 hits, by cell type × test × % dropped:\n" + pivot.to_string())

    pivot_q = sweep_df.pivot_table(
        index=["cell_type", "test"],
        columns="pct_dropped",
        values="best_q_filtered",
        aggfunc="first",
    )
    logger.info("\nbest q achieved, same indexing:\n" + pivot_q.round(4).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
