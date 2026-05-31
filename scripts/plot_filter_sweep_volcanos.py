"""Filter-sweep volcano plots, styled like the coefficients-folder volcanos.

For each cell type × OLS predictor, draw a volcano plot at each
independent-filter level X% ∈ {0, 25, 50, 75, 90}.  The filter drops
edges with the lowest |overall mean weight| (a condition-label-
independent statistic, so FDR control on the filtered subset is valid).

Predictors (positions 1..5 of the design matrix in compute_all_coefficients.py):
    adnc_ordinal, age, sex_Male, LATE_present, LBD_present

Source CSV: ``output/coefficients_all_<celltype>.csv`` — has per-edge
``{pred}_coef`` / ``{pred}_p`` / ``{pred}_q`` plus ``mean_control`` and
``mean_ad`` used to derive the filter statistic.

Outputs (per cell type × predictor):
- 5 individual PNGs (``volcano_<ct>_<predictor>_drop<NN>.png``)
- 1 combined 1×5 panel (``volcano_sweep_combined_<ct>_<predictor>.png``)
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from screni.data.differential import benjamini_hochberg

OUT_DIR = os.path.join(REPO_ROOT, "src", "screni", "data", "output")
PLOTS_DIR = os.path.join(OUT_DIR, "plots", "filter_sweep")

CELL_TYPES = [("Microglia-PVM", "Microglia_PVM"),
              ("L2/3 IT",       "L2_3_IT")]
DROP_PERCENTILES = [0, 25, 50, 75, 90]

# Same predictor list as compute_all_coefficients.py.  Each entry:
# (column-prefix in CSV, human-readable label for axis/title)
PREDICTORS = [
    ("adnc_ordinal",  "AD severity (ADNC ordinal)"),
    ("age",           "Age at death"),
    ("sex_Male",      "Sex (Male=1)"),
    ("LATE_present",  "LATE co-pathology"),
    ("LBD_present",   "Lewy body co-pathology"),
]

N_CONTROL = 7
N_AD = 20
N_TOTAL = N_CONTROL + N_AD

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _overall_mean(df: pd.DataFrame) -> np.ndarray:
    """|overall mean weight| across all 27 donors (condition-independent)."""
    return (df["mean_control"] * N_CONTROL + df["mean_ad"] * N_AD) / N_TOTAL


def _apply_filter(df: pd.DataFrame, pct_drop: int, predictor: str) -> pd.DataFrame:
    """Filter df by |overall mean weight| percentile, then recompute BH-FDR
    on the kept subset for the given predictor's p-values.
    """
    if pct_drop <= 0:
        out = df.copy()
    else:
        threshold = np.nanpercentile(np.abs(_overall_mean(df)), pct_drop)
        out = df[np.abs(_overall_mean(df)) >= threshold].copy()
    p_col = f"{predictor}_p"
    out[f"{predictor}_q_filtered"] = benjamini_hochberg(out[p_col].values)
    return out


def _draw_volcano(ax, df, predictor, label, title, n_total,
                  x_lo, x_hi, label_top=15, show_legend=True):
    coef_col = f"{predictor}_coef"
    p_col = f"{predictor}_p"
    q_col = f"{predictor}_q_filtered"

    x = df[coef_col].values
    p = df[p_col].clip(lower=1e-30).values
    y = -np.log10(p)
    q = df[q_col].values

    x_clip = np.clip(x, x_lo, x_hi)

    colors = np.where(q < 0.05, "#d63838",
              np.where(q < 0.10, "#f0a020",
              np.where(q < 0.25, "#a0a020", "#888888")))
    sizes = np.where(q < 0.05, 22, np.where(q < 0.10, 14, 6))
    ax.scatter(x_clip, y, c=colors, s=sizes, alpha=0.6, edgecolors="none")

    n_kept = len(df)
    p_for_q005 = 0.05 / max(n_kept, 1)
    p_for_q010 = 0.10 / max(n_kept, 1)
    ax.axhline(-np.log10(p_for_q005), color="#d63838", ls="--", lw=1, alpha=0.7,
               label=f"top-edge q<0.05 (p < {p_for_q005:.2e})" if show_legend else None)
    ax.axhline(-np.log10(p_for_q010), color="#f0a020", ls="--", lw=1, alpha=0.7,
               label=f"top-edge q<0.10 (p < {p_for_q010:.2e})" if show_legend else None)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.5)

    top = df.nsmallest(label_top, q_col)
    for _, row in top.iterrows():
        ax.annotate(
            f"{row['TF']}→{row['target']}",
            (np.clip(row[coef_col], x_lo, x_hi),
             -np.log10(max(row[p_col], 1e-30))),
            textcoords="offset points", xytext=(4, 4),
            fontsize=8, alpha=0.85,
        )

    n_q05 = int((q < 0.05).sum())
    n_q10 = int((q < 0.10).sum())
    ax.set_title(f"{title}\n{n_kept}/{n_total} edges kept | "
                 f"q<0.05: {n_q05}, q<0.10: {n_q10}", fontsize=11)
    ax.set_xlabel(f"{label} coefficient")
    ax.set_ylabel("−log10(p-value)")
    ax.set_xlim(x_lo, x_hi)
    if show_legend:
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.2)


def main() -> int:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    logger.info(f"plots -> {PLOTS_DIR}")

    for ct_pretty, ct_safe in CELL_TYPES:
        path = os.path.join(OUT_DIR, f"coefficients_all_{ct_safe}.csv")
        if not os.path.exists(path):
            logger.warning(f"missing {path}, skipping {ct_pretty}")
            continue
        df_full = pd.read_csv(path)
        n_total = len(df_full)
        logger.info(f"\n=== {ct_pretty} ({n_total} edges) ===")

        for predictor, label in PREDICTORS:
            coef_col = f"{predictor}_coef"
            if coef_col not in df_full.columns:
                logger.warning(f"  {predictor}: missing {coef_col}, skipping")
                continue
            logger.info(f"  -- {predictor} --")

            # Shared x-axis range across all 5 panels for visual comparability.
            # Use the unfiltered coefficient distribution's 0.1/99.9 quantiles.
            x_vals = df_full[coef_col].values
            x_lo = float(np.nanquantile(x_vals, 0.001))
            x_hi = float(np.nanquantile(x_vals, 0.999))
            if x_lo == x_hi:
                # Degenerate; pad a hair to avoid xlim collapse
                x_lo, x_hi = x_lo - 1e-9, x_hi + 1e-9

            # 1×5 combined panel
            fig, axes = plt.subplots(1, len(DROP_PERCENTILES),
                                      figsize=(5.0 * len(DROP_PERCENTILES), 5.2),
                                      sharey=True)
            for ax, pct in zip(axes, DROP_PERCENTILES):
                sub = _apply_filter(df_full, pct, predictor)
                _draw_volcano(
                    ax, sub, predictor=predictor, label=label,
                    title=f"Filter: drop bottom {pct}%",
                    n_total=n_total, x_lo=x_lo, x_hi=x_hi,
                    show_legend=(pct == 0),
                )
            fig.suptitle(
                f"Volcano sweep — {ct_pretty}, {predictor} coefficient  "
                f"(filter = drop lowest |overall mean weight|)",
                fontsize=14, y=1.02,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            combined_path = os.path.join(
                PLOTS_DIR, f"volcano_sweep_combined_{ct_safe}_{predictor}.png"
            )
            fig.savefig(combined_path)
            plt.close(fig)
            logger.info(f"    wrote {combined_path}")

            # Individual per-filter PNGs
            for pct in DROP_PERCENTILES:
                sub = _apply_filter(df_full, pct, predictor)
                fig, ax = plt.subplots(figsize=(8.5, 6))
                _draw_volcano(
                    ax, sub, predictor=predictor, label=label,
                    title=f"{ct_pretty} | {predictor} | drop {pct}%",
                    n_total=n_total, x_lo=x_lo, x_hi=x_hi,
                )
                fig.tight_layout()
                path_out = os.path.join(
                    PLOTS_DIR,
                    f"volcano_{ct_safe}_{predictor}_drop{pct:02d}.png",
                )
                fig.savefig(path_out)
                plt.close(fig)
                logger.info(f"    wrote {path_out}")

    logger.info("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
