"""Volcano plots for each (cell type × filter level).

For the ordinal-ADNC test (the only one producing q<0.05 hits), draw a
volcano plot at each independent-filter level X% ∈ {0, 25, 50, 75, 90}.
The filter drops edges with the lowest |overall mean weight| (a
condition-label-independent statistic, so FDR control is preserved).

Two outputs per cell type:
- one PNG per filter level (5 files per cell type)
- a 1×5 combined panel showing the progression as the filter tightens
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


def _overall_mean(df):
    return (df["mean_control"] * N_CONTROL + df["mean_ad"] * N_AD) / N_TOTAL


def _apply_filter(df, pct_drop):
    """Return a filtered copy with q recomputed via BH-FDR on the kept subset."""
    if pct_drop <= 0:
        out = df.copy()
    else:
        threshold = np.nanpercentile(np.abs(_overall_mean(df)), pct_drop)
        out = df[np.abs(_overall_mean(df)) >= threshold].copy()
    out["q_value_filtered"] = benjamini_hochberg(out["p_value"].values)
    return out


def _draw_volcano(ax, df, title, n_total, label_top=10, show_legend=True):
    log2fc = df["log2FC"].clip(-6, 6).values
    p = df["p_value"].clip(lower=1e-30).values
    y = -np.log10(p)
    q = df["q_value_filtered"].values

    colors = np.where(q < 0.05, "#d63838",
              np.where(q < 0.10, "#f0a020",
              np.where(q < 0.25, "#a0a020", "#888888")))
    sizes = np.where(q < 0.05, 22, np.where(q < 0.10, 14, 6))
    ax.scatter(log2fc, y, c=colors, s=sizes, alpha=0.6, edgecolors="none")

    n_kept = len(df)
    p_for_q005 = 0.05 / max(n_kept, 1)
    p_for_q010 = 0.10 / max(n_kept, 1)
    ax.axhline(-np.log10(p_for_q005), color="#d63838", ls="--", lw=1, alpha=0.7,
               label="q<0.05 threshold" if show_legend else None)
    ax.axhline(-np.log10(p_for_q010), color="#f0a020", ls="--", lw=1, alpha=0.7,
               label="q<0.10 threshold" if show_legend else None)
    ax.axvline(0, color="grey", lw=0.5, alpha=0.5)

    # Label top edges by filtered-q
    top = df.nsmallest(label_top, "q_value_filtered")
    for _, row in top.iterrows():
        ax.annotate(
            f"{row['TF']}→{row['target']}",
            (np.clip(row["log2FC"], -6, 6),
             -np.log10(max(row["p_value"], 1e-30))),
            textcoords="offset points", xytext=(4, 4),
            fontsize=8, alpha=0.85,
        )

    n_q05 = int((q < 0.05).sum())
    n_q10 = int((q < 0.10).sum())
    ax.set_title(f"{title}\n{n_kept}/{n_total} edges kept | "
                 f"q<0.05: {n_q05}, q<0.10: {n_q10}", fontsize=11)
    ax.set_xlabel("log2 fold change (mean_ad / mean_control)")
    ax.set_ylabel("−log10(p-value)")
    if show_legend:
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.2)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    logger.info(f"plots -> {PLOTS_DIR}")

    for ct_pretty, ct_safe in CELL_TYPES:
        path = os.path.join(OUT_DIR, f"differential_edges_{ct_safe}_ordinal.csv")
        if not os.path.exists(path):
            logger.warning(f"missing {path}, skipping {ct_pretty}")
            continue
        df = pd.read_csv(path)
        n_total = len(df)
        logger.info(f"\n=== {ct_pretty} ({n_total} edges) ===")

        # 1×5 combined panel
        fig, axes = plt.subplots(1, len(DROP_PERCENTILES),
                                  figsize=(5.0 * len(DROP_PERCENTILES), 5.2),
                                  sharey=True)
        for ax, pct in zip(axes, DROP_PERCENTILES):
            sub = _apply_filter(df, pct)
            _draw_volcano(ax, sub,
                           title=f"Filter: drop bottom {pct}%",
                           n_total=n_total,
                           show_legend=(pct == 0))
        fig.suptitle(
            f"Volcano sweep — {ct_pretty}, ordinal ADNC regression  "
            f"(filter = drop lowest |overall mean weight|)",
            fontsize=14, y=1.02,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        combined_path = os.path.join(
            PLOTS_DIR, f"volcano_sweep_combined_{ct_safe}.png"
        )
        fig.savefig(combined_path)
        plt.close(fig)
        logger.info(f"  wrote {combined_path}")

        # Individual per-filter PNGs
        for pct in DROP_PERCENTILES:
            sub = _apply_filter(df, pct)
            fig, ax = plt.subplots(figsize=(8.5, 6))
            _draw_volcano(ax, sub,
                           title=f"{ct_pretty} | ordinal ADNC | drop {pct}%",
                           n_total=n_total)
            fig.tight_layout()
            path_out = os.path.join(
                PLOTS_DIR, f"volcano_{ct_safe}_drop{pct:02d}.png"
            )
            fig.savefig(path_out)
            plt.close(fig)
            logger.info(f"  wrote {path_out}")

    logger.info("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
