"""Volcano + gradient-equivalent plots for every coefficient in the OLS.

Reads the outputs of ``compute_all_coefficients.py``:
- ``output/coefficients_all_<celltype>.csv``  (per-edge stats for all 5 predictors)
- ``output/per_edge_per_donor_<celltype>.npz`` (per-donor weight cache)

For each predictor (adnc_ordinal, age, sex_Male, LATE_present, LBD_present):

* **Volcano plot** — x = coefficient, y = −log10(p), points colored by
  q-value, threshold lines for top-edge q<0.05 and q<0.10, top-15 edges
  labeled.

* **Gradient-equivalent plot** — for the top-6 edges by q-value, a small
  panel showing per-donor edge weight against that predictor's value.
  For continuous predictors (adnc_ordinal, age) → scatter with linear
  fit; for binary (sex_Male, LATE_present, LBD_present) → strip plot of
  the two groups with means + linear-fit slope shown.

Saves all plots under ``output/plots/coefficients/``.
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO_ROOT, "src", "screni", "data", "output")
PLOTS_DIR = os.path.join(OUT_DIR, "plots", "coefficients")

CELL_TYPES = [("Microglia-PVM", "Microglia_PVM"), ("L2/3 IT", "L2_3_IT")]

# Predictor metadata: (column-prefix in CSV, kind, x-axis label, key in cache,
#                     binary-tick-labels-if-applicable)
PREDICTORS = [
    ("adnc_ordinal",  "ordinal",    "AD severity (Not AD → High)",
     "adnc_ordinal",  ["Not AD", "Low", "Int", "High"]),
    ("age",           "continuous", "Age at death (years)",
     "age",           None),
    ("sex_Male",      "binary",     "Sex",
     "sex_Male",      ["Female", "Male"]),
    ("LATE_present",  "binary",     "LATE co-pathology",
     "LATE_present",  ["absent", "present"]),
    ("LBD_present",   "binary",     "Lewy body co-pathology",
     "LBD_present",   ["absent", "present"]),
]

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


# ---------------------------------------------------------------------------
# Volcano per coefficient
# ---------------------------------------------------------------------------


def volcano_for_coefficient(df: pd.DataFrame, predictor: str, label: str,
                             title: str, save_path: str, label_top: int = 15):
    coef_col = f"{predictor}_coef"
    p_col = f"{predictor}_p"
    q_col = f"{predictor}_q"

    if coef_col not in df.columns or p_col not in df.columns:
        logger.warning(f"missing columns for {predictor}, skipping")
        return

    fig, ax = plt.subplots(figsize=(8.5, 6))

    x = df[coef_col].values
    p = df[p_col].clip(lower=1e-30).values
    y = -np.log10(p)
    q = df[q_col].values

    # Clip x for plotting to avoid extreme outliers blowing the axis
    x_clip = np.clip(x, np.nanquantile(x, 0.001), np.nanquantile(x, 0.999))

    colors = np.where(q < 0.05, "#d63838",
              np.where(q < 0.10, "#f0a020",
              np.where(q < 0.25, "#a0a020", "#888888")))
    sizes = np.where(q < 0.05, 20, np.where(q < 0.10, 14, 6))
    ax.scatter(x_clip, y, c=colors, s=sizes, alpha=0.6, edgecolors="none")

    n_tests = len(df)
    p_for_q005 = 0.05 / max(n_tests, 1)
    p_for_q010 = 0.10 / max(n_tests, 1)
    ax.axhline(-np.log10(p_for_q005), color="#d63838", ls="--", lw=1, alpha=0.7,
               label=f"top-edge q<0.05 (p < {p_for_q005:.2e})")
    ax.axhline(-np.log10(p_for_q010), color="#f0a020", ls="--", lw=1, alpha=0.7,
               label=f"top-edge q<0.10 (p < {p_for_q010:.2e})")
    ax.axvline(0, color="grey", lw=0.5, alpha=0.5)

    # Label top edges
    top = df.nsmallest(label_top, q_col)
    for _, row in top.iterrows():
        ax.annotate(
            f"{row['TF']}→{row['target']}",
            (np.clip(row[coef_col],
                      np.nanquantile(x, 0.001),
                      np.nanquantile(x, 0.999)),
             -np.log10(max(row[p_col], 1e-30))),
            textcoords="offset points", xytext=(4, 4),
            fontsize=8, alpha=0.85,
        )

    ax.set_xlabel(f"{label} coefficient")
    ax.set_ylabel("−log10(p-value)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Gradient-equivalent plots (continuous + binary)
# ---------------------------------------------------------------------------


def gradient_for_coefficient(df: pd.DataFrame, cache: dict, predictor: str,
                              kind: str, x_label: str, binary_labels,
                              title: str, save_path: str, top_n: int = 6):
    q_col = f"{predictor}_q"
    p_col = f"{predictor}_p"
    coef_col = f"{predictor}_coef"

    if q_col not in df.columns:
        return

    top = df.nsmallest(top_n, q_col).reset_index(drop=True)
    if len(top) == 0:
        return

    n_cols = 3
    n_rows = int(np.ceil(top_n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 3.4 * n_rows),
                              squeeze=False)

    edge_weights = cache["edge_weights"]       # (n_edges, n_donors)
    cached_tfs = cache["tfs"]
    cached_targets = cache["targets"]
    pred_values = cache[predictor]              # (n_donors,)
    condition = cache["condition"]              # array of "control"/"ad"

    # Build lookup from (TF, target) → row index in cache
    pair_to_idx = {(str(tf), str(t)): i
                    for i, (tf, t) in enumerate(zip(cached_tfs, cached_targets))}

    for ti, edge in top.iterrows():
        ax = axes[ti // n_cols, ti % n_cols]
        key = (edge["TF"], edge["target"])
        if key not in pair_to_idx:
            ax.text(0.5, 0.5, "(not in cache)", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_axis_off()
            continue
        w = edge_weights[pair_to_idx[key]]      # (n_donors,)
        x = pred_values

        # Color by binary condition for visual reference
        colors = ["#3366aa" if c == "control" else "#d63838" for c in condition]

        if kind == "binary":
            # Strip plot: jitter on the binary axis
            rng = np.random.default_rng(42 + ti)
            jitter = rng.uniform(-0.08, 0.08, size=len(x))
            ax.scatter(x + jitter, w, c=colors, s=44, alpha=0.8,
                       edgecolors="white", linewidths=0.5, zorder=3)
            # Group means
            for level in [0, 1]:
                m = x == level
                if m.any():
                    ax.hlines(np.mean(w[m]), level - 0.2, level + 0.2,
                              colors="black", linestyles="-", lw=2, zorder=4)
            # Linear fit (same as regression, no covariates)
            slope, intercept = np.polyfit(x, w, 1)
            xfit = np.array([-0.2, 1.2])
            ax.plot(xfit, slope * xfit + intercept, color="black", lw=1.5,
                    alpha=0.6, zorder=2,
                    label=f"slope = {slope:.2e}")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(binary_labels, fontsize=9)
            ax.set_xlim(-0.4, 1.4)
        else:
            # Continuous (scatter)
            ax.scatter(x, w, c=colors, s=44, alpha=0.8,
                       edgecolors="white", linewidths=0.5, zorder=3)
            # Linear fit
            mask_ok = ~(np.isnan(x) | np.isnan(w))
            if mask_ok.sum() >= 3:
                slope, intercept = np.polyfit(x[mask_ok], w[mask_ok], 1)
                xfit = np.array([x.min() - 0.05 * (x.max() - x.min()),
                                  x.max() + 0.05 * (x.max() - x.min())])
                ax.plot(xfit, slope * xfit + intercept, color="black", lw=1.5,
                        alpha=0.7, zorder=2,
                        label=f"slope = {slope:.2e}")
            if kind == "ordinal":
                ax.set_xticks([0, 1, 2, 3])
                ax.set_xticklabels(binary_labels, rotation=20, ha="right", fontsize=9)
                ax.set_xlim(-0.3, 3.3)

        ax.set_ylabel("edge weight" if ti % n_cols == 0 else "")
        ax.set_title(
            f"{edge['TF']} → {edge['target']}\n"
            f"q={edge[q_col]:.3g}, p={edge[p_col]:.2g}",
            fontsize=10,
        )
        ax.grid(alpha=0.2, axis="y")
        ax.legend(loc="best", fontsize=8, frameon=False)

    # Hide leftover axes if top_n < n_rows*n_cols
    for k in range(len(top), n_rows * n_cols):
        axes[k // n_cols, k % n_cols].set_axis_off()

    legend = [
        mpatches.Patch(color="#3366aa", label="control donor"),
        mpatches.Patch(color="#d63838", label="AD donor"),
    ]
    fig.legend(handles=legend, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=13, y=1.04)
    fig.supxlabel(x_label, fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    logger.info(f"plots → {PLOTS_DIR}")

    for ct_pretty, ct_safe in CELL_TYPES:
        logger.info(f"\n=== {ct_pretty} ===")
        csv_path = os.path.join(OUT_DIR, f"coefficients_all_{ct_safe}.csv")
        cache_path = os.path.join(OUT_DIR, f"per_edge_per_donor_{ct_safe}.npz")
        if not os.path.exists(csv_path):
            logger.warning(f"missing {csv_path}; run compute_all_coefficients.py first")
            continue
        if not os.path.exists(cache_path):
            logger.warning(f"missing {cache_path}")
            continue
        df = pd.read_csv(csv_path)
        cache = np.load(cache_path, allow_pickle=True)
        logger.info(f"  loaded {len(df)} edges, cache shape {cache['edge_weights'].shape}")

        for prefix, kind, xlabel, cache_key, binary_labels in PREDICTORS:
            logger.info(f"\n  -- {prefix} ({kind}) --")

            volcano_for_coefficient(
                df, predictor=prefix, label=prefix.replace("_", " "),
                title=f"Volcano — {ct_pretty}, {prefix} coefficient",
                save_path=os.path.join(PLOTS_DIR,
                                       f"volcano_{ct_safe}_{prefix}.png"),
            )

            gradient_for_coefficient(
                df, cache, predictor=prefix, kind=kind, x_label=xlabel,
                binary_labels=binary_labels,
                title=f"Top edges by {prefix} — {ct_pretty}",
                save_path=os.path.join(PLOTS_DIR,
                                       f"gradient_{ct_safe}_{prefix}.png"),
            )

    logger.info("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
