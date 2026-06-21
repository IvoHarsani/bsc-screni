"""Generate presentation-quality plots from the SQ1 differential CSVs.

Produces, for each cell type:

* Volcano plot                — log2FC vs -log10(p), top edges labeled
* P-value histogram           — raw p distribution
* Top-N differential edges    — horizontal bar chart of |coef|, signed

Plus shared (cohort-level) plots:

* Cohort demographics         — age boxplot, sex stacked bar, co-pathology bars
* Edge-count comparison       — # tested, # q<0.05, # q<0.10, etc.

All outputs land under ``src/screni/data/output/plots/`` as PNG.
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
OUT_DIR = os.path.join(REPO_ROOT, "src", "screni", "data", "output")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed", "seaad")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

CELL_TYPES = [("Microglia-PVM", "Microglia_PVM"), ("L2/3 IT", "L2_3_IT")]
# tests/variants to include if their CSVs exist
TESTS = [
    ("ols", "OLS (binary)", ""),
    ("wilcoxon", "Wilcoxon (binary)", "_wilcoxon"),
    ("ordinal", "OLS (ordinal ADNC)", "_ordinal"),
    ("cps", "OLS (continuous CPS)", "_cps"),
]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace("-", "_")


def _load_results(ct_safe: str) -> dict[str, pd.DataFrame]:
    out = {}
    for tag, _label, suffix in TESTS:
        path = os.path.join(OUT_DIR, f"differential_edges_{ct_safe}{suffix}.csv")
        if os.path.exists(path):
            out[tag] = pd.read_csv(path)
            logger.info(f"  loaded {tag}: {len(out[tag])} edges from {os.path.basename(path)}")
    return out


# ---------------------------------------------------------------------------
# Volcano plot
# ---------------------------------------------------------------------------


def volcano_plot(df: pd.DataFrame, title: str, save_path: str, label_top: int = 15):
    """log2FC × -log10(p) with q<0.05 and q<0.10 thresholds marked."""
    fig, ax = plt.subplots(figsize=(8, 6))

    log2fc = df["log2FC"].clip(-6, 6).values
    neg_log_p = -np.log10(df["p_value"].clip(lower=1e-30)).values
    q = df["q_value"].values

    # Color: q-value bands
    colors = np.where(q < 0.05, "#d63838",
              np.where(q < 0.10, "#f0a020",
              np.where(q < 0.25, "#a0a020", "#888888")))
    sizes = np.where(q < 0.05, 18, np.where(q < 0.10, 12, 6))
    ax.scatter(log2fc, neg_log_p, c=colors, s=sizes, alpha=0.6, edgecolors="none")

    # FDR threshold lines (informative): show what p would need to be for q<0.05 on the top edge
    # Equivalent: rank-1 p such that q = 0.05 → p = 0.05 / N
    n_tests = len(df)
    p_for_q005 = 0.05 / max(n_tests, 1)
    p_for_q010 = 0.10 / max(n_tests, 1)
    ax.axhline(-np.log10(p_for_q005), color="#d63838", linestyle="--", linewidth=1, alpha=0.7,
               label=f"top-edge q<0.05 threshold  (p < {p_for_q005:.2e})")
    ax.axhline(-np.log10(p_for_q010), color="#f0a020", linestyle="--", linewidth=1, alpha=0.7,
               label=f"top-edge q<0.10 threshold  (p < {p_for_q010:.2e})")
    ax.axvline(0, color="grey", linewidth=0.5, alpha=0.5)

    # Label top edges by q-value
    top = df.nsmallest(label_top, "q_value")
    for _, row in top.iterrows():
        ax.annotate(
            f"{row['TF']}→{row['target']}",
            (np.clip(row["log2FC"], -6, 6), -np.log10(max(row["p_value"], 1e-30))),
            textcoords="offset points", xytext=(4, 4), fontsize=8, alpha=0.85,
        )

    ax.set_xlabel("log2 fold change  (mean_ad / mean_control, signed)")
    ax.set_ylabel("−log10(p-value)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# P-value histogram
# ---------------------------------------------------------------------------


def pvalue_histogram(df: pd.DataFrame, title: str, save_path: str):
    """Raw p-value distribution vs. uniform expectation under null."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    p = df["p_value"].dropna().values
    n = len(p)
    bins = np.linspace(0, 1, 21)
    ax.hist(p, bins=bins, color="#3366aa", alpha=0.8, edgecolor="white")
    # uniform null expectation
    expected = n / 20
    ax.axhline(expected, color="black", linestyle="--", linewidth=1, alpha=0.6,
               label=f"uniform null ({expected:.0f}/bin)")
    ax.set_xlabel("raw p-value")
    ax.set_ylabel("# edges")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Top edges horizontal bar
# ---------------------------------------------------------------------------


def top_edges_barplot(df: pd.DataFrame, title: str, save_path: str, top_n: int = 15):
    top = df.nsmallest(top_n, "q_value").iloc[::-1]   # reverse for plot ordering
    labels = [f"{r['TF']} → {r['target']}" for _, r in top.iterrows()]
    log2fc = top["log2FC"].clip(-6, 6).values
    colors = ["#d63838" if v > 0 else "#3366aa" for v in log2fc]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(labels) + 1.5))
    ax.barh(range(len(labels)), log2fc, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("log2 fold change  (positive = stronger in AD)")
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.5)
    # annotate with q-value
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(log2fc[i] + 0.05 * np.sign(log2fc[i] or 1), i,
                f"q={r['q_value']:.3f}", va="center", fontsize=8, alpha=0.8)
    ax.grid(alpha=0.2, axis="x")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Cohort demographics
# ---------------------------------------------------------------------------


def cohort_demographics(df: pd.DataFrame, title: str, save_path: str):
    """4-panel figure: age boxplot, sex stacked bar, LATE bar, LBD bar."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))

    # 1. Age by condition
    ctrl = df[df["condition"] == "control"]["age"]
    ad = df[df["condition"] == "ad"]["age"]
    axes[0].boxplot([ctrl, ad], labels=["control", "AD"], widths=0.6)
    for i, vals in enumerate([ctrl, ad]):
        axes[0].scatter(np.full(len(vals), i + 1) + np.random.uniform(-0.08, 0.08, len(vals)),
                        vals, alpha=0.6, s=18, color=["#3366aa", "#d63838"][i])
    axes[0].set_ylabel("Age at death")
    axes[0].set_title("Age (n_ctrl={}, n_ad={})".format(len(ctrl), len(ad)))
    axes[0].grid(alpha=0.2, axis="y")

    # 2. Sex stacked bar
    sex_ct = (df.groupby(["condition", "sex"]).size().unstack(fill_value=0)
              .reindex(["control", "ad"]).fillna(0))
    sex_ct.plot(kind="bar", stacked=True, ax=axes[1],
                color=["#88b0d6", "#d68888"], width=0.6, edgecolor="white")
    axes[1].set_title("Sex composition")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("# donors")
    axes[1].legend(title="", fontsize=9)
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].grid(alpha=0.2, axis="y")

    # 3. LATE present
    late = (df.groupby(["condition", "LATE_present"]).size().unstack(fill_value=0)
            .reindex(["control", "ad"]).fillna(0))
    late.plot(kind="bar", stacked=True, ax=axes[2],
              color=["#bbbbbb", "#9d6dd3"], width=0.6, edgecolor="white")
    axes[2].set_title("LATE co-pathology")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("# donors")
    axes[2].legend(title="LATE", fontsize=9)
    axes[2].tick_params(axis="x", rotation=0)
    axes[2].grid(alpha=0.2, axis="y")

    # 4. LBD present
    lbd = (df.groupby(["condition", "LBD_present"]).size().unstack(fill_value=0)
           .reindex(["control", "ad"]).fillna(0))
    lbd.plot(kind="bar", stacked=True, ax=axes[3],
             color=["#bbbbbb", "#d39d6d"], width=0.6, edgecolor="white")
    axes[3].set_title("Lewy-body co-pathology")
    axes[3].set_xlabel("")
    axes[3].set_ylabel("# donors")
    axes[3].legend(title="LBD", fontsize=9)
    axes[3].tick_params(axis="x", rotation=0)
    axes[3].grid(alpha=0.2, axis="y")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Cohort table as styled image
# ---------------------------------------------------------------------------


def cohort_table_image(df: pd.DataFrame, title: str, save_path: str):
    """Render the donor metadata table as a styled image."""
    df_disp = df.copy()
    df_disp = df_disp.sort_values(["condition", "donor_id"]).reset_index()
    df_disp = df_disp[["donor_id", "condition", "n_cells", "age", "sex",
                       "LATE_present", "LBD_present"]]
    df_disp["age"] = df_disp["age"].round(0).astype("Int64")
    df_disp["LATE_present"] = df_disp["LATE_present"].map({True: "Y", False: "—"})
    df_disp["LBD_present"] = df_disp["LBD_present"].map({True: "Y", False: "—"})

    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(df_disp) + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=df_disp.values,
        colLabels=df_disp.columns.tolist(),
        loc="center", cellLoc="center", colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.3)
    # Row tinting by condition
    for i, row in df_disp.iterrows():
        color = "#e8f2fb" if row["condition"] == "control" else "#fbeaea"
        for j in range(len(df_disp.columns)):
            tbl[(i + 1, j)].set_facecolor(color)
    for j in range(len(df_disp.columns)):
        tbl[(0, j)].set_facecolor("#cccccc")
        tbl[(0, j)].set_text_props(weight="bold")

    ax.set_title(title, fontsize=12, pad=10)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Cross-test overlap matrix
# ---------------------------------------------------------------------------


def cross_test_overlap(results: dict[str, pd.DataFrame], title: str, save_path: str,
                        top_k: int = 50):
    if len(results) < 2:
        logger.info("  skipping overlap plot (need >=2 tests)")
        return
    names = list(results.keys())
    top_sets = {n: set(zip(results[n].head(top_k)["TF"], results[n].head(top_k)["target"]))
                for n in names}
    mat = np.zeros((len(names), len(names)), dtype=int)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            mat[i, j] = len(top_sets[a] & top_sets[b])
    fig, ax = plt.subplots(figsize=(0.9 * len(names) + 2, 0.9 * len(names) + 2))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=top_k)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, mat[i, j], ha="center", va="center",
                    color="white" if mat[i, j] > top_k / 2 else "black", fontsize=10)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=f"# shared in top {top_k}", shrink=0.7)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"  -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    logger.info(f"output dir: {PLOTS_DIR}")

    # 1. Cohort plot (uses the donor metadata from any one CSV — first n_donors rows)
    # We reconstruct a donor table from one of the OLS CSVs' per-edge n_donors field,
    # but more reliably we re-derive it from the RNA sub h5ad metadata.
    try:
        import anndata as ad
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        from screni.data.loading_seaad import (
            add_condition_column, add_copathology_columns, select_eligible_donors,
        )
        rna_sub_path = os.path.join(DATA_DIR, "seaad_paired_rna_sq1.h5ad")
        if os.path.exists(rna_sub_path):
            rna = ad.read_h5ad(rna_sub_path)
            add_condition_column(rna)
            add_copathology_columns(rna)
            for ct_pretty, ct_safe in CELL_TYPES:
                mask = (
                    (rna.obs["cell_type"].astype(str) == ct_pretty)
                    & rna.obs["condition"].notna()
                )
                rna_ct = rna[mask].copy()
                donor_meta = select_eligible_donors(
                    rna_ct, cell_type=ct_pretty, min_cells_per_donor=1,
                ).set_index("donor_id")
                cohort_demographics(
                    donor_meta,
                    f"Cohort demographics — {ct_pretty}  (n={len(donor_meta)})",
                    os.path.join(PLOTS_DIR, f"cohort_{ct_safe}.png"),
                )
                cohort_table_image(
                    donor_meta,
                    f"Donor metadata — {ct_pretty}",
                    os.path.join(PLOTS_DIR, f"cohort_table_{ct_safe}.png"),
                )
        else:
            logger.warning("RNA sub not found; skipping cohort plots")
    except Exception as e:
        logger.warning(f"cohort plot block failed: {e}")

    # 2. Per-cell-type, per-test differential plots
    for ct_pretty, ct_safe in CELL_TYPES:
        logger.info(f"\n=== {ct_pretty} ===")
        results = _load_results(ct_safe)
        if not results:
            logger.warning(f"no CSVs found for {ct_pretty}; skipping")
            continue

        for tag, df in results.items():
            label = dict((t[0], t[1]) for t in TESTS)[tag]
            volcano_plot(
                df, f"Volcano — {ct_pretty}, {label}",
                os.path.join(PLOTS_DIR, f"volcano_{ct_safe}_{tag}.png"),
            )
            pvalue_histogram(
                df, f"P-value distribution — {ct_pretty}, {label}",
                os.path.join(PLOTS_DIR, f"pvalues_{ct_safe}_{tag}.png"),
            )
            top_edges_barplot(
                df, f"Top differential edges — {ct_pretty}, {label}",
                os.path.join(PLOTS_DIR, f"top_edges_{ct_safe}_{tag}.png"),
            )

        cross_test_overlap(
            results, f"Top-50 overlap across tests — {ct_pretty}",
            os.path.join(PLOTS_DIR, f"overlap_{ct_safe}.png"),
        )

    logger.info(f"\nAll plots written to {PLOTS_DIR}")
    logger.info("\nTo copy them to your laptop:")
    logger.info(
        "  scp -r daic:/tudelft.net/staff-umbrella/ScReNI/iharsani/"
        "bsc-screni/src/screni/data/output/plots/ ~/Downloads/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
