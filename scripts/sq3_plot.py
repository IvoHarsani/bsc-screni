"""SQ3 figures. Recomputes the reported quantities and saves PNGs to output/plots/sq3/.

  fig1_preservation_null   : per-module Zsummary control->control vs control->AD (the negative result)
  fig2_global_activity     : per-donor total regulatory activity vs AD severity, per cell type
  fig3_module_differential : per-module significance raw vs global-trend-controlled
  fig4_module_heatmap      : control co-regulation adjacency reordered by module (block structure)

Run:  PYTHONPATH=src pixi run python scripts/sq3_plot.py
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from screni.data.module_preservation import leiden_modules, module_preservation
from screni.data.differential import _ols_with_numeric_predictor, benjamini_hochberg

OUT = os.path.join(REPO, "src", "screni", "data", "output")
FIG = os.path.join(OUT, "plots", "sq3")
os.makedirs(FIG, exist_ok=True)
COV = ("age", "sex_Male", "LATE_present", "LBD_present")
EPS = 1e-12
CTS = [("L2_3_IT", "L2/3 IT"), ("Microglia_PVM", "Microglia-PVM")]
BETA = 2


def load(ct):
    z = np.load(f"{OUT}/sq3_incoming_activity_{ct}.npz", allow_pickle=True)
    act = z["activity"].astype(float); genes = np.asarray(z["gene_names"]).astype(str)
    cond = np.asarray(z["condition"]).astype(str); cdon = np.asarray(z["donor_ids"]).astype(str)
    pe = np.load(f"{OUT}/per_edge_per_donor_{ct}.npz", allow_pickle=True)
    tgt = set(np.unique(np.asarray(pe["targets"]).astype(str)))
    uni = np.array([g in tgt for g in genes])
    meta = pd.DataFrame({k: pe[k].astype(float) for k in
                         ["adnc_ordinal", "age", "sex_Male", "LATE_present", "LBD_present"]},
                        index=np.asarray(pe["donor_ids"]).astype(str))
    meta["condition"] = np.asarray(pe["condition"]).astype(str)
    return act[:, uni], genes[uni], cond, cdon, meta


def adj(cells, beta=BETA):
    X = cells - cells.mean(1, keepdims=True)
    S = np.nan_to_num(np.corrcoef(X.T), nan=0.0)
    A = np.abs(S) ** beta; A = (A + A.T) / 2; np.fill_diagonal(A, 1.0)
    return A


# ---------------- Fig 1: preservation null ----------------
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (ct, pretty) in zip(axes, CTS):
        act, genes, cond, cdon, meta = load(ct)
        ci = np.where(cond == "control")[0]; ai = np.where(cond == "ad")[0]
        cdonors = np.unique(cdon[ci]); m_side = len(ci) // 2
        keep = (act[ci].var(0) > EPS) & (act[ai].var(0) > EPS)
        zcc, zca = [], []
        for boot in range(4):
            rng = np.random.default_rng(boot)
            perm = rng.permutation(cdonors)
            A_d, B_d = set(perm[:len(cdonors)//2]), set(perm[len(cdonors)//2:])
            Ac = ci[np.array([cdon[i] in A_d for i in ci])]
            Bc = ci[np.array([cdon[i] in B_d for i in ci])]
            m = min(len(Ac), len(Bc), m_side)
            Ac = rng.choice(Ac, m, replace=False); Bc = rng.choice(Bc, m, replace=False)
            adc = rng.choice(ai, m, replace=False)
            A_ref = adj(act[Ac][:, keep]); A_B = adj(act[Bc][:, keep]); A_ad = adj(act[adc][:, keep])
            lab = leiden_modules(A_ref, resolution=1.0, seed=42)
            big = np.bincount(lab) >= 10
            r_cc = module_preservation(A_ref, A_B, lab, n_permutations=100, seed=1)
            r_ca = module_preservation(A_ref, A_ad, lab, n_permutations=100, seed=1)
            sel = r_cc.module_sizes >= 10
            zcc.extend(r_cc.Zsummary[sel]); zca.extend(r_ca.Zsummary[sel])
        zcc, zca = np.array(zcc), np.array(zca)
        lim = [min(zcc.min(), zca.min()) - 1, max(zcc.max(), zca.max()) + 1]
        ax.plot(lim, lim, "k--", lw=1, alpha=.6, label="y = x (no AD effect)")
        ax.axhspan(-100, 2, color="red", alpha=.05); ax.axvspan(-100, 2, color="red", alpha=.05)
        ax.scatter(zcc, zca, s=55, alpha=.75, edgecolor="k", linewidth=.4)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Zsummary  control → control\n(natural variation)")
        ax.set_ylabel("Zsummary  control → AD")
        ax.set_title(f"{pretty}\n(each point = one module, 4 donor splits)")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Module preservation: AD ≈ healthy-vs-healthy  →  module structure conserved",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig1_preservation_null.png", dpi=200)
    plt.close(fig)
    print("fig1 done")


# ---------------- Fig 2: global activity vs severity ----------------
def fig2():
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, (ct, pretty) in zip(axes, CTS):
        act, genes, cond, cdon, meta = load(ct)
        donors = meta.index.values
        total = np.array([act[cdon == d].mean() for d in donors])
        sev = meta["adnc_ordinal"].values
        # covariate-adjusted ordinal trend, and binary control/AD contrast for comparison
        r_ord = _ols_with_numeric_predictor(total, meta, "adnc_ordinal", covariates=COV)
        meta_b = meta.copy(); meta_b["binary_ad"] = (sev >= 2).astype(float)
        r_bin = _ols_with_numeric_predictor(total, meta_b, "binary_ad", covariates=COV)
        # individual donor dots (faded), coloured by control/AD
        colors = np.where(meta["condition"].values == "ad", "#c0392b", "#2980b9")
        jit = np.random.default_rng(0).normal(0, 0.06, len(donors))
        ax.scatter(sev + jit, total, c=colors, s=42, alpha=.55, edgecolor="k", linewidth=.3, zorder=2)
        # per-severity-level mean +/- SD (honest view of the spread)
        lvls = [0, 1, 2, 3]
        means = [total[sev == l].mean() for l in lvls]
        sds = [total[sev == l].std() for l in lvls]
        ns = [int((sev == l).sum()) for l in lvls]
        ax.errorbar(lvls, means, yerr=sds, fmt="o-", color="black", lw=1.6, ms=8,
                    capsize=5, zorder=3)
        for l, m, n in zip(lvls, means, ns):
            ax.annotate(f"n={n}", (l, m), textcoords="offset points", xytext=(9, 7), fontsize=7.5)
        ax.set_xticks(lvls); ax.set_xlim(-0.4, 3.4)
        ax.set_xlabel("ADNC severity (0=Not AD … 3=High)")
        ax.set_ylabel("mean incoming regulatory activity per donor")
        ax.set_title(f"{pretty}\ngraded (ordinal) p={r_ord.p_value:.3f}   |   "
                     f"binary control-vs-AD p={r_bin.p_value:.3f}")
    handles = [Patch(color="#2980b9", label="control donor"),
               Patch(color="#c0392b", label="AD donor"),
               Line2D([0], [0], color="black", marker="o", label="level mean ± SD")]
    fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("L2/3 activity declines gradually with severity (ordinal sig., binary n.s.); "
                 "microglia flat — note the wide within-level spread",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig2_global_activity.png", dpi=200)
    plt.close(fig)
    print("fig2 done")


# ---------------- Fig 3: module-differential collapse ----------------
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (ct, pretty) in zip(axes, CTS):
        act, genes, cond, cdon, meta = load(ct)
        donors = meta.index.values
        pb = np.vstack([act[cdon == d].mean(0) for d in donors])
        total = pb.mean(1); meta2 = meta.copy(); meta2["total_activity"] = total
        ci = cond == "control"
        keep = (act[ci] - act[ci].mean(1, keepdims=True)).var(0) > EPS
        A = adj(act[ci][:, keep])
        lab = leiden_modules(A, resolution=1.5, seed=42)
        sizes = np.bincount(lab); mods = [m for m in np.unique(lab) if sizes[m] >= 10]
        pbk = pb[:, keep]
        rawp, ctrlp = [], []
        for m in mods:
            ma = pbk[:, lab == m].mean(1)
            rawp.append(_ols_with_numeric_predictor(ma, meta, "adnc_ordinal", covariates=COV).p_value)
            ctrlp.append(_ols_with_numeric_predictor(ma, meta2, "adnc_ordinal", covariates=COV + ("total_activity",)).p_value)
        rq = benjamini_hochberg(np.array(rawp)); cq = benjamini_hochberg(np.array(ctrlp))
        x = np.arange(len(mods))
        ax.scatter(x, -np.log10(rq), s=60, c="#e67e22", label="raw (no global control)", edgecolor="k", linewidth=.4)
        ax.scatter(x, -np.log10(cq), s=60, c="#16a085", marker="s", label="controlling for global trend", edgecolor="k", linewidth=.4)
        ax.axhline(-np.log10(0.05), color="k", ls="--", lw=1, alpha=.7)
        ax.text(0, -np.log10(0.05) + .05, "q = 0.05", fontsize=8)
        ax.set_xlabel("module"); ax.set_ylabel("-log10(q)")
        ax.set_title(f"{pretty} ({len(mods)} modules)")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Apparent module hits vanish once the global activity trend is removed",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_module_differential.png", dpi=200)
    plt.close(fig)
    print("fig3 done")


# ---------------- Fig 4: adjacency heatmap by module ----------------
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (ct, pretty) in zip(axes, CTS):
        act, genes, cond, cdon, meta = load(ct)
        ci = cond == "control"
        keep = (act[ci] - act[ci].mean(1, keepdims=True)).var(0) > EPS
        X = act[ci][:, keep] - act[ci][:, keep].mean(1, keepdims=True)
        S = np.abs(np.nan_to_num(np.corrcoef(X.T), nan=0.0))   # |corr| (more visible than |corr|^beta)
        A = adj(act[ci][:, keep])
        lab = leiden_modules(A, resolution=1.0, seed=42)
        # order genes by module, only modules with >=10 genes shown contiguously first
        sizes = np.bincount(lab)
        big = [m for m in np.argsort(sizes)[::-1] if sizes[m] >= 10]
        order = np.concatenate([np.where(lab == m)[0] for m in big] +
                               [np.where(~np.isin(lab, big))[0]])
        Sord = S[np.ix_(order, order)]
        np.fill_diagonal(Sord, np.nan)
        im = ax.imshow(Sord, cmap="magma", vmin=0, vmax=0.5, aspect="auto", interpolation="nearest")
        # module boundary lines
        b = np.cumsum([sizes[m] for m in big])
        for x in b[:-1]:
            ax.axhline(x - .5, color="cyan", lw=.6, alpha=.7); ax.axvline(x - .5, color="cyan", lw=.6, alpha=.7)
        ax.set_title(f"{pretty}\ncontrol co-regulation network, genes ordered by module\n({len(big)} modules ≥10 genes, cyan lines)")
        ax.set_xlabel("genes"); ax.set_ylabel("genes")
        fig.colorbar(im, ax=ax, fraction=.046, label="|correlation|")
    fig.suptitle("Co-regulation modules: modest but real block structure in the control network",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig4_module_heatmap.png", dpi=200)
    plt.close(fig)
    print("fig4 done")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print(f"\nall figures in {FIG}")
