"""SQ3 module-differential pipeline — stage-by-stage diagnostics (incremental validation).

Stage 1: incoming activity (the per-cell, per-gene input)
Stage 2: co-regulation network (correlation -> adjacency)
(Stages 3-5 added incrementally.)

Run:  PYTHONPATH=src pixi run python scripts/sq3_diagnostics.py
Figures -> output/plots/sq3_diag/ ; key numbers printed to stdout.
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from screni.data.differential import _ols_with_numeric_predictor, benjamini_hochberg
from screni.data.module_preservation import leiden_modules
OUT = os.path.join(REPO, "src", "screni", "data", "output")
FIG = os.path.join(OUT, "plots", "sq3_diag")
os.makedirs(FIG, exist_ok=True)
CTS = [("L2_3_IT", "L2/3 IT"), ("Microglia_PVM", "Microglia-PVM")]
EPS = 1e-12
COV = ("age", "sex_Male", "LATE_present", "LBD_present")


def load(ct):
    z = np.load(f"{OUT}/sq3_incoming_activity_{ct}.npz", allow_pickle=True)
    act = z["activity"].astype(float); genes = np.asarray(z["gene_names"]).astype(str)
    cond = np.asarray(z["condition"]).astype(str); cdon = np.asarray(z["donor_ids"]).astype(str)
    tgt = set(np.unique(np.asarray(np.load(f"{OUT}/per_edge_per_donor_{ct}.npz", allow_pickle=True)["targets"]).astype(str)))
    uni = np.array([g in tgt for g in genes])
    return act[:, uni], genes[uni], cond, cdon


def build_adj(cells, beta=2):
    """Centered cross-cell correlation -> |corr|^beta adjacency. Returns (A, keep_mask over input genes)."""
    X = cells - cells.mean(1, keepdims=True)
    keep = X.var(0) > EPS
    S = np.nan_to_num(np.corrcoef(X[:, keep].T), nan=0.0)
    A = np.abs(S) ** beta
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 1.0)
    return A, keep


def meta_of(ct):
    pe = np.load(f"{OUT}/per_edge_per_donor_{ct}.npz", allow_pickle=True)
    return pd.DataFrame({k: pe[k].astype(float) for k in
                         ["adnc_ordinal", "age", "sex_Male", "LATE_present", "LBD_present"]},
                        index=np.asarray(pe["donor_ids"]).astype(str))


# ============================= STAGE 1 =============================
def stage1():
    print("\n========== STAGE 1: incoming activity ==========")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (ct, pretty) in enumerate(CTS):
        act, genes, cond, cdon = load(ct)
        N, G = act.shape
        gm = act.mean(0)  # per-gene mean incoming activity

        # (a) per-gene mean activity distribution + top genes
        ax = axes[row, 0]
        ax.hist(gm, bins=40, color="#34495e")
        top = np.argsort(gm)[::-1][:8]
        ax.set_title(f"{pretty}: per-gene mean incoming activity\n(top: {', '.join(genes[top][:6])})", fontsize=9)
        ax.set_xlabel("mean incoming activity"); ax.set_ylabel("# genes")

        # (b) variance decomposition: between-cell vs between-gene vs residual
        m = act.mean()
        cell_means = act.mean(1); gene_means = act.mean(0)
        SST = ((act - m) ** 2).sum()
        SS_cell = G * ((cell_means - m) ** 2).sum()
        SS_gene = N * ((gene_means - m) ** 2).sum()
        SS_res = SST - SS_cell - SS_gene
        fr = np.array([SS_cell, SS_gene, SS_res]) / SST
        ax = axes[row, 1]
        ax.bar(["between-cell\n(global level)", "between-gene", "residual\n(interaction)"], fr,
               color=["#c0392b", "#2980b9", "#7f8c8d"])
        for i, v in enumerate(fr):
            ax.text(i, v + .01, f"{v:.0%}", ha="center", fontsize=9)
        ax.set_ylim(0, 1); ax.set_ylabel("fraction of total variance")
        ax.set_title(f"{pretty}: variance of incoming activity", fontsize=9)
        print(f"{pretty}: variance  between-cell={fr[0]:.1%}  between-gene={fr[1]:.1%}  residual={fr[2]:.1%}")

        # (c) split-half reproducibility of per-gene mean
        rng = np.random.default_rng(0)
        perm = rng.permutation(N)
        h1, h2 = perm[:N // 2], perm[N // 2:]
        g1, g2 = act[h1].mean(0), act[h2].mean(0)
        r = np.corrcoef(g1, g2)[0, 1]
        ax = axes[row, 2]
        ax.scatter(g1, g2, s=6, alpha=.4, color="#16a085")
        ax.set_xlabel("per-gene mean (cells half 1)"); ax.set_ylabel("per-gene mean (cells half 2)")
        ax.set_title(f"{pretty}: split-half reproducibility\nr = {r:.3f}", fontsize=9)
        print(f"{pretty}: split-half per-gene reproducibility r={r:.3f}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/stage1_incoming_activity.png", dpi=180)
    plt.close(fig)
    print(f"-> {FIG}/stage1_incoming_activity.png")


# ============================= STAGE 2 =============================
def stage2(beta=2):
    print("\n========== STAGE 2: co-regulation network ==========")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (ct, pretty) in enumerate(CTS):
        act, genes, cond, cdon = load(ct)
        ctrl = act[cond == "control"]; adc = act[cond == "ad"]
        keep = (ctrl - ctrl.mean(1, keepdims=True)).var(0) > EPS

        # correlations WITHOUT and WITH per-cell centering (control cells)
        Xraw = ctrl[:, keep]
        Xc = Xraw - Xraw.mean(1, keepdims=True)
        Sraw = np.nan_to_num(np.corrcoef(Xraw.T), nan=0.0)
        S = np.nan_to_num(np.corrcoef(Xc.T), nan=0.0)
        iu = np.triu_indices(S.shape[0], k=1)

        # (a) eigen-spectrum of the (centered) correlation matrix -> PC1 dominance
        ev = np.linalg.eigvalsh(S)[::-1]
        pc1 = ev[0] / ev.sum()
        ax = axes[row, 0]
        ax.bar(range(1, 11), ev[:10] / ev.sum(), color="#8e44ad")
        ax.set_title(f"{pretty}: correlation eigen-spectrum\nPC1 = {pc1:.0%} of variance", fontsize=9)
        ax.set_xlabel("component"); ax.set_ylabel("fraction of variance")
        print(f"{pretty}: PC1 of co-regulation correlation = {pc1:.1%}  (PC1-3 = {ev[:3].sum()/ev.sum():.1%})")

        # (b) correlation distribution before vs after per-cell centering
        ax = axes[row, 1]
        ax.hist(Sraw[iu], bins=60, alpha=.55, color="#e67e22", label="raw (no centering)")
        ax.hist(S[iu], bins=60, alpha=.55, color="#16a085", label="per-cell centered")
        ax.axvline(0, color="k", lw=.8)
        ax.set_title(f"{pretty}: gene-gene correlations\nmean raw={Sraw[iu].mean():.2f} → centered={S[iu].mean():.2f}", fontsize=9)
        ax.set_xlabel("correlation"); ax.set_ylabel("# gene pairs"); ax.legend(fontsize=8)
        print(f"{pretty}: corr mean raw={Sraw[iu].mean():.3f} -> centered={S[iu].mean():.3f}")

        # (c) control vs AD adjacency similarity (foreshadows the preservation null)
        kad = (adc - adc.mean(1, keepdims=True)).var(0) > EPS
        both = keep & kad
        Xc2 = ctrl[:, both] - ctrl[:, both].mean(1, keepdims=True)
        Xa2 = adc[:, both] - adc[:, both].mean(1, keepdims=True)
        A_ctrl = np.abs(np.nan_to_num(np.corrcoef(Xc2.T), nan=0.0)) ** beta
        A_ad = np.abs(np.nan_to_num(np.corrcoef(Xa2.T), nan=0.0)) ** beta
        iu2 = np.triu_indices(A_ctrl.shape[0], k=1)
        ac, aa = A_ctrl[iu2], A_ad[iu2]
        radj = np.corrcoef(ac, aa)[0, 1]
        sub = np.random.default_rng(0).choice(len(ac), min(4000, len(ac)), replace=False)
        ax = axes[row, 2]
        ax.scatter(ac[sub], aa[sub], s=4, alpha=.25, color="#2c3e50")
        lim = [0, max(ac[sub].max(), aa[sub].max())]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_title(f"{pretty}: control vs AD adjacency\nr = {radj:.2f} (edge weights ~unchanged)", fontsize=9)
        ax.set_xlabel("adjacency (control)"); ax.set_ylabel("adjacency (AD)")
        print(f"{pretty}: control-vs-AD adjacency correlation r={radj:.3f}")
    fig.tight_layout()
    fig.savefig(f"{FIG}/stage2_network.png", dpi=180)
    plt.close(fig)
    print(f"-> {FIG}/stage2_network.png")


# ============================= STAGE 3 =============================
def stage3(beta=2, resolution=1.5):
    print("\n========== STAGE 3: module definition ==========")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, (ct, pretty) in enumerate(CTS):
        act, genes, cond, cdon = load(ct)
        ctrl = act[cond == "control"]; adc = act[cond == "ad"]
        A, keepc = build_adj(ctrl, beta)
        genes_c = genes[keepc]
        lab = leiden_modules(A, resolution=resolution, seed=42)
        sizes = np.bincount(lab); big = sizes >= 10

        # (a) module sizes
        ax = axes[row, 0]
        ssort = np.sort(sizes[sizes >= 3])[::-1]
        ax.bar(range(len(ssort)), ssort, color=["#2980b9" if s >= 10 else "#bdc3c7" for s in ssort])
        ax.axhline(10, color="r", ls="--", lw=1)
        ax.set_title(f"{pretty}: module sizes (res={resolution})\n{int(big.sum())} modules ≥10 genes", fontsize=9)
        ax.set_xlabel("module (sorted)"); ax.set_ylabel("# genes")

        # SEED stability (same cells, different seed) — average over many seed PAIRS, not one
        labs = [leiden_modules(A, resolution=resolution, seed=s) for s in range(6)]
        seed_ari = np.mean([adjusted_rand_score(labs[i], labs[j]) for i in range(6) for j in range(i + 1, 6)])
        # SAMPLE stability — control->control vs control->AD partition ARI (size-matched, donor-disjoint).
        # This is the proper baseline: low control->AD only means "disease" if it is below control->control.
        ci = np.where(cond == "control")[0]; ai = np.where(cond == "ad")[0]; cdu = np.unique(cdon[ci])
        cc, ca = [], []
        for rep in range(3):
            rng = np.random.default_rng(rep); perm = rng.permutation(cdu)
            dA, dB = set(perm[:len(cdu) // 2]), set(perm[len(cdu) // 2:])
            Ac = ci[np.array([cdon[i] in dA for i in ci])]; Bc = ci[np.array([cdon[i] in dB for i in ci])]
            m = min(len(Ac), len(Bc))
            Ac = rng.choice(Ac, m, replace=False); Bc = rng.choice(Bc, m, replace=False); ad = rng.choice(ai, m, replace=False)
            keep = ((act[Ac] - act[Ac].mean(1, keepdims=True)).var(0) > EPS) & \
                   ((act[Bc] - act[Bc].mean(1, keepdims=True)).var(0) > EPS) & \
                   ((act[ad] - act[ad].mean(1, keepdims=True)).var(0) > EPS)
            pA = leiden_modules(build_adj(act[Ac][:, keep], beta)[0], resolution=resolution, seed=42)
            pB = leiden_modules(build_adj(act[Bc][:, keep], beta)[0], resolution=resolution, seed=42)
            pAD = leiden_modules(build_adj(act[ad][:, keep], beta)[0], resolution=resolution, seed=42)
            cc.append(adjusted_rand_score(pA, pB)); ca.append(adjusted_rand_score(pA, pAD))
        print(f"{pretty}: {int(big.sum())} modules≥10; seed-ARI(multi)={seed_ari:.2f}; "
              f"ctrl→ctrl partition ARI={np.mean(cc):.2f}; ctrl→AD={np.mean(ca):.2f}")

        # (b) resolution sweep: # modules>=10 and MULTI-SEED seed-ARI
        ax = axes[row, 1]
        res_list = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]; ncount, rari = [], []
        for r in res_list:
            ls = [leiden_modules(A, resolution=r, seed=s) for s in range(5)]
            ncount.append(int((np.bincount(ls[0]) >= 10).sum()))
            rari.append(np.mean([adjusted_rand_score(ls[i], ls[j]) for i in range(5) for j in range(i + 1, 5)]))
        ax.plot(res_list, ncount, "o-", color="#2980b9"); ax.set_ylabel("# modules ≥10", color="#2980b9")
        ax2 = ax.twinx(); ax2.plot(res_list, rari, "s--", color="#e67e22"); ax2.set_ylabel("seed-ARI (multi-seed)", color="#e67e22"); ax2.set_ylim(0, 1)
        ax.set_xlabel("Leiden resolution"); ax.set_title(f"{pretty}: resolution sweep", fontsize=9)

        # enrichment (guarded — gprofiler may be absent; then dump gene lists for external enrichment)
        try:
            from gprofiler import GProfiler
            gp = GProfiler(return_dataframe=True)
            print(f"  {pretty} top GO:BP per module:")
            for m in np.where(big)[0]:
                r = gp.profile(organism="hsapiens", query=genes_c[lab == m].tolist(), sources=["GO:BP"])
                print(f"    module {m} (n={sizes[m]}): {r.sort_values('p_value')['name'].iloc[0] if len(r) else '(none)'}")
        except Exception as e:
            path = f"{OUT}/sq3_module_genes_{ct}.txt"
            with open(path, "w") as f:
                for m in np.where(big)[0]:
                    f.write(f"module {m} (n={sizes[m]}): {' '.join(genes_c[lab == m])}\n")
            print(f"  [enrichment skipped: {type(e).__name__}; module gene lists -> {path}]")
    fig.tight_layout(); fig.savefig(f"{FIG}/stage3_modules.png", dpi=180); plt.close(fig)
    print(f"-> {FIG}/stage3_modules.png")


# ============================= STAGE 4 =============================
def stage4(beta=2, resolution=1.5):
    print("\n========== STAGE 4: per-donor module activity ==========")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, (ct, pretty) in enumerate(CTS):
        act, genes, cond, cdon = load(ct)
        meta = meta_of(ct); donors = meta.index.values; sev = meta["adnc_ordinal"].values
        ctrl = act[cond == "control"]
        A, keepc = build_adj(ctrl, beta); lab = leiden_modules(A, resolution=resolution, seed=42)
        sizes = np.bincount(lab); mods = [m for m in np.unique(lab) if sizes[m] >= 10]
        actk = act[:, keepc]
        pb = np.vstack([actk[cdon == d].mean(0) for d in donors])           # donors x kept genes
        modact = np.column_stack([pb[:, lab == m].mean(1) for m in mods])    # donors x modules
        order = np.argsort(sev)

        # (a) donors (severity-ordered) x modules, z-scored per module
        Z = (modact - modact.mean(0)) / (modact.std(0) + 1e-12)
        ax = axes[row, 0]
        im = ax.imshow(Z[order], aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_yticks(range(len(donors))); ax.set_yticklabels(sev[order].astype(int), fontsize=6)
        ax.set_xlabel("module"); ax.set_ylabel("donor severity (0 top → 3 bottom)")
        ax.set_title(f"{pretty}: per-donor module activity (z-scored)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.046)

        # (b) inter-module correlation across donors
        C = np.corrcoef(modact.T)
        offmean = C[np.triu_indices_from(C, 1)].mean()
        ax = axes[row, 1]
        im = ax.imshow(C, cmap="magma", vmin=-1, vmax=1)
        ax.set_title(f"{pretty}: inter-module activity correlation\nmean off-diag = {offmean:.2f}", fontsize=9)
        ax.set_xlabel("module"); ax.set_ylabel("module")
        fig.colorbar(im, ax=ax, fraction=.046)
        print(f"{pretty}: mean inter-module correlation across donors = {offmean:.2f}  ({len(mods)} modules)")
    fig.tight_layout(); fig.savefig(f"{FIG}/stage4_module_activity.png", dpi=180); plt.close(fig)
    print(f"-> {FIG}/stage4_module_activity.png")


# ============================= STAGE 5 =============================
def stage5(beta=2, resolution=1.5):
    print("\n========== STAGE 5: differential test ==========")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, (ct, pretty) in enumerate(CTS):
        act, genes, cond, cdon = load(ct)
        meta = meta_of(ct); donors = meta.index.values
        ctrl = act[cond == "control"]
        A, keepc = build_adj(ctrl, beta); lab = leiden_modules(A, resolution=resolution, seed=42)
        sizes = np.bincount(lab); mods = [m for m in np.unique(lab) if sizes[m] >= 10]
        actk = act[:, keepc]
        pb = np.vstack([actk[cdon == d].mean(0) for d in donors]); total = pb.mean(1)
        meta2 = meta.copy(); meta2["total_activity"] = total
        coefs, rawp, ctrlp = [], [], []
        for m in mods:
            ma = pb[:, lab == m].mean(1)
            coefs.append(_ols_with_numeric_predictor(ma, meta, "adnc_ordinal", covariates=COV).coef)
            rawp.append(_ols_with_numeric_predictor(ma, meta, "adnc_ordinal", covariates=COV).p_value)
            ctrlp.append(_ols_with_numeric_predictor(ma, meta2, "adnc_ordinal", covariates=COV + ("total_activity",)).p_value)
        coefs = np.array(coefs); rq = benjamini_hochberg(np.array(rawp)); cq = benjamini_hochberg(np.array(ctrlp))
        x = np.arange(len(mods))
        same = max((coefs < 0).mean(), (coefs > 0).mean())

        # (a) per-module coefficient (sign-coloured) — clustered same-sign => global, not specific
        ax = axes[row, 0]
        ax.bar(x, coefs, color=["#c0392b" if c < 0 else "#27ae60" for c in coefs])
        ax.axhline(0, color="k", lw=.8)
        ax.set_title(f"{pretty}: per-module ADNC coefficient\n{same:.0%} share the same sign", fontsize=9)
        ax.set_xlabel("module"); ax.set_ylabel("ADNC coefficient")

        # (b) raw vs global-controlled significance
        ax = axes[row, 1]
        ax.scatter(x, -np.log10(rq), color="#e67e22", label="raw")
        ax.scatter(x, -np.log10(cq), color="#16a085", marker="s", label="global-controlled")
        ax.axhline(-np.log10(0.05), color="k", ls="--", lw=1); ax.text(0, -np.log10(0.05) + .03, "q=0.05", fontsize=7)
        ax.set_xlabel("module"); ax.set_ylabel("-log10(q)"); ax.legend(fontsize=8)
        ax.set_title(f"{pretty}: significance, raw vs global-controlled", fontsize=9)
        print(f"{pretty}: {same:.0%} same-sign; median coef={np.median(coefs):.2g}; "
              f"q<0.05 raw={int((rq < 0.05).sum())}, controlled={int((cq < 0.05).sum())}")
    fig.tight_layout(); fig.savefig(f"{FIG}/stage5_differential.png", dpi=180); plt.close(fig)
    print(f"-> {FIG}/stage5_differential.png")


if __name__ == "__main__":
    stage1()
    stage2()
    stage3()
    stage4()
    stage5()
