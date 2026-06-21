"""Module-preservation statistics for ScReNI GRNs (SQ3 — module-level differential analysis).

Reimplements the **general-network path** of WGCNA's ``modulePreservation`` function
(method: Langfelder, Luo, Oldham & Horvath, "Is My Network Module Preserved and
Reproducible?", *PLoS Comput Biol* 2011, 7(1):e1001057 — *not* the 2008 WGCNA package
paper). For a bare adjacency input (no underlying per-sample expression), stock WGCNA
sets ``restrictSummaryForGeneralNetworks = TRUE`` (the default) and reduces ``Zsummary``
to exactly three adjacency-based statistics. We reproduce that path.

Provenance of the three statistics (read off WGCNA ``R/modulePreservation.R``):

    density:       meanAdj   (preservation matrix col 4)
    connectivity:  cor.kIM   (col 7)  and  cor.adj  (col 10)
    Zdensity      = Z(meanAdj)                        # single density stat
    Zconnectivity = median(Z(cor.kIM), Z(cor.adj))    # 2 stats -> median == mean
    Zsummary      = mean(Zdensity, Zconnectivity)      #  == 1/2 Z_meanAdj + 1/4 Z_corKIM + 1/4 Z_corAdj
    thresholds:   Zsummary < 2 = not preserved (disrupted); 2-10 = weak/moderate; > 10 = strong

The preservation statistics are **module-definition agnostic** — they take a gene->module
label vector as input and do not care how the modules were produced. Module definition
(``leiden_modules``) and adjacency construction (``coregulation_adjacency``) are provided
here for the ScReNI use-case but are separable concerns.

SQ3 framing: build a control (reference) and an AD (test) co-regulation adjacency per cell
type, define modules on the control adjacency, then test whether each control module is
preserved in AD. A low ``Zsummary`` module = regulatory wiring disrupted in AD.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adjacency construction (ScReNI-specific; separable from the stats)
# ---------------------------------------------------------------------------


def coregulation_adjacency(
    weights: np.ndarray,
    beta: int = 6,
    row_center: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a symmetric gene-gene co-regulation adjacency from a directed weight matrix.

    A wScReNI weight matrix is ``regulator(row) x target(col)``: ``weights[i, j]`` is the
    inferred importance of regulator ``i`` for target ``j``. Two genes are "co-regulated"
    when they receive weight from the **same regulators with similar strength** — i.e. their
    *incoming-weight profiles* (their columns) are correlated.

    Parameters
    ----------
    weights
        ``(n_regulators, n_genes)`` directed weight matrix (rows = regulators, cols = targets).
    beta
        Soft-thresholding power (WGCNA-style). ``A = |corr|**beta``. Default 6.
    row_center
        If True, subtract each regulator's mean weight across targets before correlating.
        This removes the "hub regulator weights nearly everyone" baseline that otherwise
        makes every gene's incoming profile look alike (one dominant global axis). Strongly
        recommended — without it the similarity is near-degenerate.
    eps
        Genes whose incoming profile has variance <= eps are dropped (constant/zero columns
        carry no co-regulation signal).

    Returns
    -------
    ``(A, keep_idx)``
        ``A`` is the ``(m, m)`` symmetric adjacency in [0, 1] with unit diagonal, over the
        ``m`` retained genes; ``keep_idx`` are the original column indices kept (use to map
        back to gene names). Build the reference and test adjacencies over the **same**
        ``keep_idx`` (intersection) before computing preservation.
    """
    W = np.asarray(weights, dtype=float)
    # Drop dead (constant/zero) incoming columns on the RAW weights, BEFORE row-centering.
    # Centering an all-zero column turns it into -row_means, which is non-constant and would
    # (a) survive a post-centering variance filter and (b) make all such genes look spuriously
    # co-regulated. So the filter must run first.
    keep = np.where(W.var(axis=0) > eps)[0]
    Wk = W[:, keep]
    if row_center:
        Wk = Wk - Wk.mean(axis=1, keepdims=True)
    S = np.corrcoef(Wk.T)
    S = np.nan_to_num(S, nan=0.0)
    A = np.abs(S) ** beta
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 1.0)
    return A, keep


def leiden_modules(
    A: np.ndarray,
    resolution: float = 2.0,
    seed: int = 42,
    edge_threshold: float = 0.02,
) -> np.ndarray:
    """Define modules by Leiden community detection on a co-regulation adjacency.

    WGCNA's own pipeline uses hierarchical clustering + dynamic tree cut; on a dense
    co-regulation adjacency that degenerates into one giant cluster, so we use Leiden
    (robust to dense weighted graphs, already a project dependency via scanpy). The
    preservation statistics are agnostic to this choice.

    Returns a length-``n`` integer label vector (0-based; community ids).
    """
    import igraph as ig
    import leidenalg as la

    n = A.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    w = A[iu, ju]
    keep = w > edge_threshold
    g = ig.Graph(n=n, edges=list(zip(iu[keep].tolist(), ju[keep].tolist())))
    g.es["weight"] = w[keep].tolist()
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.asarray(part.membership, dtype=int)


# ---------------------------------------------------------------------------
# The three preservation statistics (the validated core)
# ---------------------------------------------------------------------------


def _offdiag(M: np.ndarray) -> np.ndarray:
    """Upper-triangular off-diagonal entries (R's ``as.dist``)."""
    return M[np.triu_indices(M.shape[0], k=1)]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, NaN if undefined (n<3 or a constant input)."""
    if len(x) < 3:
        return np.nan
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def preservation_stats_for_set(
    A_ref: np.ndarray, A_test: np.ndarray, idx: np.ndarray
) -> tuple[float, float, float]:
    """The three WGCNA general-network statistics for one node set.

    Returns ``(meanAdj, cor_kIM, cor_adj)`` for the node set ``idx``:

    * ``meanAdj``  = mean off-diagonal adjacency among the nodes in the **test** network
      (is the module still densely connected in AD?).
    * ``cor_kIM``  = Pearson corr between intramodular connectivity ``kIM`` in ref vs test,
      across the nodes (are the same genes the hubs in both?). ``kIM_i = sum_{j in set} A_ij - A_ii``.
    * ``cor_adj``  = Pearson corr between the ref and test adjacency entries over node pairs
      (are the individual edge weights preserved?).
    """
    sub_ref = A_ref[np.ix_(idx, idx)]
    sub_test = A_test[np.ix_(idx, idx)]

    mean_adj = float(np.mean(_offdiag(sub_test)))

    kim_ref = sub_ref.sum(axis=1) - np.diag(sub_ref)
    kim_test = sub_test.sum(axis=1) - np.diag(sub_test)
    cor_kim = _safe_corr(kim_ref, kim_test)

    cor_adj = _safe_corr(_offdiag(sub_ref), _offdiag(sub_test))
    return mean_adj, cor_kim, cor_adj


@dataclass
class ModulePreservationResult:
    """Per-module preservation results (one row per module).

    All arrays are aligned and indexed by position; ``module_ids[k]`` is the label of the
    k-th module. ``Zsummary`` is the headline; ``< 2`` = disrupted, ``> 10`` = strongly
    preserved.
    """

    module_ids: np.ndarray
    module_sizes: np.ndarray
    meanAdj: np.ndarray
    cor_kIM: np.ndarray
    cor_adj: np.ndarray
    Z_meanAdj: np.ndarray
    Z_corKIM: np.ndarray
    Z_corAdj: np.ndarray
    Zdensity: np.ndarray
    Zconnectivity: np.ndarray
    Zsummary: np.ndarray
    medianRank: np.ndarray
    n_permutations: int = 0

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "module": self.module_ids,
                "size": self.module_sizes,
                "meanAdj": self.meanAdj,
                "cor_kIM": self.cor_kIM,
                "cor_adj": self.cor_adj,
                "Z_meanAdj": self.Z_meanAdj,
                "Z_corKIM": self.Z_corKIM,
                "Z_corAdj": self.Z_corAdj,
                "Zdensity": self.Zdensity,
                "Zconnectivity": self.Zconnectivity,
                "Zsummary": self.Zsummary,
                "medianRank": self.medianRank,
            }
        ).sort_values("Zsummary").reset_index(drop=True)


def module_preservation(
    A_ref: np.ndarray,
    A_test: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 200,
    seed: int = 12345,
    min_size: int = 3,
    exclude_labels: Optional[set] = None,
) -> ModulePreservationResult:
    """Compute WGCNA general-network module-preservation Zsummary for every module.

    Modules are defined on the **reference** (control) network; preservation is assessed in
    the **test** (AD) network. ``A_ref`` and ``A_test`` must be symmetric adjacencies over
    the same node set and ordering, and ``labels`` a length-n module assignment for those
    nodes (e.g. from :func:`leiden_modules` on ``A_ref``).

    Null model: for each module of size ``s``, draw ``n_permutations`` random node-subsets of
    size ``s`` and recompute the three statistics → null mean/sd per statistic. This is the
    direct per-module equivalent of WGCNA permuting the gene->module labels (under the null of
    no preservation, membership is exchangeable). We do not use WGCNA's module-size
    regression-interpolation shortcut; per-module permutation is exact at the cost of more draws.

    ``Z = (observed - null_mean) / null_sd``;
    ``Zdensity = Z_meanAdj``;
    ``Zconnectivity = median(Z_corKIM, Z_corAdj)``;
    ``Zsummary = mean(Zdensity, Zconnectivity)``.
    """
    A_ref = np.asarray(A_ref, dtype=float)
    A_test = np.asarray(A_test, dtype=float)
    labels = np.asarray(labels)
    n = A_ref.shape[0]
    if A_ref.shape != A_test.shape or A_ref.shape[0] != A_ref.shape[1]:
        raise ValueError("A_ref and A_test must be square and the same shape")
    if labels.shape[0] != n:
        raise ValueError("labels must have one entry per node")

    exclude = set(exclude_labels or set())
    uniq = [m for m in np.unique(labels) if m not in exclude]

    rng = np.random.default_rng(seed)

    ids, sizes = [], []
    obs_ma, obs_ck, obs_ca = [], [], []
    z_ma, z_ck, z_ca = [], [], []
    zden, zcon, zsum = [], [], []

    for m in uniq:
        idx = np.where(labels == m)[0]
        s = len(idx)
        if s < min_size:
            continue
        ma, ck, ca = preservation_stats_for_set(A_ref, A_test, idx)

        null = np.full((n_permutations, 3), np.nan)
        for p in range(n_permutations):
            ridx = rng.choice(n, size=s, replace=False)
            null[p] = preservation_stats_for_set(A_ref, A_test, ridx)

        mu = np.nanmean(null, axis=0)
        sd = np.nanstd(null, axis=0, ddof=0)
        obs = np.array([ma, ck, ca])
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (obs - mu) / sd
        # If a statistic has zero null variance, Z is undefined -> treat as 0 contribution.
        z = np.where(sd == 0, 0.0, z)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        z_density = z[0]
        z_connectivity = np.median([z[1], z[2]])
        z_summary = np.mean([z_density, z_connectivity])

        ids.append(m)
        sizes.append(s)
        obs_ma.append(ma); obs_ck.append(ck); obs_ca.append(ca)
        z_ma.append(z[0]); z_ck.append(z[1]); z_ca.append(z[2])
        zden.append(z_density); zcon.append(z_connectivity); zsum.append(z_summary)

    ids = np.array(ids)
    obs_ma = np.array(obs_ma); obs_ck = np.array(obs_ck); obs_ca = np.array(obs_ca)

    # medianRank: rank modules by each observed stat (higher = better preserved = rank 1),
    # then take the median rank across the three stats. Size-robust, permutation-free.
    def _rank_desc(x):
        order = np.argsort(-np.nan_to_num(x, nan=-np.inf))
        r = np.empty(len(x))
        r[order] = np.arange(1, len(x) + 1)
        return r

    if len(ids):
        rank_mat = np.vstack([_rank_desc(obs_ma), _rank_desc(obs_ck), _rank_desc(obs_ca)])
        median_rank = np.median(rank_mat, axis=0)
    else:
        median_rank = np.array([])

    logger.info(
        "module_preservation: %d modules (size>=%d), %d permutations",
        len(ids), min_size, n_permutations,
    )
    return ModulePreservationResult(
        module_ids=ids,
        module_sizes=np.array(sizes),
        meanAdj=obs_ma,
        cor_kIM=obs_ck,
        cor_adj=obs_ca,
        Z_meanAdj=np.array(z_ma),
        Z_corKIM=np.array(z_ck),
        Z_corAdj=np.array(z_ca),
        Zdensity=np.array(zden),
        Zconnectivity=np.array(zcon),
        Zsummary=np.array(zsum),
        medianRank=median_rank,
        n_permutations=n_permutations,
    )
