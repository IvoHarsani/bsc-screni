"""Tests for module_preservation.py — the WGCNA general-network Zsummary engine.

The headline test plants modules with known structure: a reference network with k dense
blocks, and a test network identical except ONE block is scrambled. The engine must flag
exactly that block with a low Zsummary while the preserved blocks score high. This is the
correctness check that stands in for a round-trip against R's modulePreservation.
"""

import numpy as np
import pytest

from screni.data.module_preservation import (
    coregulation_adjacency,
    module_preservation,
    preservation_stats_for_set,
    _offdiag,
    _safe_corr,
)


def make_block_adjacency(n_per=40, k=5, within=0.85, between=0.04, noise=0.02, seed=0):
    """k equal-sized dense blocks; returns (A, labels). Symmetric, [0,1], unit diagonal."""
    rng = np.random.default_rng(seed)
    n = n_per * k
    labels = np.repeat(np.arange(k), n_per)
    A = np.full((n, n), between)
    for m in range(k):
        idx = np.where(labels == m)[0]
        A[np.ix_(idx, idx)] = within
    A = A + rng.normal(0, noise, (n, n))
    A = np.clip((A + A.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(A, 1.0)
    return A, labels


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def test_offdiag_extracts_upper_triangle():
    M = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 4.0, 1.0]])
    assert sorted(_offdiag(M).tolist()) == [2.0, 3.0, 4.0]


def test_safe_corr_undefined_cases():
    assert np.isnan(_safe_corr(np.array([1.0, 2.0]), np.array([1.0, 2.0])))  # n<3
    assert np.isnan(_safe_corr(np.ones(5), np.arange(5.0)))  # constant input
    assert _safe_corr(np.arange(5.0), np.arange(5.0)) == pytest.approx(1.0)


def test_preservation_stats_identical_block_is_dense_and_correlated():
    A, labels = make_block_adjacency(n_per=30, k=3, noise=0.0, seed=1)
    idx = np.where(labels == 0)[0]
    mean_adj, cor_kim, cor_adj = preservation_stats_for_set(A, A, idx)
    assert mean_adj > 0.8                 # dense block
    assert cor_kim == pytest.approx(1.0)  # identical ref/test -> perfect correlation
    assert cor_adj == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Headline: planted disrupted module
# ---------------------------------------------------------------------------


def test_scrambled_module_flagged_others_preserved():
    A_ref, labels = make_block_adjacency(n_per=40, k=5, seed=0)
    rng = np.random.default_rng(123)

    # test network = reference, but module 0's internal wiring is destroyed (made sparse/random)
    A_test = A_ref.copy()
    idx0 = np.where(labels == 0)[0]
    block = rng.uniform(0.0, 0.08, size=(len(idx0), len(idx0)))
    block = (block + block.T) / 2.0
    A_test[np.ix_(idx0, idx0)] = block
    np.fill_diagonal(A_test, 1.0)

    res = module_preservation(A_ref, A_test, labels, n_permutations=100, seed=7)
    z = dict(zip(res.module_ids.tolist(), res.Zsummary.tolist()))

    # module 0 is the disrupted one: lowest Zsummary, below the "not preserved" threshold
    assert min(z, key=z.get) == 0
    assert z[0] < 2.0
    # the four untouched modules are strongly preserved
    for m in (1, 2, 3, 4):
        assert z[m] > 2.0
        assert z[m] > z[0] + 5.0

    # medianRank: the disrupted module ranks worst (largest median rank)
    mr = dict(zip(res.module_ids.tolist(), res.medianRank.tolist()))
    assert max(mr, key=mr.get) == 0


def test_identical_networks_all_preserved():
    A, labels = make_block_adjacency(n_per=30, k=4, seed=2)
    res = module_preservation(A, A, labels, n_permutations=80, seed=3)
    # every module is perfectly preserved -> all clearly above the disruption threshold
    assert np.all(res.Zsummary > 2.0)


def test_unrelated_test_network_nothing_preserved():
    A_ref, labels = make_block_adjacency(n_per=30, k=4, seed=4)
    # a test network with the same marginal structure but no relation to the reference blocks
    A_test, _ = make_block_adjacency(n_per=30, k=4, seed=999)
    rng = np.random.default_rng(5)
    perm = rng.permutation(A_test.shape[0])  # shuffle so blocks don't line up with labels
    A_test = A_test[np.ix_(perm, perm)]
    res = module_preservation(A_ref, A_test, labels, n_permutations=100, seed=6)
    # no module should look strongly preserved
    assert np.all(res.Zsummary < 10.0)


# ---------------------------------------------------------------------------
# Adjacency construction
# ---------------------------------------------------------------------------


def test_coregulation_adjacency_shape_symmetry_and_drops_dead_columns():
    rng = np.random.default_rng(0)
    # 10 regulators x 8 genes; genes 6 and 7 receive no weight (dead columns)
    W = np.abs(rng.normal(size=(10, 8)))
    W[:, 6] = 0.0
    W[:, 7] = 0.0
    A, keep = coregulation_adjacency(W, beta=6)
    assert keep.tolist() == [0, 1, 2, 3, 4, 5]      # dead columns dropped
    assert A.shape == (6, 6)
    assert np.allclose(A, A.T)                       # symmetric
    assert np.allclose(np.diag(A), 1.0)              # unit diagonal
    assert A.min() >= 0.0 and A.max() <= 1.0


def test_coregulation_row_centering_changes_similarity():
    rng = np.random.default_rng(1)
    base = np.abs(rng.normal(size=(20, 30)))
    hub = np.abs(rng.normal(2.0, 0.1, size=(1, 30)))  # one regulator weights everyone strongly
    W = np.vstack([hub, base])
    A_raw, _ = coregulation_adjacency(W, beta=6, row_center=False)
    A_rc, _ = coregulation_adjacency(W, beta=6, row_center=True)
    # the hub baseline inflates raw similarity; centering removes it
    assert _offdiag(A_raw).mean() > _offdiag(A_rc).mean()
