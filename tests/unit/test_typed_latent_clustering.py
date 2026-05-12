"""Unit tests for nga.arch.typed_latent_clustering.

Validates the Riemannian k-means clustering routine constrained to type
orbits in the Poincare ball. The clustering layer is the state-induction
half of the architecture: random latent labels migrate iteratively into
stable clusters, while type IDs (the equivariant half) are taken as-is.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from nga.arch.hyperbolic_embedding import (
    embed_euclidean_to_poincare,
    exp_map_zero,
    poincare_distance,
    project,
)
from nga.arch.typed_latent_clustering import (
    ClusteringResult,
    cluster_typed_latents,
    riemannian_centroid,
)


# ---------------------------------------------------------------------------
# Test fixtures: synthetic typed mixture in the Poincare ball.
# ---------------------------------------------------------------------------


def _make_typed_mixture(
    *,
    n_types: int = 3,
    clusters_per_type: int = 4,
    samples_per_cluster: int = 25,
    dim: int = 4,
    centre_radius: float = 0.5,
    noise_scale: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic typed mixture and return (obs, types, gt_labels, centres).

    Each (type, cluster) gets its own random centre on the Poincare ball at
    Euclidean radius ``centre_radius``. Samples are drawn around each centre
    as exp_0(centre_tangent + noise) where the noise is small Gaussian in the
    tangent space at zero.
    """
    rng = np.random.default_rng(seed)
    centres = np.zeros((n_types, clusters_per_type, dim), dtype=float)
    for q in range(n_types):
        for k in range(clusters_per_type):
            v = rng.normal(size=dim)
            v = v / np.linalg.norm(v) * centre_radius
            # exp_map_zero(v) keeps us inside the ball.
            centres[q, k] = exp_map_zero(v)

    obs_list: list[np.ndarray] = []
    types_list: list[int] = []
    gt_list: list[int] = []
    for q in range(n_types):
        for k in range(clusters_per_type):
            for _ in range(samples_per_cluster):
                # Tangent at zero of the centre, plus small noise:
                centre = centres[q, k]
                # Approximate "tangent at centre" by a small Euclidean perturb
                # then re-project. This is enough to test purity at this scale.
                noise = rng.normal(scale=noise_scale, size=dim)
                pt = project(centre + noise)
                obs_list.append(pt)
                types_list.append(q)
                gt_list.append(k)

    obs = np.array(obs_list, dtype=float)
    types = np.array(types_list, dtype=int)
    gt = np.array(gt_list, dtype=int)
    return obs, types, gt, centres


def _purity_within_type(
    pred: np.ndarray, gt: np.ndarray, types: np.ndarray
) -> float:
    """Cluster purity computed independently per type, then averaged.

    For each type-orbit, find the best label permutation by greedy majority
    voting and compute the fraction of correctly assigned points. Return
    the mean over types.
    """
    purities: list[float] = []
    for q in np.unique(types):
        mask = types == q
        p = pred[mask]
        g = gt[mask]
        # For each predicted label, look at the dominant ground-truth label.
        correct = 0
        for lab in np.unique(p):
            sub = g[p == lab]
            if sub.size == 0:
                continue
            most_common = Counter(sub.tolist()).most_common(1)[0][1]
            correct += most_common
        purities.append(correct / max(p.size, 1))
    return float(np.mean(purities)) if purities else 0.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cluster_recovers_synthetic_mixture() -> None:
    obs, types, gt, _ = _make_typed_mixture(
        n_types=3,
        clusters_per_type=4,
        samples_per_cluster=25,
        dim=4,
        centre_radius=0.5,
        noise_scale=0.04,
        seed=0,
    )
    res = cluster_typed_latents(
        observations_in_poincare=obs,
        types=types,
        L_per_type=4,
        max_iter=50,
        seed=42,
    )
    assert isinstance(res, ClusteringResult)
    purity = _purity_within_type(res.labels, gt, types)
    assert purity >= 0.90, f"expected purity >= 0.90, got {purity:.3f}"


def test_cluster_constrained_to_type_orbit() -> None:
    obs, types, _, _ = _make_typed_mixture(
        n_types=3, clusters_per_type=3, samples_per_cluster=10, seed=1
    )
    res = cluster_typed_latents(
        observations_in_poincare=obs,
        types=types,
        L_per_type=3,
        max_iter=20,
        seed=7,
    )

    for t in range(obs.shape[0]):
        q = int(types[t])
        lab = int(res.labels[t])
        protos_q = res.prototypes[q]
        assert protos_q.shape[0] > 0, f"type {q} had no prototypes"
        assert 0 <= lab < protos_q.shape[0], (
            f"observation {t} of type {q} got latent {lab} outside its orbit"
        )
        assigned_proto = protos_q[lab]
        # Sanity: the assigned prototype is the *nearest* one in the type
        # orbit, hence necessarily inside the type's prototype matrix.
        own_dist = float(poincare_distance(obs[t], assigned_proto))
        if protos_q.shape[0] > 1:
            all_d = poincare_distance(obs[t][None, :], protos_q)
            # poincare_distance with (1, d) vs (m, d) returns (m,)
            best = float(np.min(np.atleast_1d(all_d)))
            assert own_dist <= best + 1e-9


def test_cluster_deterministic_under_seed() -> None:
    obs, types, _, _ = _make_typed_mixture(
        n_types=2, clusters_per_type=3, samples_per_cluster=15, seed=2
    )
    r1 = cluster_typed_latents(
        observations_in_poincare=obs,
        types=types,
        L_per_type=3,
        max_iter=30,
        seed=11,
    )
    r2 = cluster_typed_latents(
        observations_in_poincare=obs,
        types=types,
        L_per_type=3,
        max_iter=30,
        seed=11,
    )
    np.testing.assert_array_equal(r1.labels, r2.labels)
    assert set(r1.prototypes.keys()) == set(r2.prototypes.keys())
    for q in r1.prototypes:
        np.testing.assert_allclose(r1.prototypes[q], r2.prototypes[q], atol=0.0)


def test_cluster_handles_empty_type() -> None:
    # Generate observations only for types 0 and 1, but tell the clustering
    # routine that L_per_type covers type 2 as well.
    obs, types, _, _ = _make_typed_mixture(
        n_types=2, clusters_per_type=2, samples_per_cluster=8, seed=3
    )
    # Types are still 0 and 1 in `types`. The dict here covers an extra
    # never-seen type 2 -- the routine should not crash.
    res = cluster_typed_latents(
        observations_in_poincare=obs,
        types=types,
        L_per_type={0: 2, 1: 2, 2: 4},
        max_iter=20,
        seed=5,
    )
    # Type 2 must be present with a 0-row prototype matrix.
    assert 2 in res.prototypes
    assert res.prototypes[2].shape == (0, obs.shape[1])
    # Types 0 and 1 should still have valid prototypes.
    assert res.prototypes[0].shape == (2, obs.shape[1])
    assert res.prototypes[1].shape == (2, obs.shape[1])


def test_cluster_converges() -> None:
    obs, types, _, _ = _make_typed_mixture(
        n_types=2,
        clusters_per_type=3,
        samples_per_cluster=20,
        noise_scale=0.03,
        seed=4,
    )
    res = cluster_typed_latents(
        observations_in_poincare=obs,
        types=types,
        L_per_type=3,
        max_iter=50,
        seed=13,
        tol=1e-4,
    )
    assert res.converged is True
    assert res.n_iter < 50


def test_riemannian_centroid_recovers_origin() -> None:
    # Four symmetric points around origin in 2D: (+/-r, 0), (0, +/-r).
    r = 0.3
    pts = np.array(
        [[+r, 0.0], [-r, 0.0], [0.0, +r], [0.0, -r]],
        dtype=float,
    )
    c = riemannian_centroid(pts, max_inner_iter=20, lr=0.5)
    # The centroid should be at the origin within tight tolerance.
    origin = np.zeros(2, dtype=float)
    d = float(poincare_distance(c, origin))
    assert d < 1e-3, f"centroid drift {d:.6f} > 1e-3"


def test_cluster_per_type_l_dict() -> None:
    obs, types, _, _ = _make_typed_mixture(
        n_types=3, clusters_per_type=5, samples_per_cluster=8, seed=6
    )
    L = {0: 3, 1: 5, 2: 2}
    res = cluster_typed_latents(
        observations_in_poincare=obs,
        types=types,
        L_per_type=L,
        max_iter=20,
        seed=17,
    )
    assert res.prototypes[0].shape[0] == 3
    assert res.prototypes[1].shape[0] == 5
    assert res.prototypes[2].shape[0] == 2


def test_cluster_distortion_decreases() -> None:
    """Run iteration-by-iteration and assert distortion is non-increasing.

    We use the existence of the public API by calling ``cluster_typed_latents``
    with successively larger ``max_iter`` budgets and reading the reported
    ``final_distortion``. Lloyd's algorithm with a Frechet-mean update on a
    Riemannian manifold is monotone, so the sequence should be
    non-increasing.
    """
    obs, types, _, _ = _make_typed_mixture(
        n_types=2,
        clusters_per_type=3,
        samples_per_cluster=15,
        noise_scale=0.04,
        seed=8,
    )
    distortions: list[float] = []
    for budget in [1, 2, 3, 5, 8, 13]:
        res = cluster_typed_latents(
            observations_in_poincare=obs,
            types=types,
            L_per_type=3,
            max_iter=budget,
            seed=23,
            # Set a tighter tol so the algorithm doesn't early-stop and let
            # us see the full monotone trace at small budgets.
            tol=0.0,
        )
        distortions.append(res.final_distortion)
    # Allow a tiny numerical slack for the inner-iteration approximation.
    for a, b in zip(distortions, distortions[1:]):
        assert b <= a + 1e-6, (
            f"distortion not monotone: trace = {distortions}"
        )
