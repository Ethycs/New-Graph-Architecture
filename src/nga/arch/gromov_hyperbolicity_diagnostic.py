"""Gromov delta diagnostic: estimate how tree-like a (finite) hyperbolic
embedding is. Smaller delta = more tree-like = better fit for the underlying
hierarchy.

Algorithm (4-point condition):
  For every quadruple (a, b, c, d), compute the three pairs of sums:
    s1 = d(a,b) + d(c,d)
    s2 = d(a,c) + d(b,d)
    s3 = d(a,d) + d(b,c)
  Sort to get s1 >= s2 >= s3. Then delta(quad) = (s1 - s2) / 2.

  delta = max over all quadruples of delta(quad).

In practice we estimate via a sample: pick K random quadruples, compute the
delta of each, return the max and the histogram. K=200 is fine for V <= 50.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Lazy import shim - same pattern as graph_extrusion.py so this module is
# importable even before hyperbolic_embedding.py lands.
def _get_poincare_distance():
    """Return the poincare_distance callable from hyperbolic_embedding."""
    try:
        from nga.arch.hyperbolic_embedding import poincare_distance  # type: ignore[import]
        return poincare_distance
    except ImportError as exc:
        raise ImportError(
            "nga.arch.hyperbolic_embedding is not yet available. "
            "Ensure it is present before calling estimate_gromov_delta."
        ) from exc


@dataclass
class DeltaEstimate:
    """Result of a sample-based Gromov-delta estimation.

    Attributes
    ----------
    delta_max:
        Maximum delta observed across all sampled quadruples.  This is the
        primary summary statistic: the smaller it is, the more tree-like the
        embedding.
    delta_mean:
        Mean delta across sampled quadruples.
    n_quadruples:
        Number of quadruples that were actually evaluated (may be less than the
        requested n_quadruples when comb(N, 4) < n_quadruples).
    samples:
        Array of shape (n_quadruples,) containing the per-quadruple delta
        values.
    """

    delta_max: float
    delta_mean: float
    n_quadruples: int
    samples: np.ndarray  # shape (n_quadruples,) of delta values


def _build_distance_matrix(points: np.ndarray) -> np.ndarray:
    """Compute the full pairwise Poincare distance matrix.

    Parameters
    ----------
    points:
        Array of shape (N, d) of points in the Poincare ball.

    Returns
    -------
    np.ndarray
        Symmetric array of shape (N, N) with D[i, j] = poincare_distance(i, j).
    """
    poincare_distance = _get_poincare_distance()
    N = points.shape[0]
    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            d = float(poincare_distance(points[i], points[j]))
            D[i, j] = d
            D[j, i] = d
    return D


def estimate_gromov_delta(
    points: np.ndarray,             # shape (N, d), assumed in Poincare ball
    *,
    n_quadruples: int = 200,
    seed: int = 0,
) -> DeltaEstimate:
    """Sample-based estimate of Gromov delta.

    If N < 4, returns DeltaEstimate(delta_max=0.0, delta_mean=0.0, n=0,
    samples=[]).  Otherwise samples min(n_quadruples, comb(N, 4)) distinct
    quadruples (with a deterministic RNG) and computes delta per quadruple.

    The 4-point condition formula used here is the standard one from
    Mathematics.md Section "Gromov Hyperbolicity":
      For each quadruple (a, b, c, d):
        s1 = D(a,b) + D(c,d)
        s2 = D(a,c) + D(b,d)
        s3 = D(a,d) + D(b,c)
      Sort descending: top >= mid >= bot.
      delta(quadruple) = (top - mid) / 2.
    Global delta = max over all sampled quadruples.

    Parameters
    ----------
    points:
        Array of shape (N, d) of points in the Poincare ball.
    n_quadruples:
        Target number of quadruples to sample (or all comb(N,4) if smaller).
    seed:
        Integer seed for the NumPy default_rng, ensuring deterministic output.

    Returns
    -------
    DeltaEstimate
        Summary statistics over the sampled quadruples.
    """
    N = points.shape[0]

    # Degenerate case: too few points to form a quadruple.
    if N < 4:
        return DeltaEstimate(
            delta_max=0.0,
            delta_mean=0.0,
            n_quadruples=0,
            samples=np.array([], dtype=float),
        )

    # Cap sample count at total number of distinct quadruples.
    total_quads = math.comb(N, 4)
    actual_n = min(n_quadruples, total_quads)

    # Pre-compute full distance matrix to avoid O(K * N^2) redundancy.
    D = _build_distance_matrix(points)

    rng = np.random.default_rng(seed)

    if actual_n == total_quads:
        # Enumerate all quadruples deterministically.
        indices_list = [
            (a, b, c, d)
            for a in range(N)
            for b in range(a + 1, N)
            for c in range(b + 1, N)
            for d in range(c + 1, N)
        ]
    else:
        # Sample without replacement using reservoir / permutation approach.
        # We generate random quadruples until we have actual_n distinct ones.
        seen: set[tuple[int, int, int, int]] = set()
        indices_list = []
        while len(indices_list) < actual_n:
            quad = tuple(sorted(rng.choice(N, size=4, replace=False).tolist()))
            quad = (quad[0], quad[1], quad[2], quad[3])
            if quad not in seen:
                seen.add(quad)
                indices_list.append(quad)

    # Compute delta for each quadruple.
    deltas = np.empty(actual_n, dtype=float)
    for k, (a, b, c, d) in enumerate(indices_list):
        s1 = D[a, b] + D[c, d]
        s2 = D[a, c] + D[b, d]
        s3 = D[a, d] + D[b, c]
        # Sort descending to find the two largest.
        top, mid, _bot = sorted([s1, s2, s3], reverse=True)
        deltas[k] = (top - mid) / 2.0

    return DeltaEstimate(
        delta_max=float(deltas.max()),
        delta_mean=float(deltas.mean()),
        n_quadruples=actual_n,
        samples=deltas,
    )


def is_tree_like(estimate: DeltaEstimate, *, threshold: float = 0.5) -> bool:
    """Return True iff delta_max < threshold, indicating approximately tree-like structure.

    threshold=0.5 is a reasonable Phase 3 default (in units where the
    "natural scale" of the embedding is roughly 1.0). Note: this is a
    heuristic - the right threshold scales with the embedding's diameter.
    Embeddings with large spread may require proportionally larger thresholds;
    embeddings concentrated near the origin may warrant smaller ones.

    Parameters
    ----------
    estimate:
        A DeltaEstimate produced by estimate_gromov_delta.
    threshold:
        The maximum acceptable delta_max for declaring the embedding tree-like.

    Returns
    -------
    bool
        True when delta_max < threshold.
    """
    return estimate.delta_max < threshold
