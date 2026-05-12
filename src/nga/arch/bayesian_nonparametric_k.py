"""Bayesian-nonparametric K selection.

Phase 20 / graph-extraction Wave A atom. The architecture's clustering
pipeline today (``typed_latent_clustering.RiemannianKMeans``) takes ``K``
as a hyperparameter; Phase 19B's diagnostic result used ``K = 41``
because we knew there were 41 diseases. The graph-extraction proposal
needs ``K`` to come from the data: cluster a corpus, choose the K that
best balances fit against complexity, return the chosen K and the
corresponding labels.

What this atom does
===================

Given an embedding matrix ``X`` of shape ``(N, d)`` and a search range
``K_range = [K_min, ..., K_max]``, ``estimate_k`` evaluates each candidate
K by fitting a clustering and scoring it with one of three criteria:

* ``"holdout_nll"`` -- partition the corpus into train / holdout, fit
  Gaussian-mixture centroids on the train rows, compute the negative
  log-likelihood of the holdout rows under each (mean, identity-covariance,
  uniform weight) component. Lower NLL = better predictive fit on unseen
  data; this is the principled "predict unseen samples" criterion the
  proposal calls for.
* ``"bic"`` -- the Bayesian Information Criterion
  ``BIC = -2 log L + p log N``, with ``p = K * d`` parameters (means only).
  Penalises complexity; classical model-selection workhorse.
* ``"elbow"`` -- within-cluster sum of squared distances; pick the K at
  the "elbow" of the distortion curve. Fast, no probabilistic model, but
  the elbow is heuristic.

Why all three? Held-out NLL is what we want philosophically; BIC is what
we want when held-out data is scarce; the elbow is the cheap sanity-check
that runs in milliseconds. The runner can compare the three and only
trust ``K_star`` when they roughly agree.

Output
======

``estimate_k`` returns a :class:`KEstimationResult` containing
``K_star``, the per-K scores (a 1-D array aligned with ``K_range``), and
the chosen-K labels. The result is deterministic under fixed seed.

Stick-breaking DP-mixture (the heavyweight variant called out in the
proposal) is intentionally not implemented in this atom: the held-out
NLL criterion captures the same intent at a fraction of the engineering
cost. If experiments show held-out NLL is unstable, the proper DP-mixture
goes in as a v2 of this atom.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["KEstimationResult", "estimate_k"]


@dataclass(frozen=True)
class KEstimationResult:
    """The output of :func:`estimate_k`.

    Attributes
    ----------
    K_star:
        The chosen K.
    K_range:
        The candidate K values evaluated, in order.
    scores:
        Per-K score under the chosen criterion, aligned with ``K_range``.
        Lower-is-better for ``holdout_nll`` and ``bic``; for ``elbow``
        the score is the within-cluster distortion (lower is better but
        the chosen K is selected by the curvature of the curve, not by
        the absolute minimum).
    labels:
        Cluster assignment per row at ``K = K_star``, shape ``(N,)``.
    centroids:
        Centroid matrix at ``K = K_star``, shape ``(K_star, d)``.
    criterion:
        Name of the criterion used.
    """

    K_star: int
    K_range: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray
    criterion: str


def _kmeanspp_init(
    X: np.ndarray,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """k-means++ initialisation: pick well-separated centroids.

    Standard k-means++ (Arthur & Vassilvitskii 2007). Numerically careful:
    distances are clipped at zero to suppress floating-point negatives, and
    the probability distribution is renormalised even when all distances
    have collapsed to zero (which can happen with duplicated rows).
    """
    n, d = X.shape
    if K > n:
        raise ValueError(f"K ({K}) cannot exceed n_rows ({n})")
    centroids = np.empty((K, d), dtype=X.dtype)
    first = rng.integers(n)
    centroids[0] = X[first]

    closest_sq = np.maximum(np.sum((X - centroids[0]) ** 2, axis=1), 0.0)
    for k in range(1, K):
        total = float(closest_sq.sum())
        if total <= 0.0:
            # All points coincide with the existing centroids; pick uniformly.
            probs = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            probs = closest_sq / total
        idx = int(rng.choice(n, p=probs))
        centroids[k] = X[idx]
        new_sq = np.maximum(np.sum((X - centroids[k]) ** 2, axis=1), 0.0)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centroids


def _lloyd_step(X: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One Lloyd iteration. Returns ``(new_centroids, labels)``."""
    # (N, K) squared distance via the identity ||x - c||^2 = ||x||^2 - 2<x, c> + ||c||^2.
    xx = np.sum(X * X, axis=1, keepdims=True)        # (N, 1)
    cc = np.sum(centroids * centroids, axis=1)       # (K,)
    cross = X @ centroids.T                           # (N, K)
    sq = xx - 2.0 * cross + cc[np.newaxis, :]
    sq = np.maximum(sq, 0.0)
    labels = np.argmin(sq, axis=1)

    K = centroids.shape[0]
    new_centroids = np.empty_like(centroids)
    for k in range(K):
        mask = labels == k
        if not np.any(mask):
            # Empty cluster: keep the current centroid (alternative: re-seed
            # from the farthest point; we leave it stable for determinism).
            new_centroids[k] = centroids[k]
        else:
            new_centroids[k] = X[mask].mean(axis=0)
    return new_centroids, labels


def _fit_kmeans(
    X: np.ndarray,
    K: int,
    rng: np.random.Generator,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Plain Euclidean k-means with k-means++ init and Lloyd iterations.

    For Riemannian / Poincaré clustering, prefer
    :class:`nga.arch.typed_latent_clustering.RiemannianKMeans`. This atom
    uses Euclidean k-means because Bayesian K-selection is upstream of
    the metric choice -- it operates on whatever coordinates the harvester
    yields, and Euclidean is the right default for arbitrary embeddings.
    """
    centroids = _kmeanspp_init(X, K, rng)
    labels = np.zeros(X.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        new_centroids, new_labels = _lloyd_step(X, centroids)
        shift = float(np.max(np.abs(new_centroids - centroids)))
        centroids = new_centroids
        labels = new_labels
        if shift < tol:
            break
    return centroids, labels


def _within_cluster_distortion(
    X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
) -> float:
    """Sum of squared distances from each point to its assigned centroid."""
    diffs = X - centroids[labels]
    return float(np.sum(diffs * diffs))


def _gaussian_nll(
    X: np.ndarray, centroids: np.ndarray, variance: float
) -> float:
    """Mean NLL of ``X`` under an equally-weighted isotropic Gaussian mixture
    with the supplied isotropic ``variance``.

    ``log p(x) = logsumexp_k(- ||x - c_k||^2 / 2sigma^2)
    - log K - (d/2) log(2 pi sigma^2)``.
    """
    _, d = X.shape
    K = centroids.shape[0]
    sq = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    log_kernel = -sq / (2.0 * variance)  # (N, K)
    m = np.max(log_kernel, axis=1, keepdims=True)
    log_sum = m.squeeze(axis=1) + np.log(np.sum(np.exp(log_kernel - m), axis=1))
    log_norm = np.log(float(K)) + 0.5 * d * np.log(2.0 * np.pi * variance)
    log_likelihood = float(np.mean(log_sum - log_norm))
    return -log_likelihood


def _train_isotropic_variance(
    X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
) -> float:
    """MLE isotropic variance per Gaussian-mixture component.

    sigma^2 = (1 / (N d)) sum_i ||x_i - c_{labels_i}||^2. Floored at a small
    positive value so the NLL stays finite even when a cluster collapses
    onto a single point.
    """
    n, d = X.shape
    diffs_sq = np.sum((X - centroids[labels]) ** 2, axis=1)
    var = float(np.sum(diffs_sq) / max(n * d, 1))
    return max(var, 1e-6)


def estimate_k(
    X: np.ndarray,
    K_range: list[int] | tuple[int, ...] | np.ndarray,
    *,
    criterion: str = "holdout_nll",
    seed: int = 0,
    holdout_fraction: float = 0.2,
    max_iter: int = 50,
) -> KEstimationResult:
    """Choose K from ``K_range`` by the named criterion.

    Parameters
    ----------
    X:
        Embedding matrix, shape ``(N, d)``. Rows are observations.
    K_range:
        Candidate K values. Must be strictly positive and at most ``N``
        (for holdout variants, at most ``N_train``).
    criterion:
        One of ``"holdout_nll"``, ``"bic"``, ``"elbow"``.
    seed:
        Drives k-means++ initialisation and the holdout split.
    holdout_fraction:
        Fraction of rows held out when criterion is ``"holdout_nll"``.
    max_iter:
        Lloyd iteration budget per K.

    Returns
    -------
    KEstimationResult.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (N, d), got shape {X.shape}")
    K_range_arr = np.asarray(list(K_range), dtype=np.int64)
    if K_range_arr.ndim != 1 or K_range_arr.size == 0:
        raise ValueError("K_range must be a non-empty 1-D iterable")
    if (K_range_arr <= 0).any():
        raise ValueError("K_range entries must be positive")
    if criterion not in {"holdout_nll", "bic", "elbow"}:
        raise ValueError(
            f"unknown criterion {criterion!r}; expected one of "
            f"'holdout_nll', 'bic', 'elbow'"
        )

    rng = np.random.default_rng(seed)
    n, d = X.shape
    n_train = n
    train_X = X
    holdout_X = None
    if criterion == "holdout_nll":
        if not (0.0 < holdout_fraction < 1.0):
            raise ValueError(
                f"holdout_fraction must be in (0, 1), got {holdout_fraction}"
            )
        perm = rng.permutation(n)
        n_holdout = max(1, int(round(holdout_fraction * n)))
        holdout_idx = perm[:n_holdout]
        train_idx = perm[n_holdout:]
        train_X = X[train_idx]
        holdout_X = X[holdout_idx]
        n_train = train_X.shape[0]
        if any(K > n_train for K in K_range_arr):
            raise ValueError(
                f"K_range contains K > n_train ({n_train}); cannot fit "
                f"with held-out NLL"
            )

    scores = np.empty(K_range_arr.shape[0], dtype=np.float64)
    fitted_centroids: list[np.ndarray] = []
    fitted_labels: list[np.ndarray] = []
    for j, K in enumerate(K_range_arr):
        K_int = int(K)
        centroids, labels = _fit_kmeans(
            train_X, K_int, np.random.default_rng(seed + j), max_iter=max_iter
        )
        fitted_centroids.append(centroids)
        fitted_labels.append(labels)
        if criterion == "elbow":
            scores[j] = _within_cluster_distortion(train_X, labels, centroids)
        elif criterion == "bic":
            # BIC = -2 log L + p log N, with isotropic Gaussian whose variance
            # is the MLE on the train labels. p = K*d means + 1 variance.
            sigma_sq = _train_isotropic_variance(train_X, labels, centroids)
            avg_nll = _gaussian_nll(train_X, centroids, variance=sigma_sq)
            log_l = -avg_nll * n_train
            p = K_int * d + 1
            scores[j] = -2.0 * log_l + p * np.log(n_train)
        else:  # holdout_nll
            sigma_sq = _train_isotropic_variance(train_X, labels, centroids)
            scores[j] = _gaussian_nll(holdout_X, centroids, variance=sigma_sq)

    if criterion in {"holdout_nll", "bic"}:
        best = int(np.argmin(scores))
    else:  # elbow: knee-finding on the distortion curve
        best = _knee_index(K_range_arr, scores)

    # Refit at K_star on the full corpus for the returned labels / centroids
    # (the holdout split was just for scoring).
    K_star = int(K_range_arr[best])
    centroids_full, labels_full = _fit_kmeans(
        X, K_star, np.random.default_rng(seed + 10_000), max_iter=max_iter
    )
    return KEstimationResult(
        K_star=K_star,
        K_range=K_range_arr,
        scores=scores,
        labels=labels_full,
        centroids=centroids_full,
        criterion=criterion,
    )


def _knee_index(K_range: np.ndarray, scores: np.ndarray) -> int:
    """Pick the elbow on a monotonically-decreasing distortion curve.

    Implements the "max distance from chord" heuristic: connect the
    endpoints of the (K, score) curve with a straight line, return the
    index whose perpendicular distance to that line is largest. Degrades
    gracefully (returns the first index) when the curve is too short.
    """
    K_range = np.asarray(K_range, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if K_range.size < 3:
        return 0
    x0, y0 = K_range[0], scores[0]
    x1, y1 = K_range[-1], scores[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = float(np.hypot(dx, dy))
    if norm == 0.0:
        return 0
    distances = np.abs(dy * K_range - dx * scores + x1 * y0 - y1 * x0) / norm
    return int(np.argmax(distances))
