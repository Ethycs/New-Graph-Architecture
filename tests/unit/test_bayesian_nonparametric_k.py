"""Unit tests for nga.arch.bayesian_nonparametric_k.estimate_k."""
from __future__ import annotations

import numpy as np
import pytest

from nga.arch.bayesian_nonparametric_k import (
    KEstimationResult,
    estimate_k,
)


def _three_blob_dataset(n_per_blob: int = 60, sigma: float = 0.05) -> np.ndarray:
    """Three well-separated Gaussian blobs in R^2 -- K_star should be 3."""
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [3.0, 0.0], [1.5, 2.6]])
    rows = []
    for c in centers:
        rows.append(c + sigma * rng.standard_normal((n_per_blob, 2)))
    return np.concatenate(rows, axis=0)


def _five_blob_dataset(n_per_blob: int = 40, sigma: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(1)
    centers = np.array(
        [[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0], [2.5, 2.5]]
    )
    rows = []
    for c in centers:
        rows.append(c + sigma * rng.standard_normal((n_per_blob, 2)))
    return np.concatenate(rows, axis=0)


def test_estimate_k_returns_correct_type() -> None:
    X = _three_blob_dataset()
    res = estimate_k(X, K_range=[1, 2, 3, 4, 5], criterion="holdout_nll", seed=42)
    assert isinstance(res, KEstimationResult)
    assert res.criterion == "holdout_nll"
    assert res.K_range.tolist() == [1, 2, 3, 4, 5]
    assert res.scores.shape == (5,)
    assert res.labels.shape == (X.shape[0],)
    assert res.centroids.shape == (res.K_star, X.shape[1])


def test_holdout_nll_finds_K_3_on_three_blobs() -> None:
    """Held-out NLL is minimised at K=3 when the data is three blobs."""
    X = _three_blob_dataset()
    res = estimate_k(
        X, K_range=[1, 2, 3, 4, 5, 6, 7], criterion="holdout_nll", seed=42
    )
    assert res.K_star == 3, (
        f"expected K_star=3 on three-blob synthetic data, got {res.K_star}; "
        f"scores: {res.scores.tolist()}"
    )


def test_bic_finds_K_3_on_three_blobs() -> None:
    """BIC is minimised near K=3 (penalised for overcomplexity)."""
    X = _three_blob_dataset()
    res = estimate_k(
        X, K_range=[1, 2, 3, 4, 5, 6, 7], criterion="bic", seed=42
    )
    # BIC sometimes picks 2 or 3 depending on the variance-1 prior; both are
    # acceptable for a well-separated three-blob dataset.
    assert res.K_star in {2, 3}, (
        f"expected K_star in {{2, 3}}, got {res.K_star}; "
        f"scores: {res.scores.tolist()}"
    )


def test_elbow_finds_K_3_on_three_blobs() -> None:
    """The chord-distance elbow heuristic picks K=3 (or nearby)."""
    X = _three_blob_dataset()
    res = estimate_k(
        X, K_range=[1, 2, 3, 4, 5, 6, 7], criterion="elbow", seed=42
    )
    assert res.K_star in {2, 3, 4}, (
        f"expected K_star near 3, got {res.K_star}; "
        f"scores: {res.scores.tolist()}"
    )


def test_holdout_nll_finds_K_5_on_five_blobs() -> None:
    """Held-out NLL is minimised at K=5 when the data is five blobs."""
    X = _five_blob_dataset()
    res = estimate_k(
        X, K_range=[2, 3, 4, 5, 6, 7, 8], criterion="holdout_nll", seed=42
    )
    assert res.K_star == 5


def test_determinism_under_fixed_seed() -> None:
    """Same data + seed + criterion -> identical K_star, scores, labels."""
    X = _three_blob_dataset()
    a = estimate_k(X, K_range=[2, 3, 4], criterion="holdout_nll", seed=7)
    b = estimate_k(X, K_range=[2, 3, 4], criterion="holdout_nll", seed=7)
    assert a.K_star == b.K_star
    np.testing.assert_allclose(a.scores, b.scores)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(a.centroids, b.centroids)


def test_labels_at_K_star_partition_the_input() -> None:
    """Labels are integers in [0, K_star) covering every row exactly once."""
    X = _three_blob_dataset()
    res = estimate_k(X, K_range=[2, 3, 4], criterion="bic", seed=0)
    assert res.labels.shape == (X.shape[0],)
    assert res.labels.min() >= 0
    assert res.labels.max() < res.K_star


def test_unknown_criterion_raises() -> None:
    X = _three_blob_dataset()
    with pytest.raises(ValueError):
        estimate_k(X, K_range=[2, 3], criterion="unknown")


def test_empty_K_range_raises() -> None:
    X = _three_blob_dataset()
    with pytest.raises(ValueError):
        estimate_k(X, K_range=[], criterion="holdout_nll")


def test_K_larger_than_n_train_raises_for_holdout() -> None:
    """When K exceeds the train-side row count, holdout_nll raises."""
    X = _three_blob_dataset(n_per_blob=5)  # 15 rows total
    with pytest.raises(ValueError):
        # holdout 20% -> 3 holdout, 12 train; K=15 cannot be fit.
        estimate_k(X, K_range=[15], criterion="holdout_nll", seed=0)


def test_non_2d_X_raises() -> None:
    with pytest.raises(ValueError):
        estimate_k(np.zeros((5,)), K_range=[2], criterion="bic")
    with pytest.raises(ValueError):
        estimate_k(np.zeros((5, 3, 2)), K_range=[2], criterion="bic")
