"""Unit tests for nga.arch.posterior_mask.PosteriorMask.

Covers the conjugate-update story (cold start, conjugacy, warm start),
batch/serial equivalence, validation of quality, mathematical sanity
(non-negative variance, finite entropy, clipped log-odds), the identity
hash, the drop-in legality_matrix threshold semantics, and convergence
on a synthetic 4-cycle graph.
"""
from __future__ import annotations

import numpy as np
import pytest

from nga.arch.posterior_mask import PosteriorMask


def test_cold_start_uniform() -> None:
    """Beta(1, 1) priors yield posterior_mean = 0.5 everywhere."""
    mask = PosteriorMask(5)
    mean = mask.posterior_mean()
    assert mean.shape == (5, 5)
    assert np.all(mean == 0.5)


def test_conjugacy() -> None:
    """7 successes + 3 failures on Beta(1, 1) gives mean (1+7)/(1+7 + 1+3)."""
    mask = PosteriorMask(3)
    for _ in range(7):
        mask.update(0, 1, 1.0)
    for _ in range(3):
        mask.update(0, 1, 0.0)
    expected = (1.0 + 7.0) / (1.0 + 7.0 + 1.0 + 3.0)  # 8 / 12
    assert mask.posterior_mean()[0, 1] == pytest.approx(expected)
    # Other edges should be untouched.
    other = mask.posterior_mean().copy()
    other[0, 1] = 0.5
    assert np.all(other == 0.5)


def test_warm_start_from_legality() -> None:
    """from_legality_matrix gives mean ~ s/(s+1) on legal, 1/(s+1) on illegal."""
    legal = np.array(
        [
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ]
    )
    s = 10.0
    mask = PosteriorMask.from_legality_matrix(legal, prior_strength=s)
    mean = mask.posterior_mean()
    legal_mean = s / (s + 1.0)
    illegal_mean = 1.0 / (s + 1.0)
    assert mean[legal] == pytest.approx(legal_mean)
    assert mean[~legal] == pytest.approx(illegal_mean)
    # And approximately 0.91 / 0.09 as the docstring promises.
    assert legal_mean == pytest.approx(0.9090909, abs=1e-4)
    assert illegal_mean == pytest.approx(0.0909091, abs=1e-4)


def test_update_batch_equivalent_to_individual() -> None:
    """Batch update produces the same posterior as a Python loop of update()."""
    rng = np.random.default_rng(seed=123)
    n = 6
    transitions = rng.integers(low=0, high=n, size=(50, 2))
    qualities = rng.random(size=50)

    serial = PosteriorMask(n)
    for (s, d), q in zip(transitions, qualities, strict=True):
        serial.update(int(s), int(d), float(q))

    batch = PosteriorMask(n)
    batch.update_batch(transitions, qualities)

    np.testing.assert_allclose(serial.alpha, batch.alpha, atol=1e-12)
    np.testing.assert_allclose(serial.beta, batch.beta, atol=1e-12)


def test_quality_clamping_or_validation() -> None:
    """Quality outside [0, 1] raises ValueError on both update paths."""
    mask = PosteriorMask(3)
    with pytest.raises(ValueError):
        mask.update(0, 1, -0.1)
    with pytest.raises(ValueError):
        mask.update(0, 1, 1.1)
    with pytest.raises(ValueError):
        mask.update_batch(np.array([[0, 1]]), np.array([1.5]))
    with pytest.raises(ValueError):
        mask.update_batch(np.array([[0, 1]]), np.array([-0.5]))


def test_posterior_variance_nonnegative() -> None:
    """Beta variance is always >= 0, including after asymmetric updates."""
    mask = PosteriorMask(4)
    rng = np.random.default_rng(seed=7)
    for _ in range(30):
        s, d = rng.integers(0, 4, size=2)
        mask.update(int(s), int(d), float(rng.random()))
    var = mask.posterior_variance()
    assert var.shape == (4, 4)
    assert np.all(var >= 0.0)
    assert np.all(np.isfinite(var))


def test_posterior_entropy_finite() -> None:
    """Differential entropy is finite for all populated edges."""
    mask = PosteriorMask(4)
    # Cold start (uniform Beta(1,1)) -> H = ln 1 = 0, finite.
    h0 = mask.posterior_entropy()
    assert np.all(np.isfinite(h0))
    # After mixed updates, still finite.
    mask.update(0, 1, 0.9)
    mask.update(0, 1, 0.1)
    mask.update(2, 3, 1.0)
    mask.update(2, 3, 0.0)
    h1 = mask.posterior_entropy()
    assert np.all(np.isfinite(h1))


def test_legality_bias_clipped() -> None:
    """Log-odds is finite at cold start and clipped to [-30, +30] at extremes."""
    mask = PosteriorMask(3)
    bias = mask.legality_bias()
    # Cold start: alpha == beta == 1, log-odds = 0.
    assert np.all(bias == 0.0)
    assert np.all(np.isfinite(bias))

    # Drive one edge strongly-legal and another strongly-illegal by setting
    # alpha/beta to extreme pseudo-counts directly (a uniform-batch worth of
    # updates; equivalent to many calls to update() but fast and exact).
    mask.alpha[0, 1] = 1e20
    mask.beta[0, 1] = 1.0
    mask.alpha[1, 2] = 1.0
    mask.beta[1, 2] = 1e20
    bias = mask.legality_bias()
    assert np.all(np.isfinite(bias))
    assert np.all(bias <= 30.0 + 1e-9)
    assert np.all(bias >= -30.0 - 1e-9)
    # The two extreme edges should be saturated at the clip.
    assert bias[0, 1] == pytest.approx(30.0)
    assert bias[1, 2] == pytest.approx(-30.0)


def test_mask_version_id_changes_after_update() -> None:
    """Hash before vs after an update differs; identical state -> identical hash."""
    mask = PosteriorMask(3)
    h_before = mask.mask_version_id()
    assert len(h_before) == 16
    mask.update(0, 1, 0.9)
    h_after = mask.mask_version_id()
    assert h_before != h_after
    # Same state -> same hash.
    twin = PosteriorMask(3)
    twin.update(0, 1, 0.9)
    assert twin.mask_version_id() == h_after


def test_legality_matrix_threshold() -> None:
    """Skeptical-prior contract: strict inequality, so cold-start uniform
    prior (mean = 0.5 exactly) is illegal at threshold=0.5; only net-positive
    evidence (mean > 0.5) commits to legal. At threshold=0.49 the uniform
    prior is legal; at threshold=0.5 illegal."""
    mask = PosteriorMask(4)
    assert not np.any(mask.legality_matrix(threshold=0.5))
    assert np.all(mask.legality_matrix(threshold=0.49))


def test_convergence_on_synthetic_data() -> None:
    """A 4-cycle ground truth is recovered exactly by 200 updates per edge."""
    n = 4
    true_legal = np.zeros((n, n), dtype=bool)
    cycle = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for s, d in cycle:
        true_legal[s, d] = True

    mask = PosteriorMask(n)
    for s in range(n):
        for d in range(n):
            q = 0.95 if true_legal[s, d] else 0.05
            for _ in range(200):
                mask.update(s, d, q)

    recovered = mask.legality_matrix(threshold=0.5)
    hamming = int(np.sum(recovered != true_legal))
    assert hamming == 0, f"Expected Hamming distance 0, got {hamming}"
