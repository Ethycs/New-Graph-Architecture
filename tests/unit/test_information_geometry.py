"""Unit tests for nga.arch.information_geometry.

Covers Fisher information for Bernoulli edges, KL divergences (categorical,
Beta, Bernoulli), the Cramer-Rao sample-complexity floor, the diagonal
natural-gradient rescaling, and the hyperbolic Fisher-Rao geodesic distance
between Bernoulli distributions.
"""
from __future__ import annotations

import numpy as np
import pytest

from nga.arch.information_geometry import (
    cramer_rao_bound,
    crb_confidence_for_observations,
    crb_n_for_target_variance,
    fisher_information_bernoulli,
    fisher_information_diagonal_mask,
    fisher_rao_distance_bernoulli,
    kl_bernoulli,
    kl_beta,
    kl_categorical,
    natural_gradient_diagonal,
)


# ---------------------------------------------------------------------------
# Fisher information
# ---------------------------------------------------------------------------


def test_fisher_bernoulli_at_half_is_4() -> None:
    """I(0.5) = 1 / (0.5 * 0.5) = 4."""
    assert fisher_information_bernoulli(0.5) == pytest.approx(4.0)


def test_fisher_bernoulli_diverges_at_boundary() -> None:
    """Near p = 1 the Bernoulli Fisher info blows up."""
    info = fisher_information_bernoulli(0.999)
    # 1 / (0.999 * 0.001) ~ 1001
    assert info > 100.0
    assert np.isfinite(info)


def test_fisher_diagonal_mask_shape_matches_alpha() -> None:
    """The diagonal-Fisher mask preserves the shape of alpha/beta."""
    alpha = np.full((4, 4), 2.0)
    beta = np.full((4, 4), 3.0)
    f_diag = fisher_information_diagonal_mask(alpha, beta)
    assert f_diag.shape == (4, 4)


def test_fisher_diagonal_mask_alpha_plus_beta() -> None:
    """The diagonal-Fisher mask equals alpha + beta elementwise."""
    rng = np.random.default_rng(0)
    alpha = rng.uniform(0.1, 5.0, size=(3, 3))
    beta = rng.uniform(0.1, 5.0, size=(3, 3))
    f_diag = fisher_information_diagonal_mask(alpha, beta)
    np.testing.assert_allclose(f_diag, alpha + beta)


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------


def test_kl_categorical_zero_for_identical_distributions() -> None:
    """KL(p || p) = 0 within float tolerance."""
    p = np.array([0.1, 0.3, 0.6])
    assert kl_categorical(p, p) == pytest.approx(0.0, abs=1e-12)


def test_kl_categorical_positive_for_different() -> None:
    """KL(p || q) > 0 when distributions differ."""
    p = np.array([0.1, 0.3, 0.6])
    q = np.array([0.6, 0.3, 0.1])
    assert kl_categorical(p, q) > 0.0


def test_kl_categorical_asymmetric() -> None:
    """KL is not symmetric in general.

    Reverse-permutation distributions accidentally yield equal KLs, so we
    pick a genuinely asymmetric pair.
    """
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.2, 0.5, 0.3])
    kl_pq = kl_categorical(p, q)
    kl_qp = kl_categorical(q, p)
    assert kl_pq != pytest.approx(kl_qp, rel=1e-3)


def test_kl_beta_zero_for_identical() -> None:
    """KL(Beta(2, 3) || Beta(2, 3)) ~ 0."""
    assert kl_beta(2.0, 3.0, 2.0, 3.0) == pytest.approx(0.0, abs=1e-10)


def test_kl_beta_positive_for_different() -> None:
    """KL(Beta(2, 3) || Beta(5, 1)) > 0."""
    assert kl_beta(2.0, 3.0, 5.0, 1.0) > 0.0


def test_kl_bernoulli_matches_kl_categorical_2_classes() -> None:
    """KL(Bern(0.3) || Bern(0.7)) equals KL([0.3, 0.7] || [0.7, 0.3])."""
    p, q = 0.3, 0.7
    bern_kl = kl_bernoulli(p, q)
    cat_kl = kl_categorical(np.array([p, 1.0 - p]), np.array([q, 1.0 - q]))
    assert bern_kl == pytest.approx(cat_kl, rel=1e-9)


# ---------------------------------------------------------------------------
# Cramer-Rao bound
# ---------------------------------------------------------------------------


def test_cramer_rao_decreases_with_n() -> None:
    """CRB shrinks as n grows: CRB(I, 10) > CRB(I, 100)."""
    info = 4.0
    assert cramer_rao_bound(info, 10) > cramer_rao_bound(info, 100)


def test_crb_n_for_target_variance() -> None:
    """Required N = 1 / (target * I)."""
    info = 4.0
    target = 0.01
    expected = 1.0 / (target * info)
    assert crb_n_for_target_variance(info, target) == pytest.approx(expected)


def test_crb_confidence_at_threshold() -> None:
    """confidence == 1.0 when n equals the required sample size exactly."""
    info = 4.0
    target = 0.01
    n_req = 1.0 / (target * info)
    conf = crb_confidence_for_observations(info, int(round(n_req)), target)
    assert conf == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Natural-gradient
# ---------------------------------------------------------------------------


def test_natural_gradient_scales_inversely_with_fisher() -> None:
    """High Fisher info shrinks the natural gradient."""
    grad = np.array([1.0, 1.0, 1.0])
    f_low = np.array([0.5, 0.5, 0.5])
    f_high = np.array([1000.0, 1000.0, 1000.0])
    nat_low = natural_gradient_diagonal(grad, f_low)
    nat_high = natural_gradient_diagonal(grad, f_high)
    # All entries of nat_high should be much smaller than nat_low.
    assert np.all(np.abs(nat_high) < np.abs(nat_low))
    # And specifically high-Fisher gradient is ~1/1000.
    assert np.all(nat_high < 1e-2)


def test_natural_gradient_handles_zero_fisher_via_damping() -> None:
    """Zero Fisher info does not produce NaN/inf thanks to damping."""
    grad = np.array([1.0, -2.0, 0.5])
    f = np.array([0.0, 0.0, 0.0])
    nat = natural_gradient_diagonal(grad, f, damping=1e-6)
    assert np.all(np.isfinite(nat))
    # With damping=1e-6, |nat| ~ |grad| / 1e-6.
    np.testing.assert_allclose(nat, grad / 1e-6, rtol=1e-9)


# ---------------------------------------------------------------------------
# Fisher-Rao geodesic on Bernoulli
# ---------------------------------------------------------------------------


def test_fisher_rao_distance_bernoulli_zero_for_same() -> None:
    """FR(0.5, 0.5) = 0."""
    assert fisher_rao_distance_bernoulli(0.5, 0.5) == pytest.approx(0.0)


def test_fisher_rao_distance_bernoulli_symmetric() -> None:
    """FR(0.3, 0.7) = FR(0.7, 0.3)."""
    d_pq = fisher_rao_distance_bernoulli(0.3, 0.7)
    d_qp = fisher_rao_distance_bernoulli(0.7, 0.3)
    assert d_pq == pytest.approx(d_qp)


def test_fisher_rao_distance_grows_with_separation() -> None:
    """FR(0.1, 0.9) > FR(0.4, 0.6) -- more separated Bernoullis are farther."""
    d_far = fisher_rao_distance_bernoulli(0.1, 0.9)
    d_close = fisher_rao_distance_bernoulli(0.4, 0.6)
    assert d_far > d_close
