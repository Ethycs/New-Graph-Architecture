"""Unit tests for the Phase 7 KL-surprise extension of singularity_detector.

These tests cover the new ``compute_kl_surprise`` helper, the new
``SignalContributions.kl_surprise`` field, and the optional ``kl_surprise``
keyword on ``SingularityDetector.compute``. Backward compatibility is also
verified: the default behaviour of ``compute()`` is unchanged when callers
do not opt in.
"""
from __future__ import annotations

import numpy as np
import pytest

from nga.arch.singularity_detector import (
    SignalContributions,
    SingularityDetector,
    compute_kl_surprise,
)


# ---------------------------------------------------------------------------
# compute_kl_surprise: standalone behaviour
# ---------------------------------------------------------------------------


def test_kl_surprise_signal_zero_for_matching_distributions() -> None:
    """When empirical and predicted distributions match, surprise ~ 0."""
    p = np.array([0.2, 0.3, 0.5])
    surprise = compute_kl_surprise(p, p)
    assert surprise == pytest.approx(0.0, abs=1e-12)


def test_kl_surprise_signal_increases_with_divergence() -> None:
    """KL surprise grows monotonically as distributions diverge."""
    empirical = np.array([0.5, 0.5])
    near = np.array([0.55, 0.45])
    mid = np.array([0.7, 0.3])
    far = np.array([0.95, 0.05])

    s_near = compute_kl_surprise(empirical, near)
    s_mid = compute_kl_surprise(empirical, mid)
    s_far = compute_kl_surprise(empirical, far)

    assert s_near < s_mid < s_far


def test_kl_surprise_bounded_in_unit_interval() -> None:
    """0 <= surprise < 1 for all inputs, including extreme divergences."""
    # Identical -> 0.
    p_same = np.array([0.25, 0.25, 0.25, 0.25])
    s0 = compute_kl_surprise(p_same, p_same)
    assert 0.0 <= s0 < 1.0

    # Strongly divergent -- empirical concentrated where predicted is small.
    empirical = np.array([0.99, 0.005, 0.005])
    predicted = np.array([0.005, 0.005, 0.99])
    s_high = compute_kl_surprise(empirical, predicted)
    assert 0.0 <= s_high < 1.0

    # A range of random pairs -- all must be in [0, 1).
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.dirichlet(np.ones(5))
        q = rng.dirichlet(np.ones(5))
        s = compute_kl_surprise(p, q)
        assert 0.0 <= s < 1.0


# ---------------------------------------------------------------------------
# SignalContributions: new field
# ---------------------------------------------------------------------------


def test_signal_contributions_has_kl_surprise_field() -> None:
    """SignalContributions instances expose the new kl_surprise field."""
    sc = SignalContributions(
        margin_signal=0.1,
        decision_tie_signal=0.2,
        illegal_signal=0.0,
        loop_signal=0.0,
        stabilizer_signal=0.0,
        catastrophe_bias=0.0,
    )
    # Default value is 0.0 so existing callers don't need to pass it.
    assert hasattr(sc, "kl_surprise")
    assert sc.kl_surprise == 0.0

    # Explicitly setting it should round-trip.
    sc2 = SignalContributions(
        margin_signal=0.1,
        decision_tie_signal=0.2,
        illegal_signal=0.0,
        loop_signal=0.0,
        stabilizer_signal=0.0,
        catastrophe_bias=0.0,
        kl_surprise=0.42,
    )
    assert sc2.kl_surprise == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# SingularityDetector.compute: opt-in behaviour
# ---------------------------------------------------------------------------


def test_compute_with_kl_surprise_opt_in() -> None:
    """Passing kl_surprise + a positive weight raises sigma above the
    no-kl-surprise sigma for the same call."""
    weights = {
        "margin": 0.6,
        "decision_tie": 0.1,
        "illegal": 0.2,
        "loop": 0.05,
        "stabilizer": 0.05,
        "kl_surprise": 1.0,
    }
    detector = SingularityDetector(weights=weights)

    sigma_no_kl, _ = detector.compute(margin=0.5)
    sigma_with_kl, contribs = detector.compute(margin=0.5, kl_surprise=0.5)

    assert sigma_with_kl > sigma_no_kl
    assert contribs.kl_surprise == pytest.approx(0.5)


def test_compute_default_weights_have_kl_surprise_zero() -> None:
    """Existing behaviour is preserved by default: even with kl_surprise
    passed, the default weight is zero so sigma is unchanged."""
    detector = SingularityDetector()  # default weights, kl_surprise weight = 0.0

    sigma_a, _ = detector.compute(margin=0.5)
    sigma_b, _ = detector.compute(margin=0.5, kl_surprise=0.9)

    assert sigma_a == pytest.approx(sigma_b)
    # And the public DEFAULT_WEIGHTS map exposes the zero default.
    assert SingularityDetector.DEFAULT_WEIGHTS["kl_surprise"] == 0.0


def test_compute_with_kl_surprise_none_or_zero_unchanged() -> None:
    """Passing kl_surprise=None or 0.0 produces the same sigma as omitting
    the argument entirely."""
    detector = SingularityDetector(weights={"kl_surprise": 1.0})

    sigma_omitted, _ = detector.compute(margin=0.5)
    sigma_none, _ = detector.compute(margin=0.5, kl_surprise=None)
    sigma_zero, _ = detector.compute(margin=0.5, kl_surprise=0.0)

    assert sigma_omitted == pytest.approx(sigma_none)
    assert sigma_omitted == pytest.approx(sigma_zero)


def test_compute_with_high_kl_surprise_dominates_at_high_weight() -> None:
    """With weights={kl_surprise: 10.0, others: 0.0}, sigma tracks
    kl_surprise directly (clipped to [0, 1])."""
    weights = {
        "margin": 0.0,
        "decision_tie": 0.0,
        "illegal": 0.0,
        "loop": 0.0,
        "stabilizer": 0.0,
        "kl_surprise": 10.0,
    }
    detector = SingularityDetector(weights=weights)

    # Small kl_surprise * 10 = 0.5: well below the clip ceiling.
    sigma_small, _ = detector.compute(margin=1.0, kl_surprise=0.05)
    assert sigma_small == pytest.approx(0.5)

    # Larger kl_surprise * 10 saturates the clip ceiling of 1.0.
    sigma_large, _ = detector.compute(margin=1.0, kl_surprise=0.5)
    assert sigma_large == pytest.approx(1.0)

    # Sigma is monotone in kl_surprise with this weighting.
    sigma_a, _ = detector.compute(margin=1.0, kl_surprise=0.02)
    sigma_b, _ = detector.compute(margin=1.0, kl_surprise=0.04)
    assert sigma_b > sigma_a


def test_kl_surprise_independent_of_mask() -> None:
    """Two predictions with the same KL but different is_illegal flags
    contribute the same kl_surprise amount -- the surprise is purely
    informational, not mask-dependent."""
    weights = {
        "margin": 0.0,
        "decision_tie": 0.0,
        "illegal": 0.0,  # zero out the illegal contribution to isolate kl_surprise
        "loop": 0.0,
        "stabilizer": 0.0,
        "kl_surprise": 1.0,
    }
    detector = SingularityDetector(weights=weights)

    empirical = np.array([0.6, 0.3, 0.1])
    predicted = np.array([0.2, 0.3, 0.5])
    surprise = compute_kl_surprise(empirical, predicted)

    sigma_legal, contribs_legal = detector.compute(
        margin=1.0, is_illegal=False, kl_surprise=surprise
    )
    sigma_illegal, contribs_illegal = detector.compute(
        margin=1.0, is_illegal=True, kl_surprise=surprise
    )

    # The kl_surprise contribution is identical regardless of legality.
    assert contribs_legal.kl_surprise == pytest.approx(contribs_illegal.kl_surprise)
    # And, with illegal weight zeroed, the resulting sigma is also identical.
    assert sigma_legal == pytest.approx(sigma_illegal)
