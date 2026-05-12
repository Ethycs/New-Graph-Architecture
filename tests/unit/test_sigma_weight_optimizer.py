"""Unit tests for the sigma-weight optimiser (Phase 18 Track 1)."""
from __future__ import annotations

import numpy as np
import pytest

from nga.arch.failure_margin_auroc import binary_auroc
from nga.arch.sigma_weight_optimizer import (
    DEFAULT_OPTIMIZER_GRID,
    compute_sigma_from_weights,
    optimize_sigma_weights,
    sigma_auroc_for_weights,
)


def _toy_signals(n: int, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Build a (signals, error_labels) pair where ``margin`` perfectly
    discriminates errors -- so the optimiser should put nearly all weight
    on margin."""
    rng = np.random.default_rng(seed)
    error_labels = rng.integers(0, 2, size=n).astype(np.int64)
    # margin: high for errors, low for non-errors -> perfect AUROC ~ 1.0
    margin_signal = np.where(error_labels == 1, 0.9, 0.1).astype(np.float64)
    margin_signal = margin_signal + 0.01 * rng.standard_normal(n)
    margin_signal = np.clip(margin_signal, 0.0, 1.0)
    signals = {
        "margin": margin_signal,
        "decision_tie": rng.uniform(0.0, 1.0, size=n),
        "illegal": rng.uniform(0.0, 1.0, size=n),
        "loop_risk": rng.uniform(0.0, 1.0, size=n),
        "kl_surprise": rng.uniform(0.0, 1.0, size=n),
    }
    return signals, error_labels


def test_optimize_returns_dict():
    """``optimize_sigma_weights`` returns a dict with the canonical keys."""
    signals, labels = _toy_signals(n=80, seed=0)
    out = optimize_sigma_weights(signals=signals, error_labels=labels)
    assert isinstance(out, dict)
    expected_keys = set(DEFAULT_OPTIMIZER_GRID)
    assert set(out) == expected_keys, (
        f"expected keys {expected_keys!r}; got {set(out)!r}"
    )
    for k, v in out.items():
        assert isinstance(v, float)
        assert v >= 0.0


def test_optimize_increases_auroc():
    """Tuned weights produce sigma_AUROC at least as good as the default
    weight dict on the same data. (>= because the grid may already
    contain the default-equivalent point.)"""
    signals, labels = _toy_signals(n=200, seed=1)

    default_weights = {
        "margin": 0.6,
        "decision_tie": 0.1,
        "illegal": 0.2,
        "loop_risk": 0.05,
        "kl_surprise": 0.5,
    }
    auroc_default = sigma_auroc_for_weights(default_weights, signals, labels)

    tuned = optimize_sigma_weights(signals=signals, error_labels=labels)
    auroc_tuned = sigma_auroc_for_weights(tuned, signals, labels)
    assert auroc_tuned >= auroc_default - 1e-9, (
        f"tuned AUROC {auroc_tuned:.4f} should be >= default {auroc_default:.4f}"
    )


def test_optimize_handles_random_data():
    """On a no-signal random dataset the best AUROC should still be near
    0.5 (no weight combination can manufacture signal where there is
    none)."""
    rng = np.random.default_rng(42)
    n = 300
    signals = {k: rng.uniform(0.0, 1.0, size=n) for k in DEFAULT_OPTIMIZER_GRID}
    labels = rng.integers(0, 2, size=n).astype(np.int64)

    tuned = optimize_sigma_weights(signals=signals, error_labels=labels)
    auroc = sigma_auroc_for_weights(tuned, signals, labels)
    # Allow a generous slack: the grid optimizes against THIS sample, so
    # some sample-optimum overshoot above 0.5 is expected. But it should
    # not be near 1.0.
    assert 0.40 <= auroc <= 0.75, (
        f"random-data tuned AUROC {auroc:.4f} should be near 0.5, not extreme"
    )


def test_compute_sigma_from_weights_linear():
    """``compute_sigma_from_weights`` returns the linear blend of
    signals weighted by ``weights`` (clipped to [0, 1])."""
    signals = {
        "margin": np.array([0.2, 0.4, 0.6]),
        "kl_surprise": np.array([0.1, 0.5, 0.0]),
    }
    weights = {"margin": 0.5, "kl_surprise": 1.0}
    out = compute_sigma_from_weights(weights, signals)
    expected = np.clip(
        0.5 * signals["margin"] + 1.0 * signals["kl_surprise"], 0.0, 1.0
    )
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_sigma_auroc_for_weights_matches_compute_then_auroc():
    """``sigma_auroc_for_weights`` is equivalent to building sigma via
    ``compute_sigma_from_weights`` then calling ``binary_auroc``."""
    signals, labels = _toy_signals(n=100, seed=7)
    weights = {
        "margin": 1.0,
        "decision_tie": 0.5,
        "illegal": 0.5,
        "loop_risk": 0.25,
        "kl_surprise": 0.5,
    }
    sigma = compute_sigma_from_weights(weights, signals)
    expected = float(binary_auroc(sigma, labels.astype(bool)))
    got = sigma_auroc_for_weights(weights, signals, labels)
    assert got == pytest.approx(expected, abs=1e-12)
