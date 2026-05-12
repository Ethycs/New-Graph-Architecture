"""Unit tests for `structural_ambiguity_auroc`.

This metric is the "abstain from grammar" cousin of the existing
margin/sigma AUROC pair. Where the failure-margin AUROCs measure
"does this signal predict prediction errors?", this metric measures
"does this signal predict grammar-level ambiguity points?". The two
questions have different ground-truth labels and different (legitimate)
answers; both belong in the architecture's metric inventory.

Tests are pure-numpy and do not depend on torch, sklearn, or any
upstream pipeline; they exercise the metric's contract directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from nga.arch.failure_margin_auroc import (
    binary_auroc,
    structural_ambiguity_auroc,
)


def test_structural_auroc_perfect_score():
    """Score == label everywhere => perfect ranking => AUROC = 1.0."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200).astype(bool)
    # Ensure both classes present so the metric is well-defined.
    if labels.all() or (~labels).all():
        labels[0] = False
        labels[1] = True
    scores = labels.astype(float)
    auroc = structural_ambiguity_auroc(scores, labels)
    # With ties, average ranks make AUROC == 1.0 only when ALL positives
    # outrank ALL negatives. Score == label has positives at score 1 and
    # negatives at score 0, no cross-class ties => AUROC = 1.0 exactly.
    assert auroc == pytest.approx(1.0, abs=1e-12)


def test_structural_auroc_inverted_score():
    """Score == 1 - label => perfectly inverted ranking => AUROC = 0.0."""
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=200).astype(bool)
    if labels.all() or (~labels).all():
        labels[0] = False
        labels[1] = True
    scores = (1.0 - labels.astype(float))
    auroc = structural_ambiguity_auroc(scores, labels)
    assert auroc == pytest.approx(0.0, abs=1e-12)


def test_structural_auroc_random_scores_near_half():
    """Scores independent of labels => AUROC ~ 0.5 within sampling noise."""
    rng = np.random.default_rng(42)
    n = 1000
    labels = rng.integers(0, 2, size=n).astype(bool)
    scores = rng.standard_normal(n).astype(float)
    auroc = structural_ambiguity_auroc(scores, labels)
    # n=1000 with balanced labels gives a se ~ 0.018 on random AUROC.
    # 0.05 tolerance is several SE wide, conservative against seed flake.
    assert abs(auroc - 0.5) < 0.05, f"random AUROC drifted: {auroc:.4f}"


def test_structural_auroc_handles_all_positive():
    """All labels positive => AUROC undefined => returns 0.5."""
    n = 50
    labels = np.ones(n, dtype=bool)
    scores = np.linspace(0.0, 1.0, n)
    auroc = structural_ambiguity_auroc(scores, labels)
    assert auroc == 0.5


def test_structural_auroc_handles_all_negative():
    """All labels negative => AUROC undefined => returns 0.5."""
    n = 50
    labels = np.zeros(n, dtype=bool)
    scores = np.linspace(0.0, 1.0, n)
    auroc = structural_ambiguity_auroc(scores, labels)
    assert auroc == 0.5


def test_structural_auroc_matches_binary_auroc():
    """structural_ambiguity_auroc is a wrapper around binary_auroc on
    well-defined inputs; the two must agree to floating-point tolerance.
    """
    rng = np.random.default_rng(7)
    n = 500
    labels = rng.integers(0, 2, size=n).astype(bool)
    scores = rng.standard_normal(n).astype(float)
    expected = binary_auroc(scores, labels)
    actual = structural_ambiguity_auroc(scores, labels)
    assert actual == pytest.approx(expected, abs=1e-12)
