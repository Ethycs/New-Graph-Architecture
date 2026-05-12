"""Unit tests for nga.arch.catastrophe_labels.

These cover the numerical fold/cusp/none detector that classifies the local
geometry of a sampled energy curve via central finite differences. The legacy
FSM-edge-label lookup surface is exercised elsewhere; here we pin down the
math: which polynomial shapes get which catastrophe label, where the eps
threshold bites, and the input-shape contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from nga.arch.catastrophe_labels import (
    CatastropheLabel,
    classify_curvature_1d,
    classify_edge_catastrophe,
    edge_catastrophe_label,
)


def _sample(f, centre: float, h: float = 0.1, n: int = 5) -> np.ndarray:
    """Sample f at n equally-spaced points centred on `centre` with spacing h."""
    half = (n - 1) // 2
    xs = centre + h * np.arange(-half, half + 1)
    return np.array([f(x) for x in xs], dtype=float)


def test_fold_detection() -> None:
    # E(x) = (x - 0.5)**2 has a clean quadratic minimum at x = 0.5.
    samples = _sample(lambda x: (x - 0.5) ** 2, centre=0.5)
    assert classify_curvature_1d(samples) == CatastropheLabel.FOLD


def test_cusp_detection() -> None:
    # E(x) = x**3 is the textbook degenerate critical point at x = 0:
    # second derivative vanishes, third derivative does not.
    samples = _sample(lambda x: x ** 3, centre=0.0, h=0.1)
    assert classify_curvature_1d(samples) == CatastropheLabel.CUSP


def test_no_catastrophe_on_linear() -> None:
    # E(x) = 2x + 1: no critical structure, every finite difference of order
    # >= 2 is exactly zero.
    samples = _sample(lambda x: 2.0 * x + 1.0, centre=0.0)
    assert classify_curvature_1d(samples) == CatastropheLabel.NONE


def test_classify_edge_catastrophe_finds_minimum() -> None:
    # Parabolic-ish minimum at index 2: gradient is smallest there, and the
    # 5-sample classifier should label the local geometry FOLD.
    path = np.array([3.0, 2.0, 1.0001, 2.0, 3.0])
    assert classify_edge_catastrophe(path) == CatastropheLabel.FOLD


def test_eps_threshold_respected() -> None:
    # With a huge eps, even a clean fold falls below the threshold and the
    # detector falls all the way through to NONE.
    samples = _sample(lambda x: (x - 0.5) ** 2, centre=0.5)
    assert classify_curvature_1d(samples, eps=10.0) == CatastropheLabel.NONE
    path = np.array([3.0, 2.0, 1.0001, 2.0, 3.0])
    assert classify_edge_catastrophe(path, eps=10.0) == CatastropheLabel.NONE


def test_handles_short_path() -> None:
    # The local 5-sample window is mandatory; shorter inputs raise ValueError.
    with pytest.raises(ValueError):
        classify_curvature_1d(np.array([0.0, 1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        classify_edge_catastrophe(np.array([0.0, 1.0, 2.0, 3.0]))


def test_classify_curvature_1d_symmetry() -> None:
    # The classifier looks at |d2E|, not its sign: a clean maximum is also
    # a fold catastrophe (A_2), just with opposite curvature.
    minimum = _sample(lambda x: (x - 0.5) ** 2, centre=0.5)
    maximum = _sample(lambda x: -((x - 0.5) ** 2), centre=0.5)
    assert classify_curvature_1d(minimum) == CatastropheLabel.FOLD
    assert classify_curvature_1d(maximum) == CatastropheLabel.FOLD


def test_butterfly_swallowtail_remain_named_but_not_detected() -> None:
    # Both higher-order catastrophes are part of the public taxonomy, but the
    # numerical detector in this wave only emits NONE / FOLD / CUSP. We probe
    # a representative spread of inputs and assert no detector call returns
    # SWALLOWTAIL or BUTTERFLY.
    assert CatastropheLabel.SWALLOWTAIL.value == "swallowtail"
    assert CatastropheLabel.BUTTERFLY.value == "butterfly"

    detectable = {CatastropheLabel.NONE, CatastropheLabel.FOLD, CatastropheLabel.CUSP}
    candidates = [
        _sample(lambda x: (x - 0.5) ** 2, centre=0.5),       # fold
        _sample(lambda x: x ** 3, centre=0.0),                # cusp
        _sample(lambda x: 2.0 * x + 1.0, centre=0.0),         # none
        _sample(lambda x: x ** 4, centre=0.0),                # also degenerate
        _sample(lambda x: x ** 5, centre=0.0),
    ]
    for sample in candidates:
        assert classify_curvature_1d(sample) in detectable
        # classify_edge_catastrophe takes the same length-5 array as a path.
        assert classify_edge_catastrophe(sample) in detectable


def test_edge_catastrophe_label_concatenates_endpoints() -> None:
    # Sanity-check the convenience wrapper end-to-end: a parabolic well with
    # endpoints at 3 and three quadratic-mid samples should still resolve to
    # FOLD via edge_catastrophe_label.
    mids = np.array([2.0, 1.0001, 2.0])
    assert (
        edge_catastrophe_label(3.0, 3.0, mids) == CatastropheLabel.FOLD
    )
    # A linear ramp through endpoints + linear mids has no curvature -> NONE.
    linear_mids = np.array([1.0, 2.0, 3.0])
    assert (
        edge_catastrophe_label(0.0, 4.0, linear_mids) == CatastropheLabel.NONE
    )
