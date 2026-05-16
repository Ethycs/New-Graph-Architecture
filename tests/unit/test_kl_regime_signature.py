"""Unit tests for ``nga.arch.kl_regime_signature``.

Acceptance bar A4 from ``docs/proposals/labelled-hypergraph.md``: on a
hand-constructed two-regime case the signature distance matches the
analytic KL to numerical tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

from nga.arch.kl_regime_signature import KLRegimeSignature


def _analytic_kl(p: np.ndarray, q: np.ndarray, eps: float = 1.0e-12) -> float:
    p = np.clip(p / p.sum(), eps, None)
    q = np.clip(q / q.sum(), eps, None)
    return float(np.sum(p * (np.log(p) - np.log(q))))


# ---------------------------------------------------------------------------
# A4: two-regime analytic match
# ---------------------------------------------------------------------------


def test_two_regime_kl_matches_analytic() -> None:
    p1 = np.array([0.9, 0.1])
    p2 = np.array([0.1, 0.9])
    sigs = KLRegimeSignature.compute({"r1": p1, "r2": p2}, symmetric=False)
    # IDs are sorted: r1, r2.
    assert list(sigs) == ["r1", "r2"]
    expected_12 = _analytic_kl(p1, p2)
    expected_21 = _analytic_kl(p2, p1)
    assert sigs["r1"][0] == pytest.approx(0.0, abs=1e-9)
    assert sigs["r1"][1] == pytest.approx(expected_12, abs=1e-6)
    assert sigs["r2"][0] == pytest.approx(expected_21, abs=1e-6)
    assert sigs["r2"][1] == pytest.approx(0.0, abs=1e-9)


def test_symmetric_kl_is_sum_of_directed() -> None:
    p1 = np.array([0.7, 0.3])
    p2 = np.array([0.2, 0.8])
    asym = KLRegimeSignature.compute({"r1": p1, "r2": p2}, symmetric=False)
    sym = KLRegimeSignature.compute({"r1": p1, "r2": p2}, symmetric=True)
    expected_off = asym["r1"][1] + asym["r2"][0]
    assert sym["r1"][1] == pytest.approx(expected_off, abs=1e-9)
    assert sym["r2"][0] == pytest.approx(expected_off, abs=1e-9)


def test_identical_distributions_have_zero_distance() -> None:
    p = np.array([0.3, 0.3, 0.4])
    sigs = KLRegimeSignature.compute({"a": p, "b": p.copy()})
    assert sigs["a"][1] == pytest.approx(0.0, abs=1e-9)
    assert sigs["b"][0] == pytest.approx(0.0, abs=1e-9)


def test_compute_empty_input_returns_empty() -> None:
    assert KLRegimeSignature.compute({}) == {}


def test_compute_rejects_mismatched_support() -> None:
    with pytest.raises(ValueError, match="support"):
        KLRegimeSignature.compute(
            {
                "a": np.array([0.5, 0.5]),
                "b": np.array([0.3, 0.3, 0.4]),
            }
        )


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def test_cluster_identifies_close_regimes() -> None:
    # Three regimes; r1 and r2 are nearly identical; r3 is far.
    p1 = np.array([0.5, 0.5])
    p2 = np.array([0.5001, 0.4999])
    p3 = np.array([0.99, 0.01])
    sigs = KLRegimeSignature.compute({"r1": p1, "r2": p2, "r3": p3})
    # Threshold large enough to merge r1,r2 but not r3.
    clusters = KLRegimeSignature.cluster(sigs, threshold=1e-3)
    assert clusters["r1"] == clusters["r2"]
    assert clusters["r3"] != clusters["r1"]


def test_cluster_at_zero_threshold_keeps_distinct() -> None:
    p1 = np.array([0.5, 0.5])
    p2 = np.array([0.51, 0.49])
    sigs = KLRegimeSignature.compute({"r1": p1, "r2": p2})
    clusters = KLRegimeSignature.cluster(sigs, threshold=0.0)
    assert clusters["r1"] != clusters["r2"]


def test_cluster_canonical_is_sorted_first_id() -> None:
    p = np.array([0.5, 0.5])
    sigs = KLRegimeSignature.compute({"z_regime": p, "a_regime": p.copy()})
    clusters = KLRegimeSignature.cluster(sigs, threshold=0.1)
    assert clusters["a_regime"] == "a_regime"
    assert clusters["z_regime"] == "a_regime"


def test_cluster_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        KLRegimeSignature.cluster({}, threshold=-0.1)


# ---------------------------------------------------------------------------
# Threshold suggestion
# ---------------------------------------------------------------------------


def test_suggest_threshold_finds_gap() -> None:
    # Two clusters: {r1, r2} close together; {r3, r4} close together; gap between.
    p1 = np.array([0.6, 0.4])
    p2 = np.array([0.61, 0.39])
    p3 = np.array([0.1, 0.9])
    p4 = np.array([0.11, 0.89])
    sigs = KLRegimeSignature.compute({"r1": p1, "r2": p2, "r3": p3, "r4": p4})
    thresh = KLRegimeSignature.suggest_threshold(sigs, method="gap")
    # The suggested threshold should merge intra-cluster but not inter-cluster.
    clusters = KLRegimeSignature.cluster(sigs, threshold=thresh)
    assert clusters["r1"] == clusters["r2"]
    assert clusters["r3"] == clusters["r4"]
    assert clusters["r1"] != clusters["r3"]


def test_suggest_threshold_fewer_than_three_returns_zero() -> None:
    assert KLRegimeSignature.suggest_threshold({}) == 0.0
    p = np.array([0.5, 0.5])
    sigs = KLRegimeSignature.compute({"a": p, "b": p})
    assert KLRegimeSignature.suggest_threshold(sigs) == 0.0


def test_suggest_threshold_unknown_method_raises() -> None:
    p = np.array([0.5, 0.5])
    sigs = KLRegimeSignature.compute({"a": p, "b": p, "c": p})
    with pytest.raises(ValueError, match="Unknown method"):
        KLRegimeSignature.suggest_threshold(sigs, method="banana")
