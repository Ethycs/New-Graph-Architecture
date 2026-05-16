"""Unit tests for ``nga.arch.labelled_hypergraph``.

Acceptance bars A1-A2, A6 from ``docs/proposals/labelled-hypergraph.md``:
- A1: JSON round-trip preserves structure.
- A2: Discrete-graph projection recovers the original PCG-X output.
- A6: Construction from PCG-X-style inputs produces a well-formed hypergraph.
"""
from __future__ import annotations

import numpy as np
import pytest

from nga.arch.labelled_hypergraph import (
    Hyperedge,
    LabelledHypergraph,
    Regime,
)


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------


def test_regime_defaults_are_empty() -> None:
    r = Regime(regime_id="r0")
    assert r.canonical_signature.shape == (0,)
    assert r.named == {}
    assert r.residual == []
    assert r.p_lambda is None
    assert r.support_count == 0


def test_regime_signature_hash_is_stable_and_short() -> None:
    sig = np.array([0.1, 0.2, 0.3])
    r1 = Regime(regime_id="r0", canonical_signature=sig)
    r2 = Regime(regime_id="r0", canonical_signature=sig.copy())
    h1 = r1.signature_hash()
    h2 = r2.signature_hash()
    assert h1 == h2
    assert len(h1) == 12
    # Different signature ⇒ different hash.
    r3 = Regime(regime_id="r0", canonical_signature=np.array([0.4, 0.5, 0.6]))
    assert r3.signature_hash() != h1


def test_regime_rejects_multidim_signature() -> None:
    with pytest.raises(ValueError):
        Regime(regime_id="r0", canonical_signature=np.zeros((2, 3)))


def test_regime_to_from_dict_round_trip() -> None:
    r = Regime(
        regime_id="r7",
        canonical_signature=np.array([0.0, 1.5, 0.3]),
        named={"fsm_state": "q_open", "sae_dominant": "f_142"},
        residual=["f_3017", "f_4422"],
        p_lambda=0.18,
        support_count=42,
    )
    d = r.to_dict()
    r2 = Regime.from_dict(d)
    assert r2.regime_id == r.regime_id
    assert np.allclose(r2.canonical_signature, r.canonical_signature)
    assert r2.named == r.named
    assert r2.residual == r.residual
    assert r2.p_lambda == r.p_lambda
    assert r2.support_count == r.support_count


# ---------------------------------------------------------------------------
# Hyperedge
# ---------------------------------------------------------------------------


def test_hyperedge_defaults_are_empty() -> None:
    e = Hyperedge(src_regime_id="r0", dst_regime_id="r1")
    assert e.feature_delta == {}
    assert e.boundary_geometry is None
    assert e.sigma_at_crossing is None
    assert e.beta_posterior is None
    assert e.traversal_count == 0


def test_hyperedge_to_from_dict_round_trip() -> None:
    e = Hyperedge(
        src_regime_id="r0",
        dst_regime_id="r1",
        feature_delta={"f_3": 0.5, "f_8": -0.2},
        boundary_geometry={"normal": [0.1, 0.9], "offset": 0.3},
        sigma_at_crossing=0.42,
        beta_posterior=(3.0, 1.0),
        traversal_count=11,
    )
    d = e.to_dict()
    e2 = Hyperedge.from_dict(d)
    assert e2.src_regime_id == e.src_regime_id
    assert e2.dst_regime_id == e.dst_regime_id
    assert e2.feature_delta == e.feature_delta
    assert e2.boundary_geometry == e.boundary_geometry
    assert e2.sigma_at_crossing == e.sigma_at_crossing
    assert e2.beta_posterior == e.beta_posterior
    assert e2.traversal_count == e.traversal_count


# ---------------------------------------------------------------------------
# LabelledHypergraph — A6: construction degrades gracefully
# ---------------------------------------------------------------------------


def test_from_pcg_graph_minimal_input() -> None:
    """Only regime_ids supplied — everything else defaults."""
    h = LabelledHypergraph.from_pcg_graph(["r0", "r1", "r2"])
    assert set(h.regimes) == {"r0", "r1", "r2"}
    assert h.hyperedges == []
    for r in h.regimes.values():
        assert r.canonical_signature.shape == (0,)
        assert r.named == {}
        assert r.residual == []


def test_from_pcg_graph_full_enrichment() -> None:
    h = LabelledHypergraph.from_pcg_graph(
        regime_ids=["r0", "r1"],
        edges=[("r0", "r1"), ("r1", "r0")],
        edge_counts={("r0", "r1"): 5, ("r1", "r0"): 3},
        canonical_signatures={"r0": np.array([0.0, 1.0]), "r1": np.array([1.0, 0.0])},
        named_labels={"r0": {"fsm_state": "q_open"}, "r1": {"fsm_state": "q_close"}},
        residual_features={"r0": ["f_99"], "r1": ["f_42"]},
        feature_deltas={("r0", "r1"): {"f_99": -1.0, "f_42": 1.0}},
        p_lambda={"r0": 0.6, "r1": 0.4},
        support_counts={"r0": 100, "r1": 50},
        metadata={"substrate": "gpt2", "layer": 10},
    )
    assert h.regimes["r0"].named == {"fsm_state": "q_open"}
    assert h.regimes["r1"].residual == ["f_42"]
    assert h.regimes["r0"].p_lambda == 0.6
    assert h.regimes["r1"].support_count == 50
    assert h.metadata == {"substrate": "gpt2", "layer": 10}
    # Edges are in input order.
    assert (h.hyperedges[0].src_regime_id, h.hyperedges[0].dst_regime_id) == ("r0", "r1")
    assert h.hyperedges[0].feature_delta == {"f_99": -1.0, "f_42": 1.0}
    assert h.hyperedges[0].traversal_count == 5
    assert h.hyperedges[1].traversal_count == 3


def test_from_pcg_graph_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        LabelledHypergraph.from_pcg_graph(["r0", "r0", "r1"])


def test_from_pcg_graph_rejects_dangling_edge() -> None:
    with pytest.raises(ValueError, match="unknown regime"):
        LabelledHypergraph.from_pcg_graph(
            regime_ids=["r0", "r1"],
            edges=[("r0", "r_missing")],
        )


# ---------------------------------------------------------------------------
# A2: Discrete-graph projection recovers the original PCG-X output
# ---------------------------------------------------------------------------


def test_as_discrete_graph_recovers_inputs() -> None:
    regime_ids = ["r0", "r1", "r2"]
    edges = [("r0", "r1"), ("r1", "r2"), ("r2", "r0")]
    h = LabelledHypergraph.from_pcg_graph(
        regime_ids=regime_ids,
        edges=edges,
        canonical_signatures={"r0": np.array([0.0, 1.0, 1.0])},
        named_labels={"r0": {"x": "y"}},
        residual_features={"r0": ["f_1"]},
    )
    rids, e = h.as_discrete_graph()
    assert rids == regime_ids
    assert e == edges


# ---------------------------------------------------------------------------
# Interpretation coverage
# ---------------------------------------------------------------------------


def test_interpretation_coverage_named_residual_split() -> None:
    h = LabelledHypergraph.from_pcg_graph(
        regime_ids=["fully_named", "fully_residual", "mixed", "empty"],
        named_labels={
            "fully_named": {"a": "x", "b": "y"},
            "mixed": {"a": "x"},
        },
        residual_features={
            "fully_residual": ["f_1", "f_2"],
            "mixed": ["f_3"],
        },
    )
    cov = h.interpretation_coverage()
    assert cov["fully_named"] == 1.0
    assert cov["fully_residual"] == 0.0
    assert cov["mixed"] == pytest.approx(0.5)
    assert np.isnan(cov["empty"])


# ---------------------------------------------------------------------------
# A1: JSON round-trip preserves structure
# ---------------------------------------------------------------------------


def test_json_round_trip_preserves_structure() -> None:
    h1 = LabelledHypergraph.from_pcg_graph(
        regime_ids=["r0", "r1"],
        edges=[("r0", "r1")],
        edge_counts={("r0", "r1"): 7},
        canonical_signatures={
            "r0": np.array([0.0, 0.5]),
            "r1": np.array([0.5, 0.0]),
        },
        named_labels={"r0": {"axis1": "alpha"}},
        residual_features={"r1": ["f_1", "f_2"]},
        feature_deltas={("r0", "r1"): {"f_1": 0.3}},
        p_lambda={"r0": 0.6, "r1": 0.4},
        support_counts={"r0": 12, "r1": 8},
        metadata={"layer": 10},
    )
    payload = h1.to_json()
    h2 = LabelledHypergraph.from_json(payload)
    # Canonical-form re-emission is bitwise identical.
    assert h2.to_json() == payload
    # Per-regime fields preserved.
    assert h2.regimes["r0"].named == {"axis1": "alpha"}
    assert h2.regimes["r1"].residual == ["f_1", "f_2"]
    assert np.allclose(h2.regimes["r0"].canonical_signature, [0.0, 0.5])
    # Edge fields preserved.
    assert h2.hyperedges[0].traversal_count == 7
    assert h2.hyperedges[0].feature_delta == {"f_1": 0.3}
    # Metadata preserved.
    assert h2.metadata == {"layer": 10}


def test_hash_is_stable_for_same_structure() -> None:
    h1 = LabelledHypergraph.from_pcg_graph(
        regime_ids=["r0", "r1"],
        edges=[("r0", "r1")],
    )
    h2 = LabelledHypergraph.from_pcg_graph(
        regime_ids=["r0", "r1"],
        edges=[("r0", "r1")],
    )
    assert h1.hash() == h2.hash()
    h3 = LabelledHypergraph.from_pcg_graph(
        regime_ids=["r0", "r1"],
        edges=[("r1", "r0")],
    )
    assert h1.hash() != h3.hash()
