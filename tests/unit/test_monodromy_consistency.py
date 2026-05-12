"""Unit tests for ``nga.arch.monodromy_consistency``.

Covers closed-walk enumeration over a runtime FSM, the drift utilities, and
both the basic and intermediate-drift consistency-loss flavours.
"""
from __future__ import annotations

import numpy as np

from nga.arch.graph_fsm import GraphFSM
from nga.arch.monodromy_consistency import (
    closed_walks_in_fsm,
    consistency_loss_with_intermediate_drift,
    monodromy_consistency_loss,
    walk_energy_drift,
)
from nga.drivers.graph_fsm_spec import (
    Coordinates,
    Edge,
    GraphFSMSpec,
    Vertex,
)


# ---------------------------------------------------------------------------
# FSM helpers
# ---------------------------------------------------------------------------


def _make_cyclic4_fsm() -> GraphFSM:
    """4-vertex directed cycle: s0 -> s1 -> s2 -> s3 -> s0."""
    vertices = [Vertex(id=f"s{i}", label=f"S{i}") for i in range(4)]
    edges = [
        Edge(source="s0", target="s1"),
        Edge(source="s1", target="s2"),
        Edge(source="s2", target="s3"),
        Edge(source="s3", target="s0"),
    ]
    spec = GraphFSMSpec(
        name="cyclic4",
        vertex_count=4,
        edge_count=4,
        vertices=vertices,
        edges=edges,
        coordinates=Coordinates(space="euclidean", dimension=1),
    )
    return GraphFSM(spec)


def _make_dag3_fsm() -> GraphFSM:
    """Acyclic 3-vertex graph: s0 -> s1 -> s2 (no closed walks)."""
    vertices = [Vertex(id=f"s{i}", label=f"S{i}") for i in range(3)]
    edges = [
        Edge(source="s0", target="s1"),
        Edge(source="s1", target="s2"),
    ]
    spec = GraphFSMSpec(
        name="dag3",
        vertex_count=3,
        edge_count=2,
        vertices=vertices,
        edges=edges,
        coordinates=Coordinates(space="euclidean", dimension=1),
    )
    return GraphFSM(spec)


def _is_cyclic_rotation(a: list[int], b: list[int]) -> bool:
    """True iff the cycle (a[0], ..., a[-2]) is a cyclic rotation of (b[0], ..., b[-2]).

    Both walks are assumed to be in the [v_0, ..., v_{n-1}, v_0] convention.
    """
    if len(a) != len(b):
        return False
    body_a = a[:-1]
    body_b = b[:-1]
    n = len(body_a)
    if n != len(body_b):
        return False
    for i in range(n):
        if body_a[i:] + body_a[:i] == body_b:
            return True
    return False


# ---------------------------------------------------------------------------
# Closed-walk enumeration
# ---------------------------------------------------------------------------


def test_closed_walks_simple_cycle() -> None:
    """A 4-vertex directed cycle has the unique closed walk 0->1->2->3->0 (mod rotation)."""
    fsm = _make_cyclic4_fsm()
    walks = closed_walks_in_fsm(fsm, max_length=4)
    expected = [0, 1, 2, 3, 0]
    assert any(_is_cyclic_rotation(w, expected) for w in walks), (
        f"expected to find cyclic rotation of {expected} in walks={walks}"
    )


def test_closed_walks_acyclic_graph_has_none() -> None:
    """A DAG has no closed walks."""
    fsm = _make_dag3_fsm()
    walks = closed_walks_in_fsm(fsm, max_length=6)
    assert walks == []


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def test_walk_energy_drift_zero_on_closed_field() -> None:
    """Static energy field with E[start] == E[end] yields drift exactly zero."""
    walk = [0, 1, 2, 3, 0]
    energies_by_state = {0: 0.5, 1: 0.7, 2: 0.9, 3: 0.6}
    drift = walk_energy_drift(walk, energies_by_state)
    assert drift == 0.0


def test_walk_energy_drift_nonzero_on_drift() -> None:
    """Open energy at endpoints gives nonzero drift; squared-drift matches expected."""
    walk = [0, 1, 2, 3, 0]
    # Pretend the energy field assigned 1 to walk[0] but sampled 5 at walk[-1].
    energies_by_state = {0: 1.0}  # endpoint energies handled by an artificial dict
    # Inject end-state energy via shadowing key (start and end are both vertex 0
    # in a closed walk, but for this drift test we use a synthetic mapping).
    # Use distinct keys to make the test direct: build a per-step mapping.
    # The simplest direct drift test is via the per-step energy array used by
    # the loss functions; here we exercise walk_energy_drift indirectly.
    step_energies = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    drift = float(step_energies[0] - step_energies[-1])
    assert drift == -4.0
    assert drift * drift == 16.0
    # Round-trip via walk_energy_drift using a dict keyed by walk position
    # (vertex 0 maps to start energy; we use a sentinel "end" key).
    energies_by_state = {walk[0]: 1.0, -1: 5.0}
    walk_with_sentinel = [walk[0], -1]
    drift_via_helper = walk_energy_drift(walk_with_sentinel, energies_by_state)
    assert drift_via_helper == -4.0


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def test_monodromy_consistency_loss_zero_for_consistent_walks() -> None:
    """Three walks with matching endpoint energies -> loss = 0."""
    walks = [
        [0, 1, 2, 0],
        [3, 4, 3],
        [5, 6, 7, 8, 5],
    ]
    energies_by_walk = [
        np.array([1.0, 2.0, 3.0, 1.0]),
        np.array([0.5, 0.9, 0.5]),
        np.array([2.0, 3.0, 4.0, 5.0, 2.0]),
    ]
    loss = monodromy_consistency_loss(walks, energies_by_walk)
    assert loss == 0.0


def test_monodromy_consistency_loss_positive_for_drift() -> None:
    """A mix of consistent and drifting walks gives mean(squared_drift) > 0."""
    walks = [
        [0, 1, 2, 0],
        [3, 4, 3],
        [5, 6, 7, 8, 5],
    ]
    energies_by_walk = [
        np.array([1.0, 2.0, 3.0, 1.0]),  # drift 0 -> 0
        np.array([0.5, 0.9, 1.5]),       # drift -1.0 -> 1.0
        np.array([2.0, 3.0, 4.0, 5.0, 4.0]),  # drift -2.0 -> 4.0
    ]
    loss = monodromy_consistency_loss(walks, energies_by_walk)
    expected = (0.0 + 1.0 + 4.0) / 3.0
    assert loss > 0.0
    assert loss == expected


def test_consistency_loss_with_intermediate_drift_stricter() -> None:
    """The intermediate-drift loss dominates the endpoint-only loss on the same input."""
    walks = [
        [0, 1, 2, 0],
        [3, 4, 3],
        [5, 6, 7, 8, 5],
    ]
    energies_by_walk = [
        np.array([1.0, 2.0, 3.0, 1.0]),
        np.array([0.5, 0.9, 1.5]),
        np.array([2.0, 3.0, 4.0, 5.0, 4.0]),
    ]
    base_loss = monodromy_consistency_loss(walks, energies_by_walk)
    strict_loss = consistency_loss_with_intermediate_drift(walks, energies_by_walk)
    assert strict_loss >= base_loss
    # Concretely, energies wander even on the consistent walk, so strict > base.
    assert strict_loss > base_loss


def test_handles_empty_walks_list() -> None:
    """An empty walk list is vacuously consistent: both losses return 0.0."""
    assert monodromy_consistency_loss([], []) == 0.0
    assert consistency_loss_with_intermediate_drift([], []) == 0.0


def test_walk_with_length_1() -> None:
    """A self-loop walk [0, 0] with constant energies has zero drift and zero loss."""
    walk = [0, 0]
    energies = np.array([1.0, 1.0])
    drift = walk_energy_drift(walk, {0: 1.0})
    assert drift == 0.0
    base = monodromy_consistency_loss([walk], [energies])
    strict = consistency_loss_with_intermediate_drift([walk], [energies])
    assert base == 0.0
    assert strict == 0.0
