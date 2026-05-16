"""Unit tests for ``nga.arch.control_policy.ControlPolicy``.

Verifies the σ-thresholded routing decisions, BFS-based recovery action
selection, and threshold inclusivity convention.
"""
from __future__ import annotations

import numpy as np
import pytest

from nga.arch.control_policy import ControlPolicy, ControlPolicyResult
from nga.arch.graph_fsm import GraphFSM
from nga.drivers.graph_fsm_spec import (
    Coordinates,
    Edge,
    GraphFSMSpec,
    Vertex,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cyclic_fsm() -> GraphFSM:
    """Build a 4-state cyclic FSM 0->1->2->3->0 wrapped in a ``GraphFSM``."""
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


def _make_disconnected_fsm() -> GraphFSM:
    """Build a 4-state FSM with two disconnected components.

    Component A: 0 -> 1 (no path to 2 or 3).
    Component B: 2 -> 3.
    """
    vertices = [Vertex(id=f"s{i}", label=f"S{i}") for i in range(4)]
    edges = [
        Edge(source="s0", target="s1"),
        Edge(source="s2", target="s3"),
    ]
    spec = GraphFSMSpec(
        name="disconnected4",
        vertex_count=4,
        edge_count=2,
        vertices=vertices,
        edges=edges,
        coordinates=Coordinates(space="euclidean", dimension=1),
    )
    return GraphFSM(spec)


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_low_sigma_routes_normal() -> None:
    """σ=0.1 < θ_normal → ROUTE_NORMAL with the model's prediction."""
    policy = ControlPolicy(theta_normal=0.3, theta_abstain=0.7)
    result = policy.decide(sigma=0.1, model_prediction=7, current_state=0)
    assert isinstance(result, ControlPolicyResult)
    assert result.decision == "ROUTE_NORMAL"
    assert result.chosen_action == 7
    assert "0.1" in result.reason or "0.1000" in result.reason


def test_high_sigma_abstains() -> None:
    """σ=0.9 ≥ θ_abstain → ABSTAIN with sentinel action -1."""
    policy = ControlPolicy(theta_normal=0.3, theta_abstain=0.7)
    result = policy.decide(sigma=0.9, model_prediction=7, current_state=0)
    assert result.decision == "ABSTAIN"
    assert result.chosen_action == -1


def test_mid_sigma_routes_recovery_to_goal() -> None:
    """σ=0.5 in recovery band: BFS from 0 to goal=2 yields first step 1."""
    fsm = _make_cyclic_fsm()
    policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        fsm=fsm,
        goal_states=[2],
    )
    result = policy.decide(sigma=0.5, model_prediction=99, current_state=0)
    assert result.decision == "ROUTE_RECOVERY"
    assert result.chosen_action == 1


def test_mid_sigma_no_fsm_falls_back() -> None:
    """σ=0.5 with no FSM → ROUTE_NORMAL with reason mentioning 'unavailable'."""
    policy = ControlPolicy(theta_normal=0.3, theta_abstain=0.7)
    result = policy.decide(sigma=0.5, model_prediction=4, current_state=0)
    assert result.decision == "ROUTE_NORMAL"
    assert result.chosen_action == 4
    assert "unavailable" in result.reason


def test_mid_sigma_unreachable_goal_falls_back() -> None:
    """FSM with no path to goal → ROUTE_NORMAL with 'unavailable' reason."""
    fsm = _make_disconnected_fsm()
    # From state 0 there is no path to goal=3 (different component).
    policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        fsm=fsm,
        goal_states=[3],
    )
    result = policy.decide(sigma=0.5, model_prediction=42, current_state=0)
    assert result.decision == "ROUTE_NORMAL"
    assert result.chosen_action == 42
    assert "unavailable" in result.reason


def test_thresholds_inclusive_exclusive() -> None:
    """σ exactly at θ_normal enters the recovery band (inclusive lower bound).

    σ exactly at θ_abstain enters the abstain band (inclusive lower bound of
    abstain). Recovery falls through to ROUTE_NORMAL when no FSM is provided,
    but the reason must clearly attribute the fall-through to the recovery
    band (i.e. 'recovery_unavailable'), not to being below θ_normal.
    """
    policy = ControlPolicy(theta_normal=0.3, theta_abstain=0.7)

    # σ exactly at θ_normal → recovery band; with no FSM falls to NORMAL with
    # 'recovery_unavailable' reason (proves it entered recovery band, not normal).
    at_normal = policy.decide(sigma=0.3, model_prediction=5, current_state=0)
    assert at_normal.decision == "ROUTE_NORMAL"
    assert at_normal.chosen_action == 5
    assert "unavailable" in at_normal.reason

    # σ exactly at θ_abstain → ABSTAIN.
    at_abstain = policy.decide(sigma=0.7, model_prediction=5, current_state=0)
    assert at_abstain.decision == "ABSTAIN"
    assert at_abstain.chosen_action == -1


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


def test_current_state_none_in_recovery_band_falls_back() -> None:
    """current_state=None in recovery band forces ROUTE_NORMAL."""
    fsm = _make_cyclic_fsm()
    policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        fsm=fsm,
        goal_states=[2],
    )
    result = policy.decide(sigma=0.5, model_prediction=11, current_state=None)
    assert result.decision == "ROUTE_NORMAL"
    assert result.chosen_action == 11
    assert "unavailable" in result.reason


def test_invalid_thresholds_rejected() -> None:
    """θ_normal > θ_abstain is rejected at construction time."""
    with pytest.raises(ValueError):
        ControlPolicy(theta_normal=0.8, theta_abstain=0.3)
    with pytest.raises(ValueError):
        ControlPolicy(theta_normal=-0.1, theta_abstain=0.5)
    with pytest.raises(ValueError):
        ControlPolicy(theta_normal=0.2, theta_abstain=1.5)


def test_bfs_picks_nearest_goal() -> None:
    """When multiple goals are given, BFS picks the closest one."""
    fsm = _make_cyclic_fsm()
    # From state 0, goal 1 is 1 step away and goal 3 is 3 steps away.
    policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        fsm=fsm,
        goal_states=[3, 1],
    )
    result = policy.decide(sigma=0.5, model_prediction=0, current_state=0)
    assert result.decision == "ROUTE_RECOVERY"
    # First step toward closest goal (1) is 1 itself.
    assert result.chosen_action == 1


def test_already_at_goal_falls_back() -> None:
    """If current_state is already a goal, no next step exists; fall back."""
    fsm = _make_cyclic_fsm()
    policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        fsm=fsm,
        goal_states=[0],
    )
    result = policy.decide(sigma=0.5, model_prediction=77, current_state=0)
    assert result.decision == "ROUTE_NORMAL"
    assert result.chosen_action == 77
    assert "unavailable" in result.reason


# ---------------------------------------------------------------------------
# legality_matrix path (substrate-agnostic adjacency, e.g. PCG-X regime graphs)
# ---------------------------------------------------------------------------


def test_legality_matrix_recovery_bfs_first_step() -> None:
    """Raw adjacency drives the same BFS as a GraphFSM would.

    Adjacency for a 4-node cycle 0->1->2->3->0; BFS from 0 to {2} picks 1.
    """
    legality = np.zeros((4, 4), dtype=bool)
    legality[0, 1] = True
    legality[1, 2] = True
    legality[2, 3] = True
    legality[3, 0] = True
    policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        legality_matrix=legality,
        goal_states=[2],
    )
    result = policy.decide(sigma=0.5, model_prediction=99, current_state=0)
    assert result.decision == "ROUTE_RECOVERY"
    assert result.chosen_action == 1


def test_legality_matrix_unreachable_goal_falls_back() -> None:
    """Disconnected adjacency: no path from 0 to {3} → recovery_unavailable."""
    legality = np.zeros((4, 4), dtype=bool)
    legality[0, 1] = True
    legality[2, 3] = True  # 0,1 and 2,3 are disconnected components.
    policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        legality_matrix=legality,
        goal_states=[3],
    )
    result = policy.decide(sigma=0.5, model_prediction=42, current_state=0)
    assert result.decision == "ROUTE_NORMAL"
    assert result.chosen_action == 42
    assert "unavailable" in result.reason


def test_legality_matrix_fsm_mutually_exclusive() -> None:
    """Passing both fsm and legality_matrix is rejected at construction time."""
    fsm = _make_cyclic_fsm()
    legality = np.eye(4, dtype=bool)
    with pytest.raises(ValueError, match="not both"):
        ControlPolicy(
            theta_normal=0.3,
            theta_abstain=0.7,
            fsm=fsm,
            legality_matrix=legality,
            goal_states=[1],
        )


def test_legality_matrix_must_be_square() -> None:
    """Non-square legality_matrix is rejected with a clear error."""
    bad = np.zeros((3, 5), dtype=bool)
    with pytest.raises(ValueError, match="square"):
        ControlPolicy(
            theta_normal=0.3,
            theta_abstain=0.7,
            legality_matrix=bad,
            goal_states=[1],
        )


def test_legality_matrix_equivalent_to_fsm_path() -> None:
    """For the same adjacency, FSM path and raw-matrix path produce the same decision."""
    fsm = _make_cyclic_fsm()
    legality = np.asarray(fsm.legality_matrix, dtype=bool)
    fsm_policy = ControlPolicy(
        theta_normal=0.3, theta_abstain=0.7, fsm=fsm, goal_states=[2]
    )
    raw_policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        legality_matrix=legality,
        goal_states=[2],
    )
    fsm_result = fsm_policy.decide(sigma=0.5, model_prediction=0, current_state=0)
    raw_result = raw_policy.decide(sigma=0.5, model_prediction=0, current_state=0)
    assert fsm_result.decision == raw_result.decision
    assert fsm_result.chosen_action == raw_result.chosen_action
