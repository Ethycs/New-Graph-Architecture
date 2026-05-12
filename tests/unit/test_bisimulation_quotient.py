"""Unit tests for nga.arch.bisimulation_quotient.quotient_by_bisimulation."""
from __future__ import annotations

import numpy as np
import pytest

from nga.arch.bisimulation_quotient import (
    MERGE_CRITERIA,
    QuotientResult,
    quotient_by_bisimulation,
)


def _trivial_two_state_transitions(K: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Build over-clustered counts: 2 underlying states, each split into 2
    sub-clusters whose outgoing transition distributions are identical
    WITHIN a state and distinct ACROSS states.

    Sub-A = {0, 1}: each sends 80% to sub-A members, 20% to sub-B.
    Sub-B = {2, 3}: each sends 20% to sub-A, 80% to sub-B.

    Sane bisimulation quotient should merge {0, 1} into one class and
    {2, 3} into another, recovering the 2-state structure.
    """
    counts = np.zeros((K, K), dtype=np.int64)
    # Sub-A clusters (0 and 1): high self-class density, low cross-class.
    for src in (0, 1):
        counts[src, 0] = 4
        counts[src, 1] = 4
        counts[src, 2] = 1
        counts[src, 3] = 1
    # Sub-B clusters (2 and 3): mirror image.
    for src in (2, 3):
        counts[src, 0] = 1
        counts[src, 1] = 1
        counts[src, 2] = 4
        counts[src, 3] = 4
    labels = np.concatenate(
        [np.full(5, k, dtype=np.int64) for k in range(K)]
    )
    return labels, counts


def test_quotient_returns_correct_type() -> None:
    labels, counts = _trivial_two_state_transitions()
    res = quotient_by_bisimulation(
        labels, counts, target_K=2, criterion="transition"
    )
    assert isinstance(res, QuotientResult)
    assert res.transition_counts.shape == (2, 2)
    assert res.labels.shape == labels.shape
    assert res.cluster_map.shape == (4,)
    assert res.criterion == "transition"


def test_quotient_recovers_2_state_structure_by_transition() -> None:
    """When sub-clusters of the same state have identical transition
    distributions, the transition-criterion quotient merges them."""
    labels, counts = _trivial_two_state_transitions(K=4)
    res = quotient_by_bisimulation(
        labels, counts, target_K=2, criterion="transition"
    )
    # The expected partition is {{0, 1}, {2, 3}} (sub-A together, sub-B
    # together). Equivalently, cluster_map[0] == cluster_map[1] and
    # cluster_map[2] == cluster_map[3].
    cm = res.cluster_map
    assert cm[0] == cm[1], f"sub-A clusters not merged; cluster_map={cm}"
    assert cm[2] == cm[3], f"sub-B clusters not merged; cluster_map={cm}"
    assert cm[0] != cm[2], f"sub-A and sub-B mistakenly merged; cluster_map={cm}"


def test_quotient_no_op_at_target_K_equals_K() -> None:
    """target_K = K is a no-op."""
    labels, counts = _trivial_two_state_transitions(K=4)
    res = quotient_by_bisimulation(
        labels, counts, target_K=4, criterion="transition"
    )
    assert res.transition_counts.shape == (4, 4)
    np.testing.assert_array_equal(res.labels, labels)
    np.testing.assert_array_equal(res.cluster_map, np.arange(4))
    assert res.merge_history == []


def test_quotient_target_K_1_collapses_everything() -> None:
    """target_K = 1 collapses all clusters into one."""
    labels, counts = _trivial_two_state_transitions(K=4)
    res = quotient_by_bisimulation(
        labels, counts, target_K=1, criterion="transition"
    )
    assert res.transition_counts.shape == (1, 1)
    # All labels become 0.
    assert (res.labels == 0).all()
    assert (res.cluster_map == 0).all()


def test_quotient_emission_criterion_requires_emission_counts() -> None:
    labels, counts = _trivial_two_state_transitions(K=4)
    with pytest.raises(ValueError):
        quotient_by_bisimulation(
            labels, counts, target_K=2, criterion="emission"
        )


def test_quotient_emission_criterion_merges_by_token_distribution() -> None:
    """When sub-clusters share an emission distribution, emission criterion
    merges them even if transitions are mixed."""
    K = 4
    counts = np.full((K, K), 1, dtype=np.int64)  # uniform transitions
    # Emission counts: clusters 0, 1 emit token 0; clusters 2, 3 emit token 1.
    em = np.array(
        [
            [10, 0],
            [10, 0],
            [0, 10],
            [0, 10],
        ],
        dtype=np.int64,
    )
    labels = np.arange(K, dtype=np.int64)  # one step per cluster
    res = quotient_by_bisimulation(
        labels, counts, target_K=2, criterion="emission", emission_counts=em
    )
    cm = res.cluster_map
    assert cm[0] == cm[1], f"emission-matched 0, 1 not merged; cm={cm}"
    assert cm[2] == cm[3], f"emission-matched 2, 3 not merged; cm={cm}"


def test_quotient_labels_remap_correctly() -> None:
    """After quotient, per-step labels are in [0, target_K) and consistent
    with cluster_map."""
    labels, counts = _trivial_two_state_transitions(K=4)
    res = quotient_by_bisimulation(
        labels, counts, target_K=2, criterion="transition"
    )
    assert res.labels.min() >= 0
    assert res.labels.max() < 2
    # Every step's label should equal cluster_map[original_label].
    for i in range(labels.size):
        assert res.labels[i] == res.cluster_map[labels[i]]


def test_quotient_transition_counts_conservation() -> None:
    """Total transitions are conserved through the merge."""
    labels, counts = _trivial_two_state_transitions(K=4)
    total_before = int(counts.sum())
    res = quotient_by_bisimulation(
        labels, counts, target_K=2, criterion="transition"
    )
    total_after = int(res.transition_counts.sum())
    assert total_after == total_before, (
        f"counts not conserved: {total_before} -> {total_after}"
    )


def test_quotient_invalid_target_K_raises() -> None:
    labels, counts = _trivial_two_state_transitions(K=4)
    with pytest.raises(ValueError):
        quotient_by_bisimulation(
            labels, counts, target_K=0, criterion="transition"
        )
    with pytest.raises(ValueError):
        quotient_by_bisimulation(
            labels, counts, target_K=5, criterion="transition"
        )


def test_quotient_unknown_criterion_raises() -> None:
    labels, counts = _trivial_two_state_transitions(K=4)
    with pytest.raises(ValueError):
        quotient_by_bisimulation(
            labels, counts, target_K=2, criterion="banana"
        )


def test_quotient_random_baseline_runs() -> None:
    """The random criterion is a no-op-of-meaning baseline; it must still
    produce a well-typed result."""
    labels, counts = _trivial_two_state_transitions(K=4)
    res = quotient_by_bisimulation(
        labels,
        counts,
        target_K=2,
        criterion="random",
        seed=0,
    )
    assert res.transition_counts.shape == (2, 2)
    assert res.labels.min() >= 0
    assert res.labels.max() < 2


def test_all_criteria_listed_in_merge_criteria_constant() -> None:
    """MERGE_CRITERIA enumerates every valid criterion."""
    for c in MERGE_CRITERIA:
        labels, counts = _trivial_two_state_transitions(K=4)
        em = np.full((4, 3), 1, dtype=np.int64) if c in {"emission", "full"} else None
        res = quotient_by_bisimulation(
            labels,
            counts,
            target_K=2,
            criterion=c,
            emission_counts=em,
        )
        assert res.transition_counts.shape == (2, 2)
