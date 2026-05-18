"""Unit tests for the policy-intent dataset adapter (Phase 28b).

Covers:
  * v1 and v2 FSM YAMLs load and have the expected vertex/edge counts.
  * Generated samples have correct shapes (X, y_next, prev_states) and
    aligned per-row metadata (is_adversarial, anchor_present, sequence_id).
  * The reducer (in `dataset_policy_intent._reducer`) is consistent with
    the YAML FSM edges: every accepted step's (prev_state, observed_token,
    true_next_state) triple matches a YAML edge.
  * Rejected steps (authority gate REJECT) do not flip state.
  * Anchor-present is True iff current_state in {WITHHELD (v1)} or
    {WITHHELD, CONDITIONAL (v2)}.
  * Authority modes are None for v1 and one of 4 fixed strings for v2.
  * The GRAMMAR_DISPATCH wiring round-trips both versions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nga.arch.graph_fsm import GraphFSM
from nga.drivers import graph_fsm_spec
from nga.exp.dataset_policy_intent import (
    ANCHOR_STATES,
    AUTHORITY_MODES_V2,
    POLICY_V1_TOKENS,
    POLICY_V2_TOKENS,
    PolicyDataset,
    PolicySample,
    alphabet_for,
    generate_policy_intent_dataset,
    transition_table,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_fsm(version: str) -> GraphFSM:
    path = REPO_ROOT / "tests" / "fixtures" / "graphs" / f"policy_intent_{version}.fsm.yaml"
    return GraphFSM(graph_fsm_spec.load(path))


# ---------------------------------------------------------------------------
# FSM YAML structure
# ---------------------------------------------------------------------------


def test_v1_fsm_yaml_has_2_states_6_edges() -> None:
    spec = graph_fsm_spec.load(
        REPO_ROOT / "tests" / "fixtures" / "graphs" / "policy_intent_v1.fsm.yaml"
    )
    assert spec.vertex_count == 2
    assert spec.edge_count == 6
    assert {v.id for v in spec.vertices} == {"ALLOWED", "WITHHELD"}


def test_v2_fsm_yaml_has_4_states_20_edges() -> None:
    spec = graph_fsm_spec.load(
        REPO_ROOT / "tests" / "fixtures" / "graphs" / "policy_intent_v2.fsm.yaml"
    )
    assert spec.vertex_count == 4
    assert spec.edge_count == 20
    assert {v.id for v in spec.vertices} == {
        "UNESTABLISHED",
        "GRANTED",
        "WITHHELD",
        "CONDITIONAL",
    }


# ---------------------------------------------------------------------------
# Dataset shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version,expected_feature_dim", [("v1", 3), ("v2", 5)])
def test_dataset_shape_matches_expected(version: str, expected_feature_dim: int) -> None:
    fsm = _load_fsm(version)
    ds = generate_policy_intent_dataset(fsm, n_policies=10, seed=42, version=version)
    n = len(ds.samples)
    assert n > 0
    assert ds.feature_dim == expected_feature_dim
    assert ds.X.shape == (n, expected_feature_dim)
    assert ds.y_next.shape == (n,)
    assert ds.prev_states.shape == (n,)
    assert ds.current_states.shape == (n,)
    assert ds.is_adversarial.shape == (n,)
    assert ds.anchor_present.shape == (n,)
    assert ds.authority_mode_ids.shape == (n,)
    assert ds.fsm is fsm
    # Sequence ids span 0..n_policies-1
    sids = sorted({s.sequence_id for s in ds.samples})
    assert sids == list(range(10))


# ---------------------------------------------------------------------------
# Reducer consistency with FSM YAML
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_accepted_transitions_match_yaml_edges(version: str) -> None:
    spec = graph_fsm_spec.load(
        REPO_ROOT / "tests" / "fixtures" / "graphs" / f"policy_intent_{version}.fsm.yaml"
    )
    fsm = GraphFSM(spec)
    ds = generate_policy_intent_dataset(fsm, n_policies=40, seed=42, version=version)
    table = transition_table(version)
    # Every accepted sample must agree with the reducer's table, and the
    # (prev_state, observed_token, true_next_state) must correspond to an
    # edge in the YAML FSM.
    edge_set = {(e.source, e.label, e.target) for e in spec.edges}
    n_accepted = 0
    for s in ds.samples:
        if s.is_adversarial:
            continue
        n_accepted += 1
        assert s.true_next_state == table[(s.prev_state, s.observed_token)]
        triple = (s.prev_state, s.observed_token, s.true_next_state)
        assert triple in edge_set, f"sample not in FSM edges: {triple}"
    assert n_accepted > 0


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_rejected_steps_do_not_flip_state(version: str) -> None:
    fsm = _load_fsm(version)
    ds = generate_policy_intent_dataset(
        fsm, n_policies=40, seed=42, version=version, reject_rate=0.5
    )
    rejected = [s for s in ds.samples if s.is_adversarial]
    assert len(rejected) > 0, "expected some rejected steps at reject_rate=0.5"
    for s in rejected:
        assert s.true_next_state == s.prev_state, (
            f"rejected step should not flip: {s.prev_state} + {s.observed_token} -> "
            f"{s.true_next_state} (expected stay at {s.prev_state})"
        )


# ---------------------------------------------------------------------------
# Anchor presence semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_anchor_present_matches_state_table(version: str) -> None:
    fsm = _load_fsm(version)
    ds = generate_policy_intent_dataset(fsm, n_policies=20, seed=42, version=version)
    anchor_states = ANCHOR_STATES[version]
    for s in ds.samples:
        expected = s.current_state in anchor_states
        assert s.anchor_present == expected, (
            f"anchor mismatch on state {s.current_state} (version={version}): "
            f"expected {expected}, got {s.anchor_present}"
        )


# ---------------------------------------------------------------------------
# Authority modes
# ---------------------------------------------------------------------------


def test_v1_authority_modes_are_none() -> None:
    fsm = _load_fsm("v1")
    ds = generate_policy_intent_dataset(fsm, n_policies=10, seed=42, version="v1")
    assert all(s.authority_mode is None for s in ds.samples)
    assert np.all(ds.authority_mode_ids == -1)


def test_v2_authority_modes_cover_all_four_eventually() -> None:
    fsm = _load_fsm("v2")
    # 80 policies with uniform sampling -- all 4 modes should appear.
    ds = generate_policy_intent_dataset(fsm, n_policies=80, seed=42, version="v2")
    modes_seen = {s.authority_mode for s in ds.samples}
    assert modes_seen == set(AUTHORITY_MODES_V2)
    # Authority mode is fixed per policy (one mode per sequence_id).
    by_seq: dict[int, set[str]] = {}
    for s in ds.samples:
        by_seq.setdefault(s.sequence_id, set()).add(s.authority_mode)
    assert all(len(modes) == 1 for modes in by_seq.values()), (
        "authority_mode should be fixed per (source, target, domain) instance"
    )


# ---------------------------------------------------------------------------
# Alphabet helpers
# ---------------------------------------------------------------------------


def test_alphabet_for_v1_and_v2() -> None:
    assert alphabet_for("v1") == POLICY_V1_TOKENS == ("GRANT_EVENT", "REVOKE_EVENT", "NULL")
    assert alphabet_for("v2") == POLICY_V2_TOKENS == (
        "GRANT", "REVOKE", "CONDITION", "REQUEST", "ACKNOWLEDGE"
    )


def test_alphabet_for_unknown_version_raises() -> None:
    with pytest.raises(ValueError):
        alphabet_for("v3")


def test_generate_unknown_version_raises() -> None:
    fsm = _load_fsm("v1")
    with pytest.raises(ValueError):
        generate_policy_intent_dataset(fsm, n_policies=2, seed=42, version="v3")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_dataset_is_deterministic_under_same_seed(version: str) -> None:
    fsm = _load_fsm(version)
    a = generate_policy_intent_dataset(fsm, n_policies=10, seed=42, version=version)
    b = generate_policy_intent_dataset(fsm, n_policies=10, seed=42, version=version)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y_next, b.y_next)
    np.testing.assert_array_equal(a.prev_states, b.prev_states)
    np.testing.assert_array_equal(a.is_adversarial, b.is_adversarial)
    np.testing.assert_array_equal(a.anchor_present, b.anchor_present)
    np.testing.assert_array_equal(a.authority_mode_ids, b.authority_mode_ids)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_different_seeds_produce_different_datasets(version: str) -> None:
    fsm = _load_fsm(version)
    a = generate_policy_intent_dataset(fsm, n_policies=10, seed=42, version=version)
    b = generate_policy_intent_dataset(fsm, n_policies=10, seed=43, version=version)
    assert not np.array_equal(a.X, b.X)


# ---------------------------------------------------------------------------
# Dispatch wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["policy_intent_v1", "policy_intent_v2"])
def test_grammar_dispatch_round_trip(name: str) -> None:
    from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH

    assert name in GRAMMAR_DISPATCH
    d = GRAMMAR_DISPATCH[name]
    fsm = GraphFSM(graph_fsm_spec.load(REPO_ROOT / d["fsm_yaml_path"]))
    ds = d["loader"](fsm, d["default_n"], 42)
    assert isinstance(ds, PolicyDataset)
    # The runner reads sequence ids via this attr name.
    seq_attr = d["sequence_id_attr"]
    sids = sorted({int(getattr(s, seq_attr)) for s in ds.samples})
    assert sids == list(range(d["default_n"]))


# ---------------------------------------------------------------------------
# Token weights override
# ---------------------------------------------------------------------------


def test_token_weights_override_skews_distribution() -> None:
    fsm = _load_fsm("v1")
    # Heavy NULL bias -- should produce many idempotent self-loops and few flips.
    ds = generate_policy_intent_dataset(
        fsm,
        n_policies=40,
        seed=42,
        version="v1",
        token_weights={"GRANT_EVENT": 0.05, "REVOKE_EVENT": 0.05, "NULL": 0.90},
        reject_rate=0.0,  # no rejects -- isolate the token-weight effect
    )
    counts = {t: 0 for t in POLICY_V1_TOKENS}
    for s in ds.samples:
        counts[s.observed_token] += 1
    total = sum(counts.values())
    assert counts["NULL"] / total > 0.80, f"NULL bias not applied: {counts}"


def test_token_weights_zero_sum_raises() -> None:
    fsm = _load_fsm("v1")
    with pytest.raises(ValueError):
        generate_policy_intent_dataset(
            fsm,
            n_policies=2,
            seed=42,
            version="v1",
            token_weights={"GRANT_EVENT": 0.0, "REVOKE_EVENT": 0.0, "NULL": 0.0},
        )
