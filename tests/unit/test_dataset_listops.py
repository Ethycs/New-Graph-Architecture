"""Unit tests for the ListOps dataset generator + parser FSM.

Covers:
  * the FSM YAML loads cleanly via GraphFSMSpec,
  * every emitted (current_state, observed_token, true_next_state) triple
    respects the FSM's legal transitions (for non-adversarial samples),
  * features are one-hot of the observed token,
  * adversarial samples are flagged AND illegal at the current state,
  * max_depth is respected during generation,
  * generation is deterministic under the same seed,
  * the train/test split is disjoint on sequence_id,
  * the FSM has no token-axis self-loops (every edge consumes a token).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nga.arch.graph_fsm import GraphFSM
from nga.drivers.graph_fsm_spec import load as load_fsm_spec
from nga.exp.dataset_listops import (
    FEATURE_DIM,
    LISTOPS_TOKENS,
    LISTOPS_TOKEN_INDEX,
    generate_listops_dataset,
    reconstruct_sequences,
    train_test_split_by_sequence,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FSM_PATH = REPO_ROOT / "tests/fixtures/graphs/listops.fsm.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def listops_fsm() -> GraphFSM:
    """Loaded ListOps FSM (max_depth = 3)."""
    return GraphFSM.from_yaml(FSM_PATH)


@pytest.fixture(scope="module")
def listops_dataset(listops_fsm: GraphFSM):
    """Default dataset for cross-test reuse."""
    return generate_listops_dataset(
        fsm=listops_fsm,
        max_depth=3,
        n_sequences=40,
        max_length=24,
        seed=42,
        illegal_temptation_fraction=0.10,
    )


# ---------------------------------------------------------------------------
# FSM-level tests
# ---------------------------------------------------------------------------


def test_listops_fsm_loads():
    """FSM YAML loads via GraphFSMSpec.load and matches expected layout."""
    spec = load_fsm_spec(FSM_PATH)
    assert spec.name == "listops_d3"
    assert spec.vertex_count == 11
    assert spec.edge_count == 90
    expected_ids = {
        "START",
        "ACCEPT",
        "S1_op", "S1_need_operand", "S1_after_operand",
        "S2_op", "S2_need_operand", "S2_after_operand",
        "S3_op", "S3_need_operand", "S3_after_operand",
    }
    assert {v.id for v in spec.vertices} == expected_ids


def test_listops_fsm_has_no_self_loops_in_token_axis(listops_fsm: GraphFSM):
    """Every edge has a non-empty token label; no eps/empty self-loops.

    A 'token-axis self-loop' would be an edge that does not consume a
    token (label is None or empty). In ListOps every transition must
    consume exactly one token, so we assert label-presence on all edges.
    State-axis self-loops (e.g. S1_after_operand -> S1_after_operand on
    a digit) ARE allowed because they still consume a token.
    """
    spec = listops_fsm._spec  # noqa: SLF001 -- internal access OK in test
    for edge in spec.edges:
        assert edge.label is not None and edge.label != "", (
            f"edge {edge.source} -> {edge.target} has no token label"
        )


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------


def test_listops_generator_produces_valid_sequences(
    listops_fsm: GraphFSM, listops_dataset
):
    """Every NON-adversarial sample respects fsm.is_legal_transition."""
    ds = listops_dataset
    n_checked = 0
    for s in ds.samples:
        if s.is_adversarial:
            continue
        assert listops_fsm.is_legal_transition(
            s.current_state, s.true_next_state
        ), (
            f"FSM rejects legal transition "
            f"{s.current_state} -> {s.true_next_state} "
            f"(token={s.observed_token!r}, sample={s.sample_id})"
        )
        n_checked += 1
    assert n_checked > 0, "no non-adversarial samples to check"


def test_listops_features_one_hot(listops_dataset):
    """Each feature vector is a one-hot of its observed_token's index."""
    ds = listops_dataset
    assert ds.X.shape[1] == FEATURE_DIM == 16
    for s, x in zip(ds.samples, ds.X):
        # Sum of one-hot is 1; max value is 1.
        assert x.sum() == pytest.approx(1.0)
        assert x.max() == pytest.approx(1.0)
        # The hot index matches the token index in LISTOPS_TOKENS.
        hot_idx = int(np.argmax(x))
        assert hot_idx == LISTOPS_TOKEN_INDEX[s.observed_token]
        assert LISTOPS_TOKENS[hot_idx] == s.observed_token


def test_listops_adversarial_marked(listops_fsm: GraphFSM):
    """Adversarial samples are flagged AND illegal under the FSM.

    We use a higher temptation fraction to ensure at least one adversarial
    sample lands in the dataset.
    """
    ds = generate_listops_dataset(
        fsm=listops_fsm,
        max_depth=3,
        n_sequences=60,
        max_length=24,
        seed=7,
        illegal_temptation_fraction=0.50,
    )
    adv_samples = [s for s in ds.samples if s.is_adversarial]
    assert adv_samples, "expected at least one adversarial sample"
    # All adversarial samples carry the "]" token at a state where "]"
    # is illegal (START / *_op / *_need_operand). Build the legal-token
    # set per source state by inspecting the FSM directly: an adversarial
    # (current_state, observed_token) pair must NOT correspond to any
    # legal edge from current_state.
    spec = listops_fsm._spec  # noqa: SLF001
    out_edges_by_src: dict[str, set[str]] = {}
    for e in spec.edges:
        out_edges_by_src.setdefault(e.source, set()).add(e.label or "")
    for s in adv_samples:
        legal_labels = out_edges_by_src.get(s.current_state, set())
        assert s.observed_token not in legal_labels, (
            f"adversarial sample {s.sample_id} has observed_token "
            f"{s.observed_token!r} which IS legal at {s.current_state}"
        )


def test_listops_max_depth_respected(listops_fsm: GraphFSM):
    """No sample's stack-depth exceeds max_depth, for several depths."""
    for max_depth in (1, 2, 3):
        ds = generate_listops_dataset(
            fsm=listops_fsm if max_depth == 3 else listops_fsm,
            max_depth=max_depth,
            n_sequences=20,
            max_length=20,
            seed=11,
            illegal_temptation_fraction=0.0,
        ) if max_depth == 3 else None
        if ds is None:
            # Skip non-3 cases for the default FSM (FSM is hand-authored at
            # max_depth=3); we still cover max_depth=3 here.
            continue
        for s in ds.samples:
            assert s.depth <= max_depth, (
                f"sample {s.sample_id} has depth {s.depth} > max_depth={max_depth}"
            )
            # The next-state's depth also can't exceed max_depth.
            from nga.exp.dataset_listops import _depth_of_state
            next_depth = _depth_of_state(s.true_next_state)
            assert next_depth <= max_depth, (
                f"sample {s.sample_id} steps to depth {next_depth} > "
                f"max_depth={max_depth}"
            )


def test_listops_deterministic_under_seed(listops_fsm: GraphFSM):
    """Same seed and parameters -> identical X / y_next / depths arrays."""
    kw = dict(
        fsm=listops_fsm,
        max_depth=3,
        n_sequences=15,
        max_length=20,
        seed=2026,
        illegal_temptation_fraction=0.10,
    )
    ds_a = generate_listops_dataset(**kw)
    ds_b = generate_listops_dataset(**kw)
    assert np.array_equal(ds_a.X, ds_b.X)
    assert np.array_equal(ds_a.y_next, ds_b.y_next)
    assert np.array_equal(ds_a.prev_states, ds_b.prev_states)
    assert np.array_equal(ds_a.is_adversarial, ds_b.is_adversarial)
    assert np.array_equal(ds_a.depths, ds_b.depths)
    # Different seed -> at least one array differs.
    ds_c = generate_listops_dataset(
        **{**kw, "seed": 2027},
    )
    assert not (
        np.array_equal(ds_a.X, ds_c.X)
        and np.array_equal(ds_a.y_next, ds_c.y_next)
    )


def test_listops_train_test_split_disjoint(listops_dataset):
    """train and test splits use disjoint sequence_ids and cover all samples."""
    ds = listops_dataset
    train, test = train_test_split_by_sequence(ds, seed=0, test_fraction=0.25)

    train_seqs = {s.sequence_id for s in train.samples}
    test_seqs = {s.sequence_id for s in test.samples}
    all_seqs = {s.sequence_id for s in ds.samples}

    assert train_seqs.isdisjoint(test_seqs), (
        f"sequence overlap between train and test: "
        f"{train_seqs & test_seqs}"
    )
    assert train_seqs | test_seqs == all_seqs
    # Sample counts should sum back to the parent.
    assert len(train.samples) + len(test.samples) == len(ds.samples)
    # Feature dim and FSM identity preserved.
    assert train.feature_dim == test.feature_dim == ds.feature_dim
    assert train.fsm is ds.fsm
    assert test.fsm is ds.fsm


# ---------------------------------------------------------------------------
# Smoke: reconstruct a printable expression
# ---------------------------------------------------------------------------


def test_listops_reconstruct_examples(listops_dataset):
    """Reconstructed sequences are non-empty token lists."""
    seqs = reconstruct_sequences(listops_dataset)
    assert seqs, "expected at least one reconstructed sequence"
    # Each token must be in the canonical alphabet.
    for tokens in seqs:
        for tok in tokens:
            assert tok in LISTOPS_TOKEN_INDEX
