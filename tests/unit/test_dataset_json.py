"""Unit tests for the JSON dataset generator + parser FSM.

Covers:
  * the source generator emits ONLY valid JSON (round-trips via json.loads),
  * the tokenizer's emitted alphabet is a subset of JSON_TOKENS,
  * walk_fsm_json produces only legal FSM transitions per step,
  * features are one-hot of the observed token's index,
  * adversarial samples are flagged AND are illegal at the current state,
  * generation is deterministic under the same seed,
  * train/test split is disjoint on document_id,
  * the FSM YAML loads cleanly and has between 20 and 30 vertices,
  * at least one "expecting value" state has 7+ outgoing edges
    (the load-bearing sigma-ambiguity site),
  * with default params, both objects and arrays appear in the corpus.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nga.arch.graph_fsm import GraphFSM
from nga.drivers.graph_fsm_spec import load as load_fsm_spec
from nga.exp.dataset_json import (
    FEATURE_DIM,
    JSON_TOKEN_INDEX,
    JSON_TOKENS,
    generate_json_dataset,
    generate_json_source,
    tokenize_json,
    train_test_split_by_document,
    walk_fsm_json,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FSM_PATH = REPO_ROOT / "tests/fixtures/graphs/json.fsm.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def json_fsm() -> GraphFSM:
    """Loaded JSON FSM (max_depth = 3 baked into the YAML)."""
    return GraphFSM.from_yaml(FSM_PATH)


@pytest.fixture(scope="module")
def json_dataset(json_fsm: GraphFSM):
    """Default dataset for cross-test reuse."""
    return generate_json_dataset(
        fsm=json_fsm,
        n_documents=80,
        max_depth=3,
        p_object=0.5,
        seed=42,
        illegal_temptation_fraction=0.10,
    )


# ---------------------------------------------------------------------------
# FSM-level tests
# ---------------------------------------------------------------------------


def test_json_fsm_loads_and_has_expected_vertex_count():
    """FSM YAML loads via GraphFSMSpec.load and has 20..30 vertices."""
    spec = load_fsm_spec(FSM_PATH)
    assert spec.name == "json_d3"
    assert 20 <= spec.vertex_count <= 30, (
        f"vertex_count {spec.vertex_count} outside expected [20, 30]"
    )
    assert spec.edge_count == len(spec.edges)
    # Sanity-check the canonical JSON state set is present.
    ids = {v.id for v in spec.vertices}
    assert "START" in ids
    assert "ACCEPT" in ids
    for d in (1, 2, 3):
        for suffix in (
            "obj_key",
            "obj_after_key",
            "obj_after_colon",
            "obj_after_value",
            "obj_after_comma",
            "arr_first_value",
            "arr_after_value",
            "arr_after_comma",
        ):
            assert f"S{d}_{suffix}" in ids, (
                f"S{d}_{suffix} missing from FSM vertex set"
            )


def test_json_fsm_value_states_have_seven_outgoing(json_fsm: GraphFSM):
    """At least one "expecting value" state has >= 7 outgoing edges.

    The load-bearing sigma-ambiguity prediction: at JSON's "expecting
    value" position, the parser must pick between 7 legal continuations
    (5 primitives + ``{`` + ``[``). This test asserts that the FSM YAML
    does encode that 7-way fanout for at least one such state.
    """
    spec = json_fsm._spec  # noqa: SLF001 -- internal access OK in test
    out_counts: dict[str, int] = {}
    for e in spec.edges:
        out_counts[e.source] = out_counts.get(e.source, 0) + 1

    # Every "expecting value" state. START and the *_after_colon /
    # *_first_value / *_after_comma states.
    value_states = ["START"]
    for d in (1, 2, 3):
        value_states.extend(
            [
                f"S{d}_obj_after_colon",
                f"S{d}_arr_first_value",
                f"S{d}_arr_after_comma",
            ]
        )
    seven_plus = [s for s in value_states if out_counts.get(s, 0) >= 7]
    assert seven_plus, (
        f"expected at least one 'expecting value' state with >= 7 "
        f"outgoing edges; out_counts={out_counts!r}"
    )


# ---------------------------------------------------------------------------
# Source-generator tests
# ---------------------------------------------------------------------------


def test_json_generator_produces_valid_json():
    """Every generated document parses via stdlib json.loads.

    External-benchmark contract: the source the FSM walks is REAL JSON,
    accepted by the real RFC 8259 parser.
    """
    docs = generate_json_source(
        n_documents=60, max_depth=3, p_object=0.5, seed=2026
    )
    assert len(docs) == 60
    for src in docs:
        # json.loads raises JSONDecodeError on invalid input.
        json.loads(src)


def test_json_dataset_has_objects_and_arrays():
    """With default params, both objects and arrays appear in the corpus."""
    docs = generate_json_source(
        n_documents=80, max_depth=3, p_object=0.5, seed=11
    )
    has_object = any(d.lstrip().startswith("{") for d in docs)
    has_array = any(d.lstrip().startswith("[") for d in docs)
    assert has_object, "expected at least one object document in the corpus"
    assert has_array, "expected at least one array document in the corpus"


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


def test_json_tokenizer_emits_known_alphabet():
    """Every token from the tokenizer is a member of JSON_TOKENS.

    Hits a few hand-picked source strings spanning the alphabet, plus a
    sweep over generated documents.
    """
    handcrafted = [
        '"a"',
        "42",
        "true",
        "false",
        "null",
        "[1, 2, 3]",
        '{"a": 1, "b": [true, null]}',
        "{}",
        "[]",
        '{"x": {"y": [-3, 0]}}',
    ]
    for src in handcrafted:
        for token in tokenize_json(src):
            assert token in JSON_TOKEN_INDEX, (
                f"unknown token {token!r} from source {src!r}"
            )

    # Sweep generated documents.
    docs = generate_json_source(
        n_documents=30, max_depth=3, p_object=0.5, seed=17
    )
    for src in docs:
        for token in tokenize_json(src):
            assert token in JSON_TOKEN_INDEX


# ---------------------------------------------------------------------------
# FSM walker tests
# ---------------------------------------------------------------------------


def test_json_walk_fsm_legal_per_step(json_fsm: GraphFSM):
    """Every transition produced by walk_fsm_json is legal under the FSM."""
    docs = generate_json_source(
        n_documents=40, max_depth=3, p_object=0.5, seed=5
    )
    for src in docs:
        toks = tokenize_json(src)
        trace = walk_fsm_json(toks, json_fsm)
        assert trace, f"no transitions produced for source {src!r}"
        for prev, _tok, nxt in trace:
            assert json_fsm.is_legal_transition(prev, nxt), (
                f"illegal transition {prev!r} -> {nxt!r} for source {src!r}"
            )
        # The walk must terminate in ACCEPT.
        assert trace[-1][2] == "ACCEPT", (
            f"walk did not terminate in ACCEPT for source {src!r}; "
            f"final state was {trace[-1][2]!r}"
        )


# ---------------------------------------------------------------------------
# Dataset feature tests
# ---------------------------------------------------------------------------


def test_json_dataset_features_one_hot(json_dataset):
    """Each feature vector is a one-hot of its observed_token's index."""
    ds = json_dataset
    assert ds.X.shape[1] == FEATURE_DIM == 16
    for s, x in zip(ds.samples, ds.X):
        # Sum of one-hot is 1; max value is 1.
        assert x.sum() == pytest.approx(1.0)
        assert x.max() == pytest.approx(1.0)
        # Hot index matches the token's index in JSON_TOKENS.
        hot_idx = int(np.argmax(x))
        assert hot_idx == JSON_TOKEN_INDEX[s.observed_token]
        assert JSON_TOKENS[hot_idx] == s.observed_token


def test_json_adversarial_marked(json_fsm: GraphFSM):
    """Adversarial samples are flagged AND illegal under the FSM.

    Use a higher temptation fraction to ensure at least one adversarial
    sample lands in the dataset.
    """
    ds = generate_json_dataset(
        fsm=json_fsm,
        n_documents=40,
        max_depth=3,
        p_object=0.5,
        seed=7,
        illegal_temptation_fraction=0.50,
    )
    adv_samples = [s for s in ds.samples if s.is_adversarial]
    assert adv_samples, "expected at least one adversarial sample"

    # Build legal-token set per state by inspecting FSM edges directly.
    spec = json_fsm._spec  # noqa: SLF001 -- internal access OK in test
    out_edges_by_src: dict[str, set[str]] = {}
    for e in spec.edges:
        out_edges_by_src.setdefault(e.source, set()).add(e.label or "")

    for s in adv_samples:
        legal_labels = out_edges_by_src.get(s.current_state, set())
        assert s.observed_token not in legal_labels, (
            f"adversarial sample {s.sample_id} has observed_token "
            f"{s.observed_token!r} which IS legal at {s.current_state}"
        )
    # Non-adversarial samples: their observed_token MUST be legal.
    for s in ds.samples:
        if not s.is_adversarial:
            legal_labels = out_edges_by_src.get(s.current_state, set())
            assert s.observed_token in legal_labels, (
                f"non-adversarial sample {s.sample_id} has observed_token "
                f"{s.observed_token!r} which is ILLEGAL at "
                f"{s.current_state} (legal: {sorted(legal_labels)})"
            )


def test_json_deterministic_under_seed(json_fsm: GraphFSM):
    """Same seed and parameters -> identical X / y_next / depth arrays."""
    kw = dict(
        fsm=json_fsm,
        n_documents=20,
        max_depth=3,
        p_object=0.5,
        seed=2026,
        illegal_temptation_fraction=0.10,
    )
    ds_a = generate_json_dataset(**kw)
    ds_b = generate_json_dataset(**kw)
    assert np.array_equal(ds_a.X, ds_b.X)
    assert np.array_equal(ds_a.y_next, ds_b.y_next)
    assert np.array_equal(ds_a.prev_states, ds_b.prev_states)
    assert np.array_equal(ds_a.is_adversarial, ds_b.is_adversarial)
    assert np.array_equal(ds_a.depths, ds_b.depths)
    assert ds_a.documents == ds_b.documents

    # Different seed -> at least one array differs (almost always).
    ds_c = generate_json_dataset(**{**kw, "seed": 2027})
    assert not (
        np.array_equal(ds_a.X, ds_c.X)
        and np.array_equal(ds_a.y_next, ds_c.y_next)
        and ds_a.documents == ds_c.documents
    )


def test_json_train_test_split_disjoint(json_dataset):
    """train / test splits use disjoint document_ids and cover all samples."""
    ds = json_dataset
    train, test = train_test_split_by_document(ds, seed=0, test_fraction=0.25)

    train_docs = {s.document_id for s in train.samples}
    test_docs = {s.document_id for s in test.samples}
    all_docs = {s.document_id for s in ds.samples}

    assert train_docs.isdisjoint(test_docs), (
        f"document overlap between train and test: "
        f"{train_docs & test_docs}"
    )
    assert train_docs | test_docs == all_docs
    assert len(train.samples) + len(test.samples) == len(ds.samples)
    assert train.feature_dim == test.feature_dim == ds.feature_dim
    assert train.fsm is ds.fsm
    assert test.fsm is ds.fsm
