"""Unit tests for the python_expr dataset generator + parser FSM.

Covers:
  * the source generator emits ONLY valid Python (round-trips via ast.parse),
  * the tokenizer's emitted alphabet is a subset of PYTHON_EXPR_TOKENS,
  * walk_fsm produces only legal FSM transitions per step,
  * features are one-hot of the observed token's index,
  * adversarial samples are flagged AND are illegal at the current state,
  * generation is deterministic under the same seed,
  * train/test split is disjoint on program_id,
  * the FSM YAML loads cleanly and has the expected vertex count,
  * max_depth is respected (no parser depth exceeds max_depth).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from nga.arch.graph_fsm import GraphFSM
from nga.drivers.graph_fsm_spec import load as load_fsm_spec
from nga.exp.dataset_python_expr import (
    FEATURE_DIM,
    PYTHON_EXPR_TOKEN_INDEX,
    PYTHON_EXPR_TOKENS,
    generate_python_expr_dataset,
    generate_python_source,
    tokenize_program,
    train_test_split_by_program,
    walk_fsm,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FSM_PATH = REPO_ROOT / "tests/fixtures/graphs/python_expr.fsm.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def python_expr_fsm() -> GraphFSM:
    """Loaded python_expr FSM (max_depth = 4 baked into the YAML)."""
    return GraphFSM.from_yaml(FSM_PATH)


@pytest.fixture(scope="module")
def python_expr_dataset(python_expr_fsm: GraphFSM):
    """Default dataset for cross-test reuse."""
    return generate_python_expr_dataset(
        fsm=python_expr_fsm,
        n_programs=40,
        max_depth=3,
        seed=42,
        illegal_temptation_fraction=0.10,
    )


# ---------------------------------------------------------------------------
# FSM-level tests
# ---------------------------------------------------------------------------


def test_fsm_loads_and_has_expected_vertex_count():
    """FSM YAML loads via GraphFSMSpec.load and has exactly 14 vertices."""
    spec = load_fsm_spec(FSM_PATH)
    assert spec.name == "python_expr_d4"
    assert spec.vertex_count == 14
    assert spec.edge_count == 51
    expected_ids = {
        "START",
        "S0_after_name_at_start",
        "S0_after_assign_eq",
        "S0_factor",
        "S1_factor",
        "S2_factor",
        "S3_factor",
        "S4_factor",
        "S0_after_term",
        "S1_after_term",
        "S2_after_term",
        "S3_after_term",
        "S4_after_term",
        "ACCEPT",
    }
    assert {v.id for v in spec.vertices} == expected_ids


# ---------------------------------------------------------------------------
# Source-generator tests
# ---------------------------------------------------------------------------


def test_python_source_generator_produces_valid_python():
    """Every generated program parses via stdlib ast.parse without exception.

    This is the external-benchmark contract: the source the FSM walks
    is REAL Python, accepted by the real Python parser.
    """
    programs = generate_python_source(
        n_programs=50, max_depth=3, seed=2026
    )
    assert len(programs) == 50
    for src in programs:
        # ast.parse raises SyntaxError on invalid input -- if any
        # generated program fails this, the generator is buggy.
        ast.parse(src)
        # Each program ends with a newline (required by tokenize.tokenize
        # to emit a logical NEWLINE token).
        assert src.endswith("\n")


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


def test_python_tokenizer_emits_known_alphabet():
    """Every token from the tokenizer is a member of PYTHON_EXPR_TOKENS.

    Hits a few hand-picked source strings spanning the alphabet, plus
    a sweep over generated programs.
    """
    handcrafted = [
        "a\n",
        "1\n",
        "a + b\n",
        "x = a + b * (c - 1)\n",
        "(a + b) * c\n",
        "x = ((a + 1))\n",
    ]
    for src in handcrafted:
        for canonical, _raw in tokenize_program(src):
            assert canonical in PYTHON_EXPR_TOKEN_INDEX, (
                f"unknown token {canonical!r} from source {src!r}"
            )

    # Sweep generated programs.
    progs = generate_python_source(n_programs=30, max_depth=3, seed=11)
    for src in progs:
        for canonical, _raw in tokenize_program(src):
            assert canonical in PYTHON_EXPR_TOKEN_INDEX


# ---------------------------------------------------------------------------
# FSM walker tests
# ---------------------------------------------------------------------------


def test_walk_fsm_legal_per_step(python_expr_fsm: GraphFSM):
    """Every transition produced by walk_fsm is legal under the FSM."""
    progs = generate_python_source(n_programs=20, max_depth=3, seed=5)
    for src in progs:
        toks = tokenize_program(src)
        trace = walk_fsm(toks, python_expr_fsm)
        assert trace, f"no transitions produced for source {src!r}"
        for prev, _tok, nxt in trace:
            assert python_expr_fsm.is_legal_transition(prev, nxt), (
                f"illegal transition {prev!r} -> {nxt!r} for source {src!r}"
            )


# ---------------------------------------------------------------------------
# Dataset feature tests
# ---------------------------------------------------------------------------


def test_dataset_features_one_hot(python_expr_dataset):
    """Each feature vector is a one-hot of its observed_token's index."""
    ds = python_expr_dataset
    assert ds.X.shape[1] == FEATURE_DIM == 16
    for s, x in zip(ds.samples, ds.X):
        # Sum of one-hot is 1; max value is 1.
        assert x.sum() == pytest.approx(1.0)
        assert x.max() == pytest.approx(1.0)
        # Hot index matches the token's index in PYTHON_EXPR_TOKENS.
        hot_idx = int(np.argmax(x))
        assert hot_idx == PYTHON_EXPR_TOKEN_INDEX[s.observed_token]
        assert PYTHON_EXPR_TOKENS[hot_idx] == s.observed_token


def test_dataset_adversarial_marked(python_expr_fsm: GraphFSM):
    """Adversarial samples are flagged AND illegal under the FSM.

    Use a higher temptation fraction to ensure at least one adversarial
    sample lands in the dataset.
    """
    ds = generate_python_expr_dataset(
        fsm=python_expr_fsm,
        n_programs=40,
        max_depth=3,
        seed=7,
        illegal_temptation_fraction=0.50,
    )
    adv_samples = [s for s in ds.samples if s.is_adversarial]
    assert adv_samples, "expected at least one adversarial sample"

    # Build legal-token set per state by inspecting FSM edges directly.
    spec = python_expr_fsm._spec  # noqa: SLF001 -- internal access OK in test
    out_edges_by_src: dict[str, set[str]] = {}
    for e in spec.edges:
        out_edges_by_src.setdefault(e.source, set()).add(e.label or "")

    for s in adv_samples:
        legal_labels = out_edges_by_src.get(s.current_state, set())
        assert s.observed_token not in legal_labels, (
            f"adversarial sample {s.sample_id} has observed_token "
            f"{s.observed_token!r} which IS legal at {s.current_state}"
        )
        # Non-adversarial samples should NOT be flagged.
    for s in ds.samples:
        if not s.is_adversarial:
            legal_labels = out_edges_by_src.get(s.current_state, set())
            assert s.observed_token in legal_labels, (
                f"non-adversarial sample {s.sample_id} has observed_token "
                f"{s.observed_token!r} which is ILLEGAL at "
                f"{s.current_state} (legal: {sorted(legal_labels)})"
            )


def test_dataset_deterministic_under_seed(python_expr_fsm: GraphFSM):
    """Same seed and parameters -> identical X / y_next / depth arrays."""
    kw = dict(
        fsm=python_expr_fsm,
        n_programs=15,
        max_depth=3,
        seed=2026,
        illegal_temptation_fraction=0.10,
    )
    ds_a = generate_python_expr_dataset(**kw)
    ds_b = generate_python_expr_dataset(**kw)
    assert np.array_equal(ds_a.X, ds_b.X)
    assert np.array_equal(ds_a.y_next, ds_b.y_next)
    assert np.array_equal(ds_a.prev_states, ds_b.prev_states)
    assert np.array_equal(ds_a.is_adversarial, ds_b.is_adversarial)
    assert np.array_equal(ds_a.depths, ds_b.depths)
    assert ds_a.programs == ds_b.programs

    # Different seed -> at least one array differs (almost always).
    ds_c = generate_python_expr_dataset(
        **{**kw, "seed": 2027},
    )
    assert not (
        np.array_equal(ds_a.X, ds_c.X)
        and np.array_equal(ds_a.y_next, ds_c.y_next)
        and ds_a.programs == ds_c.programs
    )


def test_train_test_split_by_program(python_expr_dataset):
    """train / test splits use disjoint program_ids and cover all samples."""
    ds = python_expr_dataset
    train, test = train_test_split_by_program(ds, seed=0, test_fraction=0.25)

    train_progs = {s.program_id for s in train.samples}
    test_progs = {s.program_id for s in test.samples}
    all_progs = {s.program_id for s in ds.samples}

    assert train_progs.isdisjoint(test_progs), (
        f"program overlap between train and test: "
        f"{train_progs & test_progs}"
    )
    assert train_progs | test_progs == all_progs
    assert len(train.samples) + len(test.samples) == len(ds.samples)
    assert train.feature_dim == test.feature_dim == ds.feature_dim
    assert train.fsm is ds.fsm
    assert test.fsm is ds.fsm


def test_dataset_max_depth_respected(python_expr_fsm: GraphFSM):
    """No sample's parser depth exceeds max_depth, for several depths."""
    for max_depth in (1, 2, 3):
        ds = generate_python_expr_dataset(
            fsm=python_expr_fsm,
            n_programs=25,
            max_depth=max_depth,
            seed=11,
            illegal_temptation_fraction=0.0,
        )
        for s in ds.samples:
            assert s.depth <= max_depth, (
                f"sample {s.sample_id} has depth {s.depth} > "
                f"max_depth={max_depth}"
            )
            # The implied next-state's depth also must not exceed max_depth.
            from nga.exp.dataset_python_expr import _depth_of_state
            next_depth = _depth_of_state(s.true_next_state)
            assert next_depth <= max_depth, (
                f"sample {s.sample_id} steps to depth {next_depth} > "
                f"max_depth={max_depth}"
            )
