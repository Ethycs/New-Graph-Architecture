"""Unit tests for the python_big dataset generator + parser FSM (Phase 14 Wave II).

Mirrors test_dataset_python_expr but with the bigger language: function
calls, function definitions, return statements. The new sigma site is
S{d}_after_name_in_factor (the "is this NAME a variable or the head of a
call?" decision).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from nga.arch.graph_fsm import GraphFSM
from nga.drivers.graph_fsm_spec import load as load_fsm_spec
from nga.exp.dataset_python_big import (
    FEATURE_DIM,
    PYTHON_BIG_TOKEN_INDEX,
    PYTHON_BIG_TOKENS,
    generate_python_big_dataset,
    generate_python_big_source,
    tokenize_program_big,
    train_test_split_by_program,
    walk_fsm_big,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FSM_PATH = REPO_ROOT / "tests/fixtures/graphs/python_big.fsm.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def python_big_fsm() -> GraphFSM:
    """Loaded python_big FSM (max_paren_depth = 3 baked into the YAML)."""
    return GraphFSM.from_yaml(FSM_PATH)


@pytest.fixture(scope="module")
def python_big_dataset(python_big_fsm: GraphFSM):
    """Default dataset for cross-test reuse."""
    return generate_python_big_dataset(
        fsm=python_big_fsm,
        n_programs=40,
        max_paren_depth=3,
        p_def=0.25,
        p_call=0.4,
        seed=42,
        illegal_temptation_fraction=0.10,
    )


# ---------------------------------------------------------------------------
# FSM-level tests
# ---------------------------------------------------------------------------


def test_python_big_fsm_loads_and_has_expected_vertex_count():
    """FSM YAML loads via GraphFSMSpec.load and vertex count is in [24, 32]."""
    spec = load_fsm_spec(FSM_PATH)
    assert spec.name == "python_big_d3"
    assert 24 <= spec.vertex_count <= 32, (
        f"expected vertex_count in [24, 32], got {spec.vertex_count}"
    )
    # Spot-check a few load-bearing vertex ids are present.
    ids = {v.id for v in spec.vertices}
    assert "START" in ids
    assert "ACCEPT" in ids
    assert "S0_after_def_kw" in ids
    assert "S0_after_return_kw" in ids


def test_python_big_call_ambiguity_state_present():
    """At least one S{d}_after_name_in_factor state exists for d in 0..3.

    This is the new sigma-load-bearing site: the parser must choose
    whether a NAME is a variable reference or the head of a call.
    """
    spec = load_fsm_spec(FSM_PATH)
    ids = {v.id for v in spec.vertices}
    found = [d for d in range(4) if f"S{d}_after_name_in_factor" in ids]
    assert found, (
        f"expected at least one S{{d}}_after_name_in_factor for d in 0..3, "
        f"got vertex set {sorted(ids)}"
    )


# ---------------------------------------------------------------------------
# Source-generator tests
# ---------------------------------------------------------------------------


def test_python_big_generator_validates_via_ast():
    """Every generated program parses via stdlib ast.parse without exception."""
    programs = generate_python_big_source(
        n_programs=60,
        max_paren_depth=3,
        p_def=0.25,
        p_call=0.4,
        seed=2026,
    )
    assert len(programs) == 60
    for src in programs:
        # Will raise SyntaxError if invalid -- that's an immediate failure.
        ast.parse(src)
        assert src.endswith("\n")


def test_python_big_includes_calls_and_defs():
    """With p_def and p_call elevated, the corpus contains BOTH defs and calls.

    "At least one def" = at least one program contains the keyword 'def'.
    "At least one call" = at least one program contains a NAME-followed-by-'('
    pattern that is NOT the def's parameter list.
    """
    programs = generate_python_big_source(
        n_programs=80,
        max_paren_depth=3,
        p_def=0.3,
        p_call=0.5,
        seed=7,
    )
    has_def = any("def " in src for src in programs)
    assert has_def, "expected at least one def_stmt in the generated corpus"

    # Detect a call: walk tokens and look for an EXPRESSION-context NAME
    # followed by '(' (not the def's NAME, which is followed by '(' but is
    # the parameter list, not a call). We scan for any '(' token whose
    # immediately preceding token is a NAME AND for which the token before
    # that NAME is NOT 'def'.
    has_call = False
    for src in programs:
        toks = tokenize_program_big(src)
        for i in range(2, len(toks)):
            if toks[i][0] == "(" and toks[i - 1][0] == "NAME":
                if toks[i - 2][0] != "def":
                    has_call = True
                    break
        # Edge case: NAME at index 0 followed by '(' at index 1 is a call
        # at top-level (e.g. `f(x)\n`). Index 2 loop misses i=1; check it.
        if not has_call and len(toks) >= 2:
            if toks[0][0] == "NAME" and toks[1][0] == "(":
                has_call = True
        if has_call:
            break
    assert has_call, "expected at least one call site in the generated corpus"


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


def test_python_big_tokenizer_emits_known_alphabet():
    """Every token from the tokenizer is a member of PYTHON_BIG_TOKENS."""
    handcrafted = [
        "a\n",
        "1\n",
        "f()\n",
        "f(a, b)\n",
        "x = f(a + 1)\n",
        "def g(x): return x + 1\n",
        "def h(): return 0\n",
        "def foo(a, b, c): return f(a) + g(b, c)\n",
    ]
    for src in handcrafted:
        # Sanity: also valid Python.
        ast.parse(src)
        for canonical, _raw in tokenize_program_big(src):
            assert canonical in PYTHON_BIG_TOKEN_INDEX, (
                f"unknown token {canonical!r} from source {src!r}"
            )

    progs = generate_python_big_source(
        n_programs=30,
        max_paren_depth=3,
        p_def=0.3,
        p_call=0.4,
        seed=11,
    )
    for src in progs:
        for canonical, _raw in tokenize_program_big(src):
            assert canonical in PYTHON_BIG_TOKEN_INDEX


# ---------------------------------------------------------------------------
# FSM walker tests
# ---------------------------------------------------------------------------


def test_python_big_walk_fsm_legal_per_step(python_big_fsm: GraphFSM):
    """Every transition produced by walk_fsm_big is FSM-legal."""
    progs = generate_python_big_source(
        n_programs=25,
        max_paren_depth=3,
        p_def=0.3,
        p_call=0.4,
        seed=5,
    )
    for src in progs:
        toks = tokenize_program_big(src)
        trace = walk_fsm_big(toks, python_big_fsm)
        assert trace, f"no transitions produced for source {src!r}"
        for prev, _tok, nxt in trace:
            assert python_big_fsm.is_legal_transition(prev, nxt), (
                f"illegal transition {prev!r} -> {nxt!r} for source {src!r}"
            )


# ---------------------------------------------------------------------------
# Dataset feature tests
# ---------------------------------------------------------------------------


def test_python_big_dataset_features_one_hot(python_big_dataset):
    """Each feature vector is a one-hot of its observed_token's index."""
    ds = python_big_dataset
    assert ds.X.shape[1] == FEATURE_DIM == 18
    for s, x in zip(ds.samples, ds.X):
        assert x.sum() == pytest.approx(1.0)
        assert x.max() == pytest.approx(1.0)
        hot_idx = int(np.argmax(x))
        assert hot_idx == PYTHON_BIG_TOKEN_INDEX[s.observed_token]
        assert PYTHON_BIG_TOKENS[hot_idx] == s.observed_token


def test_python_big_adversarial_marked(python_big_fsm: GraphFSM):
    """Adversarial samples are flagged AND illegal under the FSM."""
    ds = generate_python_big_dataset(
        fsm=python_big_fsm,
        n_programs=40,
        max_paren_depth=3,
        p_def=0.25,
        p_call=0.4,
        seed=7,
        illegal_temptation_fraction=0.50,
    )
    adv_samples = [s for s in ds.samples if s.is_adversarial]
    assert adv_samples, "expected at least one adversarial sample"

    spec = python_big_fsm._spec  # noqa: SLF001
    out_edges_by_src: dict[str, set[str]] = {}
    for e in spec.edges:
        out_edges_by_src.setdefault(e.source, set()).add(e.label or "")

    for s in adv_samples:
        legal_labels = out_edges_by_src.get(s.current_state, set())
        assert s.observed_token not in legal_labels, (
            f"adversarial sample {s.sample_id} has observed_token "
            f"{s.observed_token!r} which IS legal at {s.current_state}"
        )
    for s in ds.samples:
        if not s.is_adversarial:
            legal_labels = out_edges_by_src.get(s.current_state, set())
            assert s.observed_token in legal_labels, (
                f"non-adversarial sample {s.sample_id} has observed_token "
                f"{s.observed_token!r} which is ILLEGAL at "
                f"{s.current_state} (legal: {sorted(legal_labels)})"
            )


def test_python_big_deterministic_under_seed(python_big_fsm: GraphFSM):
    """Same seed and parameters -> identical X / y_next / depth arrays."""
    kw = dict(
        fsm=python_big_fsm,
        n_programs=15,
        max_paren_depth=3,
        p_def=0.25,
        p_call=0.4,
        seed=2026,
        illegal_temptation_fraction=0.10,
    )
    ds_a = generate_python_big_dataset(**kw)
    ds_b = generate_python_big_dataset(**kw)
    assert np.array_equal(ds_a.X, ds_b.X)
    assert np.array_equal(ds_a.y_next, ds_b.y_next)
    assert np.array_equal(ds_a.prev_states, ds_b.prev_states)
    assert np.array_equal(ds_a.is_adversarial, ds_b.is_adversarial)
    assert np.array_equal(ds_a.depths, ds_b.depths)
    assert ds_a.programs == ds_b.programs

    ds_c = generate_python_big_dataset(**{**kw, "seed": 2027})
    assert not (
        np.array_equal(ds_a.X, ds_c.X)
        and np.array_equal(ds_a.y_next, ds_c.y_next)
        and ds_a.programs == ds_c.programs
    )


def test_python_big_train_test_split_disjoint(python_big_dataset):
    """train / test splits use disjoint program_ids and cover all samples."""
    ds = python_big_dataset
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
