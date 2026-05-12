"""Unit tests for the python_control dataset generator + parser FSM
(Phase 18 Track 2).

Mirrors test_dataset_python_big but with control flow added: single-line
if / else / while statements. The new sigma site is S0_after_if_body
(the "is the next token 'else' or a fresh top-level stmt?" branching
ambiguity).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from nga.arch.graph_fsm import GraphFSM
from nga.drivers.graph_fsm_spec import load as load_fsm_spec
from nga.exp.dataset_python_control import (
    FEATURE_DIM,
    PYTHON_CONTROL_TOKEN_INDEX,
    PYTHON_CONTROL_TOKENS,
    generate_python_control_dataset,
    generate_python_control_source,
    tokenize_program_control,
    train_test_split_by_program,
    walk_fsm_control,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FSM_PATH = REPO_ROOT / "tests/fixtures/graphs/python_control.fsm.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def python_control_fsm() -> GraphFSM:
    """Loaded python_control FSM (max_paren_depth = 3 baked into the YAML)."""
    return GraphFSM.from_yaml(FSM_PATH)


@pytest.fixture(scope="module")
def python_control_dataset(python_control_fsm: GraphFSM):
    """Default dataset for cross-test reuse."""
    return generate_python_control_dataset(
        fsm=python_control_fsm,
        n_programs=40,
        max_paren_depth=3,
        p_def=0.15,
        p_call=0.30,
        p_if=0.30,
        p_while=0.20,
        seed=42,
        illegal_temptation_fraction=0.10,
    )


# ---------------------------------------------------------------------------
# FSM-level tests
# ---------------------------------------------------------------------------


def test_python_control_fsm_loads_and_has_expected_vertex_count():
    """FSM YAML loads via GraphFSMSpec.load and vertex count is in [32, 40]."""
    spec = load_fsm_spec(FSM_PATH)
    assert spec.name == "python_control_d3"
    assert 32 <= spec.vertex_count <= 40, (
        f"expected vertex_count in [32, 40], got {spec.vertex_count}"
    )
    ids = {v.id for v in spec.vertices}
    # Inherited python_big states.
    assert "START" in ids
    assert "ACCEPT" in ids
    assert "S0_after_def_kw" in ids
    assert "S0_after_return_kw" in ids
    # NEW control-flow states.
    assert "S0_after_if_kw" in ids
    assert "S0_after_while_kw" in ids
    assert "S0_after_else_kw" in ids
    assert "S0_after_if_body" in ids


def test_python_control_branching_state_present():
    """S0_after_if_body is the new sigma-load-bearing branching site."""
    spec = load_fsm_spec(FSM_PATH)
    ids = {v.id for v in spec.vertices}
    assert "S0_after_if_body" in ids, (
        f"expected S0_after_if_body in the vertex set; got {sorted(ids)}"
    )


# ---------------------------------------------------------------------------
# Source-generator tests
# ---------------------------------------------------------------------------


def test_python_control_validates_via_ast():
    """Every generated program parses via stdlib ast.parse without exception."""
    programs = generate_python_control_source(
        n_programs=80,
        max_paren_depth=3,
        p_def=0.15,
        p_call=0.30,
        p_if=0.30,
        p_while=0.20,
        seed=2026,
    )
    assert len(programs) == 80
    for src in programs:
        ast.parse(src)
        assert src.endswith("\n")


def test_python_control_includes_if_else_while():
    """With elevated branch probabilities the corpus contains all three keywords."""
    programs = generate_python_control_source(
        n_programs=120,
        max_paren_depth=3,
        p_def=0.10,
        p_call=0.30,
        p_if=0.35,
        p_while=0.25,
        seed=7,
    )
    has_if = any("if " in src for src in programs)
    has_else = any("\nelse" in src or src.startswith("else") for src in programs)
    has_while = any("while " in src for src in programs)
    assert has_if, "expected at least one if-stmt in the generated corpus"
    assert has_else, "expected at least one else-clause in the generated corpus"
    assert has_while, "expected at least one while-stmt in the generated corpus"


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


def test_python_control_tokenizer_emits_known_alphabet():
    """Every token from the tokenizer is a member of PYTHON_CONTROL_TOKENS."""
    handcrafted = [
        "a\n",
        "1\n",
        "f()\n",
        "x = f(a + 1)\n",
        "def g(x): return x + 1\n",
        "if 1: x = 5\n",
        "if a: y = 1\nelse: y = 2\n",
        "while 0: x = 1\n",
        "while a: b = c + d\n",
        "if 1: x = 1\nelse: y = 2\nz = 3\n",
    ]
    for src in handcrafted:
        ast.parse(src)
        for canonical, _raw in tokenize_program_control(src):
            assert canonical in PYTHON_CONTROL_TOKEN_INDEX, (
                f"unknown token {canonical!r} from source {src!r}"
            )

    progs = generate_python_control_source(
        n_programs=30,
        max_paren_depth=3,
        p_def=0.20,
        p_call=0.30,
        p_if=0.30,
        p_while=0.20,
        seed=11,
    )
    for src in progs:
        for canonical, _raw in tokenize_program_control(src):
            assert canonical in PYTHON_CONTROL_TOKEN_INDEX


def test_python_control_tokenizer_maps_keywords():
    """The 'if', 'else', 'while' keyword names map to dedicated tokens."""
    src = "if a: b = 1\nelse: c = 2\nwhile d: e = f\n"
    toks = tokenize_program_control(src)
    canon = [c for c, _ in toks]
    assert "if" in canon
    assert "else" in canon
    assert "while" in canon


# ---------------------------------------------------------------------------
# FSM walker tests
# ---------------------------------------------------------------------------


def test_python_control_walk_fsm_legal_per_step(python_control_fsm: GraphFSM):
    """Every transition produced by walk_fsm_control is FSM-legal."""
    progs = generate_python_control_source(
        n_programs=30,
        max_paren_depth=3,
        p_def=0.15,
        p_call=0.30,
        p_if=0.30,
        p_while=0.20,
        seed=5,
    )
    for src in progs:
        toks = tokenize_program_control(src)
        trace = walk_fsm_control(toks, python_control_fsm)
        assert trace, f"no transitions produced for source {src!r}"
        for prev, _tok, nxt in trace:
            assert python_control_fsm.is_legal_transition(prev, nxt), (
                f"illegal transition {prev!r} -> {nxt!r} for source {src!r}"
            )


def test_python_control_after_if_body_reached(python_control_fsm: GraphFSM):
    """At least one trace lands in the new S0_after_if_body branching state."""
    progs = generate_python_control_source(
        n_programs=80,
        max_paren_depth=3,
        p_def=0.05,
        p_call=0.10,
        p_if=0.50,
        p_while=0.10,
        seed=3,
    )
    visited_after_if_body = False
    for src in progs:
        toks = tokenize_program_control(src)
        trace = walk_fsm_control(toks, python_control_fsm)
        for prev, _tok, nxt in trace:
            if nxt == "S0_after_if_body" or prev == "S0_after_if_body":
                visited_after_if_body = True
                break
        if visited_after_if_body:
            break
    assert visited_after_if_body, (
        "no trace reached S0_after_if_body; the if-body NEWLINE "
        "branching state must be exercised by the corpus"
    )


# ---------------------------------------------------------------------------
# Dataset feature tests
# ---------------------------------------------------------------------------


def test_python_control_dataset_features_one_hot(python_control_dataset):
    """Each feature vector is a one-hot of its observed_token's index."""
    ds = python_control_dataset
    assert ds.X.shape[1] == FEATURE_DIM == 22
    for s, x in zip(ds.samples, ds.X):
        assert x.sum() == pytest.approx(1.0)
        assert x.max() == pytest.approx(1.0)
        hot_idx = int(np.argmax(x))
        assert hot_idx == PYTHON_CONTROL_TOKEN_INDEX[s.observed_token]
        assert PYTHON_CONTROL_TOKENS[hot_idx] == s.observed_token


def test_python_control_adversarial_marked(python_control_fsm: GraphFSM):
    """Adversarial samples are flagged AND illegal under the FSM."""
    ds = generate_python_control_dataset(
        fsm=python_control_fsm,
        n_programs=40,
        max_paren_depth=3,
        p_def=0.15,
        p_call=0.30,
        p_if=0.30,
        p_while=0.20,
        seed=7,
        illegal_temptation_fraction=0.50,
    )
    adv_samples = [s for s in ds.samples if s.is_adversarial]
    assert adv_samples, "expected at least one adversarial sample"

    spec = python_control_fsm._spec  # noqa: SLF001
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


def test_python_control_deterministic_under_seed(python_control_fsm: GraphFSM):
    """Same seed and parameters -> identical X / y_next / depth arrays."""
    kw = dict(
        fsm=python_control_fsm,
        n_programs=15,
        max_paren_depth=3,
        p_def=0.15,
        p_call=0.30,
        p_if=0.30,
        p_while=0.20,
        seed=2026,
        illegal_temptation_fraction=0.10,
    )
    ds_a = generate_python_control_dataset(**kw)
    ds_b = generate_python_control_dataset(**kw)
    assert np.array_equal(ds_a.X, ds_b.X)
    assert np.array_equal(ds_a.y_next, ds_b.y_next)
    assert np.array_equal(ds_a.prev_states, ds_b.prev_states)
    assert np.array_equal(ds_a.is_adversarial, ds_b.is_adversarial)
    assert np.array_equal(ds_a.depths, ds_b.depths)
    assert ds_a.programs == ds_b.programs

    ds_c = generate_python_control_dataset(**{**kw, "seed": 2027})
    assert not (
        np.array_equal(ds_a.X, ds_c.X)
        and np.array_equal(ds_a.y_next, ds_c.y_next)
        and ds_a.programs == ds_c.programs
    )


def test_python_control_train_test_split_disjoint(python_control_dataset):
    """train / test splits use disjoint program_ids and cover all samples."""
    ds = python_control_dataset
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
