"""Unit tests for nga.arch.grammar_compiler.

Roundtrip equivalence: a hand-authored .fsm.yaml and a grammar DSL file that
declare the same states + transitions must produce identical legality matrices
once vertices are sorted in a canonical order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from nga.arch.grammar_compiler import (
    GrammarSpec,
    compile_grammar,
    compile_to_yaml,
    load_grammar,
)
from nga.drivers.graph_fsm_spec import GraphFSMSpec
from nga.drivers.graph_fsm_spec import load as load_fsm

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAMMAR_PATH = REPO_ROOT / "tests" / "fixtures" / "grammars" / "babyai.grammar.yaml"
HAND_FSM_PATH = REPO_ROOT / "tests" / "fixtures" / "graphs" / "babyai_cyclic.fsm.yaml"


def _legality_matrix_sorted(spec: GraphFSMSpec) -> np.ndarray:
    """Return the legality matrix with rows/cols permuted into vertex-name order.

    Sorting by vertex id makes the comparison invariant to how states were
    declared in the grammar vs. the hand-authored FSM.
    """
    raw = np.asarray(spec.build_legality_matrix(), dtype=bool)
    ids = [v.id for v in spec.vertices]
    order = np.argsort(ids)
    return raw[np.ix_(order, order)]


def test_grammar_compiler_roundtrip() -> None:
    """Compiled grammar and hand-authored FSM yield identical legality matrices."""
    grammar = load_grammar(GRAMMAR_PATH)
    compiled = compile_grammar(grammar)
    hand_authored = load_fsm(HAND_FSM_PATH)

    compiled_matrix = _legality_matrix_sorted(compiled)
    hand_matrix = _legality_matrix_sorted(hand_authored)

    assert compiled_matrix.shape == hand_matrix.shape
    assert np.array_equal(compiled_matrix, hand_matrix), (
        f"legality mismatch\ncompiled=\n{compiled_matrix.astype(int)}\n"
        f"hand=\n{hand_matrix.astype(int)}"
    )

    # Sanity: vertex sets agree.
    assert sorted(v.id for v in compiled.vertices) == sorted(
        v.id for v in hand_authored.vertices
    )


def test_grammar_rejects_unknown_state() -> None:
    """A transition referencing an undeclared state raises ValidationError."""
    bad_doc = {
        "schema_version": "1.0",
        "name": "broken",
        "states": [
            {"name": "A"},
            {"name": "B"},
        ],
        "transitions": [
            {"from": "A", "to": "Ghost"},  # Ghost is not declared
        ],
        "embedding": {"dimension": 4},
    }
    with pytest.raises(ValidationError):
        GrammarSpec.model_validate(bad_doc)


def test_grammar_compiler_preserves_dimension() -> None:
    """compile_grammar carries the embedding dimension through to the GraphFSMSpec."""
    grammar = load_grammar(GRAMMAR_PATH)
    compiled = compile_grammar(grammar)

    assert compiled.coordinates.dimension == grammar.embedding.dimension
    assert compiled.coordinates.space == grammar.embedding.space

    # Every vertex got a default zero coordinate of length `dimension`.
    assert compiled.coordinates.node_embeddings is not None
    assert set(compiled.coordinates.node_embeddings.keys()) == {
        s.name for s in grammar.states
    }
    for vid, row in compiled.coordinates.node_embeddings.items():
        assert len(row) == grammar.embedding.dimension, vid
        assert all(x == 0.0 for x in row)


def test_compile_to_yaml_roundtrips_through_disk(tmp_path: Path) -> None:
    """compile_to_yaml writes a loadable GraphFSMSpec with matching legality."""
    out = tmp_path / "compiled.fsm.yaml"
    compiled = compile_to_yaml(GRAMMAR_PATH, out)
    reloaded = load_fsm(out)

    assert np.array_equal(
        _legality_matrix_sorted(compiled), _legality_matrix_sorted(reloaded)
    )
