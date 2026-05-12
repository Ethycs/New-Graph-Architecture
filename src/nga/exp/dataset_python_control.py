"""Python "control" dataset generator (Phase 18 Track 2).

Extends ``dataset_python_big`` from arithmetic + assignment + calls + defs
to ALSO cover single-line control flow: ``if cond: stmt``,
``while cond: stmt``, and the optional ``else: stmt`` clause that may
follow an ``if`` body. Bodies are SINGLE-LINE simple statements (no
indentation, no DEDENT) so every program parses cleanly via stdlib
``ast.parse``.

Grammar (mirrors ``tests/fixtures/graphs/python_control.fsm.yaml``):

    program     := stmt+
    stmt        := expr_stmt | assign_stmt | def_stmt | if_stmt | while_stmt
    expr_stmt   := expr NEWLINE
    assign_stmt := NAME '=' expr NEWLINE
    def_stmt    := 'def' NAME '(' params ')' ':' return_stmt
    params      := empty | NAME (',' NAME)*
    return_stmt := 'return' expr NEWLINE
    if_stmt     := 'if' cond_atom ':' simple_body ('else' ':' simple_body)?
    while_stmt  := 'while' cond_atom ':' simple_body
    cond_atom   := NAME | NUMBER
    simple_body := simple_assign | simple_expr
    simple_assign := NAME '=' simple_expr
    simple_expr := simple_factor (('+' | '-' | '*' | '/') simple_factor)*
    simple_factor := NAME | NUMBER         # NO parens / calls in the body
    expr        := term (('+' | '-') term)*
    term        := factor (('*' | '/') factor)*
    factor      := NUMBER | name_or_call | '(' expr ')'
    name_or_call := NAME ('(' args ')')?
    args        := empty | expr (',' expr)*

Token alphabet (17 token types, padded to 22 dims):

    NAME, NUMBER, +, -, *, /, (, ), =, NEWLINE,
    def, return, ',', ':',
    if, else, while                                                # NEW

The structurally interesting addition over python_big is the
**branching ambiguity at S0_after_if_body**: after an if's body NEWLINE,
the next token may be ``else`` (continue the if-stmt) or any of the
START-token alternatives (begin a fresh top-level stmt). BOTH branches
are legal Python; the parser cannot decide which one was meant until it
sees the next token. The architectural prediction: this is exactly the
kind of structure σ should fire on, and σ_uplift should swing back
positive on python_control vs. python_big.

External-benchmark contract: every generated program is asserted to
``ast.parse`` cleanly.
"""
from __future__ import annotations

import ast
import io
import keyword
import random
import tokenize
from dataclasses import dataclass

import numpy as np

from nga.arch.graph_fsm import GraphFSM


__all__ = [
    "PYTHON_CONTROL_TOKENS",
    "PYTHON_CONTROL_TOKEN_INDEX",
    "FEATURE_DIM",
    "PythonControlSample",
    "PythonControlDataset",
    "generate_python_control_source",
    "tokenize_program_control",
    "walk_fsm_control",
    "generate_python_control_dataset",
    "train_test_split_by_program",
]


# ---------------------------------------------------------------------------
# Token alphabet
# ---------------------------------------------------------------------------

PYTHON_CONTROL_TOKENS: list[str] = [
    "NAME",
    "NUMBER",
    "+",
    "-",
    "*",
    "/",
    "(",
    ")",
    "=",
    "NEWLINE",
    "def",
    "return",
    ",",
    ":",
    "if",
    "else",
    "while",
]
"""Fixed canonical token order (17 entries). The one-hot feature index of
a token is its position in this list; the feature vector is padded with
zeros up to FEATURE_DIM = 22 (alphabet 17 + 5 padding for symmetry with
the python_big convention)."""

PYTHON_CONTROL_TOKEN_INDEX: dict[str, int] = {
    tok: i for i, tok in enumerate(PYTHON_CONTROL_TOKENS)
}

FEATURE_DIM: int = 22
"""Feature vector dimension; 17 active alphabet entries plus 5 zero pad."""


# ---------------------------------------------------------------------------
# Generator pools / constants
# ---------------------------------------------------------------------------

_NAME_POOL: tuple[str, ...] = ("a", "b", "c", "x", "y", "z", "f", "g", "h")
"""Fixed identifier pool. None of these collide with Python keywords."""

_NUMBER_POOL: tuple[str, ...] = tuple(str(d) for d in range(10))
_OPS_ADD: tuple[str, ...] = ("+", "-")
_OPS_MUL: tuple[str, ...] = ("*", "/")
_OPS_ALL: tuple[str, ...] = ("+", "-", "*", "/")

_RESERVED: frozenset[str] = frozenset({"def", "return", "if", "else", "while"})
"""Our explicit keywords (a strict subset of ``keyword.kwlist`` that the
grammar uses). Used by the tokenizer to decide whether a tokenize.NAME is
a keyword token or an identifier (``NAME``)."""


# ---------------------------------------------------------------------------
# Sample / dataset records
# ---------------------------------------------------------------------------


@dataclass
class PythonControlSample:
    """One transition sample in a Python-control parse trace."""

    sample_id: str
    features: np.ndarray
    observed_token: str
    prev_state: str
    current_state: str
    true_next_state: str
    is_adversarial: bool
    program_id: int
    depth: int


@dataclass
class PythonControlDataset:
    """Container for python_control samples + materialised numpy arrays."""

    samples: list[PythonControlSample]
    feature_dim: int
    fsm: GraphFSM
    max_paren_depth: int
    programs: list[str]
    X: np.ndarray
    y_next: np.ndarray
    prev_states: np.ndarray
    current_states: np.ndarray
    is_adversarial: np.ndarray
    depths: np.ndarray


# ---------------------------------------------------------------------------
# Token / state helpers
# ---------------------------------------------------------------------------


def _one_hot_token(token: str) -> np.ndarray:
    """Return a 22-dim zero-padded one-hot float64 vector for ``token``."""
    if token not in PYTHON_CONTROL_TOKEN_INDEX:
        raise KeyError(f"Unknown python_control token {token!r}")
    vec = np.zeros(FEATURE_DIM, dtype=np.float64)
    vec[PYTHON_CONTROL_TOKEN_INDEX[token]] = 1.0
    return vec


def _depth_of_state(state: str) -> int:
    """Extract the parenthesis depth from a state id.

    Control-scaffolding states (START, ACCEPT, S0_after_*, S0_def_*,
    S0_body_*, S0_if_*, S0_while_*, S0_simple_*) all live at depth 0.
    Numbered states (S{d}_factor / S{d}_after_term / S{d}_after_name_in_factor)
    map to depth d.
    """
    if state in ("START", "ACCEPT"):
        return 0
    if not state.startswith("S"):
        raise ValueError(f"Unrecognised python_control state id {state!r}")
    rest = state[1:]
    digit_part = rest.split("_", 1)[0]
    if not digit_part.isdigit():
        return 0
    return int(digit_part)


# ---------------------------------------------------------------------------
# Source generator (random programs that parse via ast.parse)
# ---------------------------------------------------------------------------


def _gen_factor(
    rng: random.Random,
    depth: int,
    max_paren_depth: int,
    p_call: float,
) -> str:
    """factor := NUMBER | name_or_call | '(' expr ')' (full grammar)."""
    p_paren = max(0.05, 0.25 - 0.07 * depth)
    can_open = depth < max_paren_depth
    r = rng.random()
    if can_open and r < p_paren:
        inner = _gen_expr(rng, depth + 1, max_paren_depth, p_call)
        return f"({inner})"

    if can_open and rng.random() < p_call:
        name = rng.choice(_NAME_POOL)
        return f"{name}({_gen_args(rng, depth + 1, max_paren_depth, p_call)})"

    if rng.random() < 0.5:
        return rng.choice(_NAME_POOL)
    return rng.choice(_NUMBER_POOL)


def _gen_args(
    rng: random.Random,
    depth: int,
    max_paren_depth: int,
    p_call: float,
) -> str:
    """args := empty | expr (',' expr)*."""
    if rng.random() < 0.30:
        return ""
    parts = [_gen_expr(rng, depth, max_paren_depth, p_call)]
    while rng.random() < 0.30:
        parts.append(_gen_expr(rng, depth, max_paren_depth, p_call))
    return ", ".join(parts)


def _gen_term(
    rng: random.Random,
    depth: int,
    max_paren_depth: int,
    p_call: float,
) -> str:
    """term := factor (('*' | '/') factor)*."""
    parts = [_gen_factor(rng, depth, max_paren_depth, p_call)]
    while rng.random() < 0.25:
        op = rng.choice(_OPS_MUL)
        parts.append(op)
        parts.append(_gen_factor(rng, depth, max_paren_depth, p_call))
    return " ".join(parts)


def _gen_expr(
    rng: random.Random,
    depth: int,
    max_paren_depth: int,
    p_call: float,
) -> str:
    """expr := term (('+' | '-') term)*."""
    parts = [_gen_term(rng, depth, max_paren_depth, p_call)]
    while rng.random() < 0.25:
        op = rng.choice(_OPS_ADD)
        parts.append(op)
        parts.append(_gen_term(rng, depth, max_paren_depth, p_call))
    return " ".join(parts)


def _gen_simple_factor(rng: random.Random) -> str:
    """simple_factor := NAME | NUMBER (no parens / calls in the body)."""
    if rng.random() < 0.5:
        return rng.choice(_NAME_POOL)
    return rng.choice(_NUMBER_POOL)


def _gen_simple_expr(rng: random.Random) -> str:
    """simple_expr := simple_factor ((+|-|*|/) simple_factor)*."""
    parts = [_gen_simple_factor(rng)]
    while rng.random() < 0.30:
        parts.append(rng.choice(_OPS_ALL))
        parts.append(_gen_simple_factor(rng))
    return " ".join(parts)


def _gen_simple_body(rng: random.Random) -> str:
    """simple_body := simple_assign | simple_expr.

    simple_assign starts with a bare NAME followed by '='; simple_expr
    is just a chain of NAME/NUMBER atoms joined by operators.
    """
    if rng.random() < 0.5:
        target = rng.choice(_NAME_POOL)
        return f"{target} = {_gen_simple_expr(rng)}"
    return _gen_simple_expr(rng)


def _gen_def_stmt(
    rng: random.Random,
    max_paren_depth: int,
    p_call: float,
) -> str:
    """def_stmt := 'def' NAME '(' params ')' ':' return_stmt."""
    name = rng.choice(_NAME_POOL)
    n_params = rng.randint(0, 3)
    if n_params == 0:
        params = ""
    else:
        params = ", ".join(rng.choices(_NAME_POOL, k=n_params))
    body_expr = _gen_expr(rng, depth=0, max_paren_depth=max_paren_depth, p_call=p_call)
    return f"def {name}({params}): return {body_expr}"


def _gen_if_stmt(rng: random.Random, p_else: float = 0.5) -> str:
    """if_stmt := 'if' cond_atom ':' simple_body ('else' ':' simple_body)?.

    The condition is a single NAME or NUMBER atom (matching the FSM's
    restriction). The body is a single simple-stmt (no parens / calls).
    """
    cond = (
        rng.choice(_NAME_POOL) if rng.random() < 0.5
        else rng.choice(_NUMBER_POOL)
    )
    body = _gen_simple_body(rng)
    if rng.random() < p_else:
        else_body = _gen_simple_body(rng)
        return f"if {cond}: {body}\nelse: {else_body}"
    return f"if {cond}: {body}"


def _gen_while_stmt(rng: random.Random) -> str:
    """while_stmt := 'while' cond_atom ':' simple_body."""
    cond = (
        rng.choice(_NAME_POOL) if rng.random() < 0.5
        else rng.choice(_NUMBER_POOL)
    )
    body = _gen_simple_body(rng)
    return f"while {cond}: {body}"


def _gen_top_stmt(
    rng: random.Random,
    max_paren_depth: int,
    p_def: float,
    p_call: float,
    p_if: float,
    p_while: float,
) -> str:
    """Generate one top-level statement.

    Top-level grammar: stmt := expr_stmt | assign_stmt | def_stmt
                              | if_stmt | while_stmt.
    """
    r = rng.random()
    if r < p_def:
        return _gen_def_stmt(rng, max_paren_depth, p_call)
    r2 = rng.random()
    if r2 < p_if:
        return _gen_if_stmt(rng)
    r3 = rng.random()
    if r3 < p_while:
        return _gen_while_stmt(rng)
    if rng.random() < 0.5:
        target = rng.choice(_NAME_POOL)
        rhs = _gen_expr(rng, depth=0, max_paren_depth=max_paren_depth, p_call=p_call)
        return f"{target} = {rhs}"
    return _gen_expr(rng, depth=0, max_paren_depth=max_paren_depth, p_call=p_call)


def generate_python_control_source(
    *,
    n_programs: int,
    max_paren_depth: int = 3,
    p_def: float = 0.15,
    p_call: float = 0.25,
    p_if: float = 0.20,
    p_while: float = 0.10,
    seed: int,
    n_stmts_min: int = 1,
    n_stmts_max: int = 3,
) -> list[str]:
    """Generate ``n_programs`` random valid Python "control" programs.

    Every program is asserted to ``ast.parse`` cleanly.

    Parameters
    ----------
    n_programs:
        Number of programs.
    max_paren_depth:
        Maximum parenthesis nesting depth in non-body expressions
        (1..3). The if-body itself never opens parens by construction.
    p_def, p_call, p_if, p_while:
        Branch probabilities for the top-level stmt dispatch.
    seed:
        RNG seed.
    n_stmts_min, n_stmts_max:
        Inclusive bounds on the number of statements per program.
    """
    if n_programs < 1:
        raise ValueError(f"n_programs must be >= 1, got {n_programs}")
    if not 1 <= max_paren_depth <= 3:
        raise ValueError(
            f"max_paren_depth must be in [1, 3], got {max_paren_depth}"
        )
    for name, val in (
        ("p_def", p_def),
        ("p_call", p_call),
        ("p_if", p_if),
        ("p_while", p_while),
    ):
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {val}")
    if not 1 <= n_stmts_min <= n_stmts_max:
        raise ValueError(
            f"n_stmts bounds invalid: [{n_stmts_min}, {n_stmts_max}]"
        )

    rng = random.Random(seed)
    programs: list[str] = []
    for _ in range(n_programs):
        n_stmts = rng.randint(n_stmts_min, n_stmts_max)
        stmts = [
            _gen_top_stmt(rng, max_paren_depth, p_def, p_call, p_if, p_while)
            for _ in range(n_stmts)
        ]
        source = "\n".join(stmts) + "\n"
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(
                f"Generated program failed ast.parse: {exc!r}\n{source!r}"
            ) from exc
        programs.append(source)
    return programs


# ---------------------------------------------------------------------------
# Tokenizer adapter (stdlib tokenize -> our 17-token alphabet)
# ---------------------------------------------------------------------------

_TOKENIZE_OP_TO_TOKEN: dict[str, str] = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "(": "(",
    ")": ")",
    "=": "=",
    ",": ",",
    ":": ":",
}

_DROP_TOKEN_TYPES: frozenset[int] = frozenset({
    tokenize.ENCODING,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.COMMENT,
    tokenize.ENDMARKER,
    tokenize.NL,
})


def tokenize_program_control(source: str) -> list[tuple[str, str]]:
    """Tokenize ``source`` and adapt to the 17-token python_control alphabet.

    Returns ``[(canonical_token, raw_string), ...]``. NAME tokens whose
    raw string is one of our reserved keywords (``def`` / ``return`` /
    ``if`` / ``else`` / ``while``) are emitted with the keyword as the
    canonical token rather than ``NAME``. All other ``tokenize.NAME``s
    map to ``NAME``.
    """
    out: list[tuple[str, str]] = []
    bytestream = io.BytesIO(source.encode("utf-8"))
    for tok in tokenize.tokenize(bytestream.readline):
        if tok.type in _DROP_TOKEN_TYPES:
            continue
        if tok.type == tokenize.NAME:
            raw = tok.string
            if raw in _RESERVED:
                assert keyword.iskeyword(raw), (
                    f"Internal: {raw!r} expected to be a Python keyword"
                )
                out.append((raw, raw))
                continue
            if keyword.iskeyword(raw):
                raise ValueError(
                    f"tokenize emitted unexpected keyword NAME {raw!r}; the "
                    f"python_control grammar only handles {sorted(_RESERVED)}."
                )
            out.append(("NAME", raw))
            continue
        if tok.type == tokenize.NUMBER:
            out.append(("NUMBER", tok.string))
            continue
        if tok.type == tokenize.NEWLINE:
            out.append(("NEWLINE", "\n"))
            continue
        if tok.type == tokenize.OP:
            mapped = _TOKENIZE_OP_TO_TOKEN.get(tok.string)
            if mapped is None:
                raise ValueError(
                    f"tokenize emitted OP {tok.string!r} which is not in the "
                    f"python_control alphabet (allowed: "
                    f"{sorted(_TOKENIZE_OP_TO_TOKEN)})."
                )
            out.append((mapped, tok.string))
            continue
        raise ValueError(
            f"tokenize emitted unexpected token type "
            f"{tokenize.tok_name[tok.type]} ({tok.string!r}); python_control "
            f"alphabet cannot represent it."
        )
    return out


# ---------------------------------------------------------------------------
# FSM walker
# ---------------------------------------------------------------------------


def _build_edge_index(fsm: GraphFSM) -> dict[tuple[str, str], str]:
    """Return ``{(source, label): target}`` for the FSM's edges."""
    spec = fsm._spec  # noqa: SLF001
    out: dict[tuple[str, str], str] = {}
    for edge in spec.edges:
        if edge.label is None:
            continue
        key = (edge.source, edge.label)
        if key in out and out[key] != edge.target:
            raise ValueError(
                f"FSM is non-deterministic on (source, label)={key!r}: "
                f"both {out[key]!r} and {edge.target!r} are edges."
            )
        out[key] = edge.target
    return out


def walk_fsm_control(
    token_stream: list[tuple[str, str]],
    fsm: GraphFSM,
) -> list[tuple[str, str, str]]:
    """Walk the FSM step-by-step over ``token_stream``.

    Returns ``[(prev_state, observed_token, next_state), ...]``. The
    initial ``prev_state`` is ``"START"``; after a NEWLINE that lands in
    ``ACCEPT``, the next stmt re-enters from ``"START"``. After a NEWLINE
    that lands in ``S0_after_if_body`` (the if-body branching state) the
    walker stays put -- the NEXT token transitions directly out of
    ``S0_after_if_body``, either via ``else`` (continue the if) or via a
    START-equivalent dispatch (start a fresh stmt).
    """
    edges_by_src_token = _build_edge_index(fsm)
    out: list[tuple[str, str, str]] = []
    state = "START"
    for canonical, _raw in token_stream:
        next_state = edges_by_src_token.get((state, canonical))
        assert next_state is not None, (
            f"FSM has no edge from {state!r} on token {canonical!r}; "
            f"the source is not in the python_control subset or the FSM is "
            f"missing an edge."
        )
        assert fsm.is_legal_transition(state, next_state), (
            f"FSM legality matrix rejects edge {state!r} -> {next_state!r}"
        )
        out.append((state, canonical, next_state))
        if next_state == "ACCEPT":
            state = "START"
        else:
            state = next_state
    return out


# ---------------------------------------------------------------------------
# Adversarial token picker
# ---------------------------------------------------------------------------


def _illegal_token_at(
    state: str,
    edges_by_src_token: dict[tuple[str, str], str],
    rng: random.Random,
) -> str | None:
    """Return a token from the alphabet that is ILLEGAL at ``state``."""
    legal = {tok for (src, tok) in edges_by_src_token if src == state}
    illegal = [tok for tok in PYTHON_CONTROL_TOKENS if tok not in legal]
    if not illegal:
        return None
    return rng.choice(illegal)


# ---------------------------------------------------------------------------
# Public dataset generator
# ---------------------------------------------------------------------------


def _expected_state_set() -> list[str]:
    """The vertex ids the FSM yaml is expected to provide (in spec order)."""
    return [
        "START",
        "S0_after_name_at_start",
        "S0_after_assign_eq",
        "S0_after_def_kw",
        "S0_after_def_name",
        "S0_def_param_first",
        "S0_def_after_param",
        "S0_def_param_after_comma",
        "S0_def_after_close",
        "S0_def_after_colon",
        "S0_after_return_kw",
        "S0_factor",
        "S1_factor",
        "S2_factor",
        "S3_factor",
        "S0_after_name_in_factor",
        "S1_after_name_in_factor",
        "S2_after_name_in_factor",
        "S3_after_name_in_factor",
        "S0_after_term",
        "S1_after_term",
        "S2_after_term",
        "S3_after_term",
        "S0_after_if_kw",
        "S0_after_while_kw",
        "S0_after_else_kw",
        "S0_if_cond_atom",
        "S0_while_cond_atom",
        "S0_if_after_colon",
        "S0_simple_body_start",
        "S0_after_if_body",
        "S0_body_after_name_at_start",
        "S0_body_after_assign_eq",
        "S0_body_factor",
        "S0_body_after_name_in_factor",
        "S0_body_after_term",
        "ACCEPT",
    ]


def generate_python_control_dataset(
    *,
    fsm: GraphFSM,
    n_programs: int = 100,
    max_paren_depth: int = 3,
    p_def: float = 0.15,
    p_call: float = 0.25,
    p_if: float = 0.20,
    p_while: float = 0.10,
    seed: int = 42,
    illegal_temptation_fraction: float = 0.10,
    n_stmts_min: int = 1,
    n_stmts_max: int = 3,
) -> PythonControlDataset:
    """Generate ``n_programs`` random valid python_control programs and walk them.

    For each program:
      1. Generate via ``generate_python_control_source`` (asserted to
         ``ast.parse`` cleanly).
      2. Tokenize via stdlib ``tokenize`` (17-token alphabet).
      3. Walk the FSM from START; at each step optionally corrupt the
         observed token to an illegal alternative (the walker still
         advances along the legal token).
    """
    if n_programs < 1:
        raise ValueError(f"n_programs must be >= 1, got {n_programs}")
    if not 1 <= max_paren_depth <= 3:
        raise ValueError(
            f"max_paren_depth must be in [1, 3], got {max_paren_depth}"
        )
    if not 0.0 <= illegal_temptation_fraction <= 1.0:
        raise ValueError(
            f"illegal_temptation_fraction must be in [0, 1], "
            f"got {illegal_temptation_fraction}"
        )

    expected = _expected_state_set()
    if fsm.vertex_ids != expected:
        raise ValueError(
            "FSM vertex set does not match the python_control canonical layout. "
            f"Expected {expected!r}, got {fsm.vertex_ids!r}."
        )

    rng = random.Random(seed)
    programs = generate_python_control_source(
        n_programs=n_programs,
        max_paren_depth=max_paren_depth,
        p_def=p_def,
        p_call=p_call,
        p_if=p_if,
        p_while=p_while,
        seed=seed,
        n_stmts_min=n_stmts_min,
        n_stmts_max=n_stmts_max,
    )

    edges_by_src_token = _build_edge_index(fsm)
    samples: list[PythonControlSample] = []

    for prog_idx, source in enumerate(programs):
        token_stream = tokenize_program_control(source)
        state = "START"
        for step_idx, (canonical, _raw) in enumerate(token_stream):
            legal_next = edges_by_src_token.get((state, canonical))
            assert legal_next is not None, (
                f"prog{prog_idx} step{step_idx}: FSM has no edge from "
                f"{state!r} on token {canonical!r} (source: {source!r})"
            )

            observed_token = canonical
            is_adv = False
            if illegal_temptation_fraction > 0.0:
                gate = float(rng.random())
                if gate < illegal_temptation_fraction:
                    illegal_tok = _illegal_token_at(
                        state, edges_by_src_token, rng
                    )
                    if illegal_tok is not None:
                        observed_token = illegal_tok
                        is_adv = True

            features = _one_hot_token(observed_token)
            depth_at_start = _depth_of_state(state)

            samples.append(
                PythonControlSample(
                    sample_id=f"prog{prog_idx}_step{step_idx}",
                    features=features,
                    observed_token=observed_token,
                    prev_state=state,
                    current_state=state,
                    true_next_state=legal_next,
                    is_adversarial=is_adv,
                    program_id=prog_idx,
                    depth=depth_at_start,
                )
            )

            if legal_next == "ACCEPT":
                state = "START"
            else:
                state = legal_next

    return _materialize(
        samples,
        fsm=fsm,
        max_paren_depth=max_paren_depth,
        programs=programs,
    )


# ---------------------------------------------------------------------------
# Materialization + helpers
# ---------------------------------------------------------------------------


def _materialize(
    samples: list[PythonControlSample],
    *,
    fsm: GraphFSM,
    max_paren_depth: int,
    programs: list[str],
) -> PythonControlDataset:
    """Pack a sample list into the numpy arrays the runner consumes."""
    n = len(samples)
    X = np.empty((n, FEATURE_DIM), dtype=np.float64)
    y_next = np.empty(n, dtype=np.int64)
    prev_states = np.empty(n, dtype=np.int64)
    is_adv = np.empty(n, dtype=bool)
    depths = np.empty(n, dtype=np.int64)

    vidx = fsm.vertex_index
    for i, s in enumerate(samples):
        X[i] = s.features
        y_next[i] = vidx[s.true_next_state]
        prev_states[i] = vidx[s.current_state]
        is_adv[i] = s.is_adversarial
        depths[i] = s.depth

    return PythonControlDataset(
        samples=samples,
        feature_dim=FEATURE_DIM,
        fsm=fsm,
        max_paren_depth=max_paren_depth,
        programs=list(programs),
        X=X,
        y_next=y_next,
        prev_states=prev_states,
        current_states=prev_states.copy(),
        is_adversarial=is_adv,
        depths=depths,
    )


def train_test_split_by_program(
    ds: PythonControlDataset,
    *,
    seed: int,
    test_fraction: float = 0.25,
) -> tuple[PythonControlDataset, PythonControlDataset]:
    """Partition by ``program_id`` so whole programs are exclusive."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            f"test_fraction must be in (0, 1), got {test_fraction}"
        )

    prog_ids = sorted({s.program_id for s in ds.samples})
    rng = np.random.default_rng(seed)
    shuffled = list(prog_ids)
    rng.shuffle(shuffled)  # type: ignore[arg-type]

    n_test = max(1, round(len(shuffled) * test_fraction))
    test_set = set(shuffled[:n_test])
    train_set = set(shuffled[n_test:])

    train_samples = [s for s in ds.samples if s.program_id in train_set]
    test_samples = [s for s in ds.samples if s.program_id in test_set]

    train_programs = [
        ds.programs[i] for i in range(len(ds.programs)) if i in train_set
    ]
    test_programs = [
        ds.programs[i] for i in range(len(ds.programs)) if i in test_set
    ]

    train_ds = _materialize(
        train_samples,
        fsm=ds.fsm,
        max_paren_depth=ds.max_paren_depth,
        programs=train_programs,
    )
    test_ds = _materialize(
        test_samples,
        fsm=ds.fsm,
        max_paren_depth=ds.max_paren_depth,
        programs=test_programs,
    )
    return train_ds, test_ds
