"""Python "big" dataset generator (Phase 14 Wave II).

Extends ``dataset_python_expr`` from arithmetic + assignment to a richer
Python subset that ALSO covers function calls, function definitions, and
return statements. Single-line def bodies only (``def f(a): return a+1``)
so ast.parse never sees an INDENT/DEDENT.

Grammar (mirrors ``tests/fixtures/graphs/python_big.fsm.yaml``):

    program     := stmt+
    stmt        := expr_stmt | assign_stmt | def_stmt
    expr_stmt   := expr NEWLINE
    assign_stmt := NAME '=' expr NEWLINE
    def_stmt    := 'def' NAME '(' params ')' ':' return_stmt
    params      := empty | NAME (',' NAME)*
    return_stmt := 'return' expr NEWLINE
    expr        := term (('+' | '-') term)*
    term        := factor (('*' | '/') factor)*
    factor      := NUMBER | name_or_call | '(' expr ')'
    name_or_call := NAME ('(' args ')')?            # CALL AMBIGUITY
    args        := empty | expr (',' expr)*

Token alphabet (14 token types, padded to 18 dims for parity with ListOps):

    NAME, NUMBER, +, -, *, /, (, ), =, NEWLINE,
    def, return, ',', ':'

The structurally interesting addition over python_expr is the
**call ambiguity**: every NAME in factor position now lands in
``S{d}_after_name_in_factor`` — a new sigma-load-bearing site where the
parser must choose between ``(`` (it's a function call) and any other
continuation (it's a variable reference). Both branches are legal Python.

External-benchmark contract: every generated program is asserted to
``ast.parse`` cleanly (``return`` only appears inside def bodies, never
top-level, since bare top-level ``return`` is a SyntaxError in real
Python).
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
    "PYTHON_BIG_TOKENS",
    "PYTHON_BIG_TOKEN_INDEX",
    "FEATURE_DIM",
    "PythonBigSample",
    "PythonBigDataset",
    "generate_python_big_source",
    "tokenize_program_big",
    "walk_fsm_big",
    "generate_python_big_dataset",
    "train_test_split_by_program",
]


# ---------------------------------------------------------------------------
# Token alphabet
# ---------------------------------------------------------------------------

PYTHON_BIG_TOKENS: list[str] = [
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
]
"""Fixed canonical token order (14 entries). The one-hot feature index of
a token is its position in this list; the feature vector is padded with
zeros up to FEATURE_DIM = 18 (alphabet 14 + 4 padding for parity with the
ListOps / Dyck convention)."""

PYTHON_BIG_TOKEN_INDEX: dict[str, int] = {
    tok: i for i, tok in enumerate(PYTHON_BIG_TOKENS)
}

FEATURE_DIM: int = 18
"""Feature vector dimension; 14 active alphabet entries plus 4 zero pad."""

# ---------------------------------------------------------------------------
# Generator pools / constants
# ---------------------------------------------------------------------------

_NAME_POOL: tuple[str, ...] = ("a", "b", "c", "x", "y", "z", "f", "g", "h")
"""Fixed identifier pool. None of these collide with Python keywords."""

_NUMBER_POOL: tuple[str, ...] = tuple(str(d) for d in range(10))
_OPS_ADD: tuple[str, ...] = ("+", "-")
_OPS_MUL: tuple[str, ...] = ("*", "/")

_RESERVED: frozenset[str] = frozenset({"def", "return"})
"""Our explicit keywords (a strict subset of ``keyword.kwlist`` that the
grammar uses). Used by the tokenizer to decide whether a tokenize.NAME is
a keyword token (``def`` / ``return``) or an identifier (``NAME``)."""


# ---------------------------------------------------------------------------
# Sample / dataset records
# ---------------------------------------------------------------------------


@dataclass
class PythonBigSample:
    """One transition sample in a Python-big parse trace.

    Same shape as ``PythonExprSample`` -- an additional ``in_def`` flag is
    NOT included because the FSM trajectory itself encodes whether we are
    inside the def-stmt scaffolding (via the ``S0_after_def_*`` states).
    """

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
class PythonBigDataset:
    """Container for python_big samples + materialised numpy arrays."""

    samples: list[PythonBigSample]
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
    """Return an 18-dim zero-padded one-hot float64 vector for ``token``."""
    if token not in PYTHON_BIG_TOKEN_INDEX:
        raise KeyError(f"Unknown python_big token {token!r}")
    vec = np.zeros(FEATURE_DIM, dtype=np.float64)
    vec[PYTHON_BIG_TOKEN_INDEX[token]] = 1.0
    return vec


def _depth_of_state(state: str) -> int:
    """Extract the parenthesis depth from a state id.

    Control-scaffolding states (START, ACCEPT, S0_after_name_at_start,
    S0_after_assign_eq, S0_after_def_*, S0_def_*, S0_after_return_kw)
    all map to depth 0. Numbered states (S{d}_factor /
    S{d}_after_name_in_factor / S{d}_after_term) map to depth d.
    """
    if state in ("START", "ACCEPT"):
        return 0
    if not state.startswith("S"):
        raise ValueError(f"Unrecognised python_big state id {state!r}")
    rest = state[1:]
    digit_part = rest.split("_", 1)[0]
    if not digit_part.isdigit():
        # Defensive -- should never happen because every legitimate state
        # starts with S<digit>_.
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
    """Generate a factor.

    factor := NUMBER | name_or_call | '(' expr ')'
    name_or_call := NAME ('(' args ')')?

    With probability ~p_paren we open a parenthesised sub-expression
    (only when depth permits another '('); otherwise we emit either a
    bare NUMBER, a bare NAME, or a CALL (NAME '(' args ')'). The call
    probability is gated by p_call AND by whether opening another paren
    would exceed max_paren_depth.
    """
    p_paren = max(0.05, 0.25 - 0.07 * depth)
    can_open = depth < max_paren_depth
    r = rng.random()
    if can_open and r < p_paren:
        inner = _gen_expr(rng, depth + 1, max_paren_depth, p_call)
        return f"({inner})"

    # Decide between NUMBER, NAME, or CALL.
    # Bias toward leaves so generation terminates; calls increase
    # nesting depth by 1, so suppress them at max_paren_depth.
    if can_open and rng.random() < p_call:
        # CALL.
        name = rng.choice(_NAME_POOL)
        return f"{name}({_gen_args(rng, depth + 1, max_paren_depth, p_call)})"

    # Plain NAME or NUMBER.
    if rng.random() < 0.5:
        return rng.choice(_NAME_POOL)
    return rng.choice(_NUMBER_POOL)


def _gen_args(
    rng: random.Random,
    depth: int,
    max_paren_depth: int,
    p_call: float,
) -> str:
    """Generate a call's argument list (possibly empty).

    args := empty | expr (',' expr)*
    """
    # Empty args ~30% of the time.
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


def _gen_def_stmt(
    rng: random.Random,
    max_paren_depth: int,
    p_call: float,
) -> str:
    """def_stmt := 'def' NAME '(' params ')' ':' return_stmt.

    Body is single-line ``return expr`` (NO indented block). Real Python
    accepts ``def f(): return 1`` as a compound stmt on one line.
    """
    name = rng.choice(_NAME_POOL)
    n_params = rng.randint(0, 3)
    if n_params == 0:
        params = ""
    else:
        params = ", ".join(rng.choices(_NAME_POOL, k=n_params))
    body_expr = _gen_expr(rng, depth=0, max_paren_depth=max_paren_depth, p_call=p_call)
    return f"def {name}({params}): return {body_expr}"


def _gen_top_stmt(
    rng: random.Random,
    max_paren_depth: int,
    p_def: float,
    p_call: float,
) -> str:
    """Generate one top-level statement.

    Top-level grammar: stmt := expr_stmt | assign_stmt | def_stmt.
    NOTE: top-level return_stmt is excluded -- bare ``return`` outside a
    function is a SyntaxError in real Python and would break the
    ast.parse contract.
    """
    r = rng.random()
    if r < p_def:
        return _gen_def_stmt(rng, max_paren_depth, p_call)
    # Use the remaining probability mass for assign / expr (50/50 within).
    if rng.random() < 0.5:
        target = rng.choice(_NAME_POOL)
        rhs = _gen_expr(rng, depth=0, max_paren_depth=max_paren_depth, p_call=p_call)
        return f"{target} = {rhs}"
    return _gen_expr(rng, depth=0, max_paren_depth=max_paren_depth, p_call=p_call)


def generate_python_big_source(
    *,
    n_programs: int,
    max_paren_depth: int = 3,
    p_def: float = 0.2,
    p_call: float = 0.3,
    seed: int,
    n_stmts_min: int = 1,
    n_stmts_max: int = 3,
) -> list[str]:
    """Generate ``n_programs`` random valid Python "big" programs.

    Every program is asserted to ``ast.parse`` cleanly -- if a generated
    program fails this round-trip the generator has a bug and we raise
    AssertionError immediately.

    Parameters
    ----------
    n_programs:
        Number of programs.
    max_paren_depth:
        Maximum parenthesis nesting depth. Must be in [1, 3] to stay
        within the FSM's vertex set.
    p_def:
        Probability a top-level statement is a def_stmt.
    p_call:
        Probability that a NAME inside a factor is followed by a call's
        argument list (turning it into a function call).
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
    if not 0.0 <= p_def <= 1.0:
        raise ValueError(f"p_def must be in [0, 1], got {p_def}")
    if not 0.0 <= p_call <= 1.0:
        raise ValueError(f"p_call must be in [0, 1], got {p_call}")
    if not 1 <= n_stmts_min <= n_stmts_max:
        raise ValueError(
            f"n_stmts bounds invalid: [{n_stmts_min}, {n_stmts_max}]"
        )

    rng = random.Random(seed)
    programs: list[str] = []
    for _ in range(n_programs):
        n_stmts = rng.randint(n_stmts_min, n_stmts_max)
        stmts = [
            _gen_top_stmt(rng, max_paren_depth, p_def, p_call)
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
# Tokenizer adapter (stdlib tokenize -> our 14-token alphabet)
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
"""Operator strings emitted by stdlib tokenize that map 1:1 to our
alphabet. Note that '=' / ':' / ',' are all tok.type == OP from the
stdlib tokenizer -- we just remap by string."""

_DROP_TOKEN_TYPES: frozenset[int] = frozenset({
    tokenize.ENCODING,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.COMMENT,
    tokenize.ENDMARKER,
    tokenize.NL,
})
"""Token types from stdlib ``tokenize`` that we drop. Single-line def
bodies never emit INDENT/DEDENT but we list them for safety."""


def tokenize_program_big(source: str) -> list[tuple[str, str]]:
    """Tokenize ``source`` and adapt to the 14-token python_big alphabet.

    Returns ``[(canonical_token, raw_string), ...]``. NAME tokens whose
    raw string is one of our reserved keywords (``def`` / ``return``) are
    emitted with the keyword as the canonical token rather than ``NAME``.
    All other ``tokenize.NAME``s map to ``NAME``.

    Sanity-checked against ``keyword.iskeyword`` so that any future Python
    keyword introduction (or user identifier collision) is caught loudly.
    """
    out: list[tuple[str, str]] = []
    bytestream = io.BytesIO(source.encode("utf-8"))
    for tok in tokenize.tokenize(bytestream.readline):
        if tok.type in _DROP_TOKEN_TYPES:
            continue
        if tok.type == tokenize.NAME:
            raw = tok.string
            if raw in _RESERVED:
                # Sanity: stdlib agrees these are keywords.
                assert keyword.iskeyword(raw), (
                    f"Internal: {raw!r} expected to be a Python keyword"
                )
                out.append((raw, raw))
                continue
            # Python's ``True`` / ``False`` / ``None`` etc. would also be
            # NAMEs that iskeyword reports True; if any of those appear in
            # generated source we want a loud failure.
            if keyword.iskeyword(raw):
                raise ValueError(
                    f"tokenize emitted unexpected keyword NAME {raw!r}; the "
                    f"python_big grammar only handles {sorted(_RESERVED)}."
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
                    f"python_big alphabet (allowed: "
                    f"{sorted(_TOKENIZE_OP_TO_TOKEN)})."
                )
            out.append((mapped, tok.string))
            continue
        raise ValueError(
            f"tokenize emitted unexpected token type "
            f"{tokenize.tok_name[tok.type]} ({tok.string!r}); python_big "
            f"alphabet cannot represent it."
        )
    return out


# ---------------------------------------------------------------------------
# FSM walker
# ---------------------------------------------------------------------------


def _build_edge_index(fsm: GraphFSM) -> dict[tuple[str, str], str]:
    """Return ``{(source, label): target}`` for the FSM's edges.

    Asserts the FSM is deterministic on ``(source, label)`` pairs --
    python_big IS deterministic by construction.
    """
    spec = fsm._spec  # noqa: SLF001 -- internal access required to read labels
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


def walk_fsm_big(
    token_stream: list[tuple[str, str]],
    fsm: GraphFSM,
) -> list[tuple[str, str, str]]:
    """Walk the FSM step-by-step over ``token_stream``.

    Returns ``[(prev_state, observed_token, next_state), ...]``. The first
    ``prev_state`` is always ``"START"``; after a NEWLINE that lands in
    ``ACCEPT``, the next statement re-enters from ``"START"``.

    Asserts every step is FSM-legal.
    """
    edges_by_src_token = _build_edge_index(fsm)
    out: list[tuple[str, str, str]] = []
    state = "START"
    for canonical, _raw in token_stream:
        next_state = edges_by_src_token.get((state, canonical))
        assert next_state is not None, (
            f"FSM has no edge from {state!r} on token {canonical!r}; "
            f"the source is not in the python_big subset or the FSM is "
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
    """Return a token from the alphabet that is ILLEGAL at ``state``.

    Returns ``None`` if every token in the alphabet is legal at this
    state (would be unusual for our grammar but kept for safety).
    """
    legal = {tok for (src, tok) in edges_by_src_token if src == state}
    illegal = [tok for tok in PYTHON_BIG_TOKENS if tok not in legal]
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
        "ACCEPT",
    ]


def generate_python_big_dataset(
    *,
    fsm: GraphFSM,
    n_programs: int = 100,
    max_paren_depth: int = 3,
    p_def: float = 0.2,
    p_call: float = 0.3,
    seed: int = 42,
    illegal_temptation_fraction: float = 0.10,
    n_stmts_min: int = 1,
    n_stmts_max: int = 3,
) -> PythonBigDataset:
    """Generate ``n_programs`` random valid python_big programs and walk them.

    For each program:
      1. Generate a source string via ``generate_python_big_source``
         (asserted to ``ast.parse`` cleanly).
      2. Tokenize via stdlib ``tokenize`` and adapt to the 14-token alphabet.
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
            "FSM vertex set does not match the python_big canonical layout. "
            f"Expected {expected!r}, got {fsm.vertex_ids!r}."
        )

    rng = random.Random(seed)
    programs = generate_python_big_source(
        n_programs=n_programs,
        max_paren_depth=max_paren_depth,
        p_def=p_def,
        p_call=p_call,
        seed=seed,
        n_stmts_min=n_stmts_min,
        n_stmts_max=n_stmts_max,
    )

    edges_by_src_token = _build_edge_index(fsm)
    samples: list[PythonBigSample] = []

    for prog_idx, source in enumerate(programs):
        token_stream = tokenize_program_big(source)
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
                PythonBigSample(
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
    samples: list[PythonBigSample],
    *,
    fsm: GraphFSM,
    max_paren_depth: int,
    programs: list[str],
) -> PythonBigDataset:
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

    return PythonBigDataset(
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
    ds: PythonBigDataset,
    *,
    seed: int,
    test_fraction: float = 0.25,
) -> tuple[PythonBigDataset, PythonBigDataset]:
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
