"""Python expression-subset dataset generator (first external benchmark).

Real Python source code -- validated via stdlib ``ast.parse`` -- tokenized
via stdlib ``tokenize`` and walked through a hand-authored parser FSM. Each
sample is one transition through the FSM:
``(prev_state, current_state, observed_token, features, true_next_state,
is_adversarial)``.

This atom is the first time the architecture's claims (mask / monodromy /
sigma at structural ambiguity / hyperbolic depth) fire on a published
language grammar instead of synthetic Dyck-k or generated ListOps.

Grammar (mirrors ``tests/fixtures/graphs/python_expr.fsm.yaml``):

    program     := stmt+
    stmt        := expr_stmt | assign_stmt
    expr_stmt   := expr NEWLINE
    assign_stmt := NAME '=' expr NEWLINE
    expr        := term (('+' | '-') term)*
    term        := factor (('*' | '/') factor)*
    factor      := NUMBER | NAME | '(' expr ')'

Token alphabet (10 token types -- identifier and number classes are
collapsed to a single token type each):

    NAME, NUMBER, +, -, *, /, (, ), =, NEWLINE

Feature design
--------------
Each step's feature vector is a 16-dim one-hot of the OBSERVED token
(matching the tokenizer index, padded with zeros up to 16 dims to match
the ListOps convention). No noise -- the dataset is purely syntactic;
the classifier's job is to predict the next FSM state given the current
state and the observed token, masked by the FSM's legal out-edges.
Noise lives in the adversarial-temptation samples, not in the feature
vectors.

State labels
------------
The FSM walker is the trajectory through states defined in
``python_expr.fsm.yaml``:

    START, ACCEPT,
    S0_after_name_at_start, S0_after_assign_eq,
    S{d}_factor       for d in [0, 4],
    S{d}_after_term   for d in [0, 4].

Why a Python-expression subset lets each architecture claim fire (better
than Dyck-k):
  * **Mask**: ``)`` is illegal at depth 0, ``(`` is illegal at max depth,
    ``=`` is legal ONLY at ``S0_after_name_at_start``, NEWLINE is legal
    only at ``S0_after_term`` and ``S0_after_name_at_start``. Mask is
    state-conditioned in several distinct ways.
  * **Hyperbolic**: parenthesis nesting depth is genuine parse-tree depth.
  * **sigma**: at ``S0_after_name_at_start`` the parser truly chooses
    between starting an assignment (``=``) and continuing an
    expression-statement (operator / NEWLINE) -- both legal, real
    structural ambiguity in Python.
  * **Monodromy**: any matched ``(...)`` is a closed walk through
    ``S{d}_after_term -> S{d+1}_factor -> ... -> S{d}_after_term``.

Adversarial samples
-------------------
At a configurable fraction (``illegal_temptation_fraction``), the
OBSERVED token is replaced with a token that is ILLEGAL at the current
FSM state (e.g. ``*`` immediately after ``(`` -- illegal because expr
must start with a factor not an operator). The walker advances along
the LEGAL token, so the trajectory stays in the FSM. ``is_adversarial``
is set on the sample.

External-benchmark contract
---------------------------
``generate_python_source`` produces strings that are real Python: each
generated program is asserted to ``ast.parse`` cleanly. The tokenizer
is the stdlib ``tokenize.tokenize`` (NOT a hand-rolled regex), so the
"external benchmark" claim is structural -- this is real Python, parsed
by real Python.
"""
from __future__ import annotations

import ast
import io
import random
import tokenize
from dataclasses import dataclass

import numpy as np

from nga.arch.graph_fsm import GraphFSM


__all__ = [
    "PYTHON_EXPR_TOKENS",
    "PYTHON_EXPR_TOKEN_INDEX",
    "FEATURE_DIM",
    "PythonExprSample",
    "PythonExprDataset",
    "generate_python_source",
    "tokenize_program",
    "walk_fsm",
    "generate_python_expr_dataset",
    "train_test_split_by_program",
]


# ---------------------------------------------------------------------------
# Token alphabet
# ---------------------------------------------------------------------------

PYTHON_EXPR_TOKENS: list[str] = [
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
]
"""Fixed canonical token order (10 entries). The one-hot feature index of
a token is its position in this list; the feature vector is padded with
zeros up to FEATURE_DIM = 16 to match the ListOps / Dyck convention."""

PYTHON_EXPR_TOKEN_INDEX: dict[str, int] = {
    tok: i for i, tok in enumerate(PYTHON_EXPR_TOKENS)
}

FEATURE_DIM: int = 16

# Token-class subsets used by the walker / generator.
_NAME_POOL: tuple[str, ...] = ("a", "b", "c", "x", "y", "z")
_NUMBER_POOL: tuple[str, ...] = tuple(str(d) for d in range(10))
_OPS_ADD: tuple[str, ...] = ("+", "-")
_OPS_MUL: tuple[str, ...] = ("*", "/")
_OPS_ALL: tuple[str, ...] = _OPS_ADD + _OPS_MUL


# ---------------------------------------------------------------------------
# Sample / dataset records
# ---------------------------------------------------------------------------


@dataclass
class PythonExprSample:
    """One transition sample in a Python-expression parse trace.

    Attributes
    ----------
    sample_id:
        Stable identifier ``prog{program_idx}_step{step_idx}``.
    features:
        Shape ``(16,)`` float64 one-hot vector over PYTHON_EXPR_TOKENS for
        the OBSERVED token (zero-padded to 16 dims; alphabet has 10).
    observed_token:
        The token symbol the classifier sees at this step. For
        adversarial samples this is the illegal "tempting" token; for
        ordinary samples it equals the FSM-legal token from the parsed
        source.
    prev_state:
        Vertex id at the START of this transition (alias of
        ``current_state`` -- kept distinct for symmetry with the
        runner's mask call site).
    current_state:
        Vertex id at the START of this transition.
    true_next_state:
        Vertex id at the END of this transition (after the LEGAL token
        is consumed, even for adversarial samples).
    is_adversarial:
        True iff ``observed_token`` is illegal at ``current_state``.
    program_id:
        Integer index of the parent program (one program may contain
        multiple statements, all sharing this id).
    depth:
        Parser parenthesis depth at the START of this step (extracted
        from the state id; ``START`` and ``ACCEPT`` map to depth 0).
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
class PythonExprDataset:
    """Container for Python-expression samples + materialized numpy arrays.

    Same shape as ``ListOpsDataset`` so downstream runners can swap the
    dataset without changes.

    Attributes
    ----------
    samples:
        Flat list of ``PythonExprSample`` records in generation order.
    feature_dim:
        Always 16.
    fsm:
        The GraphFSM used to validate transitions.
    max_depth:
        Maximum parenthesis nesting depth used during generation.
    programs:
        The original Python source strings (in generation order). Useful
        for human inspection and to verify the round-trip via ast.
    X:
        ``(N, 16)`` float64 one-hot feature array.
    y_next:
        ``(N,)`` int64 indices into ``fsm.vertex_ids`` of true next states.
    prev_states:
        ``(N,)`` int64 indices of the start state of each transition.
    current_states:
        Alias of ``prev_states`` (same content; kept distinct for
        runner clarity).
    is_adversarial:
        ``(N,)`` bool array.
    depths:
        ``(N,)`` int64 array of start-of-step parenthesis depths.
    """

    samples: list[PythonExprSample]
    feature_dim: int
    fsm: GraphFSM
    max_depth: int
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
    """Return a 16-dim zero-padded one-hot float64 vector for ``token``."""
    if token not in PYTHON_EXPR_TOKEN_INDEX:
        raise KeyError(f"Unknown Python-expression token {token!r}")
    vec = np.zeros(FEATURE_DIM, dtype=np.float64)
    vec[PYTHON_EXPR_TOKEN_INDEX[token]] = 1.0
    return vec


def _depth_of_state(state: str) -> int:
    """Extract the parenthesis depth from a state id.

    ``START``, ``ACCEPT``, and the special S0_* states all map to depth 0;
    ``S{d}_factor`` and ``S{d}_after_term`` map to depth ``d``.
    """
    if state in ("START", "ACCEPT"):
        return 0
    if not state.startswith("S"):
        raise ValueError(f"Unrecognised python_expr state id {state!r}")
    rest = state[1:]
    digit_part = rest.split("_", 1)[0]
    return int(digit_part)


# ---------------------------------------------------------------------------
# Source generator (random programs that PARSE via ast.parse)
# ---------------------------------------------------------------------------


def _gen_factor(rng: random.Random, depth: int, max_depth: int) -> str:
    """Recursive-descent generator for ``factor``.

    factor := NUMBER | NAME | '(' expr ')'

    Bias toward leaves grows with depth so generation always terminates.
    """
    # If we are already at max_depth we cannot open another paren, so
    # only NAME / NUMBER are legal here.
    if depth >= max_depth:
        if rng.random() < 0.5:
            return rng.choice(_NAME_POOL)
        return rng.choice(_NUMBER_POOL)

    # Bias toward leaves so trees terminate. p_paren falls off with depth.
    p_paren = max(0.05, 0.30 - 0.07 * depth)
    r = rng.random()
    if r < p_paren:
        inner = _gen_expr(rng, depth + 1, max_depth)
        return f"({inner})"
    if r < (1.0 + p_paren) / 2.0:
        return rng.choice(_NAME_POOL)
    return rng.choice(_NUMBER_POOL)


def _gen_term(rng: random.Random, depth: int, max_depth: int) -> str:
    """term := factor (('*' | '/') factor)* (small mean length)."""
    parts = [_gen_factor(rng, depth, max_depth)]
    while rng.random() < 0.30:
        op = rng.choice(_OPS_MUL)
        parts.append(op)
        parts.append(_gen_factor(rng, depth, max_depth))
    return " ".join(parts)


def _gen_expr(rng: random.Random, depth: int, max_depth: int) -> str:
    """expr := term (('+' | '-') term)* (small mean length)."""
    parts = [_gen_term(rng, depth, max_depth)]
    while rng.random() < 0.30:
        op = rng.choice(_OPS_ADD)
        parts.append(op)
        parts.append(_gen_term(rng, depth, max_depth))
    return " ".join(parts)


def _gen_stmt(rng: random.Random, max_depth: int) -> str:
    """stmt := expr_stmt | assign_stmt -- returns one source line."""
    if rng.random() < 0.5:
        # assign_stmt: NAME '=' expr
        target = rng.choice(_NAME_POOL)
        rhs = _gen_expr(rng, depth=0, max_depth=max_depth)
        return f"{target} = {rhs}"
    # expr_stmt
    return _gen_expr(rng, depth=0, max_depth=max_depth)


def generate_python_source(
    *,
    n_programs: int,
    max_depth: int = 3,
    seed: int,
    n_stmts_min: int = 1,
    n_stmts_max: int = 3,
) -> list[str]:
    """Generate ``n_programs`` random valid Python expression-only programs.

    Each program is ``n_stmts`` statements (uniform between
    ``n_stmts_min`` and ``n_stmts_max``) joined by newlines, terminated
    by a trailing newline. Every program is asserted to ``ast.parse``
    cleanly -- that's the external-benchmark contract.

    Parameters
    ----------
    n_programs:
        Number of programs to produce.
    max_depth:
        Maximum parenthesis nesting depth. Must be in [1, 4] to stay
        within the FSM's vertex set.
    seed:
        RNG seed for the random module's Random instance.
    n_stmts_min, n_stmts_max:
        Inclusive bounds on the number of statements per program.

    Returns
    -------
    list[str]
        Each entry is a Python source string, one program per entry.
        Each ends with ``\\n``.
    """
    if n_programs < 1:
        raise ValueError(f"n_programs must be >= 1, got {n_programs}")
    if not 1 <= max_depth <= 4:
        raise ValueError(f"max_depth must be in [1, 4], got {max_depth}")
    if not 1 <= n_stmts_min <= n_stmts_max:
        raise ValueError(
            f"n_stmts bounds invalid: [{n_stmts_min}, {n_stmts_max}]"
        )

    rng = random.Random(seed)
    programs: list[str] = []
    for _ in range(n_programs):
        n_stmts = rng.randint(n_stmts_min, n_stmts_max)
        stmts = [_gen_stmt(rng, max_depth) for _ in range(n_stmts)]
        source = "\n".join(stmts) + "\n"
        # Round-trip through real Python's parser. If this fails the
        # generator has a bug.
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(
                f"Generated program failed ast.parse: {exc!r}\n{source!r}"
            ) from exc
        programs.append(source)
    return programs


# ---------------------------------------------------------------------------
# Tokenizer adapter (stdlib tokenize -> our 10-token alphabet)
# ---------------------------------------------------------------------------


_TOKENIZE_OP_TO_TOKEN: dict[str, str] = {
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "(": "(",
    ")": ")",
    "=": "=",
}
"""Operator strings emitted by stdlib tokenize that map 1:1 to our alphabet."""

_DROP_TOKEN_TYPES: frozenset[int] = frozenset({
    tokenize.ENCODING,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.COMMENT,
    tokenize.ENDMARKER,
    tokenize.NL,  # non-logical newline (e.g. blank lines); our grammar uses NEWLINE only
})
"""Token types from stdlib ``tokenize`` that we drop before walking the FSM.

Subtleties:
  * ``ENCODING`` is the first token from ``tokenize.tokenize`` -- it
    encodes the source's declared encoding (e.g. utf-8) and is NOT a
    Python token. Always drop.
  * ``INDENT``/``DEDENT`` cannot occur in the expression subset (no
    blocks), but we list them for safety.
  * ``COMMENT`` is dropped because our grammar has no comment rule.
  * ``ENDMARKER`` is the EOF marker; not part of the alphabet.
  * ``NL`` is a non-logical newline (e.g. inside parentheses or a
    blank line). Our grammar uses logical NEWLINE only -- generated
    programs do not produce ``NL`` tokens, but we drop just in case.
"""


def tokenize_program(source: str) -> list[tuple[str, str]]:
    """Tokenize ``source`` via stdlib ``tokenize.tokenize``.

    Returns a list of ``(canonical_token, raw_string)`` pairs where
    ``canonical_token`` is one of PYTHON_EXPR_TOKENS. The raw string is
    kept for diagnostics / reconstruction (e.g. which NAME, which
    NUMBER literal); only ``canonical_token`` enters the feature vector.

    Parameters
    ----------
    source:
        A Python source string. Must end with a newline (the stdlib
        tokenizer requires this to emit a NEWLINE token).
    """
    out: list[tuple[str, str]] = []
    bytestream = io.BytesIO(source.encode("utf-8"))
    for tok in tokenize.tokenize(bytestream.readline):
        if tok.type in _DROP_TOKEN_TYPES:
            continue
        if tok.type == tokenize.NAME:
            out.append(("NAME", tok.string))
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
                    f"Tokenize emitted OP {tok.string!r} which is not in the "
                    f"python_expr alphabet (allowed: {sorted(_TOKENIZE_OP_TO_TOKEN)})."
                )
            out.append((mapped, tok.string))
            continue
        raise ValueError(
            f"Tokenize emitted unexpected token type "
            f"{tokenize.tok_name[tok.type]} ({tok.string!r}); expression "
            f"subset cannot represent it."
        )
    return out


# ---------------------------------------------------------------------------
# FSM walker
# ---------------------------------------------------------------------------


def _legal_next_state(
    fsm: GraphFSM,
    edges_by_src_token: dict[tuple[str, str], str],
    src: str,
    token: str,
) -> str | None:
    """Look up the unique next state for ``(src, token)`` if it exists."""
    return edges_by_src_token.get((src, token))


def _build_edge_index(fsm: GraphFSM) -> dict[tuple[str, str], str]:
    """Return ``{(source, label): target}`` for the FSM's edges.

    Assumes the FSM is deterministic on (source, label) pairs -- the
    python_expr FSM IS deterministic by construction.
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


def walk_fsm(
    token_stream: list[tuple[str, str]],
    fsm: GraphFSM,
) -> list[tuple[str, str, str]]:
    """Walk the FSM step-by-step over ``token_stream``.

    Parameters
    ----------
    token_stream:
        Output of ``tokenize_program``: a list of ``(canonical_token,
        raw_string)`` pairs. Only the canonical tokens drive the walk.
    fsm:
        The python_expr GraphFSM. Used both for legality lookup and to
        assert each step is legal.

    Returns
    -------
    list[tuple[str, str, str]]
        ``[(prev_state, observed_token, next_state), ...]`` for each
        token consumed. The first ``prev_state`` is always ``"START"``;
        after a NEWLINE that lands in ``ACCEPT``, the next statement
        re-enters from ``"START"``.
    """
    edges_by_src_token = _build_edge_index(fsm)
    out: list[tuple[str, str, str]] = []
    state = "START"
    for canonical, _raw in token_stream:
        next_state = _legal_next_state(fsm, edges_by_src_token, state, canonical)
        assert next_state is not None, (
            f"FSM has no edge from {state!r} on token {canonical!r}; "
            f"the source is not in the expression subset or the FSM is "
            f"missing an edge."
        )
        # Sanity-check via the FSM's own legality matrix.
        assert fsm.is_legal_transition(state, next_state), (
            f"FSM legality matrix rejects edge {state!r} -> {next_state!r}"
        )
        out.append((state, canonical, next_state))
        if next_state == "ACCEPT":
            # Multi-statement program: re-enter from START.
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
    legal = {
        tok for (src, tok) in edges_by_src_token if src == state
    }
    illegal = [tok for tok in PYTHON_EXPR_TOKENS if tok not in legal]
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
    ]


def generate_python_expr_dataset(
    *,
    fsm: GraphFSM,
    n_programs: int = 100,
    max_depth: int = 3,
    seed: int = 42,
    illegal_temptation_fraction: float = 0.10,
    n_stmts_min: int = 1,
    n_stmts_max: int = 3,
) -> PythonExprDataset:
    """Generate ``n_programs`` random valid Python programs and walk them.

    For each program:
      1. Generate a source string via ``generate_python_source`` (asserted
         to ``ast.parse`` cleanly).
      2. Tokenize via stdlib ``tokenize.tokenize`` and adapt to our 10-token
         alphabet.
      3. Walk the FSM from START; at each step optionally corrupt the
         observed token to an illegal alternative (the walker still
         advances along the legal token).

    Parameters
    ----------
    fsm:
        ``GraphFSM`` whose vertex_ids must match the python_expr canonical
        layout. ``max_depth=4`` is hard-coded into the YAML; the runtime
        ``max_depth`` here only bounds the SOURCE generator.
    n_programs:
        How many programs to produce.
    max_depth:
        Maximum parenthesis depth used by the source generator. Must be
        in [1, 4].
    seed:
        RNG seed.
    illegal_temptation_fraction:
        Fraction of samples whose ``observed_token`` is replaced with an
        illegal alternative (the walker still advances along the legal
        token). Set to 0.0 to disable.
    n_stmts_min, n_stmts_max:
        Inclusive bounds on statements per program.

    Returns
    -------
    PythonExprDataset
        Container with all samples and materialized numpy arrays.
    """
    if n_programs < 1:
        raise ValueError(f"n_programs must be >= 1, got {n_programs}")
    if not 1 <= max_depth <= 4:
        raise ValueError(f"max_depth must be in [1, 4], got {max_depth}")
    if not 0.0 <= illegal_temptation_fraction <= 1.0:
        raise ValueError(
            f"illegal_temptation_fraction must be in [0, 1], "
            f"got {illegal_temptation_fraction}"
        )

    expected = _expected_state_set()
    if fsm.vertex_ids != expected:
        raise ValueError(
            "FSM vertex set does not match the python_expr canonical layout. "
            f"Expected {expected!r}, got {fsm.vertex_ids!r}."
        )

    rng = random.Random(seed)
    programs = generate_python_source(
        n_programs=n_programs,
        max_depth=max_depth,
        seed=seed,
        n_stmts_min=n_stmts_min,
        n_stmts_max=n_stmts_max,
    )

    edges_by_src_token = _build_edge_index(fsm)
    samples: list[PythonExprSample] = []

    for prog_idx, source in enumerate(programs):
        token_stream = tokenize_program(source)
        # Walk the FSM. We need both the (state, next_state) per token and
        # also the OBSERVED-token field which may be corrupted -- so we
        # inline the walk here rather than calling walk_fsm (which
        # already asserts legality but doesn't expose the per-step
        # corruption hook).
        state = "START"
        for step_idx, (canonical, _raw) in enumerate(token_stream):
            legal_next = edges_by_src_token.get((state, canonical))
            assert legal_next is not None, (
                f"prog{prog_idx} step{step_idx}: FSM has no edge from "
                f"{state!r} on token {canonical!r} (source: {source!r})"
            )

            # Decide whether to corrupt this step. We always draw the
            # gating uniform first (to keep RNG consumption stable across
            # corruption / non-corruption paths up to that point) and then
            # only draw an illegal token if the gate says corrupt.
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
                PythonExprSample(
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

            # Advance along the LEGAL transition (regardless of corruption).
            if legal_next == "ACCEPT":
                # Multi-statement: next stmt re-enters from START.
                state = "START"
            else:
                state = legal_next

    return _materialize(
        samples,
        fsm=fsm,
        max_depth=max_depth,
        programs=programs,
    )


# ---------------------------------------------------------------------------
# Materialization + helpers
# ---------------------------------------------------------------------------


def _materialize(
    samples: list[PythonExprSample],
    *,
    fsm: GraphFSM,
    max_depth: int,
    programs: list[str],
) -> PythonExprDataset:
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

    return PythonExprDataset(
        samples=samples,
        feature_dim=FEATURE_DIM,
        fsm=fsm,
        max_depth=max_depth,
        programs=list(programs),
        X=X,
        y_next=y_next,
        prev_states=prev_states,
        current_states=prev_states.copy(),
        is_adversarial=is_adv,
        depths=depths,
    )


def train_test_split_by_program(
    ds: PythonExprDataset,
    *,
    seed: int,
    test_fraction: float = 0.25,
) -> tuple[PythonExprDataset, PythonExprDataset]:
    """Partition by ``program_id`` so whole programs are exclusive.

    Mirrors ``train_test_split_by_sequence`` from ``dataset_listops`` --
    splits at program granularity so the classifier never sees a
    transition whose siblings appear in the other split.
    """
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

    # Subset the corresponding source programs (preserves program_id ordering).
    # programs[i] corresponds to program_id == i in the parent dataset.
    train_programs = [
        ds.programs[i] for i in range(len(ds.programs)) if i in train_set
    ]
    test_programs = [
        ds.programs[i] for i in range(len(ds.programs)) if i in test_set
    ]

    train_ds = _materialize(
        train_samples,
        fsm=ds.fsm,
        max_depth=ds.max_depth,
        programs=train_programs,
    )
    test_ds = _materialize(
        test_samples,
        fsm=ds.fsm,
        max_depth=ds.max_depth,
        programs=test_programs,
    )
    return train_ds, test_ds
