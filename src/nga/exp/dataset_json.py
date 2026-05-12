"""JSON dataset generator (RFC 8259 subset, fourth grammar class).

Phase 16's role is to produce a 4th sigma-saturation data point that is
*not* Python and *not* the synthetic Dyck-k / ListOps grammars. The
architectural prediction: JSON's "expecting value" position has 7 legal
continuations (`{`, `[`, STRING, NUMBER, true, false, null) -- substantial
ambiguity. sigma should fire there. Whether sigma_uplift on failure-AUROC
is positive or negative is the load-bearing question.

Real RFC 8259 JSON, validated via stdlib ``json.loads``, tokenized by a
hand-written tokenizer (stdlib ``tokenize`` is for Python; JSON has its
own simple lexical structure).

Grammar (mirrors ``tests/fixtures/graphs/json.fsm.yaml``):

    value      := object | array | STRING | NUMBER | "true" | "false" | "null"
    object     := '{' (STRING ':' value (',' STRING ':' value)*)? '}'
    array      := '[' (value (',' value)*)? ']'

Token alphabet (11 tokens, padded to 16 dims for parity with the other
grammar classes):

    {, }, [, ], ,, :, STRING, NUMBER, true, false, null

Feature design
--------------
Each step's feature vector is a 16-dim one-hot of the OBSERVED token
(11 active alphabet entries plus 5 zero pad). No noise -- the dataset is
purely syntactic; the classifier's job is to predict the next FSM state
given the current state and the observed token, masked by the FSM's legal
out-edges.

State labels
------------
The FSM walker is the trajectory through states defined in
``json.fsm.yaml``:

    START, ACCEPT,
    S{d}_obj_key / _after_key / _after_colon / _after_value / _after_comma
        for d in [1, 3],
    S{d}_arr_first_value / _after_value / _after_comma for d in [1, 3].

Why a JSON parser FSM lets each architecture claim fire:
  * **Mask**: ``:`` is legal ONLY at S{d}_obj_after_key. ``,`` is legal
    ONLY at S{d}_obj_after_value or S{d}_arr_after_value. ``[`` / ``{``
    are forbidden at max-depth value sites. STRING is legal as a key in
    some contexts and as a value in others. State-conditioned in several
    distinct ways.
  * **Hyperbolic**: container nesting depth equals stack depth.
  * **sigma**: at every "expecting value" state (START, S{d}_obj_after_colon,
    S{d}_arr_first_value, S{d}_arr_after_comma) the parser truly chooses
    between 7 (or more) legal continuations -- the load-bearing prediction
    for Phase 16.
  * **Monodromy**: any matched ``{...}`` or ``[...]`` is a closed walk
    that descends and pops back.

Deterministic-walker subtlety
-----------------------------
The FSM is non-deterministic on close transitions: a ``}`` (or ``]``) at
depth d>=2 has multiple legal targets in the YAML (``S{d-1}_obj_after_value``
and ``S{d-1}_arr_after_value``) because the parent context cannot be
recovered from the current state alone. The walker maintains its own
parse-context stack to disambiguate -- it picks the unique edge whose
target matches the parent kind on the top of the stack. Every
walker-emitted transition still satisfies ``fsm.is_legal_transition``;
the FSM itself remains the single source of truth for legality.

External-benchmark contract
---------------------------
``generate_json_source`` produces strings that are real JSON: each
generated document is asserted to ``json.loads`` cleanly.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np

from nga.arch.graph_fsm import GraphFSM


__all__ = [
    "JSON_TOKENS",
    "JSON_TOKEN_INDEX",
    "FEATURE_DIM",
    "JsonSample",
    "JsonDataset",
    "generate_json_source",
    "tokenize_json",
    "walk_fsm_json",
    "generate_json_dataset",
    "train_test_split_by_document",
]


# ---------------------------------------------------------------------------
# Token alphabet
# ---------------------------------------------------------------------------

JSON_TOKENS: list[str] = [
    "{",
    "}",
    "[",
    "]",
    ",",
    ":",
    "STRING",
    "NUMBER",
    "true",
    "false",
    "null",
]
"""Fixed canonical token order (11 entries). The one-hot feature index of
a token is its position in this list; the feature vector is padded with
zeros up to FEATURE_DIM = 16 to match the ListOps / Dyck / python_expr
convention."""

JSON_TOKEN_INDEX: dict[str, int] = {tok: i for i, tok in enumerate(JSON_TOKENS)}

FEATURE_DIM: int = 16

# Token-class subsets used by the walker / generator.
_STRING_POOL: tuple[str, ...] = ("a", "key", "x", "name", "value")
_NUMBER_POOL: tuple[str, ...] = ("0", "1", "2", "42", "-3")
_PRIMITIVE_VALUE_TOKENS: tuple[str, ...] = (
    "STRING",
    "NUMBER",
    "true",
    "false",
    "null",
)


# ---------------------------------------------------------------------------
# Sample / dataset records
# ---------------------------------------------------------------------------


@dataclass
class JsonSample:
    """One transition sample in a JSON parse trace.

    Attributes
    ----------
    sample_id:
        Stable identifier ``doc{document_idx}_step{step_idx}``.
    features:
        Shape ``(16,)`` float64 one-hot vector over JSON_TOKENS for the
        OBSERVED token (zero-padded to 16 dims; alphabet has 11).
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
    document_id:
        Integer index of the parent document.
    depth:
        Parser nesting depth at the START of this step (extracted from
        the state id; ``START`` and ``ACCEPT`` map to depth 0).
    """

    sample_id: str
    features: np.ndarray
    observed_token: str
    prev_state: str
    current_state: str
    true_next_state: str
    is_adversarial: bool
    document_id: int
    depth: int


@dataclass
class JsonDataset:
    """Container for JSON samples + materialized numpy arrays.

    Same shape as ``PythonExprDataset`` so downstream runners can swap the
    dataset without changes.

    Attributes
    ----------
    samples:
        Flat list of ``JsonSample`` records in generation order.
    feature_dim:
        Always 16.
    fsm:
        The GraphFSM used to validate transitions.
    max_depth:
        Maximum container nesting depth used during generation (1..3).
    documents:
        The original JSON source strings (in generation order). Useful
        for human inspection and to verify the round-trip via json.loads.
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
        ``(N,)`` int64 array of start-of-step nesting depths.
    """

    samples: list[JsonSample]
    feature_dim: int
    fsm: GraphFSM
    max_depth: int
    documents: list[str]
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
    if token not in JSON_TOKEN_INDEX:
        raise KeyError(f"Unknown JSON token {token!r}")
    vec = np.zeros(FEATURE_DIM, dtype=np.float64)
    vec[JSON_TOKEN_INDEX[token]] = 1.0
    return vec


def _depth_of_state(state: str) -> int:
    """Extract the nesting depth from a state id.

    ``START`` and ``ACCEPT`` map to depth 0; ``S{d}_*`` maps to depth d.
    """
    if state in ("START", "ACCEPT"):
        return 0
    if not state.startswith("S"):
        raise ValueError(f"Unrecognised JSON state id {state!r}")
    rest = state[1:]
    digit_part = rest.split("_", 1)[0]
    return int(digit_part)


# ---------------------------------------------------------------------------
# Source generator (random documents that PARSE via json.loads)
# ---------------------------------------------------------------------------


def _gen_string_literal(rng: random.Random) -> str:
    """Emit a quoted JSON string literal from the fixed pool.

    Strings in the pool are simple ASCII identifiers, so they require
    no escaping. We re-use ``json.dumps`` to be explicit about correctness.
    """
    return json.dumps(rng.choice(_STRING_POOL))


def _gen_number_literal(rng: random.Random) -> str:
    """Emit a JSON number literal from the fixed pool."""
    return rng.choice(_NUMBER_POOL)


def _gen_value(
    rng: random.Random,
    depth: int,
    max_depth: int,
    p_object: float,
) -> str:
    """Recursive-descent generator for ``value``.

    At ``depth >= max_depth`` only primitives are produced (we cannot open
    another container). Otherwise we toss a coin biased toward containers
    via ``p_container`` that decays with depth, picking object vs array
    with ``p_object``.
    """
    if depth >= max_depth:
        return _gen_primitive(rng)

    # Bias toward leaves so trees terminate. p_container falls off with depth.
    p_container = max(0.20, 0.65 - 0.15 * depth)
    if rng.random() >= p_container:
        return _gen_primitive(rng)

    if rng.random() < p_object:
        return _gen_object(rng, depth + 1, max_depth, p_object)
    return _gen_array(rng, depth + 1, max_depth, p_object)


def _gen_primitive(rng: random.Random) -> str:
    """Emit a primitive JSON literal: STRING, NUMBER, true, false, or null."""
    kind = rng.randrange(5)
    if kind == 0:
        return _gen_string_literal(rng)
    if kind == 1:
        return _gen_number_literal(rng)
    if kind == 2:
        return "true"
    if kind == 3:
        return "false"
    return "null"


def _gen_object(
    rng: random.Random,
    depth: int,
    max_depth: int,
    p_object: float,
) -> str:
    """object := '{' (STRING ':' value (',' STRING ':' value)*)? '}'."""
    # 25% chance of an empty object.
    if rng.random() < 0.25:
        return "{}"

    n_pairs = 1
    while rng.random() < 0.45 and n_pairs < 4:
        n_pairs += 1

    # JSON spec forbids duplicate keys at the parser level (json.loads
    # actually accepts them and keeps the last), but we avoid them anyway
    # for clarity. With a 5-element pool this is easy.
    chosen_keys: set[str] = set()
    parts: list[str] = []
    while len(parts) < n_pairs:
        key = rng.choice(_STRING_POOL)
        if key in chosen_keys and len(chosen_keys) < len(_STRING_POOL):
            continue
        chosen_keys.add(key)
        key_lit = json.dumps(key)
        val = _gen_value(rng, depth, max_depth, p_object)
        parts.append(f"{key_lit}: {val}")
    return "{" + ", ".join(parts) + "}"


def _gen_array(
    rng: random.Random,
    depth: int,
    max_depth: int,
    p_object: float,
) -> str:
    """array := '[' (value (',' value)*)? ']'."""
    if rng.random() < 0.25:
        return "[]"

    n_elems = 1
    while rng.random() < 0.45 and n_elems < 5:
        n_elems += 1

    parts = [_gen_value(rng, depth, max_depth, p_object) for _ in range(n_elems)]
    return "[" + ", ".join(parts) + "]"


def generate_json_source(
    *,
    n_documents: int,
    max_depth: int = 3,
    p_object: float = 0.5,
    seed: int,
) -> list[str]:
    """Generate ``n_documents`` random valid JSON documents.

    Each document is asserted to ``json.loads`` cleanly -- that is the
    external-benchmark contract.

    The top-level value is biased toward a container (object or array)
    rather than a bare primitive, since most of the FSM trace lives inside
    containers and a bare-primitive top-level produces only one
    transition. With ``p_top_container = 0.95`` we still occasionally emit
    a primitive top-level for coverage of the START -> ACCEPT path.

    Parameters
    ----------
    n_documents:
        Number of documents to produce.
    max_depth:
        Maximum container nesting depth. Must be in [1, 3] to stay
        within the FSM's vertex set.
    p_object:
        Probability that a chosen container is an object (vs an array).
    seed:
        RNG seed.

    Returns
    -------
    list[str]
        Each entry is a JSON source string.
    """
    if n_documents < 1:
        raise ValueError(f"n_documents must be >= 1, got {n_documents}")
    if not 1 <= max_depth <= 3:
        raise ValueError(f"max_depth must be in [1, 3], got {max_depth}")
    if not 0.0 <= p_object <= 1.0:
        raise ValueError(f"p_object must be in [0, 1], got {p_object}")

    rng = random.Random(seed)
    documents: list[str] = []
    for _ in range(n_documents):
        # 95% containers, 5% bare primitives.
        if rng.random() < 0.95:
            if rng.random() < p_object:
                source = _gen_object(rng, depth=1, max_depth=max_depth, p_object=p_object)
            else:
                source = _gen_array(rng, depth=1, max_depth=max_depth, p_object=p_object)
        else:
            source = _gen_primitive(rng)

        # Round-trip through stdlib json.loads. If this fails the generator
        # has a bug.
        try:
            json.loads(source)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"Generated document failed json.loads: {exc!r}\n{source!r}"
            ) from exc
        documents.append(source)
    return documents


# ---------------------------------------------------------------------------
# Tokenizer (hand-written -- JSON has its own simple lexical structure)
# ---------------------------------------------------------------------------


_PUNCT_TOKENS: frozenset[str] = frozenset({"{", "}", "[", "]", ",", ":"})


def tokenize_json(source: str) -> list[str]:
    """Tokenize a JSON source string into the canonical 11-token alphabet.

    Skips whitespace; for STRING and NUMBER, skips the literal content and
    emits just the token kind. For ``true`` / ``false`` / ``null`` emits
    the keyword.

    Parameters
    ----------
    source:
        A JSON source string. Must be RFC 8259-valid (we do not perform
        full validation here -- the source is expected to come from
        ``generate_json_source`` which round-trips via ``json.loads``).

    Returns
    -------
    list[str]
        Token strings, each a member of JSON_TOKENS.
    """
    tokens: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        c = source[i]
        # Skip whitespace per RFC 8259: space, tab, LF, CR.
        if c in " \t\n\r":
            i += 1
            continue
        if c in _PUNCT_TOKENS:
            tokens.append(c)
            i += 1
            continue
        if c == '"':
            # STRING: scan to the matching unescaped close quote.
            j = i + 1
            while j < n:
                if source[j] == "\\":
                    # Skip the escape and the next char.
                    j += 2
                    continue
                if source[j] == '"':
                    break
                j += 1
            if j >= n:
                raise ValueError(
                    f"Unterminated JSON string starting at offset {i}: "
                    f"{source[i:i + 20]!r}..."
                )
            tokens.append("STRING")
            i = j + 1
            continue
        if c == "-" or c.isdigit():
            # NUMBER: scan a JSON number (integer / fraction / exponent).
            # We do not validate strictly; the source is known-valid.
            j = i + 1
            while j < n and source[j] in "0123456789.+-eE":
                j += 1
            tokens.append("NUMBER")
            i = j
            continue
        if c == "t" and source.startswith("true", i):
            tokens.append("true")
            i += 4
            continue
        if c == "f" and source.startswith("false", i):
            tokens.append("false")
            i += 5
            continue
        if c == "n" and source.startswith("null", i):
            tokens.append("null")
            i += 4
            continue
        raise ValueError(
            f"Unexpected character {c!r} at offset {i} in JSON source: "
            f"{source[max(0, i - 10):i + 20]!r}"
        )
    return tokens


# ---------------------------------------------------------------------------
# FSM walker (stack-aware to disambiguate close transitions)
# ---------------------------------------------------------------------------


def _build_edge_index(
    fsm: GraphFSM,
) -> dict[tuple[str, str], list[str]]:
    """Return ``{(source, label): [target, ...]}`` for the FSM's edges.

    Unlike python_expr's edge index (which assumed determinism), the JSON
    FSM is non-deterministic on close transitions: ``S{d}_obj_after_value
    --}-->`` can land in either ``S{d-1}_obj_after_value`` or
    ``S{d-1}_arr_after_value`` (or ``ACCEPT`` for d=1). The walker uses
    its own parse-context stack to pick the right target.
    """
    spec = fsm._spec  # noqa: SLF001 -- internal access required to read labels
    out: dict[tuple[str, str], list[str]] = {}
    for edge in spec.edges:
        if edge.label is None:
            continue
        key = (edge.source, edge.label)
        out.setdefault(key, []).append(edge.target)
    return out


def _resolve_close_target(
    candidates: list[str],
    parent_kind: str,
) -> str:
    """Pick the unique target whose state matches the parent kind.

    Parameters
    ----------
    candidates:
        Possible target state ids from the FSM edge index.
    parent_kind:
        One of ``"top"``, ``"obj"``, ``"arr"`` -- the immediate parent
        context popped off the walker's stack.
    """
    if parent_kind == "top":
        if "ACCEPT" in candidates:
            return "ACCEPT"
        raise AssertionError(
            f"close transition expected ACCEPT for top-level parent; "
            f"candidates={candidates!r}"
        )
    if parent_kind == "obj":
        for c in candidates:
            if c.endswith("_obj_after_value"):
                return c
        raise AssertionError(
            f"close transition expected a *_obj_after_value target; "
            f"candidates={candidates!r}"
        )
    if parent_kind == "arr":
        for c in candidates:
            if c.endswith("_arr_after_value"):
                return c
        raise AssertionError(
            f"close transition expected a *_arr_after_value target; "
            f"candidates={candidates!r}"
        )
    raise ValueError(f"unknown parent_kind {parent_kind!r}")


def _parent_kind_of_state(state: str) -> str:
    """Return the parent kind ("top", "obj", "arr") implied by ``state``.

    Used at open transitions ('{' / '[') to record what kind of container
    THE NEW container is nested inside -- the value to push onto the
    parse stack so a later close can pop it and disambiguate the target.
    """
    if state == "START":
        return "top"
    # *_obj_after_colon -> we're opening a new container as an object's value.
    # *_arr_first_value / *_arr_after_comma -> we're opening as an array elem.
    if state.endswith("_obj_after_colon"):
        return "obj"
    if state.endswith("_arr_first_value") or state.endswith("_arr_after_comma"):
        return "arr"
    raise ValueError(
        f"Cannot derive parent kind for open from state {state!r}"
    )


def walk_fsm_json(
    token_stream: list[str],
    fsm: GraphFSM,
) -> list[tuple[str, str, str]]:
    """Walk the JSON FSM step-by-step over ``token_stream``.

    Parameters
    ----------
    token_stream:
        Output of ``tokenize_json``: a list of canonical JSON tokens.
    fsm:
        The JSON GraphFSM. Used both for legality lookup and to assert
        each step is legal.

    Returns
    -------
    list[tuple[str, str, str]]
        ``[(prev_state, observed_token, next_state), ...]`` for each
        token consumed. The first ``prev_state`` is always ``"START"``.

    Notes
    -----
    The walker maintains a parse-context stack of parent kinds
    (``"top"``, ``"obj"``, ``"arr"``) to disambiguate close transitions
    (``}`` and ``]``) where the FSM has multiple legal targets. The kind
    pushed on open is the kind of THE CONTAINER WE ARE NESTED INSIDE
    AT TIME OF OPENING -- i.e. the parent of the newly opened container,
    not the kind of the new container itself. On close that pop yields
    the parent of the just-closed container, which is exactly what is
    needed to pick the correct close target.
    """
    edges_by_src_token = _build_edge_index(fsm)
    out: list[tuple[str, str, str]] = []
    state = "START"
    # Parse stack: parent kind for each open container (the kind of the
    # container we are NESTED INSIDE at the time the new one was opened).
    # The first push happens when we open the top-level container; "top"
    # is the parent of that root.
    parent_stack: list[str] = []

    for token in token_stream:
        candidates = edges_by_src_token.get((state, token))
        assert candidates, (
            f"FSM has no edge from {state!r} on token {token!r}; "
            f"the token stream is not in the JSON subset or the FSM is "
            f"missing an edge."
        )

        if token == "}" or token == "]":
            # Close: pop the parse stack and pick the right target.
            parent = parent_stack.pop()
            next_state = _resolve_close_target(candidates, parent)
        elif token == "{":
            # Open object: push the kind of the container we are CURRENTLY
            # NESTED INSIDE (the parent of the new object).
            assert len(candidates) == 1, (
                f"FSM expected single target for '{{' from {state!r}; "
                f"got {candidates!r}"
            )
            next_state = candidates[0]
            parent_stack.append(_parent_kind_of_state(state))
        elif token == "[":
            assert len(candidates) == 1, (
                f"FSM expected single target for '[' from {state!r}; "
                f"got {candidates!r}"
            )
            next_state = candidates[0]
            parent_stack.append(_parent_kind_of_state(state))
        else:
            # All other tokens are deterministic single-target.
            assert len(candidates) == 1, (
                f"FSM expected single target for {token!r} from {state!r}; "
                f"got {candidates!r}"
            )
            next_state = candidates[0]

        # Sanity-check via the FSM's own legality matrix.
        assert fsm.is_legal_transition(state, next_state), (
            f"FSM legality matrix rejects edge {state!r} -> {next_state!r}"
        )

        out.append((state, token, next_state))
        state = next_state

    return out


# ---------------------------------------------------------------------------
# Adversarial token picker
# ---------------------------------------------------------------------------


def _illegal_token_at(
    state: str,
    edges_by_src_token: dict[tuple[str, str], list[str]],
    rng: random.Random,
) -> str | None:
    """Return a token from the alphabet that is ILLEGAL at ``state``.

    Returns ``None`` if every token in the alphabet is legal at this
    state (would be unusual for our grammar but kept for safety).
    """
    legal = {tok for (src, tok) in edges_by_src_token if src == state}
    illegal = [tok for tok in JSON_TOKENS if tok not in legal]
    if not illegal:
        return None
    return rng.choice(illegal)


# ---------------------------------------------------------------------------
# Public dataset generator
# ---------------------------------------------------------------------------


def _expected_state_set() -> list[str]:
    """The vertex ids the FSM yaml is expected to provide (a sorted set).

    Order is not used for correctness; we compare against ``set(...)``.
    """
    expected = ["START", "ACCEPT"]
    for d in (1, 2, 3):
        expected.extend(
            [
                f"S{d}_obj_key",
                f"S{d}_obj_after_key",
                f"S{d}_obj_after_colon",
                f"S{d}_obj_after_value",
                f"S{d}_obj_after_comma",
                f"S{d}_arr_first_value",
                f"S{d}_arr_after_value",
                f"S{d}_arr_after_comma",
            ]
        )
    return expected


def generate_json_dataset(
    *,
    fsm: GraphFSM,
    n_documents: int = 200,
    max_depth: int = 3,
    p_object: float = 0.5,
    seed: int = 42,
    illegal_temptation_fraction: float = 0.10,
) -> JsonDataset:
    """Generate ``n_documents`` random valid JSON documents and walk them.

    For each document:
      1. Generate a source string via ``generate_json_source`` (asserted
         to ``json.loads`` cleanly).
      2. Tokenize via ``tokenize_json``.
      3. Walk the JSON FSM from START, maintaining a parse-context stack
         to disambiguate close transitions; at each step optionally
         corrupt the observed token to an illegal alternative (the walker
         still advances along the legal token).

    Parameters
    ----------
    fsm:
        ``GraphFSM`` whose vertex_ids must match the JSON canonical
        layout (26 states; START + 8 per depth for d in 1..3 + ACCEPT).
    n_documents:
        How many documents to produce.
    max_depth:
        Maximum container nesting depth used by the source generator.
        Must be in [1, 3].
    p_object:
        Probability that a chosen container is an object (vs an array).
    seed:
        RNG seed.
    illegal_temptation_fraction:
        Fraction of samples whose ``observed_token`` is replaced with an
        illegal alternative (the walker still advances along the legal
        token). Set to 0.0 to disable.

    Returns
    -------
    JsonDataset
        Container with all samples and materialized numpy arrays.
    """
    if n_documents < 1:
        raise ValueError(f"n_documents must be >= 1, got {n_documents}")
    if not 1 <= max_depth <= 3:
        raise ValueError(f"max_depth must be in [1, 3], got {max_depth}")
    if not 0.0 <= illegal_temptation_fraction <= 1.0:
        raise ValueError(
            f"illegal_temptation_fraction must be in [0, 1], "
            f"got {illegal_temptation_fraction}"
        )

    expected = set(_expected_state_set())
    if set(fsm.vertex_ids) != expected:
        missing = expected - set(fsm.vertex_ids)
        extra = set(fsm.vertex_ids) - expected
        raise ValueError(
            "FSM vertex set does not match the JSON canonical layout. "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )

    rng = random.Random(seed)
    documents = generate_json_source(
        n_documents=n_documents,
        max_depth=max_depth,
        p_object=p_object,
        seed=seed,
    )

    edges_by_src_token = _build_edge_index(fsm)
    samples: list[JsonSample] = []

    for doc_idx, source in enumerate(documents):
        token_stream = tokenize_json(source)
        # We inline the walk here so we can corrupt observed tokens AND
        # still maintain the parse-context stack. The walker logic mirrors
        # walk_fsm_json but exposes the per-step corruption hook.
        state = "START"
        parent_stack: list[str] = []
        for step_idx, token in enumerate(token_stream):
            candidates = edges_by_src_token.get((state, token))
            assert candidates, (
                f"doc{doc_idx} step{step_idx}: FSM has no edge from "
                f"{state!r} on token {token!r} (source: {source!r})"
            )

            # Compute the LEGAL next state (with stack disambiguation).
            if token == "}" or token == "]":
                parent = parent_stack.pop()
                legal_next = _resolve_close_target(candidates, parent)
            elif token == "{":
                legal_next = candidates[0]
                parent_stack.append(_parent_kind_of_state(state))
            elif token == "[":
                legal_next = candidates[0]
                parent_stack.append(_parent_kind_of_state(state))
            else:
                legal_next = candidates[0]

            # Decide whether to corrupt this step. We always draw the
            # gating uniform first (to keep RNG consumption stable across
            # corruption / non-corruption paths up to that point) and then
            # only draw an illegal token if the gate says corrupt.
            observed_token = token
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
                JsonSample(
                    sample_id=f"doc{doc_idx}_step{step_idx}",
                    features=features,
                    observed_token=observed_token,
                    prev_state=state,
                    current_state=state,
                    true_next_state=legal_next,
                    is_adversarial=is_adv,
                    document_id=doc_idx,
                    depth=depth_at_start,
                )
            )

            # Advance along the LEGAL transition (regardless of corruption).
            state = legal_next

    return _materialize(
        samples,
        fsm=fsm,
        max_depth=max_depth,
        documents=documents,
    )


# ---------------------------------------------------------------------------
# Materialization + helpers
# ---------------------------------------------------------------------------


def _materialize(
    samples: list[JsonSample],
    *,
    fsm: GraphFSM,
    max_depth: int,
    documents: list[str],
) -> JsonDataset:
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

    return JsonDataset(
        samples=samples,
        feature_dim=FEATURE_DIM,
        fsm=fsm,
        max_depth=max_depth,
        documents=list(documents),
        X=X,
        y_next=y_next,
        prev_states=prev_states,
        current_states=prev_states.copy(),
        is_adversarial=is_adv,
        depths=depths,
    )


def train_test_split_by_document(
    ds: JsonDataset,
    *,
    seed: int,
    test_fraction: float = 0.25,
) -> tuple[JsonDataset, JsonDataset]:
    """Partition by ``document_id`` so whole documents are exclusive.

    Mirrors ``train_test_split_by_program`` from ``dataset_python_expr``
    -- splits at document granularity so the classifier never sees a
    transition whose siblings appear in the other split.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            f"test_fraction must be in (0, 1), got {test_fraction}"
        )

    doc_ids = sorted({s.document_id for s in ds.samples})
    rng = np.random.default_rng(seed)
    shuffled = list(doc_ids)
    rng.shuffle(shuffled)  # type: ignore[arg-type]

    n_test = max(1, round(len(shuffled) * test_fraction))
    test_set = set(shuffled[:n_test])
    train_set = set(shuffled[n_test:])

    train_samples = [s for s in ds.samples if s.document_id in train_set]
    test_samples = [s for s in ds.samples if s.document_id in test_set]

    train_documents = [
        ds.documents[i] for i in range(len(ds.documents)) if i in train_set
    ]
    test_documents = [
        ds.documents[i] for i in range(len(ds.documents)) if i in test_set
    ]

    train_ds = _materialize(
        train_samples,
        fsm=ds.fsm,
        max_depth=ds.max_depth,
        documents=train_documents,
    )
    test_ds = _materialize(
        test_samples,
        fsm=ds.fsm,
        max_depth=ds.max_depth,
        documents=test_documents,
    )
    return train_ds, test_ds
