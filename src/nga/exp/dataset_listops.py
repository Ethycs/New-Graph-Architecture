"""ListOps synthetic dataset generator (Phase 8).

In-process generator for nested-operator expressions in the style of
Nangia & Bowman (2018, "ListOps: A Diagnostic Dataset for Latent Tree
Learning"). Each sample is one transition through the parser FSM:
``(prev_state, current_state, observed_token, features, true_next_state,
is_adversarial)``.

Grammar (mirrors ``tests/fixtures/graphs/listops.fsm.yaml``):

    expr         := digit | "[" op operand_list "]"
    op           := MAX | MIN | MED | SUM_MOD
    operand_list := expr (operand_list)?
    digit        := 0 | 1 | ... | 9

Token alphabet (16 tokens, fixed order):

    "[", "]", MAX, MIN, MED, SUM_MOD, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

Feature design
--------------
Each step's feature vector is a 16-dim one-hot of the OBSERVED token
(matching the tokenizer index). No noise -- the dataset is purely
syntactic; the classifier's job is to predict the next FSM state given
the current state and the observed token, masked by the FSM's legal
out-edges. Noise lives in the adversarial-temptation samples, not in
the feature vectors.

State labels
------------
The FSM walker is the trajectory through states defined in
``listops.fsm.yaml``. State ids are: ``START``, ``ACCEPT``, and
``S{d}_op`` / ``S{d}_need_operand`` / ``S{d}_after_operand`` for
``d in [1, max_depth]``.

Why ListOps lets each architecture claim fire (better than Dyck-k):
  * **Mask**: closing ``]`` is illegal at depth 0 or while still
    requiring a first operand; opening ``[`` is illegal at max depth.
    The mask is therefore state-conditioned in two distinct ways.
  * **Hyperbolic**: parse-tree depth equals stack depth.
  * **sigma**: at ``S{d}_after_operand``, the parser truly chooses
    between continuing the operand list and closing -- both legal.
  * **Monodromy**: any matched ``[ ... ]`` is a closed walk on the FSM.

Adversarial samples
-------------------
At a configurable fraction (``illegal_temptation_fraction``), when the
walker would legally close (``]`` from ``S{d}_after_operand``) the
OBSERVED token is replaced with a non-closing alternative that is
ILLEGAL at that state position. Specifically we replace ``]`` with a
random digit at moments where the FSM forbids digits -- for ListOps
we instead pick the inverse: replace a legal ``]`` close with the
illegal token ``]`` from a state where it is forbidden, OR we replace
a legal next token at an opening-bracket moment with ``]`` (which is
illegal because we just opened ``[`` and have no operand yet). The
second variant is the one we use: at ``S{d}_op`` or ``S{d}_need_operand``
positions we insert ``]`` (always illegal there) at fraction
``illegal_temptation_fraction``. The walker's INTERNAL state still
advances along the LEGAL token (the originally-sampled one); the
sample's features encode ``]`` and ``is_adversarial`` is set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nga.arch.graph_fsm import GraphFSM


__all__ = [
    "LISTOPS_TOKENS",
    "LISTOPS_TOKEN_INDEX",
    "ListOpsSample",
    "ListOpsDataset",
    "generate_listops_dataset",
    "to_arrays",
    "train_test_split_by_sequence",
]


# ---------------------------------------------------------------------------
# Token alphabet
# ---------------------------------------------------------------------------

LISTOPS_TOKENS: list[str] = [
    "[",
    "]",
    "MAX",
    "MIN",
    "MED",
    "SUM_MOD",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]
"""Fixed canonical token order. The one-hot feature index of a token is
its position in this list (16 entries)."""

LISTOPS_TOKEN_INDEX: dict[str, int] = {tok: i for i, tok in enumerate(LISTOPS_TOKENS)}

_OPS: tuple[str, ...] = ("MAX", "MIN", "MED", "SUM_MOD")
_DIGITS: tuple[str, ...] = tuple(str(d) for d in range(10))
_OPEN: str = "["
_CLOSE: str = "]"

FEATURE_DIM: int = 16


# ---------------------------------------------------------------------------
# Sample / dataset records
# ---------------------------------------------------------------------------


@dataclass
class ListOpsSample:
    """One transition sample in a ListOps parse trace.

    Attributes
    ----------
    sample_id:
        Stable identifier ``seq{seq_idx}_step{step_idx}``.
    features:
        Shape ``(16,)`` float64 one-hot vector over LISTOPS_TOKENS for
        the OBSERVED token.
    observed_token:
        The token symbol the classifier sees at this step. For
        adversarial samples this is the illegal "tempting" token; for
        ordinary samples it equals the FSM-legal token sampled by the
        walker.
    prev_state:
        Vertex id at the START of this transition (alias of
        ``current_state`` -- kept distinct for symmetry with the
        runner's mask call site).
    current_state:
        Vertex id at the START of this transition.
    true_next_state:
        Vertex id at the END of this transition (after the LEGAL token
        is consumed, even for adversarial samples). The mask zeroes the
        illegal next-state implied by the adversarial token; the label
        rewards predicting the LEGAL next-state.
    is_adversarial:
        True iff ``observed_token`` is illegal at ``current_state``.
    sequence_id:
        Integer index of the parent sequence.
    depth:
        Stack depth at the START of this step (extracted from the state
        id; ``START`` and ``ACCEPT`` map to depth 0).
    """

    sample_id: str
    features: np.ndarray
    observed_token: str
    prev_state: str
    current_state: str
    true_next_state: str
    is_adversarial: bool
    sequence_id: int
    depth: int


@dataclass
class ListOpsDataset:
    """Container for ListOps samples + materialized numpy arrays.

    Attributes
    ----------
    samples:
        Flat list of ``ListOpsSample`` records in generation order.
    feature_dim:
        Always 16 (one-hot over LISTOPS_TOKENS).
    fsm:
        The GraphFSM used to validate transitions. Defines vertex
        ordering for the index arrays.
    max_depth:
        Maximum nesting depth used during generation.
    X:
        ``(N, 16)`` float64 one-hot feature array.
    y_next:
        ``(N,)`` int64 indices into ``fsm.vertex_ids`` of true next
        states.
    prev_states:
        ``(N,)`` int64 indices of the start state of each transition.
    current_states:
        Alias of ``prev_states`` (same content; kept distinct for
        runner clarity).
    is_adversarial:
        ``(N,)`` bool array.
    depths:
        ``(N,)`` int64 array of start-of-step stack depths.
    """

    samples: list[ListOpsSample]
    feature_dim: int
    fsm: GraphFSM
    max_depth: int
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
    """Return a 16-dim one-hot float64 vector for ``token``."""
    if token not in LISTOPS_TOKEN_INDEX:
        raise KeyError(f"Unknown ListOps token {token!r}")
    vec = np.zeros(FEATURE_DIM, dtype=np.float64)
    vec[LISTOPS_TOKEN_INDEX[token]] = 1.0
    return vec


def _depth_of_state(state: str) -> int:
    """Extract the stack depth from a state id.

    ``START`` and ``ACCEPT`` are at depth 0; ``S{d}_*`` is at depth ``d``.
    """
    if state == "START" or state == "ACCEPT":
        return 0
    if not state.startswith("S"):
        raise ValueError(f"Unrecognised ListOps state id {state!r}")
    # state looks like "S2_after_operand"
    rest = state[1:]
    digit_part = rest.split("_", 1)[0]
    return int(digit_part)


# ---------------------------------------------------------------------------
# Walker: legal next tokens per state
# ---------------------------------------------------------------------------


def _legal_next_tokens(
    state: str,
    *,
    max_depth: int,
) -> list[tuple[str, str]]:
    """Return ``[(token, next_state), ...]`` legal at ``state``.

    The function encodes the same rules as ``listops.fsm.yaml``; it does
    NOT inspect the FSM (the FSM is consulted only for verification).
    Keeping this hand-written keeps the generator self-contained and
    fast while the FSM still acts as ground truth in tests.
    """
    if state == "ACCEPT":
        return []
    if state == "START":
        # digit -> ACCEPT, "[" -> S1_op (only if max_depth >= 1)
        out: list[tuple[str, str]] = [(d, "ACCEPT") for d in _DIGITS]
        if max_depth >= 1:
            out.append((_OPEN, "S1_op"))
        return out

    depth = _depth_of_state(state)

    if state == f"S{depth}_op":
        return [(op, f"S{depth}_need_operand") for op in _OPS]

    if state == f"S{depth}_need_operand":
        out = [(d, f"S{depth}_after_operand") for d in _DIGITS]
        if depth < max_depth:
            out.append((_OPEN, f"S{depth + 1}_op"))
        return out

    if state == f"S{depth}_after_operand":
        # Continue the operand list (digit or "[") OR close.
        out = [(d, f"S{depth}_after_operand") for d in _DIGITS]
        if depth < max_depth:
            out.append((_OPEN, f"S{depth + 1}_op"))
        if depth == 1:
            out.append((_CLOSE, "ACCEPT"))
        else:
            out.append((_CLOSE, f"S{depth - 1}_after_operand"))
        return out

    raise ValueError(f"Unrecognised ListOps state id {state!r}")


def _close_bias(depth: int, max_depth: int) -> float:
    """Probability of preferring a CLOSE action at S{d}_after_operand.

    Linear in depth: at depth 1 we want to CLOSE less (so sequences are
    a bit longer and richer); at max_depth we want to CLOSE more (so
    sequences eventually terminate). At ``depth == max_depth`` the
    walker cannot open further anyway; biasing toward close keeps
    sequences from being dominated by long flat operand lists at the
    shallowest depth that reached max.
    """
    if max_depth <= 1:
        return 0.5
    return 0.25 + 0.5 * (depth - 1) / max(1, max_depth - 1)


def _sample_legal_token(
    state: str,
    *,
    max_depth: int,
    rng: np.random.Generator,
) -> tuple[str, str]:
    """Sample a (token, next_state) pair uniformly from legal options.

    For ``S{d}_after_operand`` states, applies a depth-dependent bias
    toward closing (so sequences terminate). For other states the
    distribution is uniform over legal moves.
    """
    options = _legal_next_tokens(state, max_depth=max_depth)
    if not options:
        raise ValueError(
            f"No legal next-token from terminal state {state!r}"
        )

    if state.endswith("_after_operand"):
        depth = _depth_of_state(state)
        p_close = _close_bias(depth, max_depth)
        # Split options into close and non-close
        close_opts = [o for o in options if o[0] == _CLOSE]
        non_close_opts = [o for o in options if o[0] != _CLOSE]

        if close_opts and non_close_opts:
            if float(rng.random()) < p_close:
                pool = close_opts
            else:
                pool = non_close_opts
        else:
            pool = options
    else:
        pool = options

    idx = int(rng.integers(0, len(pool)))
    return pool[idx]


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------


def _expected_state_set(max_depth: int) -> list[str]:
    """The vertex ids the FSM yaml is expected to provide (in order)."""
    out: list[str] = ["START"]
    for d in range(1, max_depth + 1):
        out.append(f"S{d}_op")
        out.append(f"S{d}_need_operand")
        out.append(f"S{d}_after_operand")
    out.append("ACCEPT")
    return out


def generate_listops_dataset(
    *,
    fsm: GraphFSM,
    max_depth: int = 3,
    n_sequences: int = 200,
    max_length: int = 32,
    seed: int = 42,
    illegal_temptation_fraction: float = 0.10,
) -> ListOpsDataset:
    """Generate ``n_sequences`` random walks through the ListOps FSM.

    Parameters
    ----------
    fsm:
        ``GraphFSM`` whose vertex ids must match the ListOps canonical
        layout (``START``, ``S{d}_op``/``need_operand``/``after_operand``
        for ``d in [1, max_depth]``, ``ACCEPT``).
    max_depth:
        Maximum stack depth.
    n_sequences:
        Number of sequences to generate.
    max_length:
        Maximum number of tokens per sequence. The walker will stop
        early if it reaches ``ACCEPT``.
    seed:
        RNG seed.
    illegal_temptation_fraction:
        Fraction of ``S{d}_op`` / ``S{d}_need_operand`` steps where the
        OBSERVED token is replaced with ``]`` (illegal at those states).
        Set to 0.0 to disable adversarial samples.

    Returns
    -------
    ListOpsDataset
        Container with all samples and materialized numpy arrays.
    """
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    if n_sequences < 1:
        raise ValueError(f"n_sequences must be >= 1, got {n_sequences}")
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}")
    if not 0.0 <= illegal_temptation_fraction <= 1.0:
        raise ValueError(
            f"illegal_temptation_fraction must be in [0, 1], "
            f"got {illegal_temptation_fraction}"
        )

    expected = _expected_state_set(max_depth)
    if fsm.vertex_ids != expected:
        raise ValueError(
            "FSM vertex set does not match the ListOps canonical layout. "
            f"Expected {expected!r}, got {fsm.vertex_ids!r}."
        )

    rng = np.random.default_rng(seed)
    samples: list[ListOpsSample] = []

    for seq_idx in range(n_sequences):
        state = "START"
        for step in range(max_length):
            if state == "ACCEPT":
                break

            # Sample a legal (token, next_state) for the walker.
            legal_token, legal_next_state = _sample_legal_token(
                state, max_depth=max_depth, rng=rng
            )

            # Decide if we corrupt this step into an adversarial one.
            # Only states where "]" is ILLEGAL are valid corruption sites
            # (otherwise "]" might be a legal move, defeating the test).
            # "]" is illegal at: START, S{d}_op, S{d}_need_operand.
            corrupt = False
            if illegal_temptation_fraction > 0.0 and state != "ACCEPT":
                if state == "START" or state.endswith("_op") or \
                        state.endswith("_need_operand"):
                    if float(rng.random()) < illegal_temptation_fraction:
                        corrupt = True

            if corrupt:
                observed_token = _CLOSE
                is_adv = True
            else:
                observed_token = legal_token
                is_adv = False

            features = _one_hot_token(observed_token)
            depth_at_start = _depth_of_state(state)

            samples.append(
                ListOpsSample(
                    sample_id=f"seq{seq_idx}_step{step}",
                    features=features,
                    observed_token=observed_token,
                    prev_state=state,
                    current_state=state,
                    true_next_state=legal_next_state,
                    is_adversarial=is_adv,
                    sequence_id=seq_idx,
                    depth=depth_at_start,
                )
            )

            # Advance walker along the LEGAL transition regardless of
            # corruption (so the trajectory stays in the FSM).
            state = legal_next_state

    return _materialize(samples, fsm=fsm, max_depth=max_depth)


# ---------------------------------------------------------------------------
# Materialization + helpers
# ---------------------------------------------------------------------------


def _materialize(
    samples: list[ListOpsSample],
    *,
    fsm: GraphFSM,
    max_depth: int,
) -> ListOpsDataset:
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

    return ListOpsDataset(
        samples=samples,
        feature_dim=FEATURE_DIM,
        fsm=fsm,
        max_depth=max_depth,
        X=X,
        y_next=y_next,
        prev_states=prev_states,
        current_states=prev_states.copy(),
        is_adversarial=is_adv,
        depths=depths,
    )


def to_arrays(
    ds: ListOpsDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return ``(X, y_next, prev_state_idx, is_adversarial, depths, sample_ids)``.

    Convenience wrapper around the dataset's materialized arrays.
    """
    sample_ids = [s.sample_id for s in ds.samples]
    return (
        ds.X,
        ds.y_next,
        ds.prev_states,
        ds.is_adversarial,
        ds.depths,
        sample_ids,
    )


def train_test_split_by_sequence(
    ds: ListOpsDataset,
    *,
    seed: int,
    test_fraction: float = 0.25,
) -> tuple[ListOpsDataset, ListOpsDataset]:
    """Partition by ``sequence_id`` so whole sequences are exclusive.

    Mirrors ``train_test_split_by_sequence`` from
    ``dataset_dyck_k`` -- splits at sequence granularity so the
    classifier never sees a transition whose siblings appear in the
    other split.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            f"test_fraction must be in (0, 1), got {test_fraction}"
        )

    seq_ids = sorted({s.sequence_id for s in ds.samples})
    rng = np.random.default_rng(seed)
    shuffled = list(seq_ids)
    rng.shuffle(shuffled)  # type: ignore[arg-type]

    n_test = max(1, round(len(shuffled) * test_fraction))
    test_set = set(shuffled[:n_test])
    train_set = set(shuffled[n_test:])

    train_samples = [s for s in ds.samples if s.sequence_id in train_set]
    test_samples = [s for s in ds.samples if s.sequence_id in test_set]

    train_ds = _materialize(train_samples, fsm=ds.fsm, max_depth=ds.max_depth)
    test_ds = _materialize(test_samples, fsm=ds.fsm, max_depth=ds.max_depth)
    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Reconstruction helper (handy for tests/diagnostics)
# ---------------------------------------------------------------------------


def reconstruct_sequences(ds: ListOpsDataset) -> list[list[str]]:
    """Return the raw observed-token sequences per ``sequence_id``.

    Useful for human inspection: ``" ".join(seq)`` gives a printable
    expression like ``[ MAX 4 [ MIN 2 7 ] 1 ]``.
    """
    by_seq: dict[int, list[tuple[int, str]]] = {}
    for s in ds.samples:
        # Parse "seqK_stepJ"
        step_part = s.sample_id.split("_step")[1]
        step = int(step_part)
        by_seq.setdefault(s.sequence_id, []).append((step, s.observed_token))
    out: list[list[str]] = []
    for seq_id in sorted(by_seq.keys()):
        steps = sorted(by_seq[seq_id], key=lambda t: t[0])
        out.append([tok for _, tok in steps])
    return out
