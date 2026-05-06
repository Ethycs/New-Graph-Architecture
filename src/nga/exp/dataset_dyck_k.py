"""Dyck-k synthetic dataset generator.

In-process generator for sequences over the Dyck-k bracket language. Each
sequence is a balanced (or partially-balanced) string of opens and closes
over k bracket types, with maximum nesting depth bounded by ``max_depth``.

The FSM (see ``tests/fixtures/graphs/dyck_k.fsm.yaml``) has one state for the
empty stack ``S0`` plus ``max_depth * k`` states ``S{d}_{b}`` encoding stack
depth ``d`` and the topmost bracket ``b``. Transitions:

  * Open bracket ``b`` (when ``d < max_depth``): push ``b`` -> ``S{d+1}_{b}``.
  * Close matching bracket (when ``d >= 1`` and bracket equals top): pop.
    From ``S1_b`` close pops to ``S0``. From ``Sd_b`` (``d > 1``) close emits
    edges to BOTH ``S{d-1}_0`` and ``S{d-1}_1`` (the FSM cannot reconstruct
    the new top from the topmost-bracket abstraction alone, so this is
    non-deterministic on the underlying type).
  * Close non-matching bracket: NO edge. The legality mask must zero this.

Why this dataset matters for the architecture:

  * **Mask**: closing the wrong bracket is illegal. Adversarial-temptation
    samples force the classifier toward an illegal close; the mask must
    drive ``illegal_transition_rate`` to 0.
  * **sigma**: confidence drops as depth grows because the noisy padding
    noise is a larger fraction of the legal-class-mean signal at boundaries.
  * **Monodromy**: open(b) followed by close(b) is a length-2 closed walk
    on the FSM, producing many cycle revisits.
  * **Hyperbolic**: stack depth IS tree depth -- deep stacks are far apart
    in the underlying tree, while sibling brackets are close.

The runtime classifier features are NOT the FSM coordinates. Each step's
feature is:

  - one-hot encoding of the OBSERVED bracket symbol over 2*k entries
    (k opens, k closes), 4 dims for k=2;
  - scalar depth at this step (1 dim);
  - small Gaussian noise padding to ``feature_dim`` (default 16).

The label is the NEXT FSM state (the state after consuming this bracket).

Adversarial samples: with probability ``illegal_temptation_fraction``, a
legal close is REPLACED with a non-matching close. The bracket symbol that
the classifier sees is the wrong-type close; the true_next_state is then
ILLEGAL from the current_state. The mask zeros that transition.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nga.arch.graph_fsm import GraphFSM


__all__ = [
    "DyckSample",
    "DyckDataset",
    "generate_dyck_dataset",
    "to_arrays",
    "train_test_split_by_sequence",
]


@dataclass
class DyckSample:
    """One transition sample.

    Attributes
    ----------
    sample_id:
        Stable identifier, e.g. ``seq3_step7``.
    features:
        Shape ``(feature_dim,)`` float64 feature vector.
    prev_state:
        FSM vertex id of the state at the START of this transition (before
        consuming the observed bracket). None for step 0 if there is no
        prior state convention; in this dataset every step has a prev_state
        because each sequence starts at ``S0``.
    current_state:
        Same as ``prev_state`` -- the state from which the transition is
        leaving. Kept as a separate field for symmetry with the runner's
        legality-mask call site.
    true_next_state:
        Ground-truth FSM vertex id at the END of this transition (after
        consuming the observed bracket). For adversarial samples this is
        the FSM-illegal state implied by the wrong-bracket close; the
        ground-truth label is set to the LEGAL next state (matching close)
        so accuracy still rewards the correct choice.
    is_adversarial:
        True iff this sample's observed bracket is a non-matching close that
        violates the FSM legality from current_state.
    sequence_id:
        Integer index of the parent sequence; used by the train/test split.
    depth:
        Scalar depth (size of the stack) at the START of this step. Stored
        for the depth-vs-confidence diagnostic and used as a feature dim.
    """

    sample_id: str
    features: np.ndarray
    prev_state: str
    current_state: str
    true_next_state: str
    is_adversarial: bool
    sequence_id: int
    depth: int


@dataclass
class DyckDataset:
    """Container for Dyck samples with the FSM and feature metadata.

    Provides the materialized arrays expected by the runner:
      * X: shape (N, feature_dim)
      * y_next: shape (N,) int64 indices into fsm.vertex_ids for true_next_state
      * prev_states: shape (N,) int64 indices for current_state
      * current_states: alias for prev_states (kept for runner clarity)
      * is_adversarial: shape (N,) bool
      * depths: shape (N,) int64

    Attributes
    ----------
    samples:
        Flat list of DyckSample records in generation order.
    feature_dim:
        Dimensionality D of each features vector.
    fsm:
        The GraphFSM that defines vertex ordering.
    k:
        Number of bracket types (e.g. 2 for ``(`` and ``[``).
    max_depth:
        Maximum nesting depth allowed.
    X:
        Materialized feature array.
    y_next:
        Materialized true-next-state index array.
    prev_states:
        Materialized current-state index array (one per sample).
    current_states:
        Same content as ``prev_states``; kept as a distinct attribute for
        runner code that distinguishes the two semantically.
    is_adversarial:
        Materialized adversarial-flag array.
    depths:
        Materialized depth-at-start-of-step array.
    """

    samples: list[DyckSample]
    feature_dim: int
    fsm: GraphFSM
    k: int
    max_depth: int
    X: np.ndarray
    y_next: np.ndarray
    prev_states: np.ndarray
    current_states: np.ndarray
    is_adversarial: np.ndarray
    depths: np.ndarray


# ---------------------------------------------------------------------------
# State-encoding helpers
# ---------------------------------------------------------------------------

def _state_id(depth: int, top: int | None) -> str:
    """Return the FSM state id for ``(depth, top-of-stack)``.

    ``top`` is None iff depth == 0 (S0).
    """
    if depth == 0:
        return "S0"
    return f"S{depth}_{top}"


def _bracket_index(symbol: str, k: int) -> int:
    """Return one-hot index in [0, 2*k) for the symbol.

    open(b) maps to b (range [0, k)); close(b) maps to k + b (range [k, 2k)).
    """
    kind, b = symbol.split("_")
    b_idx = int(b)
    if kind == "open":
        return b_idx
    if kind == "close":
        return k + b_idx
    raise ValueError(f"Unrecognised bracket symbol {symbol!r}")


# ---------------------------------------------------------------------------
# Sequence sampling
# ---------------------------------------------------------------------------

def _step_action(
    rng: np.random.Generator,
    depth: int,
    max_depth: int,
) -> str:
    """Decide whether the next step is an open or a (matching) close.

    Returns "open" or "close". Probability of opening is biased toward
    ``(max_depth - depth) / max_depth`` so deep stacks tend to close. At
    depth 0, must open. At depth == max_depth, must close.
    """
    if depth == 0:
        return "open"
    if depth >= max_depth:
        return "close"
    open_p = (max_depth - depth) / max_depth
    return "open" if rng.random() < open_p else "close"


def generate_dyck_dataset(
    *,
    fsm: GraphFSM,
    k: int = 2,
    max_depth: int = 4,
    n_sequences: int = 200,
    max_length: int = 16,
    seed: int,
    illegal_temptation_fraction: float = 0.15,
    feature_dim: int = 16,
    noise_scale: float = 0.10,
) -> DyckDataset:
    """Generate ``n_sequences`` random Dyck-k walks and emit per-step samples.

    Each step produces one DyckSample. The walk:
      * starts at ``S0`` with empty stack;
      * at each step, with probability biased toward opening when depth
        is low, picks a bracket type uniformly and appends ``open_b`` or
        ``close_<top>``;
      * ends when ``max_length`` is reached.

    Adversarial transformation: at fraction ``illegal_temptation_fraction``,
    when the walker would legally emit ``close_<top>``, replace the
    OBSERVED bracket symbol with a non-matching close ``close_<other>``
    (uniform over the other k-1 bracket types). The walker's INTERNAL
    state still advances along the legal close (so the sequence keeps
    going); the sample's features encode the wrong bracket and
    ``is_adversarial`` is set. The mask zeros the implied illegal
    transition and the runner sees a tempted-but-mask-corrected step.

    Parameters
    ----------
    fsm:
        GraphFSM whose vertex_ids must match the dyck_k FSM (S0 plus
        per-(depth, top) states).
    k:
        Number of bracket types.
    max_depth:
        Maximum stack depth (inclusive).
    n_sequences:
        How many sequences to sample.
    max_length:
        Maximum number of steps per sequence.
    seed:
        Integer RNG seed.
    illegal_temptation_fraction:
        Fraction (approx) of close-steps to corrupt to non-matching close.
        Set to 0.0 to disable adversarial samples.
    feature_dim:
        Dimensionality of each features vector. Must be >= 2*k + 1.
    noise_scale:
        Std of the Gaussian noise added to the padding dims (and a small
        amount to the depth scalar).

    Returns
    -------
    DyckDataset
        Container with all samples and materialized numpy arrays.
    """
    if feature_dim < 2 * k + 1:
        raise ValueError(
            f"feature_dim={feature_dim} must be >= 2*k+1={2 * k + 1}"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")

    rng = np.random.default_rng(seed)

    # Verify FSM contains the expected states.
    expected_states = ["S0"] + [
        f"S{d}_{b}" for d in range(1, max_depth + 1) for b in range(k)
    ]
    if fsm.vertex_ids != expected_states:
        raise ValueError(
            "FSM vertex set does not match the Dyck-k canonical layout. "
            f"Expected {expected_states!r}, got {fsm.vertex_ids!r}."
        )

    samples: list[DyckSample] = []

    for seq_idx in range(n_sequences):
        # Walker tracks (depth, top-of-stack) directly per the FSM
        # abstraction. The walker is NOT a true bracket stack -- it is
        # the FSM trajectory, so its evolution matches the FSM exactly
        # (close from Sd_b lands on S{d-1}_b for d > 1, S0 for d == 1).
        depth = 0
        top: int | None = None
        for step in range(max_length):
            current_state = _state_id(depth, top)

            action = _step_action(rng, depth, max_depth)

            if action == "open":
                b = int(rng.integers(0, k))
                observed_symbol = f"open_{b}"
                next_depth = depth + 1
                next_top: int | None = b
                next_state = _state_id(next_depth, next_top)
                is_adv = False
            else:
                assert top is not None
                # Decide adversarial vs legal close.
                if k > 1 and float(rng.random()) < illegal_temptation_fraction:
                    other_options = [b for b in range(k) if b != top]
                    bad_b = int(other_options[int(rng.integers(0, len(other_options)))])
                    observed_symbol = f"close_{bad_b}"
                    is_adv = True
                else:
                    observed_symbol = f"close_{top}"
                    is_adv = False

                # The walker advances along the LEGAL close (matching top).
                # FSM convention: pop from Sd_b lands on S{d-1}_b for d>1,
                # S0 for d==1.
                next_depth = depth - 1
                if next_depth == 0:
                    next_top = None
                else:
                    next_top = top  # convention preserves bracket type
                next_state = _state_id(next_depth, next_top)

            # ---------------------------------------------------------
            # Build the feature vector.
            # Layout (feature_dim default 16, k=2):
            #   [0..2k-1]   one-hot of bracket symbol  (4 dims for k=2)
            #   [2k]        depth scalar (clipped to max_depth)
            #   [2k+1..]    Gaussian noise padding (mean 0, std noise_scale)
            # ---------------------------------------------------------
            features = np.zeros(feature_dim, dtype=np.float64)
            sym_idx = _bracket_index(observed_symbol, k)
            features[sym_idx] = 1.0
            features[2 * k] = float(depth)
            pad_n = feature_dim - (2 * k + 1)
            if pad_n > 0:
                features[2 * k + 1:] = rng.normal(0.0, noise_scale, size=pad_n)
            # Add a small jitter to the depth scalar (keeps it deterministic
            # but tiny; classifier should still learn the structure).
            features[2 * k] += float(rng.normal(0.0, noise_scale * 0.5))

            samples.append(
                DyckSample(
                    sample_id=f"seq{seq_idx}_step{step}",
                    features=features,
                    prev_state=current_state,
                    current_state=current_state,
                    true_next_state=next_state,
                    is_adversarial=is_adv,
                    sequence_id=seq_idx,
                    depth=depth,
                )
            )

            # Advance the walker state for the next step.
            depth = next_depth
            top = next_top

    return _materialize(samples, feature_dim=feature_dim, fsm=fsm, k=k, max_depth=max_depth)


def _materialize(
    samples: list[DyckSample],
    *,
    feature_dim: int,
    fsm: GraphFSM,
    k: int,
    max_depth: int,
) -> DyckDataset:
    """Pack a sample list into the numpy arrays the runner consumes."""
    n = len(samples)
    X = np.empty((n, feature_dim), dtype=np.float64)
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

    return DyckDataset(
        samples=samples,
        feature_dim=feature_dim,
        fsm=fsm,
        k=k,
        max_depth=max_depth,
        X=X,
        y_next=y_next,
        prev_states=prev_states,
        current_states=prev_states.copy(),
        is_adversarial=is_adv,
        depths=depths,
    )


# ---------------------------------------------------------------------------
# Convenience: arrays + split
# ---------------------------------------------------------------------------

def to_arrays(
    ds: DyckDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return ``(X, y_next, prev_state_idx, is_adversarial, depths, sample_ids)``.

    Mostly a convenience for the runner; the underlying arrays already live
    on the dataset.
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
    ds: DyckDataset,
    *,
    seed: int,
    test_fraction: float = 0.25,
) -> tuple[DyckDataset, DyckDataset]:
    """Partition a DyckDataset by sequence_id.

    Splits whole sequences (not individual transitions) into train/test
    so that the classifier never sees a transition whose neighbours appear
    in the other split. Mirrors ``train_test_split_by_trajectory`` from
    ``dataset_synthetic_babyai_grid``.
    """
    seq_ids = sorted({s.sequence_id for s in ds.samples})
    rng = np.random.default_rng(seed)
    shuffled = list(seq_ids)
    rng.shuffle(shuffled)  # type: ignore[arg-type]

    n_test = max(1, round(len(shuffled) * test_fraction))
    test_set = set(shuffled[:n_test])
    train_set = set(shuffled[n_test:])

    train_samples = [s for s in ds.samples if s.sequence_id in train_set]
    test_samples = [s for s in ds.samples if s.sequence_id in test_set]

    train_ds = _materialize(
        train_samples,
        feature_dim=ds.feature_dim,
        fsm=ds.fsm,
        k=ds.k,
        max_depth=ds.max_depth,
    )
    test_ds = _materialize(
        test_samples,
        feature_dim=ds.feature_dim,
        fsm=ds.fsm,
        k=ds.k,
        max_depth=ds.max_depth,
    )
    return train_ds, test_ds
