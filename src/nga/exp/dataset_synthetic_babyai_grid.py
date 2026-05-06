"""Synthetic BabyAI grid dataset generator.

In-process generator producing trajectories over a 7-state FSM:
    Parse, Navigate, ResolveDoor, Pickup, Deliver, Interact, Done
matching tests/fixtures/graphs/babyai_synthetic.fsm.yaml.

Each "trajectory" is a sequence of (feature_vector, true_state, prev_state)
triples. The classifier's job is to predict true_state given feature_vector;
prev_state is supplied to the graph mask at inference time.

The generator is deterministic given a seed. Features are simple synthetic
vectors with class-specific means + Gaussian noise; difficulty is controlled
by `noise_scale` and a fraction of "ambiguous" samples whose features are
deliberately near a class boundary (used to populate the low-margin tail).

Feature design
--------------
The per-class mean is a one-hot vector in the first V dims of a D-dimensional
feature vector, padded with zeros. This is intentionally easy so the
classifier achieves >0.8 baseline accuracy quickly, keeping the focus of the
experiment on the graph-mask and singularity-detector effects rather than
classifier tuning.

Start vertex selection
----------------------
The start vertex is the vertex with no incoming edges in the legality matrix
(column sum == 0). In the babyai_synthetic FSM this is uniquely "Parse". If
multiple vertices had no incoming edges the alphabetically-first would be
chosen; that rule is documented here for reproducibility but does not fire on
the current FSM.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nga.arch.graph_fsm import GraphFSM


__all__ = [
    "SyntheticGridSample",
    "SyntheticGridDataset",
    "generate_dataset",
    "to_features_and_labels",
    "train_test_split_by_trajectory",
]


@dataclass
class SyntheticGridSample:
    """One step in a synthetic grid trajectory.

    Attributes
    ----------
    sample_id:
        Unique identifier, e.g. "traj3_step7".
    features:
        Shape (D,) float64 feature vector for this step.
    true_state:
        The ground-truth vertex id for this step (one of the 7 FSM vertices).
    prev_state:
        The true_state of the previous step in the same trajectory, or None
        on step 0 (the first step has no prior context).
    is_ambiguous:
        True when the feature vector was drawn near the midpoint of two
        adjacent class means, making classification low-margin by design.
    is_illegal:
        True when the feature vector was drawn around an FSM-illegal target
        state (a vertex not reachable from prev_state). The classifier will be
        tempted to predict that illegal state; the graph mask must catch it.
    trajectory_id:
        Integer index of the trajectory this sample belongs to. Used by
        train_test_split_by_trajectory to partition at trajectory granularity.
    """

    sample_id: str
    features: np.ndarray
    true_state: str
    prev_state: str | None
    is_ambiguous: bool
    is_illegal: bool
    trajectory_id: int


@dataclass
class SyntheticGridDataset:
    """Collection of SyntheticGridSample records from all trajectories.

    Attributes
    ----------
    samples:
        Flat list of all samples across all trajectories, in generation order.
    feature_dim:
        Dimensionality D of each sample's feature vector.
    fsm:
        The GraphFSM instance used to generate the dataset. Callers can use
        this to recover the vertex set, legality matrix, and apply_mask.
    """

    samples: list[SyntheticGridSample]
    feature_dim: int
    fsm: GraphFSM


def _build_class_means(vertex_ids: list[str], feature_dim: int) -> dict[str, np.ndarray]:
    """Return a dict mapping each vertex id to its D-dimensional mean vector.

    The mean for vertex at index i is a one-hot vector with a 1.0 at position i
    and 0.0 elsewhere, padded to feature_dim. This guarantees that classes are
    well-separated in the first V dimensions and the classifier can achieve
    high accuracy with low noise_scale.

    Parameters
    ----------
    vertex_ids:
        Ordered list of FSM vertex ids (length V).
    feature_dim:
        Total feature dimensionality D (must be >= V).

    Returns
    -------
    dict mapping vertex_id -> np.ndarray of shape (D,).
    """
    v = len(vertex_ids)
    means: dict[str, np.ndarray] = {}
    for i, vid in enumerate(vertex_ids):
        mean = np.zeros(feature_dim, dtype=np.float64)
        mean[i % feature_dim] = 1.0  # one-hot in dim i; mod guards D < V edge case
        means[vid] = mean
    return means


def _find_start_vertex(fsm: GraphFSM) -> str:
    """Return the vertex id with no incoming edges.

    Examines fsm.legality_matrix column sums. A column sum of 0 means no
    other vertex has an edge pointing to that vertex - i.e. it is a root.
    If multiple roots exist the alphabetically-first is returned.

    Parameters
    ----------
    fsm:
        The GraphFSM whose legality_matrix is inspected.

    Returns
    -------
    str
        Vertex id of the chosen start vertex.

    Raises
    ------
    ValueError
        If the legality matrix has no column with all-False entries (i.e. all
        vertices have at least one incoming edge, which would be a fully cyclic
        graph with no valid start point).
    """
    mat = fsm.legality_matrix  # shape (V, V); mat[i, j] = edge i -> j
    col_sums = mat.sum(axis=0)  # sum over incoming edges for each column j
    root_indices = [j for j, s in enumerate(col_sums) if s == 0]
    if not root_indices:
        raise ValueError(
            "FSM has no vertex with zero incoming edges; cannot determine start vertex."
        )
    # Sort by vertex_id alphabetically and pick first.
    root_ids = sorted(fsm.vertex_ids[j] for j in root_indices)
    return root_ids[0]


def _legal_successors(fsm: GraphFSM, vertex_id: str) -> list[str]:
    """Return list of vertex ids reachable from vertex_id in one step.

    Parameters
    ----------
    fsm:
        The GraphFSM instance.
    vertex_id:
        The current vertex id.

    Returns
    -------
    list[str]
        Possibly-empty list of successor vertex ids.
    """
    i = fsm.index_of(vertex_id)
    row = fsm.legality_matrix[i]  # shape (V,)
    return [fsm.vertex_ids[j] for j in range(fsm.vertex_count) if row[j]]


def generate_dataset(
    fsm: GraphFSM,
    *,
    seed: int,
    n_trajectories: int = 50,
    steps_per_trajectory: tuple[int, int] = (4, 12),
    feature_dim: int = 32,
    noise_scale: float = 0.30,
    ambiguous_fraction: float = 0.20,
    illegal_temptation_fraction: float = 0.15,
) -> SyntheticGridDataset:
    """Build a synthetic dataset of grid trajectories.

    Each trajectory is a random walk over the FSM using only legal edges,
    starting from the start vertex (the one with no incoming edges; "Parse" in
    the babyai_synthetic FSM). The walk terminates when it reaches a vertex
    with no outgoing edges or when the step budget is exhausted.

    For each step the generator draws a feature vector as follows:

    - Normal sample: ``np.random.normal(class_mean, noise_scale, feature_dim)``.
    - Ambiguous sample (probability ``ambiguous_fraction``): feature is drawn
      near the midpoint of the true-state mean and an adjacent class mean, with
      small noise. ``is_ambiguous`` is set to True; ``true_state`` is still the
      actual FSM state at that step.
    - Illegal-temptation sample (probability ``illegal_temptation_fraction``,
      applied only when an illegal target exists): feature is drawn around the
      mean of an FSM-illegal state (one not reachable from prev_state). The
      true_state remains the actual current FSM state. ``is_illegal`` is set to
      True. These samples test whether the graph mask can override a tempting
      wrong prediction.

    The ambiguous and illegal flags are mutually exclusive: when both would
    fire on the same step, illegal takes priority.

    All RNG is derived from ``np.random.default_rng(seed)``, ensuring full
    reproducibility.

    Parameters
    ----------
    fsm:
        The GraphFSM instance whose vertex set and legality matrix define the
        state machine.
    seed:
        Integer seed for the numpy RNG.
    n_trajectories:
        Number of trajectories to generate.
    steps_per_trajectory:
        (min_steps, max_steps) pair. Actual trajectory length is drawn
        uniformly from this range.
    feature_dim:
        Dimensionality D of each feature vector (must be >= 1).
    noise_scale:
        Standard deviation of the Gaussian noise added to class means.
    ambiguous_fraction:
        Fraction of samples (approx) drawn from the mid-boundary distribution.
    illegal_temptation_fraction:
        Fraction of samples (approx) whose features are drawn around an
        FSM-illegal state, stressing the graph mask.

    Returns
    -------
    SyntheticGridDataset
        Dataset with all samples flattened across trajectories.
    """
    rng = np.random.default_rng(seed)
    vertex_ids = fsm.vertex_ids
    class_means = _build_class_means(vertex_ids, feature_dim)
    start_vertex = _find_start_vertex(fsm)

    min_steps, max_steps = steps_per_trajectory
    samples: list[SyntheticGridSample] = []

    for traj_idx in range(n_trajectories):
        n_steps = int(rng.integers(min_steps, max_steps + 1))
        current_state = start_vertex
        prev_state: str | None = None

        for step in range(n_steps):
            sample_id = f"traj{traj_idx}_step{step}"
            true_state = current_state
            mean = class_means[true_state]

            # Determine which kind of sample to generate.
            # Illegal-temptation has priority over ambiguous.
            roll = float(rng.random())

            is_illegal = False
            is_ambiguous = False

            # Collect illegal states (reachable from prev_state's perspective
            # is what the mask uses, but here we track based on prev_state).
            illegal_states: list[str] = []
            if prev_state is not None:
                legal_from_prev = set(_legal_successors(fsm, prev_state))
                illegal_states = [v for v in vertex_ids if v not in legal_from_prev]

            if roll < illegal_temptation_fraction and illegal_states:
                # Draw features around an illegal state's mean.
                is_illegal = True
                illegal_target = illegal_states[int(rng.integers(0, len(illegal_states)))]
                illegal_mean = class_means[illegal_target]
                features = rng.normal(
                    loc=illegal_mean, scale=noise_scale, size=feature_dim
                ).astype(np.float64)
            elif roll < illegal_temptation_fraction + ambiguous_fraction:
                # Draw features near the midpoint of true_state and an adjacent class.
                successors = _legal_successors(fsm, true_state)
                if successors:
                    is_ambiguous = True
                    neighbor_id = successors[int(rng.integers(0, len(successors)))]
                    neighbor_mean = class_means[neighbor_id]
                    boundary_mean = (mean + neighbor_mean) * 0.5
                    features = rng.normal(
                        loc=boundary_mean, scale=noise_scale * 0.5, size=feature_dim
                    ).astype(np.float64)
                else:
                    # No successors for ambiguous midpoint; fall back to normal sample.
                    features = rng.normal(
                        loc=mean, scale=noise_scale, size=feature_dim
                    ).astype(np.float64)
            else:
                # Normal sample.
                features = rng.normal(
                    loc=mean, scale=noise_scale, size=feature_dim
                ).astype(np.float64)

            samples.append(
                SyntheticGridSample(
                    sample_id=sample_id,
                    features=features,
                    true_state=true_state,
                    prev_state=prev_state,
                    is_ambiguous=is_ambiguous,
                    is_illegal=is_illegal,
                    trajectory_id=traj_idx,
                )
            )

            # Advance state along a random legal edge, or stop if none.
            successors = _legal_successors(fsm, current_state)
            if not successors:
                break  # terminal state reached; end trajectory early
            prev_state = current_state
            current_state = successors[int(rng.integers(0, len(successors)))]

    return SyntheticGridDataset(
        samples=samples,
        feature_dim=feature_dim,
        fsm=fsm,
    )


def to_features_and_labels(
    ds: SyntheticGridDataset,
) -> tuple[np.ndarray, np.ndarray, list[str | None], list[str]]:
    """Split a SyntheticGridDataset into arrays suitable for sklearn-style training.

    Parameters
    ----------
    ds:
        The dataset to unpack.

    Returns
    -------
    X:
        Shape (N, D) float64 array of feature vectors.
    y_idx:
        Shape (N,) int64 array of label indices into ``ds.fsm.vertex_ids``.
    prev_states:
        Length-N list of prev_state strings (or None for step-0 samples).
    sample_ids:
        Length-N list of sample_id strings.
    """
    vertex_index = ds.fsm.vertex_index
    n = len(ds.samples)
    d = ds.feature_dim

    X = np.empty((n, d), dtype=np.float64)
    y_idx = np.empty(n, dtype=np.int64)
    prev_states: list[str | None] = []
    sample_ids: list[str] = []

    for i, s in enumerate(ds.samples):
        X[i] = s.features
        y_idx[i] = vertex_index[s.true_state]
        prev_states.append(s.prev_state)
        sample_ids.append(s.sample_id)

    return X, y_idx, prev_states, sample_ids


def train_test_split_by_trajectory(
    ds: SyntheticGridDataset,
    *,
    seed: int,
    test_fraction: float = 0.25,
) -> tuple[SyntheticGridDataset, SyntheticGridDataset]:
    """Split a dataset into train and test sets at trajectory granularity.

    Splitting at the trajectory level avoids temporal leakage: each sample's
    prev_state refers to the previous step in the SAME trajectory. If samples
    were split individually, a test step's prev_state could appear in training
    data and vice versa, artificially inflating performance estimates.

    Parameters
    ----------
    ds:
        The source dataset (from generate_dataset).
    seed:
        RNG seed for the trajectory shuffle.
    test_fraction:
        Fraction of trajectories assigned to the test split (approximately).

    Returns
    -------
    train_ds, test_ds:
        Two SyntheticGridDataset instances sharing the same FSM and feature_dim
        but with disjoint trajectory sets.
    """
    # Collect all unique trajectory ids in original order.
    seen: dict[int, None] = {}
    for s in ds.samples:
        seen[s.trajectory_id] = None
    all_traj_ids = list(seen.keys())

    rng = np.random.default_rng(seed)
    shuffled = list(all_traj_ids)
    rng.shuffle(shuffled)  # type: ignore[arg-type]

    n_test = max(1, round(len(shuffled) * test_fraction))
    test_traj_set = set(shuffled[:n_test])
    train_traj_set = set(shuffled[n_test:])

    train_samples = [s for s in ds.samples if s.trajectory_id in train_traj_set]
    test_samples = [s for s in ds.samples if s.trajectory_id in test_traj_set]

    train_ds = SyntheticGridDataset(
        samples=train_samples,
        feature_dim=ds.feature_dim,
        fsm=ds.fsm,
    )
    test_ds = SyntheticGridDataset(
        samples=test_samples,
        feature_dim=ds.feature_dim,
        fsm=ds.fsm,
    )
    return train_ds, test_ds
