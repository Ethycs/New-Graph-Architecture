"""Bisimulation-style quotient of an over-clustered transition graph.

Phase 21 atom. Wave C established that a trained encoder produces
**sub-clusters within FSM states**: K-selection picks K > V because the
substrate's representation is finer-grained than the gold FSM. Forcing
K = V cuts purity (some sub-clusters survive, others get force-merged
arbitrarily) and Hamming stays well above the strict bar.

The Myhill-Nerode reframing
===========================

Two latent clusters should be **the same state** iff they are
*observationally equivalent* — their outgoing transition distributions,
incoming transition distributions, emission distributions, and
downstream predictive consequences are all indistinguishable. This is
the classical bisimulation / Myhill-Nerode equivalence on a labelled
transition system.

The extraction pipeline therefore becomes:

    overcluster -> estimate transition + emission laws -> quotient
    -> minimal graph + posterior edge confidence

This atom implements the quotient step: given an over-clustered
labelling and its observed transition / emission counts, repeatedly
merge the two clusters whose merge criterion is most similar, until
``target_K`` equivalence classes remain.

Merge criteria
==============

* ``"transition"`` -- L2 distance between row-normalised outgoing
  transition distributions. The cluster's outgoing distribution is
  ``transition_counts[i] / sum(transition_counts[i])`` with Laplace
  smoothing for empty rows.
* ``"emission"`` -- L2 distance between row-normalised token-emission
  distributions per cluster. Requires ``emission_counts``.
* ``"transition_plus_incoming"`` -- adds L2 distance on the incoming
  transition distributions (the cluster's column). This is closer to
  the formal bisimulation condition because behavioural equivalence is
  symmetric under graph reversal.
* ``"full"`` -- weighted average of transition + incoming + emission
  distances (weights default to 1/3 each).
* ``"random"`` -- uniformly random pair selection. Baseline to verify
  the criterion is doing work.

After merging, the atom returns the new labelling, new transition
counts, and (optionally) new emission counts -- all renumbered into
``[0, target_K)``. Downstream code rebuilds the Beta posterior and the
legality matrix from these.

Determinism: under fixed ``seed`` (which only affects ``"random"``
ties) the quotient is deterministic.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

__all__ = [
    "QuotientResult",
    "quotient_by_bisimulation",
    "MERGE_CRITERIA",
]


MERGE_CRITERIA = (
    "transition",
    "emission",
    "transition_plus_incoming",
    "full",
    "random",
)


@dataclass(frozen=True)
class QuotientResult:
    """The output of :func:`quotient_by_bisimulation`.

    Attributes
    ----------
    labels:
        New cluster assignment per step, shape ``(N,)``, integers in
        ``[0, target_K)``.
    transition_counts:
        New transition count matrix, shape ``(target_K, target_K)``.
    emission_counts:
        New emission count matrix, shape ``(target_K, n_tokens)`` if
        emissions were supplied; otherwise ``None``.
    cluster_map:
        Mapping from original cluster id to merged-class id, shape
        ``(K,)``, integers in ``[0, target_K)``.
    merge_history:
        List of ``(i, j)`` pairs in the order they were merged
        (working in the running-merged label space).
    criterion:
        The merge criterion used.
    """

    labels: np.ndarray
    transition_counts: np.ndarray
    emission_counts: np.ndarray | None
    cluster_map: np.ndarray
    merge_history: list[tuple[int, int]]
    criterion: str


def _row_normalize(counts: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
    """Row-stochastic normalisation with Laplace smoothing."""
    sm = counts.astype(np.float64) + smoothing
    s = sm.sum(axis=1, keepdims=True)
    s = np.where(s > 0.0, s, 1.0)
    return sm / s


def _pair_distance(
    i: int,
    j: int,
    *,
    out_probs: np.ndarray,
    in_probs: np.ndarray | None,
    em_probs: np.ndarray | None,
    criterion: str,
    rng: np.random.Generator,
) -> float:
    if criterion == "transition":
        return float(np.linalg.norm(out_probs[i] - out_probs[j]))
    if criterion == "emission":
        if em_probs is None:
            raise ValueError("emission criterion requires emission_counts")
        return float(np.linalg.norm(em_probs[i] - em_probs[j]))
    if criterion == "transition_plus_incoming":
        d_out = float(np.linalg.norm(out_probs[i] - out_probs[j]))
        if in_probs is None:
            return d_out
        d_in = float(np.linalg.norm(in_probs[i] - in_probs[j]))
        return 0.5 * (d_out + d_in)
    if criterion == "full":
        d_out = float(np.linalg.norm(out_probs[i] - out_probs[j]))
        d_in = (
            float(np.linalg.norm(in_probs[i] - in_probs[j]))
            if in_probs is not None
            else d_out
        )
        d_em = (
            float(np.linalg.norm(em_probs[i] - em_probs[j]))
            if em_probs is not None
            else 0.0
        )
        return (d_out + d_in + d_em) / 3.0
    if criterion == "random":
        return float(rng.random())
    raise ValueError(
        f"unknown criterion {criterion!r}; expected one of {MERGE_CRITERIA}"
    )


def quotient_by_bisimulation(
    labels: np.ndarray,
    transition_counts: np.ndarray,
    *,
    target_K: int,
    criterion: str = "full",
    emission_counts: np.ndarray | None = None,
    use_incoming: bool = True,
    seed: int = 0,
) -> QuotientResult:
    """Greedy agglomerative quotient by behavioural equivalence.

    Parameters
    ----------
    labels:
        Per-step cluster assignment, shape ``(N,)``, integers in
        ``[0, K)`` where ``K`` is the over-clustered count.
    transition_counts:
        ``(K, K)`` observed transition counts.
    target_K:
        Number of equivalence classes to end with. Must be in
        ``[1, K]``.
    criterion:
        One of ``MERGE_CRITERIA``.
    emission_counts:
        Optional ``(K, n_tokens)`` per-cluster emission counts; required
        for ``"emission"`` and used by ``"full"`` if present.
    use_incoming:
        If True, includes the cluster's column (incoming transition
        distribution) in ``"full"`` and ``"transition_plus_incoming"``.
    seed:
        Drives tie-breaking for ``"random"`` criterion.

    Returns
    -------
    QuotientResult with the new labelling and merged count matrices.
    """
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.asarray(transition_counts, dtype=np.int64).copy()
    K = counts.shape[0]
    if counts.shape != (K, K):
        raise ValueError(
            f"transition_counts must be square, got shape {counts.shape}"
        )
    if not 1 <= target_K <= K:
        raise ValueError(f"target_K ({target_K}) must be in [1, {K}]")
    if criterion not in MERGE_CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}")
    em_counts = (
        np.asarray(emission_counts, dtype=np.int64).copy()
        if emission_counts is not None
        else None
    )
    if em_counts is not None and em_counts.shape[0] != K:
        raise ValueError(
            f"emission_counts shape[0]={em_counts.shape[0]} != K={K}"
        )

    rng = np.random.default_rng(int(seed))
    # cluster_map[i] = current equivalence-class id of original cluster i.
    cluster_map = np.arange(K, dtype=np.int64)
    # alive[i] is True for representative class ids (originally each cluster
    # is its own class; merging removes the dissolved one).
    alive = np.ones(K, dtype=bool)
    merge_history: list[tuple[int, int]] = []

    while int(alive.sum()) > target_K:
        # Re-normalise current alive rows / columns each iteration.
        out_probs = _row_normalize(counts)
        in_probs = (
            _row_normalize(counts.T) if use_incoming else None
        )
        em_probs = _row_normalize(em_counts) if em_counts is not None else None

        # Find best pair (i, j) with i < j, both alive.
        alive_ids = np.where(alive)[0]
        best_d = np.inf
        best_pair: tuple[int, int] | None = None
        for a_idx in range(len(alive_ids)):
            for b_idx in range(a_idx + 1, len(alive_ids)):
                i = int(alive_ids[a_idx])
                j = int(alive_ids[b_idx])
                d = _pair_distance(
                    i,
                    j,
                    out_probs=out_probs,
                    in_probs=in_probs,
                    em_probs=em_probs,
                    criterion=criterion,
                    rng=rng,
                )
                if d < best_d:
                    best_d = d
                    best_pair = (i, j)
        if best_pair is None:
            break  # pragma: no cover - shouldn't happen given alive.sum > target_K
        i, j = best_pair
        merge_history.append((i, j))

        # Merge cluster j into cluster i (i.e., i absorbs j).
        counts[i, :] += counts[j, :]
        counts[:, i] += counts[:, j]
        counts[j, :] = 0
        counts[:, j] = 0
        if em_counts is not None:
            em_counts[i, :] += em_counts[j, :]
            em_counts[j, :] = 0
        alive[j] = False
        # Remap j -> i in cluster_map; this preserves the chain of merges
        # through cluster_map.
        cluster_map = np.where(cluster_map == j, i, cluster_map)

    # Renumber alive ids into [0, target_K).
    alive_ids = np.where(alive)[0]
    new_id_of_old = {int(old): new for new, old in enumerate(alive_ids)}
    final_cluster_map = np.asarray(
        [new_id_of_old[int(c)] for c in cluster_map], dtype=np.int64
    )
    # Reduced count matrices.
    n_final = len(alive_ids)
    reduced_counts = np.zeros((n_final, n_final), dtype=np.int64)
    for r_new, r_old in enumerate(alive_ids):
        for c_new, c_old in enumerate(alive_ids):
            reduced_counts[r_new, c_new] = counts[r_old, c_old]
    reduced_em = None
    if em_counts is not None:
        reduced_em = np.zeros(
            (n_final, em_counts.shape[1]), dtype=np.int64
        )
        for r_new, r_old in enumerate(alive_ids):
            reduced_em[r_new] = em_counts[r_old]

    new_labels = final_cluster_map[labels]
    return QuotientResult(
        labels=new_labels,
        transition_counts=reduced_counts,
        emission_counts=reduced_em,
        cluster_map=final_cluster_map,
        merge_history=merge_history,
        criterion=criterion,
    )
