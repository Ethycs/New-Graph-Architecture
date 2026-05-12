"""monodromy consistency: a closed walk in the FSM should leave the system invariant. Energy drift along a closed walk is the architecture's structural test of whether its world model respects its own group structure.

A closed walk v_0 -> v_1 -> ... -> v_n -> v_0 corresponds (under the dart
permutation machinery) to an element of the monodromy group <rho, tau> that
fixes the starting dart's vertex. If the world model's energy field respects
the same group structure, then traversing such a walk should return the
system to its initial energy.

When the observed energy *does not* return to baseline, the walk is a
witness for an internal contradiction: the world model has assigned
incompatible energies to states that the group structure says are linked by
an invariant. The squared drift `(E[start] - E[end])^2` is a free
regulariser - it requires no labels, only the FSM's connectivity and the
energy field on visited states.

Two flavours are exposed:

  - `monodromy_consistency_loss`: penalises only endpoint drift. This is
    the minimal "did we end where we started, energetically?" metric.
  - `consistency_loss_with_intermediate_drift`: stricter. Penalises all
    cyclic-pair drifts (sum over i of (E[i] - E[i mod n])^2 across an
    appropriate cyclic shift). Forbids the walk from wandering even if
    endpoints match.
"""
from __future__ import annotations

import numpy as np

from nga.arch.graph_fsm import GraphFSM

__all__ = [
    "closed_walks_in_fsm",
    "walk_energy_drift",
    "monodromy_consistency_loss",
    "consistency_loss_with_intermediate_drift",
]


# ---------------------------------------------------------------------------
# Walk enumeration
# ---------------------------------------------------------------------------


def _canonical_cycle_key(vertices: list[int]) -> tuple[int, ...]:
    """Canonicalise a cycle (v_0, v_1, ..., v_{n-1}) so cyclic rotations map to the same key.

    The walk format used by callers is `[v_0, v_1, ..., v_{n-1}, v_0]` where
    the last element repeats the first to close the cycle. For canonicalisation
    we strip the trailing repeat and pick the lexicographically smallest
    rotation. Direction is *not* normalised - we treat 0->1->2->0 and
    0->2->1->0 as distinct walks (they correspond to different group elements
    in general).

    Parameters
    ----------
    vertices:
        A walk in the format `[v_0, ..., v_{n-1}, v_0]`. The trailing repeat
        is stripped before computing rotations.

    Returns
    -------
    tuple[int, ...]
        The lexicographically smallest cyclic rotation of the n-vertex prefix.
    """
    if len(vertices) < 2:
        return tuple(vertices)
    # Strip the closing repeat.
    body = vertices[:-1]
    n = len(body)
    rotations = [tuple(body[i:] + body[:i]) for i in range(n)]
    return min(rotations)


def closed_walks_in_fsm(
    fsm: GraphFSM,
    *,
    max_length: int = 6,
) -> list[list[int]]:
    """Enumerate closed walks in the FSM up to ``max_length`` edges.

    A closed walk is a sequence of vertices `v_0 -> v_1 -> ... -> v_n` with
    `v_n == v_0` and every consecutive pair an edge of the FSM. Each returned
    walk is the explicit vertex sequence including the repeated start vertex
    at the end: `[v_0, v_1, ..., v_{n-1}, v_0]`.

    Cyclic rotations of the same walk are deduplicated: the walks 0->1->2->0
    and 1->2->0->1 count as one. Reversed walks are kept distinct because
    they correspond to different elements of the monodromy group in general.

    Implementation: a direct DFS over `fsm.legality_matrix`. For each starting
    vertex `s` we expand depth-first and emit a walk whenever we step back to
    `s` after at least one edge. We deduplicate via a canonical-rotation key.

    Parameters
    ----------
    fsm:
        Runtime FSM whose legality matrix defines directed adjacency.
    max_length:
        Maximum walk length, measured in edges. ``max_length >= 1``. The
        smallest non-trivial closed walks are self-loops (length 1) and
        2-cycles (length 2).

    Returns
    -------
    list[list[int]]
        Each element is `[v_0, v_1, ..., v_{n-1}, v_0]` for some 1 <= n <= max_length.
        Deduplicated by cyclic rotation.
    """
    if max_length < 1:
        return []

    adj: np.ndarray = fsm.legality_matrix  # shape (V, V), bool
    V = adj.shape[0]

    seen_keys: set[tuple[int, ...]] = set()
    walks: list[list[int]] = []

    # Pre-compute neighbour lists to avoid repeated boolean-mask scans inside the DFS.
    neighbours: list[list[int]] = [
        [int(j) for j in np.flatnonzero(adj[i])] for i in range(V)
    ]

    def dfs(start: int, current: int, path: list[int], depth: int) -> None:
        # When we arrive back at start with at least one edge consumed, emit
        # a walk and stop extending: continuing would re-visit the start
        # vertex's outgoing edges and tack on additional segments, producing
        # walks of the form [..., 0, 0, ...] that are not what callers want.
        if depth >= 1 and current == start:
            walk = list(path)  # path already terminates at `start`
            key = _canonical_cycle_key(walk)
            if key not in seen_keys:
                seen_keys.add(key)
                walks.append(walk)
            return
        if depth >= max_length:
            return
        for nxt in neighbours[current]:
            path.append(nxt)
            dfs(start, nxt, path, depth + 1)
            path.pop()

    for s in range(V):
        dfs(s, s, [s], 0)

    return walks


# ---------------------------------------------------------------------------
# Drift metrics
# ---------------------------------------------------------------------------


def walk_energy_drift(
    walk: list[int],
    energies_by_state: dict[int, float],
) -> float:
    """Energy drift of a closed walk under a static energy field.

    Returns ``E(start) - E(end)`` along the walk. Because the walk is closed
    (start vertex == end vertex), this is identically zero whenever the
    energy field is a function of vertex alone. The metric becomes meaningful
    when energies are *dynamic* - context-dependent, time-stamped, or sampled
    independently at each step.

    Parameters
    ----------
    walk:
        Vertex sequence ``[v_0, v_1, ..., v_{n-1}, v_0]``.
    energies_by_state:
        Mapping from vertex index to its (static) energy.

    Returns
    -------
    float
        ``energies_by_state[walk[0]] - energies_by_state[walk[-1]]``.
    """
    if len(walk) < 2:
        return 0.0
    e_start = float(energies_by_state[walk[0]])
    e_end = float(energies_by_state[walk[-1]])
    return e_start - e_end


def monodromy_consistency_loss(
    walks: list[list[int]],
    energies_by_walk: list[np.ndarray],
) -> float:
    """Mean squared endpoint drift across a list of closed walks.

    For each walk ``w`` with observed step-energies ``E = energies_by_walk[i]``
    of the same length as ``w``, the contribution is ``(E[0] - E[-1])^2``.
    The returned scalar is the arithmetic mean across walks.

    Trivial walks of length 1 (a single vertex, no edges) have a single energy
    sample and contribute exactly zero. Walks where energy returns exactly
    contribute zero by construction.

    Parameters
    ----------
    walks:
        List of vertex sequences. Each is ``[v_0, ..., v_{n-1}, v_0]`` (or
        a length-1 trivial walk).
    energies_by_walk:
        Parallel list; ``energies_by_walk[i]`` has the same length as
        ``walks[i]`` and gives observed energies at each step of walk i.

    Returns
    -------
    float
        Mean of ``(E[0] - E[-1])^2`` across walks. Zero when ``walks`` is empty.
    """
    if not walks:
        return 0.0
    if len(walks) != len(energies_by_walk):
        raise ValueError(
            f"walks and energies_by_walk must have same length; "
            f"got {len(walks)} vs {len(energies_by_walk)}"
        )

    drifts_sq: list[float] = []
    for walk, energies in zip(walks, energies_by_walk):
        e = np.asarray(energies, dtype=float)
        if e.size < 2:
            drifts_sq.append(0.0)
            continue
        if e.size != len(walk):
            raise ValueError(
                f"energies length {e.size} does not match walk length {len(walk)}"
            )
        drift = float(e[0] - e[-1])
        drifts_sq.append(drift * drift)

    return float(np.mean(drifts_sq))


def consistency_loss_with_intermediate_drift(
    walks: list[list[int]],
    energies_by_walk: list[np.ndarray],
) -> float:
    """Stricter variant: penalises every pairwise drift around the cycle.

    A closed walk respecting the group structure should have *every* energy
    sample equal to its cyclic counterpart, not only the endpoints. Concretely,
    for the n distinct vertex-positions of a closed walk (the body, excluding
    the repeated final vertex), this loss accumulates

        sum_i (E[i] - E[(i + 1) mod n])^2  +  (E[0] - E[-1])^2

    The first sum penalises step-wise drift (energies wandering around the
    loop), and the additive endpoint term ensures this loss dominates
    ``monodromy_consistency_loss`` on the same input. The mean is then
    taken across walks.

    For trivial walks (length-1) the contribution is zero.

    Parameters
    ----------
    walks:
        List of vertex sequences (same convention as above).
    energies_by_walk:
        Parallel list; energies aligned to each step.

    Returns
    -------
    float
        Mean accumulated squared drift across walks. Zero when ``walks`` is empty.
    """
    if not walks:
        return 0.0
    if len(walks) != len(energies_by_walk):
        raise ValueError(
            f"walks and energies_by_walk must have same length; "
            f"got {len(walks)} vs {len(energies_by_walk)}"
        )

    per_walk: list[float] = []
    for walk, energies in zip(walks, energies_by_walk):
        e = np.asarray(energies, dtype=float)
        if e.size < 2:
            per_walk.append(0.0)
            continue
        if e.size != len(walk):
            raise ValueError(
                f"energies length {e.size} does not match walk length {len(walk)}"
            )
        # Endpoint drift (matches monodromy_consistency_loss contribution).
        endpoint_drift_sq = float((e[0] - e[-1]) ** 2)
        # Intermediate drift across the body of the walk (cyclic pairs).
        # The walk format is [v_0, ..., v_{n-1}, v_0], so e has n+1 samples.
        # Pair (e[i], e[(i+1) mod n]) for i in 0..n-1 captures every step.
        body = e[:-1]
        n = body.size
        cyclic_next = np.concatenate([body[1:], body[:1]])
        intermediate_drift_sq = float(np.sum((body - cyclic_next) ** 2))
        per_walk.append(endpoint_drift_sq + intermediate_drift_sq)

    return float(np.mean(per_walk))
