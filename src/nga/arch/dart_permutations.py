"""Dart permutations (rho, tau) for monodromy-graph representation.

A dart is a directed edge: an ordered pair (vertex, edge-incidence). For a
graph with V vertices and E edges, the dart set has size 2*E (each edge has
two darts, one in each direction). The two canonical permutations on darts:

    rho: rotates darts around their base vertex (cyclic order around the vertex)
    tau: flips darts (swaps the two darts of an edge)

Together they generate the monodromy group <rho, tau> of the embedded graph.
Closed walks in the graph correspond to elements of this group (the Yuan-Wang
result: every graph is isomorphic to a monodromy graph for some triple
(rho, tau, U) with U a stabilizer subgroup).

Cycles in the action come from the orders of rho, tau, and their products:
  - rho has order = lcm of vertex-incidence cycle lengths
  - tau is an involution: order 2
  - rho*tau composed gives the "face-tracing" permutation; cycles of rho*tau
    correspond to faces of the embedded graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product as itertools_product
from math import gcd
from typing import Iterator

import numpy as np

from nga.arch.graph_fsm import GraphFSM

__all__ = ["DartSet", "build_darts_from_fsm", "build_rho", "build_tau", "compose", "permutation_order", "closed_walks_of_length"]


@dataclass
class DartSet:
    """A dart is identified by its index in [0, 2*E).

    For an edge with id e and direction d in {0, 1}, the dart index is 2*e + d.
    Direction 0 = source -> target; direction 1 = target -> source.

    `vertex_of[dart_idx]` -> vertex index where that dart sits (its base).
    `edge_of[dart_idx]` -> edge index.
    `paired[dart_idx]` -> the other dart of the same edge (the tau-image).
    """

    n_darts: int
    vertex_of: np.ndarray  # shape (n_darts,) of int
    edge_of: np.ndarray    # shape (n_darts,) of int
    paired: np.ndarray     # shape (n_darts,) of int (tau permutation, an involution)


def build_darts_from_fsm(fsm: GraphFSM) -> DartSet:
    """Build the dart set from an FSM.

    For each edge e = (u, v) in fsm.spec.edges:
      - dart 2e is at u, points u -> v.
      - dart 2e+1 is at v, points v -> u.
      paired[2e] = 2e + 1 and paired[2e+1] = 2e.

    Parameters
    ----------
    fsm:
        A loaded GraphFSM whose spec.edges provides the directed edges.

    Returns
    -------
    DartSet
        Populated dart set with vertex_of, edge_of, and paired arrays.
    """
    edges = fsm._spec.edges
    E = len(edges)
    n_darts = 2 * E

    vertex_of = np.empty(n_darts, dtype=int)
    edge_of = np.empty(n_darts, dtype=int)
    paired = np.empty(n_darts, dtype=int)

    vidx = fsm.vertex_index  # dict[str, int]

    for e, edge in enumerate(edges):
        u = vidx[edge.source]
        v = vidx[edge.target]
        d0 = 2 * e       # dart at u, pointing u->v
        d1 = 2 * e + 1   # dart at v, pointing v->u
        vertex_of[d0] = u
        vertex_of[d1] = v
        edge_of[d0] = e
        edge_of[d1] = e
        paired[d0] = d1
        paired[d1] = d0

    return DartSet(
        n_darts=n_darts,
        vertex_of=vertex_of,
        edge_of=edge_of,
        paired=paired,
    )


def build_rho(
    fsm: GraphFSM,
    darts: DartSet,
    *,
    rotation: dict[str, list[int]] | None = None,
) -> np.ndarray:
    """Build the rho permutation: rotates darts around each vertex.

    For each vertex v, the darts based at v form a cyclic order (the
    "rotation system"). rho sends each dart at v to the next one in the
    cycle.

    If `rotation` is provided, it maps vertex_id -> list of edge_ids in
    cyclic order; rho cycles through these. Otherwise, the default rotation
    is the order in which darts appear at v in `darts.vertex_of`, sorted by
    dart index.

    Parameters
    ----------
    fsm:
        The FSM providing vertex_ids and vertex_index.
    darts:
        The dart set produced by build_darts_from_fsm.
    rotation:
        Optional dict from vertex_id to list of edge indices in the desired
        cyclic order around that vertex. When None, dart indices at each
        vertex are sorted numerically to define the cycle.

    Returns
    -------
    np.ndarray
        Shape (n_darts,) of int. The rho permutation on dart indices.
    """
    n = darts.n_darts
    rho = np.arange(n, dtype=int)  # identity initially

    # Group darts by their base vertex.
    vertex_to_darts: dict[int, list[int]] = {}
    for d in range(n):
        v = int(darts.vertex_of[d])
        vertex_to_darts.setdefault(v, []).append(d)

    if rotation is not None:
        # User-supplied cyclic order by edge_id per vertex.
        for vid, edge_ids in rotation.items():
            v_idx = fsm.vertex_index[vid]
            # Build a mapping: edge_id -> list of darts at v_idx on that edge.
            edge_to_dart_at_v: dict[int, int] = {}
            for d in vertex_to_darts.get(v_idx, []):
                e = int(darts.edge_of[d])
                edge_to_dart_at_v[e] = d
            # The cyclic order of darts at this vertex follows edge_ids order.
            ordered_darts = [edge_to_dart_at_v[e] for e in edge_ids if e in edge_to_dart_at_v]
            k = len(ordered_darts)
            for i in range(k):
                rho[ordered_darts[i]] = ordered_darts[(i + 1) % k]
    else:
        # Default: sort dart indices at each vertex and cycle.
        for v_idx, dart_list in vertex_to_darts.items():
            dart_list.sort()
            k = len(dart_list)
            for i in range(k):
                rho[dart_list[i]] = dart_list[(i + 1) % k]

    return rho


def build_tau(darts: DartSet) -> np.ndarray:
    """tau permutation: just darts.paired, written as a permutation.

    tau is an involution: tau(tau(d)) = d for all d.

    Parameters
    ----------
    darts:
        The dart set produced by build_darts_from_fsm.

    Returns
    -------
    np.ndarray
        Shape (n_darts,) of int. The tau permutation (darts.paired).
    """
    return darts.paired.copy()


def compose(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compose permutations: (p o q)(x) = p(q(x)).

    Parameters
    ----------
    p:
        Left permutation, shape (n,) of int.
    q:
        Right permutation, shape (n,) of int.

    Returns
    -------
    np.ndarray
        Shape (n,) of int. The composed permutation p(q(x)).
    """
    return p[q]


def _lcm(a: int, b: int) -> int:
    """Least common multiple of two non-negative integers."""
    if a == 0 or b == 0:
        return 0
    return a * b // gcd(a, b)


def permutation_order(perm: np.ndarray) -> int:
    """Smallest k > 0 such that perm^k is the identity. Computes by iterated composition.

    The order of a permutation equals the least common multiple of its cycle
    lengths, so this is computed by finding cycles rather than blind iteration.

    Parameters
    ----------
    perm:
        Shape (n,) of int. A valid permutation on 0..n-1.

    Returns
    -------
    int
        The order of the permutation (smallest positive integer k with perm^k = id).
    """
    n = len(perm)
    visited = np.zeros(n, dtype=bool)
    order = 1
    for start in range(n):
        if visited[start]:
            continue
        # Trace the cycle from start.
        cycle_len = 0
        cur = start
        while not visited[cur]:
            visited[cur] = True
            cur = int(perm[cur])
            cycle_len += 1
        order = _lcm(order, cycle_len)
    return order


def _iter_binary_sequences(length: int) -> Iterator[tuple[int, ...]]:
    """Yield all binary sequences of a given length."""
    for seq in itertools_product((0, 1), repeat=length):
        yield seq


def closed_walks_of_length(rho: np.ndarray, tau: np.ndarray, length: int) -> list[list[int]]:
    """Enumerate distinct closed walks of length `length` starting from each dart.

    Each walk is a sequence of permutation generators (rho or tau) that
    returns to the starting dart. For length k, there are 2^k candidate
    walks; this function generates them lazily and yields only those that
    are closed.

    Returns a list of [generator_indices...] where 0=rho, 1=tau. The list is
    deduplicated by canonical rotation (a walk and its cyclic shift are
    considered equal).

    Parameters
    ----------
    rho:
        Shape (n_darts,) of int. The rho permutation.
    tau:
        Shape (n_darts,) of int. The tau permutation.
    length:
        The number of generator applications per walk.

    Returns
    -------
    list[list[int]]
        Deduplicated list of closed walk generator sequences. Each element is
        a list of length `length` drawn from {0, 1} where 0=rho and 1=tau.
    """
    generators = [rho, tau]
    n = len(rho)
    identity = np.arange(n, dtype=int)

    # Collect closed-walk sequences and deduplicate by canonical rotation.
    seen: set[tuple[int, ...]] = set()
    result: list[list[int]] = []

    for seq in _iter_binary_sequences(length):
        # Compose the sequence: apply seq[0] first, then seq[1], etc.
        # walk(d) = g_{length-1}(... g_1(g_0(d)) ...)
        combined = identity.copy()
        for gen_idx in seq:
            combined = compose(generators[gen_idx], combined)

        # A walk is closed if combined is the identity.
        if np.array_equal(combined, identity):
            # Canonical rotation: pick the lexicographically smallest rotation.
            rotations = [seq[i:] + seq[:i] for i in range(length)]
            canonical = min(rotations)
            if canonical not in seen:
                seen.add(canonical)
                result.append(list(seq))

    return result
