"""Orbit decomposition V/H: collapses equivalent vertices into orbits.

The compression ratio |V| / |V/H| is the headline payoff for Phase 4.
Each orbit is the equivalence class of vertices reachable from one another
under the group action. When the group is trivial (only the identity), every
vertex is its own orbit and the ratio is 1.0. Non-trivial symmetry collapses
the vertex set, reducing downstream attention cost by (ratio)^2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nga.arch.group_action_on_graph import GroupAction


@dataclass
class OrbitDecomposition:
    """Orbit partition of the FSM vertex set under a group action.

    Attributes
    ----------
    action:
        The GroupAction that generated this decomposition.
    orbits:
        List of orbits. Each orbit is a sorted list of vertex indices
        belonging to the same equivalence class under H.
    orbit_of:
        Mapping from vertex_index -> orbit_index for O(1) lookup.
    """

    action: GroupAction
    orbits: list[list[int]]   # each orbit is a sorted list of vertex indices
    orbit_of: dict[int, int]  # vertex_index -> orbit_index

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_orbits(self) -> int:
        """Number of orbits |V/H|."""
        return len(self.orbits)

    @property
    def vertex_count(self) -> int:
        """Number of vertices |V| in the underlying FSM."""
        return self.action.vertex_count

    @property
    def compression_ratio(self) -> float:
        """V / (V/H).

        For the trivial group this is 1.0. For a group that partitions V
        into k equal orbits of size V/k, this equals V/k. The downstream
        attention FLOPs are reduced by compression_ratio^2.
        """
        return float(self.vertex_count) / float(self.n_orbits)

    def orbit_size(self, orbit_index: int) -> int:
        """Number of vertices in orbit orbit_index.

        Parameters
        ----------
        orbit_index:
            Index into self.orbits.
        """
        return len(self.orbits[orbit_index])


# ---------------------------------------------------------------------------
# Decomposition algorithm
# ---------------------------------------------------------------------------


def decompose_orbits(action: GroupAction) -> OrbitDecomposition:
    """Compute orbits via union-find over (vertex_index, h * vertex_index) pairs.

    Algorithm
    ---------
    For each group element h and each vertex v, we union v with h(v). After
    iterating over all (h, v) pairs the union-find roots define the orbits.
    This is correct because the orbit of v is exactly {h(v) : h in H}, and
    applying every generator (in this case every explicit permutation) and
    unioning their images produces the transitive closure.

    Complexity: O(|H| * V * alpha(V)) where alpha is the inverse Ackermann
    function from union-find path compression.

    Parameters
    ----------
    action:
        The GroupAction defining the permutation group and FSM.

    Returns
    -------
    OrbitDecomposition
        Orbit partition with sorted orbit lists and O(1) orbit_of lookup.
    """
    V = action.vertex_count
    parent = list(range(V))
    rank = [0] * V

    def find(x: int) -> int:
        """Path-compressing find."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        """Union by rank."""
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    # Union each vertex with its image under every group element.
    for h_idx in range(action.order):
        for v in range(V):
            hv = action.apply(h_idx, v)
            union(v, hv)

    # Collect orbits by root.
    root_to_orbit: dict[int, int] = {}
    orbit_of: dict[int, int] = {}
    orbits: list[list[int]] = []

    for v in range(V):
        root = find(v)
        if root not in root_to_orbit:
            root_to_orbit[root] = len(orbits)
            orbits.append([])
        o_idx = root_to_orbit[root]
        orbits[o_idx].append(v)
        orbit_of[v] = o_idx

    # Sort each orbit list for deterministic output.
    for orb in orbits:
        orb.sort()

    return OrbitDecomposition(action=action, orbits=orbits, orbit_of=orbit_of)
