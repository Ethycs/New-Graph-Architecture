"""Permutation group H acting on the FSM vertex set.

A group action sigma: H x V -> V is supplied as an explicit list of
permutations on the vertex_ids order. The identity is required as the
first element. Closure under composition and existence of inverses are
verified at construction time; if the verification fails the constructor
raises ValueError naming the offending pair.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nga.arch.graph_fsm import GraphFSM


@dataclass
class GroupAction:
    """Explicit permutation-group action on the FSM vertex set.

    Parameters
    ----------
    fsm:
        The FSM whose vertex set is the domain of the action.
    permutations:
        List of permutation arrays, each of shape (V,) with dtype int.
        Entry permutations[h][v] is the image of vertex v under group
        element h. The identity permutation must appear first (index 0).
    name:
        Human-readable label for the group, e.g. "Z/3" or "S2".
    """

    fsm: GraphFSM
    permutations: list[np.ndarray]  # each shape (V,) of int
    name: str = "H"

    def __post_init__(self) -> None:
        """Validate the permutation list at construction time.

        Checks performed:
        - At least one permutation (the identity) is present.
        - The first permutation is the identity on 0..V-1.
        - Every permutation has shape (V,) and is a valid permutation.
        - The set is closed under composition.
        - Every element has its inverse in the set.

        Raises
        ------
        ValueError
            If any check fails, with a message identifying the offending
            element or pair.
        """
        V = self.fsm.vertex_count
        if len(self.permutations) == 0:
            raise ValueError("GroupAction requires at least the identity permutation.")

        identity = np.arange(V, dtype=int)

        # Validate each permutation individually.
        for h_idx, perm in enumerate(self.permutations):
            arr = np.asarray(perm, dtype=int)
            if arr.shape != (V,):
                raise ValueError(
                    f"Permutation {h_idx} has shape {arr.shape}; expected ({V},)."
                )
            if set(arr.tolist()) != set(range(V)):
                raise ValueError(
                    f"Permutation {h_idx} is not a valid permutation of 0..{V - 1}: "
                    f"{arr.tolist()}"
                )
            self.permutations[h_idx] = arr  # normalise to np.ndarray in place

        # First element must be the identity.
        if not np.array_equal(self.permutations[0], identity):
            raise ValueError(
                f"First permutation must be the identity {identity.tolist()}; "
                f"got {self.permutations[0].tolist()}."
            )

        # Verify closure and inverses.
        verify_closure(self, _already_normalised=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def order(self) -> int:
        """Cardinality |H| of the group."""
        return len(self.permutations)

    @property
    def vertex_count(self) -> int:
        """Number of vertices V in the underlying FSM."""
        return self.fsm.vertex_count

    # ------------------------------------------------------------------
    # Action methods
    # ------------------------------------------------------------------

    def apply(self, h_index: int, vertex_index: int) -> int:
        """sigma(h, v) for h = self.permutations[h_index].

        Parameters
        ----------
        h_index:
            Index into self.permutations selecting the group element h.
        vertex_index:
            The vertex v in 0..V-1 to act on.

        Returns
        -------
        int
            The image vertex index h(v).
        """
        return int(self.permutations[h_index][vertex_index])

    def apply_to_set(self, h_index: int, vertex_indices: list[int]) -> list[int]:
        """Vectorised apply over a list of vertex indices.

        Parameters
        ----------
        h_index:
            Index selecting the group element.
        vertex_indices:
            List of vertex indices to permute.

        Returns
        -------
        list[int]
            Images h(v) for each v in vertex_indices, in the same order.
        """
        perm = self.permutations[h_index]
        return [int(perm[v]) for v in vertex_indices]

    def fixes(self, h_index: int, vertex_index: int) -> bool:
        """True iff sigma(h, v) == v (h fixes vertex v).

        Parameters
        ----------
        h_index:
            Index selecting the group element h.
        vertex_index:
            The vertex v to check.
        """
        return int(self.permutations[h_index][vertex_index]) == vertex_index


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def trivial_group_action(fsm: GraphFSM, *, name: str = "1") -> GroupAction:
    """The trivial group {e}; useful as a baseline.

    Constructs the order-1 group whose only element is the identity
    permutation. Under the trivial group every vertex is its own orbit,
    so compression_ratio == 1.0 and all stabilizers are trivial. This
    baseline lets callers compare against non-trivial symmetry groups
    without special-casing the code.

    Parameters
    ----------
    fsm:
        The FSM whose vertex set is the domain.
    name:
        Label for the group (default "1" for the trivial group).
    """
    V = fsm.vertex_count
    identity = np.arange(V, dtype=int)
    return GroupAction(fsm=fsm, permutations=[identity], name=name)


def cyclic_group_action(
    fsm: GraphFSM, *, k: int, name: str | None = None
) -> GroupAction:
    """Z/k acting by cyclic shift on vertex_ids[0..k-1]; remaining vertices fixed.

    The group Z/k = {e, r, r^2, ..., r^(k-1)} acts on the first k vertices
    by the cyclic shift r: i -> (i + 1) mod k (with vertices beyond index
    k - 1 held fixed). This models symmetric tools or subtasks that appear
    k times in a rotationally equivalent arrangement.

    Design choice: we restrict the action to the first k vertices (indices
    0..k-1) and fix vertices k..V-1. This keeps the generator transparent
    and is sufficient for the Phase 4 acceptance test, which uses V=7 and
    k=3. For a full-orbit action wrapping all V vertices, pass k=V.

    Parameters
    ----------
    fsm:
        The FSM whose vertex set is the domain.
    k:
        Size of the cyclic group. Must satisfy 1 <= k <= V. If k == 1,
        the returned group is the trivial group.
    name:
        Label for the group. Defaults to "Z/{k}".

    Raises
    ------
    ValueError
        If k < 1 or k > V.
    """
    V = fsm.vertex_count
    if not (1 <= k <= V):
        raise ValueError(
            f"k={k} must satisfy 1 <= k <= V={V}."
        )
    if name is None:
        name = f"Z/{k}"

    identity = np.arange(V, dtype=int)
    perms: list[np.ndarray] = []
    for power in range(k):
        perm = identity.copy()
        # Apply power-fold cyclic shift on 0..k-1.
        for i in range(k):
            perm[i] = (i + power) % k
        perms.append(perm)

    return GroupAction(fsm=fsm, permutations=perms, name=name)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_closure(action: GroupAction, *, _already_normalised: bool = False) -> None:
    """Sanity check: every product h1 * h2 yields a permutation already in the set,
    and every permutation has its inverse present. Raise ValueError on failure.

    Composition is defined as (h1 * h2)(v) = h1(h2(v)), i.e. right-to-left
    application consistent with the group-action axiom sigma(h1 h2, v) =
    sigma(h1, sigma(h2, v)).

    Parameters
    ----------
    action:
        The GroupAction to validate.
    _already_normalised:
        Internal flag set by GroupAction.__post_init__ to skip the
        duplicate normalisation pass; callers should leave this False.

    Raises
    ------
    ValueError
        Named after the offending pair (h1_idx, h2_idx) for closure, or
        the missing inverse for the inverse check.
    """
    V = action.vertex_count
    perms = action.permutations

    # Build a lookup: tuple(perm) -> index for O(1) membership test.
    perm_to_idx: dict[tuple[int, ...], int] = {
        tuple(p.tolist()): idx for idx, p in enumerate(perms)
    }

    identity = np.arange(V, dtype=int)

    for i, h1 in enumerate(perms):
        # Check inverse: h1^{-1}(j) = the v such that h1(v) = j.
        inv = np.empty(V, dtype=int)
        inv[h1] = np.arange(V, dtype=int)
        inv_key = tuple(inv.tolist())
        if inv_key not in perm_to_idx:
            raise ValueError(
                f"GroupAction: permutation at index {i} ({h1.tolist()}) "
                f"has no inverse in the set. Inverse would be {inv.tolist()}."
            )

        for j, h2 in enumerate(perms):
            # Composition: (h1 * h2)(v) = h1[h2[v]]
            composed = h1[h2]  # numpy advanced indexing
            key = tuple(composed.tolist())
            if key not in perm_to_idx:
                raise ValueError(
                    f"GroupAction: composition of permutation {i} ({h1.tolist()}) "
                    f"and permutation {j} ({h2.tolist()}) yields {composed.tolist()}, "
                    f"which is not in the permutation set. The set is not closed."
                )
