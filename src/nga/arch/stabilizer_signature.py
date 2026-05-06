"""Per-vertex stabilizer subgroup record.

For each vertex v, stabilizer(v) = {h in H : sigma(h, v) = v}. Vertices with
non-trivial stabilizer (more than just the identity) are flagged as
candidate singularity sites - this is the Phase 4 signal that sigma(x) needs
to genuinely beat margin (q10). A vertex with stabilizer order > 1 receives
a non-zero contribution to sigma's stabilizer_risk.

In the group-action literature the stabilizer (or isotropy group) Stab_H(v)
is the subgroup of H that leaves v fixed. For a free action every stabilizer
is trivial; non-trivial stabilizers mark "cone points" in the quotient space
that cannot be faithfully embedded without special handling. Phase 4 uses this
signal to flag vertices that require extra care in the singularity detector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nga.arch.group_action_on_graph import GroupAction


@dataclass
class StabilizerSignature:
    """Per-vertex stabilizer record for a group action.

    Attributes
    ----------
    action:
        The GroupAction that was analysed.
    stabilizer_order:
        Shape (V,), int. The number of group elements that fix each
        vertex. Equals 1 for vertices with a trivial stabilizer ({e}).
    stabilizer_mask:
        Shape (V, |H|), bool. Entry (v, h_idx) is True iff
        permutations[h_idx] fixes vertex v, i.e. sigma(h, v) == v.
    """

    action: GroupAction
    stabilizer_order: np.ndarray  # shape (V,), int
    stabilizer_mask: np.ndarray   # shape (V, |H|), bool

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_nontrivial(self) -> np.ndarray:
        """Shape (V,) bool: True where stabilizer_order[v] > 1.

        A True entry marks a singular vertex - one that is fixed by at
        least one non-identity group element and therefore lies at a
        "cone point" of the quotient space.
        """
        return self.stabilizer_order > 1

    # ------------------------------------------------------------------
    # Per-vertex methods
    # ------------------------------------------------------------------

    def stabilizer_risk(self, vertex_index: int) -> float:
        """Phase 4 singularity-detector input for a single vertex.

        Returns (stabilizer_order[v] - 1) / max(|H| - 1, 1), so:
        - A vertex with trivial stabilizer scores 0.0.
        - A vertex fixed by every group element scores 1.0.
        - Intermediate values scale linearly with excess stabilizer size.

        The result is bounded in [0, 1] for the SingularityDetector to
        consume as one component of sigma(x).

        Parameters
        ----------
        vertex_index:
            Index of the vertex in 0..V-1.

        Returns
        -------
        float
            Risk score in [0.0, 1.0].
        """
        order_v = int(self.stabilizer_order[vertex_index])
        denom = max(self.action.order - 1, 1)
        return float(order_v - 1) / float(denom)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def compute_stabilizer_signature(action: GroupAction) -> StabilizerSignature:
    """Compute stabilizer for every vertex by checking which permutations fix it.

    For each vertex v and each group element h (given as permutation index),
    we test whether permutations[h][v] == v. The stabilizer_mask records the
    result for every (v, h) pair; stabilizer_order sums across h for each v.

    This is an O(V * |H|) computation with a vectorised numpy inner loop.

    Parameters
    ----------
    action:
        The GroupAction to analyse.

    Returns
    -------
    StabilizerSignature
        Populated stabilizer record for all vertices.
    """
    V = action.vertex_count
    H = action.order

    # Stack all permutations into a (H, V) matrix for vectorised comparison.
    perm_matrix = np.stack(action.permutations, axis=0)  # shape (H, V)

    # Vertex indices as a row vector.
    vertex_range = np.arange(V, dtype=int)  # shape (V,)

    # perm_matrix[h, v] == vertex_range[v] iff permutation h fixes vertex v.
    # Broadcast: compare (H, V) with (V,) -> bool (H, V).
    fixes_hv = perm_matrix == vertex_range[np.newaxis, :]  # shape (H, V)

    # Transpose to (V, H) for the dataclass convention.
    stabilizer_mask = fixes_hv.T  # shape (V, H), bool

    # stabilizer_order[v] = number of h that fix v.
    stabilizer_order = stabilizer_mask.sum(axis=1).astype(int)  # shape (V,)

    return StabilizerSignature(
        action=action,
        stabilizer_order=stabilizer_order,
        stabilizer_mask=stabilizer_mask,
    )
