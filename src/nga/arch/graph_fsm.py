"""Graph FSM arch atom - runtime wrapper around GraphFSMSpec.

Loads the persisted GraphFSMSpec (the file-format pydantic model from
drivers/graph_fsm_spec.py) and exposes the in-memory interface the rest of the
system uses during inference: stable vertex ordering, O(1) index lookup, a
numpy legality matrix, and a vectorised apply_mask operation.

The legality matrix is built once at construction time from
GraphFSMSpec.build_legality_matrix() and stored as a numpy bool array. All
subsequent mask operations are pure numpy slices with no Python loops.

See docs/arch/graph-fsm.md for the full spec.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nga.drivers.graph_fsm_spec import GraphFSMSpec
from nga.drivers.graph_fsm_spec import load as _load_spec

__all__ = ["GraphFSM"]

_MASK_FILL = -1e9


class GraphFSM:
    """Runtime wrapper around the persisted GraphFSMSpec.

    Provides:
      - vertex_ids: list[str] in stable column order (matches dist[] indexing).
      - vertex_index: dict[str, int] - O(1) id-to-column lookup.
      - legality_matrix: np.ndarray of bool, shape (V, V); True at [i, j] iff
        edge i -> j exists in the spec.
      - apply_mask(logits, current_state) -> masked logits with illegal
        next-states set to a large negative number.

    Parameters
    ----------
    spec:
        A fully validated GraphFSMSpec instance. The spec's vertex ordering
        defines the canonical column order for all distribution vectors.
    """

    def __init__(self, spec: GraphFSMSpec) -> None:
        self._spec = spec
        self._vertex_ids: list[str] = [v.id for v in spec.vertices]
        self._vertex_index: dict[str, int] = {
            vid: i for i, vid in enumerate(self._vertex_ids)
        }
        self._legality_matrix: np.ndarray = np.asarray(
            spec.build_legality_matrix(), dtype=bool
        )  # shape (V, V)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vertex_count(self) -> int:
        """Number of vertices V in this graph."""
        return len(self._vertex_ids)

    @property
    def vertex_ids(self) -> list[str]:
        """Stable list of vertex ids in spec ordering (column 0, 1, ..., V-1)."""
        return list(self._vertex_ids)

    @property
    def vertex_index(self) -> dict[str, int]:
        """Mapping from vertex id to its column index."""
        return dict(self._vertex_index)

    @property
    def legality_matrix(self) -> np.ndarray:
        """Boolean adjacency matrix of shape (V, V).

        Entry [i, j] is True iff there is a directed edge from
        vertex_ids[i] to vertex_ids[j].
        """
        return self._legality_matrix

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def index_of(self, vertex_id: str) -> int:
        """Return the column index of vertex_id, raising KeyError if unknown.

        Parameters
        ----------
        vertex_id:
            A vertex id string from the spec.
        """
        try:
            return self._vertex_index[vertex_id]
        except KeyError:
            raise KeyError(
                f"Vertex '{vertex_id}' not found. "
                f"Known vertices: {self._vertex_ids}"
            ) from None

    def is_legal_transition(self, src: str, dst: str) -> bool:
        """Return True iff there is a directed edge from src to dst.

        Parameters
        ----------
        src:
            Source vertex id.
        dst:
            Destination vertex id.
        """
        i = self.index_of(src)
        j = self.index_of(dst)
        return bool(self._legality_matrix[i, j])

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def apply_mask(
        self,
        logits: np.ndarray,
        current_state: str | None,
        *,
        enabled: bool = True,
    ) -> np.ndarray:
        """Return logits with illegal next-states set to a large negative number.

        Parameters
        ----------
        logits:
            1-D array of shape (V,) - raw scores over all vertices.
        current_state:
            The vertex id of the current state, or None if this is the first
            step (no prior state). When None the logits are returned unchanged
            because every next-state is legal on the first step.
        enabled:
            When False (ablation A1 graph_mask_enabled=False), the logits are
            returned unchanged regardless of current_state.

        Returns
        -------
        np.ndarray
            Array of shape (V,) with illegal positions set to -1e9 (or the
            original logits when masking is disabled or current_state is None).
        """
        if not enabled or current_state is None:
            return logits.copy()

        i = self.index_of(current_state)
        legal_row: np.ndarray = self._legality_matrix[i]  # shape (V,), bool

        masked = logits.copy()
        masked[~legal_row] = _MASK_FILL
        return masked

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> GraphFSM:
        """Load a GraphFSMSpec from a YAML file and wrap it in a GraphFSM.

        Parameters
        ----------
        path:
            Path to a .fsm.yaml file conforming to the GraphFSMSpec schema.
        """
        spec = _load_spec(path)
        return cls(spec)
