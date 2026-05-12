"""Lazy sparse product of typed graphs.

The system's full state is a node-tuple in this graph; storage is O(visited).

Each axis is a typed graph with a totally ordered vertex set. The product
graph's vertex set is the Cartesian product of axes, but cells are only
materialised when actually observed (or expected to be observed via
stratum-neighbour expansion). This makes the structural commitment that
"everything is a node" literal: any output of the system is a node-tuple,
one node per axis, and the system's true state space is this product graph.

Storage is O(observed traffic), not O(theoretical product), so a system with
4 axes of 5 nodes each (theoretical 4 * 5**4 = 2500 cells) costs only as
much as it actually visits.

The API does not consume orbit information directly, but is structured so a
future quotient-aware composition (using
:mod:`nga.arch.orbit_quotient_space`) can wrap this layer without changes
to the materialisation contract.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator

__all__ = ["NodeTuple", "ProductGraph"]


# A node-tuple is one node id per axis, in the order given by ``axis_names``.
NodeTuple = tuple[str, ...]


class ProductGraph:
    """Lazy sparse Cartesian product of typed graphs.

    Cells of the product are materialised on demand: only tuples passed to
    :meth:`materialise` or seen by :meth:`add_transition` consume storage.
    The theoretical size of the product (the literal Cartesian product of
    axis vertex sets) is reported by :meth:`n_theoretical` for diagnostics
    but is never allocated.

    Parameters
    ----------
    axis_names:
        Ordered axis names. Defines the canonical position of each component
        in a node-tuple.
    axis_node_lists:
        Mapping ``axis_name -> ordered list of node ids`` for that axis. The
        axis order on the typed graph is preserved here; it is what
        :meth:`expand_neighbours` uses to define "adjacent in axis order".
    """

    def __init__(
        self,
        axis_names: list[str],
        axis_node_lists: dict[str, list[str]],
    ) -> None:
        if len(axis_names) == 0:
            raise ValueError("axis_names must be non-empty")
        if len(set(axis_names)) != len(axis_names):
            raise ValueError(
                f"axis_names must be unique, got {axis_names}"
            )
        missing = [a for a in axis_names if a not in axis_node_lists]
        if missing:
            raise ValueError(
                f"axis_node_lists missing entries for axes: {missing}"
            )
        for a in axis_names:
            nodes = axis_node_lists[a]
            if len(nodes) == 0:
                raise ValueError(f"axis {a!r} has no node ids")
            if len(set(nodes)) != len(nodes):
                raise ValueError(
                    f"axis {a!r} node ids must be unique, got {nodes}"
                )

        self._axis_names: list[str] = list(axis_names)
        self._axis_index: dict[str, int] = {
            a: i for i, a in enumerate(self._axis_names)
        }
        # Per-axis: ordered node id list and a position lookup.
        self._axis_nodes: dict[str, list[str]] = {
            a: list(axis_node_lists[a]) for a in self._axis_names
        }
        self._axis_node_pos: dict[str, dict[str, int]] = {
            a: {nid: i for i, nid in enumerate(self._axis_nodes[a])}
            for a in self._axis_names
        }

        # Sparse materialisation tables.
        self._tuple_to_id: dict[NodeTuple, int] = {}
        self._id_to_tuple: list[NodeTuple] = []
        self._transitions: Counter[tuple[NodeTuple, NodeTuple]] = Counter()

    # ------------------------------------------------------------------
    # Axis introspection
    # ------------------------------------------------------------------

    @property
    def axis_names(self) -> list[str]:
        """Return the ordered axis names defining tuple component order."""
        return list(self._axis_names)

    def axis_index(self, axis_name: str) -> int:
        """Return the position of ``axis_name`` in a node-tuple.

        Raises
        ------
        KeyError
            If ``axis_name`` is not a registered axis.
        """
        try:
            return self._axis_index[axis_name]
        except KeyError:
            raise KeyError(
                f"unknown axis {axis_name!r}; "
                f"known axes: {self._axis_names}"
            ) from None

    def axis_nodes(self, axis_name: str) -> list[str]:
        """Return the ordered node-id list for ``axis_name`` (a copy)."""
        if axis_name not in self._axis_nodes:
            raise KeyError(
                f"unknown axis {axis_name!r}; "
                f"known axes: {self._axis_names}"
            )
        return list(self._axis_nodes[axis_name])

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_tuple(self, node_tuple: NodeTuple) -> NodeTuple:
        """Coerce to a tuple and validate arity / per-axis membership."""
        t = tuple(node_tuple)
        if len(t) != len(self._axis_names):
            raise ValueError(
                f"node_tuple has arity {len(t)} but product has "
                f"{len(self._axis_names)} axes ({self._axis_names})"
            )
        for axis_name, component in zip(self._axis_names, t):
            if component not in self._axis_node_pos[axis_name]:
                raise KeyError(
                    f"node id {component!r} is not a node on axis "
                    f"{axis_name!r}"
                )
        return t

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def materialise(self, node_tuple: NodeTuple) -> int:
        """Add ``node_tuple`` to the materialised set; return its cell id.

        Idempotent: a tuple that has already been materialised returns the
        same cell id without altering storage. Validates that each
        component is a registered node on the corresponding axis.

        Raises
        ------
        ValueError
            If ``node_tuple`` does not have one component per axis.
        KeyError
            If any component is not a registered node id for its axis.
        """
        t = self._validate_tuple(node_tuple)
        existing = self._tuple_to_id.get(t)
        if existing is not None:
            return existing
        cell_id = len(self._id_to_tuple)
        self._tuple_to_id[t] = cell_id
        self._id_to_tuple.append(t)
        return cell_id

    def is_materialised(self, node_tuple: NodeTuple) -> bool:
        """Return True iff ``node_tuple`` is currently materialised.

        Validates arity and per-axis membership before lookup so that a
        mistyped node id is reported rather than silently returning False.
        """
        t = self._validate_tuple(node_tuple)
        return t in self._tuple_to_id

    def cell_id(self, node_tuple: NodeTuple) -> int | None:
        """Return the cell id for ``node_tuple`` or None if not materialised."""
        t = self._validate_tuple(node_tuple)
        return self._tuple_to_id.get(t)

    def node_tuple_for(self, cell_id: int) -> NodeTuple:
        """Return the node-tuple associated with ``cell_id``.

        Raises
        ------
        IndexError
            If ``cell_id`` is out of range for the materialised cells.
        """
        if cell_id < 0 or cell_id >= len(self._id_to_tuple):
            raise IndexError(
                f"cell_id {cell_id} out of range "
                f"[0, {len(self._id_to_tuple)})"
            )
        return self._id_to_tuple[cell_id]

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def add_transition(
        self, src: NodeTuple, dst: NodeTuple
    ) -> tuple[int, int]:
        """Materialise both endpoints; record a directed transition.

        Returns the ``(src_id, dst_id)`` pair. Repeated calls with the same
        ``(src, dst)`` increment the transition count, allowing
        :meth:`transitions` to report observed traffic.
        """
        src_id = self.materialise(src)
        dst_id = self.materialise(dst)
        # Use the canonicalised tuples (post-validation) as keys.
        s = self._id_to_tuple[src_id]
        d = self._id_to_tuple[dst_id]
        self._transitions[(s, d)] += 1
        return src_id, dst_id

    def transitions(self) -> Iterator[tuple[NodeTuple, NodeTuple, int]]:
        """Yield ``(src, dst, count)`` for every observed transition."""
        for (s, d), c in self._transitions.items():
            yield s, d, c

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def n_materialised(self) -> int:
        """Return the number of materialised cells."""
        return len(self._id_to_tuple)

    def n_theoretical(self) -> int:
        """Return the size of the Cartesian product over all axes.

        This number is informational; it is never allocated. For axes of
        sizes ``s_1, s_2, ..., s_k`` the result is the product
        ``s_1 * s_2 * ... * s_k``.
        """
        prod = 1
        for a in self._axis_names:
            prod *= len(self._axis_nodes[a])
        return prod

    # ------------------------------------------------------------------
    # Stratum-neighbour expansion
    # ------------------------------------------------------------------

    def expand_neighbours(
        self,
        node_tuple: NodeTuple,
        axes: list[str] | None = None,
    ) -> list[NodeTuple]:
        """Return the node-tuples adjacent to ``node_tuple`` along one axis.

        A neighbour differs from ``node_tuple`` in exactly one component;
        the changed component must be at position +/-1 in that axis's
        ordered node list (i.e. adjacent in axis order). Boundary
        components contribute only their one valid neighbour.

        Parameters
        ----------
        node_tuple:
            The cell to expand from. Must be a valid tuple on this product;
            it does not have to be materialised.
        axes:
            If given, only axes in this list contribute neighbours. Unknown
            axis names raise KeyError. If None, all axes are eligible.

        Returns
        -------
        list[NodeTuple]
            Neighbours in a deterministic order (axis order, then -1 step
            before +1 step). Duplicates impossible by construction.
        """
        t = self._validate_tuple(node_tuple)
        if axes is None:
            eligible = list(self._axis_names)
        else:
            for a in axes:
                if a not in self._axis_index:
                    raise KeyError(
                        f"unknown axis {a!r}; "
                        f"known axes: {self._axis_names}"
                    )
            # Preserve product axis order regardless of caller order.
            eligible = [a for a in self._axis_names if a in set(axes)]

        neighbours: list[NodeTuple] = []
        for axis_name in eligible:
            i = self._axis_index[axis_name]
            axis_list = self._axis_nodes[axis_name]
            current = t[i]
            pos = self._axis_node_pos[axis_name][current]
            for step in (-1, 1):
                np_pos = pos + step
                if 0 <= np_pos < len(axis_list):
                    new_component = axis_list[np_pos]
                    new_tuple = t[:i] + (new_component,) + t[i + 1 :]
                    neighbours.append(new_tuple)
        return neighbours
