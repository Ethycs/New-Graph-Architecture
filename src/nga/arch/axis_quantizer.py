"""Axis quantiser: maps a continuous measurement to a typed-axis node ID.

The axis quantiser maps a continuous measurement (sigma, energy, margin) to a
typed-axis node ID. Each axis is a typed graph with a totally ordered vertex
set; the quantiser is a deterministic projection from R to vertex_set, so the
same input always yields the same node ID.

A bin schedule with edges ``[e_0, e_1, ..., e_n]`` produces ``n`` nodes, where
node ``i`` covers the half-open interval ``[e_i, e_{i+1})`` (the last node
also includes the right endpoint). Inputs outside ``[e_0, e_n]`` are clamped.
String IDs that already match the axis pass through unchanged, so the
projection is idempotent on quantised values.
"""
from __future__ import annotations

import numpy as np

__all__ = ["AxisQuantizer"]


class AxisQuantizer:
    """Discrete projection from a continuous axis to a typed-graph node id.

    Parameters
    ----------
    axis_name:
        Logical name of the axis (e.g. ``"sigma"``, ``"energy"``, ``"margin"``).
        Used as the default node-id prefix.
    bin_edges:
        Strictly increasing 1-D array of length ``n_bins + 1``. Edges define
        half-open intervals ``[e_i, e_{i+1})``, with the last interval closed.
    node_id_prefix:
        Optional override for node-id prefix. Defaults to ``axis_name``.

    Raises
    ------
    ValueError
        If ``bin_edges`` has fewer than 2 entries or is not strictly
        increasing.
    """

    def __init__(
        self,
        axis_name: str,
        bin_edges: np.ndarray,
        node_id_prefix: str | None = None,
    ) -> None:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1:
            raise ValueError(
                f"bin_edges must be 1-D, got shape {edges.shape}"
            )
        if edges.size < 2:
            raise ValueError(
                f"bin_edges must have length >= 2, got {edges.size}"
            )
        diffs = np.diff(edges)
        if not np.all(diffs > 0):
            raise ValueError(
                f"bin_edges must be strictly increasing, got {edges.tolist()}"
            )

        self.axis_name: str = axis_name
        self.bin_edges: np.ndarray = edges
        self._prefix: str = node_id_prefix if node_id_prefix is not None else axis_name
        self._n_bins: int = int(edges.size - 1)
        self._node_ids: list[str] = [
            f"{self._prefix}_{i}" for i in range(self._n_bins)
        ]
        self._node_id_set: frozenset[str] = frozenset(self._node_ids)

    # -- constructors ---------------------------------------------------

    @classmethod
    def from_quantiles(
        cls,
        axis_name: str,
        samples: np.ndarray,
        n_bins: int,
        node_id_prefix: str | None = None,
    ) -> "AxisQuantizer":
        """Build a quantiser whose bin edges are empirical quantiles of samples.

        The outer edges are the sample min/max; interior edges are the
        ``n_bins - 1`` interior quantiles. Each bin then contains roughly
        ``len(samples) / n_bins`` points (exactly so for uniform inputs).

        Raises
        ------
        ValueError
            If ``n_bins < 1`` or ``samples`` is empty, or if the resulting
            quantile edges are not strictly increasing (e.g. a degenerate
            sample with too many ties).
        """
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        arr = np.asarray(samples, dtype=float).ravel()
        if arr.size == 0:
            raise ValueError("samples must be non-empty")
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(arr, qs)
        return cls(axis_name, edges, node_id_prefix=node_id_prefix)

    @classmethod
    def from_uniform(
        cls,
        axis_name: str,
        low: float,
        high: float,
        n_bins: int,
        node_id_prefix: str | None = None,
    ) -> "AxisQuantizer":
        """Build a quantiser with ``n_bins`` equal-width bins on ``[low, high]``.

        Raises
        ------
        ValueError
            If ``n_bins < 1`` or ``low >= high``.
        """
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        if not (low < high):
            raise ValueError(f"require low < high, got low={low}, high={high}")
        edges = np.linspace(float(low), float(high), n_bins + 1)
        return cls(axis_name, edges, node_id_prefix=node_id_prefix)

    # -- queries --------------------------------------------------------

    def n_nodes(self) -> int:
        """Return the number of nodes (= number of bins) on this axis."""
        return self._n_bins

    def node_ids(self) -> list[str]:
        """Return the ordered list of node IDs covering the axis."""
        return list(self._node_ids)

    def bin_for(self, node_id: str) -> tuple[float, float]:
        """Return the half-open ``[low, high)`` interval covered by ``node_id``.

        The final bin is closed on the right but the upper edge is still
        returned as ``high``; clamping handles the boundary case in
        :meth:`quantise`.

        Raises
        ------
        KeyError
            If ``node_id`` does not belong to this axis.
        """
        if node_id not in self._node_id_set:
            raise KeyError(f"{node_id!r} is not a node on axis {self.axis_name!r}")
        idx = self._node_ids.index(node_id)
        return float(self.bin_edges[idx]), float(self.bin_edges[idx + 1])

    # -- core projection ------------------------------------------------

    def _quantise_scalar(self, value: float) -> str:
        clamped = float(np.clip(value, self.bin_edges[0], self.bin_edges[-1]))
        # searchsorted with side='right' on sorted edges gives index in [1, n].
        # Subtract 1 and clip to [0, n_bins - 1] to get the bin index.
        idx = int(np.searchsorted(self.bin_edges, clamped, side="right")) - 1
        if idx < 0:
            idx = 0
        if idx >= self._n_bins:
            idx = self._n_bins - 1
        return self._node_ids[idx]

    def quantise(self, value):  # type: ignore[no-untyped-def]
        """Project a continuous value (or array) to node IDs on this axis.

        Parameters
        ----------
        value:
            Either a scalar float, a 1-D numpy array of floats, or an already
            quantised string ID belonging to this axis. String IDs pass
            through unchanged so the projection is idempotent on its own
            outputs.

        Returns
        -------
        str | list[str]
            A single node ID for scalar input; a list of node IDs for array
            input.
        """
        # Idempotent pass-through for already-quantised string IDs.
        if isinstance(value, str):
            if value in self._node_id_set:
                return value
            raise ValueError(
                f"string {value!r} is not a node on axis {self.axis_name!r}"
            )

        if isinstance(value, np.ndarray):
            arr = np.asarray(value, dtype=float).ravel()
            return [self._quantise_scalar(float(v)) for v in arr]

        return self._quantise_scalar(float(value))
