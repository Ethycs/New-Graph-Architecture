"""Graph extrusion: turn the discrete graph into an abstract stratified space.

Phase 3 numpy scope: rather than building a full geometric mesh, this module
records:
  - per-node "cell radius" rho_v in the embedded space (a scalar)
  - per-edge "tube width" omega_e (a scalar)
  - a tagger that assigns each observation to the closest stratum: NODE,
    EDGE, or BOUNDARY.

The full Whitney-stratification + manifold construction lives in Phase 5
(stratified-partition-function.md). This module is the lightweight runtime
hook: given a point z in H^d and the prototypes, classify which stratum it
sits in.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from nga.arch.graph_fsm import GraphFSM

# Lazy import of poincare_distance so this module can be imported even when
# hyperbolic_embedding.py has not landed yet.  Any call path that actually
# invokes assign_stratum or assign_stratum_batch will trigger the real import
# at that point, ensuring a clear ImportError with a helpful message rather
# than a silent None.
def _poincare_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Thin shim: import and delegate to hyperbolic_embedding.poincare_distance."""
    try:
        from nga.arch.hyperbolic_embedding import poincare_distance  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "nga.arch.hyperbolic_embedding is not yet available. "
            "Ensure it is present before calling assign_stratum."
        ) from exc
    return float(poincare_distance(x, y))


class StratumKind(str, Enum):
    NODE = "node"
    EDGE = "edge"
    BOUNDARY = "boundary"


@dataclass
class ExtrusionMetadata:
    """All static data needed to classify points into strata.

    Attributes
    ----------
    fsm:
        The runtime graph FSM whose topology defines legal edges.
    prototypes:
        Array of shape (V, d) containing the Poincare-ball embedding for each
        vertex, in the same order as fsm.vertex_ids.
    cell_radii:
        Array of shape (V,) with the ball radius rho_v for each vertex.
    tube_widths:
        Mapping from (source_vertex_id, target_vertex_id) to the tube half-
        width omega_e for that directed edge.
    """

    fsm: GraphFSM
    prototypes: np.ndarray            # shape (V, d), Poincare ball
    cell_radii: np.ndarray            # shape (V,), positive
    tube_widths: dict[tuple[str, str], float]  # per-edge


@dataclass
class StratumAssignment:
    """Result of classifying a single point into a stratum.

    Attributes
    ----------
    kind:
        Which stratum the point belongs to.
    primary_vertex:
        The nearest vertex id.
    secondary_vertex:
        For EDGE and BOUNDARY assignments the second-nearest vertex id;
        None for pure NODE assignments.
    distance_to_primary:
        Poincare distance from the query point to the primary prototype.
    """

    kind: StratumKind
    primary_vertex: str               # nearest vertex_id
    secondary_vertex: str | None      # for edges/boundary: the other endpoint
    distance_to_primary: float


def build_extrusion_metadata(
    *,
    fsm: GraphFSM,
    prototypes: np.ndarray,
    cell_radius_default: float = 0.4,
    tube_width_default: float = 0.2,
) -> ExtrusionMetadata:
    """Construct ExtrusionMetadata with default radii and per-edge tube widths.

    Parameters
    ----------
    fsm:
        A loaded GraphFSM that defines the vertex set and directed-edge topology.
    prototypes:
        Array of shape (V, d) with one Poincare-ball point per vertex, in
        the same order as fsm.vertex_ids.
    cell_radius_default:
        Uniform cell radius rho_v assigned to every vertex (positive float).
    tube_width_default:
        Uniform tube half-width omega_e assigned to every directed edge
        (positive float).

    Returns
    -------
    ExtrusionMetadata
        Populated metadata object ready for stratum classification.

    Raises
    ------
    ValueError
        If prototypes.shape[0] != fsm.vertex_count.
    """
    if prototypes.shape[0] != fsm.vertex_count:
        raise ValueError(
            f"prototypes has {prototypes.shape[0]} rows but fsm has "
            f"{fsm.vertex_count} vertices."
        )

    cell_radii = np.full(fsm.vertex_count, cell_radius_default, dtype=float)

    # Build tube_widths dict from the legality matrix.
    vertex_ids = fsm.vertex_ids
    legal = fsm.legality_matrix  # shape (V, V), bool
    tube_widths: dict[tuple[str, str], float] = {}
    for i, src in enumerate(vertex_ids):
        for j, dst in enumerate(vertex_ids):
            if legal[i, j]:
                tube_widths[(src, dst)] = tube_width_default

    return ExtrusionMetadata(
        fsm=fsm,
        prototypes=prototypes,
        cell_radii=cell_radii,
        tube_widths=tube_widths,
    )


def assign_stratum(
    point: np.ndarray,                 # shape (d,)
    meta: ExtrusionMetadata,
) -> StratumAssignment:
    """Classify a point into its stratum.

    Algorithm:
      1. Compute poincare_distance(point, p_v) for every vertex v.
      2. Sort to get top-1 (primary) and top-2 (secondary) closest.
      3. If d_to_primary < cell_radius_default: kind = NODE.
      4. Else if (primary, secondary) is a legal edge AND d_to_secondary <
         d_to_primary + tube_width_default: kind = EDGE.
      5. Else: kind = BOUNDARY (unclassified).

    Parameters
    ----------
    point:
        A single point in the Poincare ball, shape (d,).
    meta:
        Precomputed ExtrusionMetadata from build_extrusion_metadata.

    Returns
    -------
    StratumAssignment
        Classification result with kind, primary vertex, optional secondary
        vertex, and distance to the primary prototype.
    """
    vertex_ids = meta.fsm.vertex_ids
    V = len(vertex_ids)

    # Step 1: compute distances to all prototypes.
    distances = np.empty(V, dtype=float)
    for i in range(V):
        distances[i] = _poincare_distance(point, meta.prototypes[i])

    # Step 2: sort to get top-2 closest.
    order = np.argsort(distances)
    idx_primary = int(order[0])
    d_primary = float(distances[idx_primary])
    vid_primary = vertex_ids[idx_primary]

    if V >= 2:
        idx_secondary = int(order[1])
        d_secondary = float(distances[idx_secondary])
        vid_secondary = vertex_ids[idx_secondary]
    else:
        idx_secondary = None
        d_secondary = float("inf")
        vid_secondary = None

    # Step 3: NODE check.
    rho = float(meta.cell_radii[idx_primary])
    if d_primary < rho:
        return StratumAssignment(
            kind=StratumKind.NODE,
            primary_vertex=vid_primary,
            secondary_vertex=None,
            distance_to_primary=d_primary,
        )

    # Step 4: EDGE check (both directions for undirected-like classification).
    if vid_secondary is not None:
        for src, dst in [(vid_primary, vid_secondary), (vid_secondary, vid_primary)]:
            edge_key = (src, dst)
            if edge_key in meta.tube_widths:
                omega = meta.tube_widths[edge_key]
                if d_secondary < d_primary + omega:
                    return StratumAssignment(
                        kind=StratumKind.EDGE,
                        primary_vertex=vid_primary,
                        secondary_vertex=vid_secondary,
                        distance_to_primary=d_primary,
                    )

    # Step 5: BOUNDARY.
    return StratumAssignment(
        kind=StratumKind.BOUNDARY,
        primary_vertex=vid_primary,
        secondary_vertex=vid_secondary,
        distance_to_primary=d_primary,
    )


def assign_stratum_batch(
    points: np.ndarray,                # shape (N, d)
    meta: ExtrusionMetadata,
) -> list[StratumAssignment]:
    """Vectorised wrapper over assign_stratum.

    Parameters
    ----------
    points:
        Array of shape (N, d) containing N points in the Poincare ball.
    meta:
        Precomputed ExtrusionMetadata from build_extrusion_metadata.

    Returns
    -------
    list[StratumAssignment]
        One StratumAssignment per input point, in the same order.
    """
    return [assign_stratum(points[i], meta) for i in range(points.shape[0])]
