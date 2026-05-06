"""Catastrophe-theoretic labels on FSM edges.

Phase 2 scope: enum + lookup. Real catastrophe theory (fold/cusp/swallowtail
geometry) lands in Phase 5 alongside the stratified partition function. Until
then this module provides defaults so singularity_detector can read priors
without crashing on un-tagged FSMs.
"""
from __future__ import annotations

from enum import Enum

from nga.arch.graph_fsm import GraphFSM

__all__ = ["CatastropheLabel", "get_edge_label", "edge_priors"]


class CatastropheLabel(str, Enum):
    NONE = "none"
    FOLD = "fold"
    CUSP = "cusp"
    SWALLOWTAIL = "swallowtail"


def get_edge_label(fsm: GraphFSM, src: str, dst: str) -> CatastropheLabel:
    """Look up the catastrophe label for an edge.

    Phase 2: returns NONE unless the FSM Edge model carries a recognised label
    in its `label` field (e.g. "fold"). Edges with unknown labels return NONE
    silently. If the transition is not legal, also returns NONE.

    Parameters
    ----------
    fsm:
        A loaded GraphFSM instance whose underlying spec provides the edge list.
    src:
        Source vertex id.
    dst:
        Destination vertex id.

    Returns
    -------
    CatastropheLabel
        The label for the matching edge, or NONE when the edge is absent,
        un-labelled, or carries an unrecognised label string.
    """
    if not fsm.is_legal_transition(src, dst):
        return CatastropheLabel.NONE
    for edge in fsm._spec.edges:
        if edge.source == src and edge.target == dst:
            label = getattr(edge, "label", None)
            if label is None:
                return CatastropheLabel.NONE
            try:
                return CatastropheLabel(label)
            except ValueError:
                return CatastropheLabel.NONE
    return CatastropheLabel.NONE


def edge_priors(label: CatastropheLabel) -> dict[str, float]:
    """Default sigma-weighting priors for an edge with the given catastrophe label.

    Phase 2 stub: FOLD/CUSP/SWALLOWTAIL edges contribute a small additive
    bias to the singularity score; NONE contributes zero. Real numeric priors
    are an open question (q02-energy-function-spec).

    Parameters
    ----------
    label:
        The catastrophe label of the edge.

    Returns
    -------
    dict[str, float]
        A mapping with at least the key "bias" (additive singularity prior).
    """
    if label == CatastropheLabel.NONE:
        return {"bias": 0.0}
    if label == CatastropheLabel.FOLD:
        return {"bias": 0.05}
    if label == CatastropheLabel.CUSP:
        return {"bias": 0.10}
    if label == CatastropheLabel.SWALLOWTAIL:
        return {"bias": 0.15}
    return {"bias": 0.0}
