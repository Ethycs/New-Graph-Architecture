"""Catalog of singularity-type tags the behavioral stratum tagger emits.

The tags are categorical labels (not topology). Phase 2 uses them as the
discrete vocabulary that singularity_detector aggregates into sigma(x), and
that behavioral_stratum_tagger writes into ResultsRecord.behavioral_stratum.
"""
from __future__ import annotations

from enum import Enum

__all__ = ["SingularityType", "PRIORITY_ORDER", "type_bit"]


class SingularityType(str, Enum):
    NOMINAL = "nominal"
    LOW_MARGIN = "low-margin"
    DECISION_TIE = "decision-tie"
    CONTRADICTION = "contradiction"
    ILLEGAL = "illegal"
    LOOP_RISK = "loop-risk"
    STABILIZER_JUMP = "stabilizer-jump"


# Priority order: highest-priority first. The tagger picks the first that fires.
# Rationale: a state that is illegal AND low-margin reports as illegal because
# downstream control should react to the legality violation first.
PRIORITY_ORDER: list[SingularityType] = [
    SingularityType.ILLEGAL,
    SingularityType.STABILIZER_JUMP,
    SingularityType.CONTRADICTION,
    SingularityType.DECISION_TIE,
    SingularityType.LOW_MARGIN,
    SingularityType.LOOP_RISK,
    SingularityType.NOMINAL,
]


def type_bit(t: SingularityType) -> int:
    """Bit index in the bitmask emitted alongside the priority tag.

    The index is the position of t in the SingularityType enum definition
    order (0-based). Used to construct the multi-label bitmask that records
    all rules that fired, independent of priority selection.

    Parameters
    ----------
    t:
        The SingularityType whose bit index to return.

    Returns
    -------
    int
        0-based bit index for t within the enum member order.
    """
    return list(SingularityType).index(t)
