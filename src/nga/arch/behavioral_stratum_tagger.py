"""Behavioral stratum tagger: assigns each step one of the SingularityType tags.

Pure rules over (TypedFieldOutput, history, FSM, margin). Outputs a (tag, bitmask)
pair per step. Phase 2: fires on margin / decision-tie / illegal / loop-risk /
stabilizer-jump signals; CONTRADICTION and STABILIZER_JUMP are stubbed (real
versions land when more arch atoms exist).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from nga.arch.graph_fsm import GraphFSM
from nga.arch.singularity_types import PRIORITY_ORDER, SingularityType, type_bit

__all__ = ["TaggerInput", "TagResult", "tag_step", "TaggerHistory"]


@dataclass
class TaggerInput:
    """Inputs to one tagger step.

    Fields
    ------
    distribution:
        Shape (V,) softmax distribution over states.
    predicted_state:
        Vertex id of the argmax prediction.
    current_state:
        Vertex id of the current FSM state, or None on the first step.
    margin:
        Top-1 minus top-2 of the distribution (from compute_margin).
    """

    distribution: np.ndarray
    predicted_state: str
    current_state: str | None
    margin: float


@dataclass
class TagResult:
    """Output of one tagger step.

    Fields
    ------
    tag:
        Highest-priority SingularityType that fired.
    bitmask:
        Integer bitmask recording every rule that fired, independently of
        priority. Bit k is set iff the rule for the k-th SingularityType
        member (in enum definition order) fired.
    confidence:
        Magnitude of the strongest firing signal in [0, 1].
    """

    tag: SingularityType
    bitmask: int
    confidence: float


class TaggerHistory:
    """Rolling window of recent predicted states for loop detection.

    Default N=5; loop-risk fires when the most recent predicted_state appears
    >= 2 times in the last N predictions (q12-loop-risk-detection default).

    Parameters
    ----------
    max_len:
        Window size. Older entries are evicted when the buffer is full.
    """

    def __init__(self, max_len: int = 5) -> None:
        self._buf: deque[str] = deque(maxlen=max_len)

    def push(self, state: str) -> None:
        """Append state to the rolling window.

        Parameters
        ----------
        state:
            The predicted state vertex id for this step.
        """
        self._buf.append(state)

    def predicted_loop_risk(self, predicted_state: str) -> float:
        """Estimate loop risk for predicted_state given the current window.

        Returns a value in [0, 1]. A value >= 0.5 (i.e. >= 2 occurrences
        in the window) indicates loop-risk by the default threshold.

        Parameters
        ----------
        predicted_state:
            The state being considered for the next step.

        Returns
        -------
        float
            0.0 when the window is empty; min(1.0, count / 2.0) otherwise.
        """
        if not self._buf:
            return 0.0
        count = sum(1 for s in self._buf if s == predicted_state)
        return min(1.0, count / 2.0)  # >= 2 occurrences => 1.0

    def __len__(self) -> int:
        return len(self._buf)


def tag_step(
    inp: TaggerInput,
    history: TaggerHistory,
    fsm: GraphFSM,
    *,
    margin_threshold: float = 0.10,
    decision_tie_threshold: float = 0.02,
    loop_threshold: float = 0.5,
) -> TagResult:
    """Run all rules; pick the highest-priority tag that fires.

    Rules (evaluated for every step, independently of priority):
      ILLEGAL         : current_state is not None AND the transition
                        current_state -> predicted_state is not legal in fsm.
      STABILIZER_JUMP : stub for Phase 2 - always False until Phase 4 lands.
      CONTRADICTION   : stub for Phase 2 - always False until grammar
                        contradictions exist.
      DECISION_TIE    : top-1 minus top-2 of distribution < decision_tie_threshold.
      LOW_MARGIN      : margin < margin_threshold (can coexist with DECISION_TIE;
                        both bits may be set even though DECISION_TIE wins priority).
      LOOP_RISK       : history.predicted_loop_risk(predicted_state) >= loop_threshold.
      NOMINAL         : no other rule fires.

    The returned tag is the first entry in PRIORITY_ORDER whose rule fired.
    The bitmask records all fired rules. The confidence is the magnitude of
    the strongest firing signal among those that fired.

    Parameters
    ----------
    inp:
        Tagger inputs for this step.
    history:
        Rolling history of recent predicted states. This function does NOT
        push the current predicted_state into history; the caller is
        responsible for calling history.push() after consuming the result.
    fsm:
        The loaded GraphFSM instance for legality checks.
    margin_threshold:
        Below this value, the LOW_MARGIN rule fires.
    decision_tie_threshold:
        If top-1 minus top-2 of the distribution is below this value, the
        DECISION_TIE rule fires.
    loop_threshold:
        If loop risk score >= this value, the LOOP_RISK rule fires.

    Returns
    -------
    TagResult
        The priority tag, the all-fired bitmask, and confidence.
    """
    dist = np.asarray(inp.distribution, dtype=float)

    # Compute top-1 and top-2 values for decision-tie.
    sorted_vals = np.sort(dist)[::-1]
    top1_val: float = float(sorted_vals[0]) if sorted_vals.size > 0 else 0.0
    top2_val: float = float(sorted_vals[1]) if sorted_vals.size > 1 else top1_val
    top1_minus_top2: float = top1_val - top2_val

    loop_risk_score: float = history.predicted_loop_risk(inp.predicted_state)

    # Evaluate all rules and compute per-rule confidence.
    fired: dict[SingularityType, float] = {}

    # ILLEGAL
    if (
        inp.current_state is not None
        and not fsm.is_legal_transition(inp.current_state, inp.predicted_state)
    ):
        fired[SingularityType.ILLEGAL] = 1.0

    # STABILIZER_JUMP - Phase 2 stub; always False.
    # fired[SingularityType.STABILIZER_JUMP] = 0.0  # not set => not fired

    # CONTRADICTION - Phase 2 stub; always False.
    # fired[SingularityType.CONTRADICTION] = 0.0  # not set => not fired

    # DECISION_TIE
    if top1_minus_top2 < decision_tie_threshold:
        confidence_tie = 1.0 - top1_minus_top2 / decision_tie_threshold
        fired[SingularityType.DECISION_TIE] = float(np.clip(confidence_tie, 0.0, 1.0))

    # LOW_MARGIN
    if inp.margin < margin_threshold:
        confidence_lm = 1.0 - inp.margin / margin_threshold
        fired[SingularityType.LOW_MARGIN] = float(np.clip(confidence_lm, 0.0, 1.0))

    # LOOP_RISK
    if loop_risk_score >= loop_threshold:
        fired[SingularityType.LOOP_RISK] = loop_risk_score

    # NOMINAL fires when nothing else fired; it always has zero confidence.
    if not fired:
        fired[SingularityType.NOMINAL] = 0.0

    # Build bitmask from all fired rules.
    bitmask: int = 0
    for st in fired:
        bitmask |= 1 << type_bit(st)

    # Select the priority tag: first in PRIORITY_ORDER that fired.
    priority_tag: SingularityType = SingularityType.NOMINAL
    for st in PRIORITY_ORDER:
        if st in fired:
            priority_tag = st
            break

    # Confidence = maximum signal among fired rules.
    confidence: float = max(fired.values()) if fired else 0.0

    return TagResult(tag=priority_tag, bitmask=bitmask, confidence=confidence)
