"""Control policy: σ-thresholded routing decisions.

Maps the singularity score σ(x) to one of three categorical decisions that
match the ``ControlAction`` schema in ``drivers/decision_trace_jsonl.py``:

  - σ < θ_normal           → ROUTE_NORMAL: trust the model's prediction.
  - θ_normal ≤ σ < θ_abstain → ROUTE_RECOVERY: override with the next state on
        the FSM-shortest-path from ``current_state`` to the nearest goal state.
  - σ ≥ θ_abstain          → ABSTAIN: emit the sentinel action ``-1``.

Threshold convention
--------------------
Both thresholds are inclusive on their upper bands. A σ exactly equal to
``θ_normal`` enters the recovery band (recovery is preferred when in doubt
about model trust); a σ exactly equal to ``θ_abstain`` enters the abstain
band (abstaining is preferred when in doubt about recovery). This makes the
two boundaries strictly monotone: increasing σ never re-grants trust.

Recovery is best-effort. When ``fsm`` or ``goal_states`` are missing, when
``current_state`` is ``None`` (no graph location yet), or when no goal is
reachable from ``current_state`` under the legality matrix, the policy falls
through to ROUTE_NORMAL with a ``reason`` string that ends in
``"recovery_unavailable"`` so downstream traces can audit why.

Dependencies: numpy + stdlib only.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from nga.arch.graph_fsm import GraphFSM

__all__ = ["ControlPolicy", "ControlPolicyResult"]


Decision = Literal["ROUTE_NORMAL", "ROUTE_RECOVERY", "ABSTAIN"]


@dataclass
class ControlPolicyResult:
    """Outcome of a single ``ControlPolicy.decide`` call.

    Fields
    ------
    decision:
        One of the three decisions in ``ControlAction.decision``.
    chosen_action:
        Integer action the runner should emit. Equals ``model_prediction``
        for ROUTE_NORMAL, the BFS first-step for ROUTE_RECOVERY, and the
        sentinel ``-1`` for ABSTAIN.
    reason:
        Short human-readable explanation of why the decision was reached;
        echoed into ``ControlAction.reason`` for the decision trace.
    """

    decision: Decision
    chosen_action: int
    reason: str


class ControlPolicy:
    """σ-thresholded control policy.

    Parameters
    ----------
    theta_normal:
        Lower σ boundary. σ < θ_normal routes the model's prediction
        unchanged. Must satisfy ``0.0 ≤ θ_normal ≤ θ_abstain ≤ 1.0``.
    theta_abstain:
        Upper σ boundary. σ ≥ θ_abstain abstains.
    fsm:
        Optional ``GraphFSM`` used by ROUTE_RECOVERY for BFS over the
        legality matrix. Required only if recovery can fire.
    goal_states:
        Optional list of integer column indices that recovery should steer
        toward. Any reachable goal will do; BFS picks the closest. Required
        only if recovery can fire.
    """

    def __init__(
        self,
        theta_normal: float = 0.3,
        theta_abstain: float = 0.7,
        fsm: GraphFSM | None = None,
        goal_states: list[int] | None = None,
    ) -> None:
        if not (0.0 <= theta_normal <= theta_abstain <= 1.0):
            raise ValueError(
                "Thresholds must satisfy 0.0 <= theta_normal <= theta_abstain <= 1.0; "
                f"got theta_normal={theta_normal}, theta_abstain={theta_abstain}."
            )
        self._theta_normal = float(theta_normal)
        self._theta_abstain = float(theta_abstain)
        self._fsm = fsm
        self._goal_states = list(goal_states) if goal_states is not None else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def theta_normal(self) -> float:
        """Lower σ boundary; σ at or above this enters the recovery band."""
        return self._theta_normal

    @property
    def theta_abstain(self) -> float:
        """Upper σ boundary; σ at or above this enters the abstain band."""
        return self._theta_abstain

    def decide(
        self,
        sigma: float,
        model_prediction: int,
        current_state: int | None,
    ) -> ControlPolicyResult:
        """Return the ``ControlPolicyResult`` for one step.

        Parameters
        ----------
        sigma:
            Singularity score in [0.0, 1.0].
        model_prediction:
            The integer action the trained model wants to emit.
        current_state:
            Integer column index of the current FSM state, or ``None`` on
            the first step (no prior state). When ``None`` recovery cannot
            fire and the policy falls through to ROUTE_NORMAL even in the
            recovery σ band.
        """
        # ABSTAIN band (σ ≥ θ_abstain). Inclusive upper boundary.
        if sigma >= self._theta_abstain:
            return ControlPolicyResult(
                decision="ABSTAIN",
                chosen_action=-1,
                reason=f"sigma={sigma:.4f} >= theta_abstain={self._theta_abstain:.4f}",
            )

        # NORMAL band (σ < θ_normal). Strict.
        if sigma < self._theta_normal:
            return ControlPolicyResult(
                decision="ROUTE_NORMAL",
                chosen_action=int(model_prediction),
                reason=f"sigma={sigma:.4f} < theta_normal={self._theta_normal:.4f}",
            )

        # RECOVERY band (θ_normal ≤ σ < θ_abstain). Try BFS.
        if self._fsm is None or self._goal_states is None or not self._goal_states:
            return ControlPolicyResult(
                decision="ROUTE_NORMAL",
                chosen_action=int(model_prediction),
                reason=(
                    f"sigma={sigma:.4f} in recovery band but no fsm/goals; "
                    "recovery_unavailable"
                ),
            )

        if current_state is None:
            return ControlPolicyResult(
                decision="ROUTE_NORMAL",
                chosen_action=int(model_prediction),
                reason=(
                    f"sigma={sigma:.4f} in recovery band but current_state is None; "
                    "recovery_unavailable"
                ),
            )

        next_step = self._compute_recovery_action(int(current_state))
        if next_step < 0:
            return ControlPolicyResult(
                decision="ROUTE_NORMAL",
                chosen_action=int(model_prediction),
                reason=(
                    f"sigma={sigma:.4f} in recovery band but no goal reachable from "
                    f"state={current_state}; recovery_unavailable"
                ),
            )

        return ControlPolicyResult(
            decision="ROUTE_RECOVERY",
            chosen_action=int(next_step),
            reason=(
                f"sigma={sigma:.4f} in [{self._theta_normal:.4f}, "
                f"{self._theta_abstain:.4f}); BFS first step from {current_state} "
                f"-> {next_step}"
            ),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_recovery_action(self, current_state: int) -> int:
        """Return the first step on the BFS-shortest-path to a goal, or ``-1``.

        ``-1`` signals "no path"; the caller falls back to ROUTE_NORMAL.
        """
        if self._fsm is None or not self._goal_states:
            return -1
        path = self._bfs_shortest_path(current_state, list(self._goal_states))
        if len(path) < 2:
            return -1
        return int(path[1])

    def _bfs_shortest_path(self, src: int, goals: list[int]) -> list[int]:
        """BFS over the FSM legality matrix; return ``[src, ..., goal]`` or ``[]``.

        The matrix entry ``legality_matrix[i, j]`` is True iff ``i -> j`` is a
        legal transition. The first goal popped from the queue is the closest
        (BFS guarantees shortest unweighted path).
        """
        if self._fsm is None:
            return []
        legality: np.ndarray = self._fsm.legality_matrix
        v = legality.shape[0]
        if not (0 <= src < v):
            return []
        goal_set = {int(g) for g in goals if 0 <= int(g) < v}
        if not goal_set:
            return []
        if src in goal_set:
            return [src]

        parent: dict[int, int] = {src: -1}
        queue: deque[int] = deque([src])
        found: int | None = None
        while queue:
            node = queue.popleft()
            # Neighbors are columns where legality[node, :] is True.
            neighbours = np.flatnonzero(legality[node])
            for nb in neighbours:
                nb_int = int(nb)
                if nb_int in parent:
                    continue
                parent[nb_int] = node
                if nb_int in goal_set:
                    found = nb_int
                    break
                queue.append(nb_int)
            if found is not None:
                break

        if found is None:
            return []

        # Reconstruct path goal -> src, then reverse.
        path: list[int] = []
        cur: int = found
        while cur != -1:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path
