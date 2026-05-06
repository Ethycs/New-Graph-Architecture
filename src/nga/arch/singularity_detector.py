"""Singularity detector sigma(x): scalar singularity score in [0, 1].

Combines margin (Phase 1), behavioral-stratum-tagger signals (Phase 2),
catastrophe-label edge priors (Phase 2 stub), and stabilizer-jump (Phase 4
stub). Higher = more singular = more likely to fail.

The composition is deliberately additive with bounded weights, so disabling
any signal (via ablation) reduces sigma smoothly rather than eliminating it.

Phase 2 monotonicity fix: signal_from_margin is now strictly monotone
decreasing across the full [0, 1] margin range. Previously the signal
saturated to 0.0 for all margins >= threshold (0.10), collapsing most
E0/E1 samples to the same sigma value and destroying AUROC discrimination
above threshold. The new piecewise formula uses a steep slope below
threshold (danger zone) and a gentle decay above it, so sigma is always
monotonically related to margin across the full range and margin's AUROC
is preserved even when sigma has no other signals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SignalContributions", "SingularityDetector", "compute_sigma"]


@dataclass
class SignalContributions:
    """Per-signal contribution to sigma before weighting."""

    margin_signal: float  # 1 - margin/threshold, clipped to [0, 1]
    decision_tie_signal: float
    illegal_signal: float  # 1.0 if illegal, else 0.0
    loop_signal: float
    stabilizer_signal: float  # Phase 4 stub: always 0.0
    catastrophe_bias: float  # additive bias from edge label, in [0, 0.15]


class SingularityDetector:
    """Configurable sigma(x) detector.

    Default weights (Phase 2): margin=0.6, decision_tie=0.1, illegal=0.2,
    loop=0.05, stabilizer=0.05. These sum to 1.0 (catastrophe_bias is added
    on top, then sigma is clipped to [0, 1]).
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "margin": 0.6,
        "decision_tie": 0.1,
        "illegal": 0.2,
        "loop": 0.05,
        "stabilizer": 0.05,
    }

    def __init__(
        self,
        *,
        margin_threshold: float = 0.10,
        decision_tie_threshold: float = 0.02,
        weights: dict[str, float] | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialise the detector.

        Parameters
        ----------
        margin_threshold:
            Margin value at or above which the margin signal is 0.0.
            Must be > 0.
        decision_tie_threshold:
            Top-1 minus top-2 gap at or above which the decision-tie
            signal is 0.0. Must be > 0.
        weights:
            Optional override of DEFAULT_WEIGHTS. Keys must be a subset
            of {"margin", "decision_tie", "illegal", "loop", "stabilizer"}.
            All values must be >= 0.
        enabled:
            When False (ablation A3 disables the detector), compute()
            returns (0.0, all-zero contributions).
        """
        if margin_threshold <= 0:
            raise ValueError(f"margin_threshold must be > 0, got {margin_threshold}")
        if decision_tie_threshold <= 0:
            raise ValueError(
                f"decision_tie_threshold must be > 0, got {decision_tie_threshold}"
            )

        self._margin_threshold = margin_threshold
        self._decision_tie_threshold = decision_tie_threshold
        self._enabled = enabled

        if weights is None:
            self._weights: dict[str, float] = dict(self.DEFAULT_WEIGHTS)
        else:
            unknown = set(weights) - set(self.DEFAULT_WEIGHTS)
            if unknown:
                raise ValueError(f"Unknown weight keys: {unknown}")
            for k, v in weights.items():
                if v < 0:
                    raise ValueError(f"Weight '{k}' must be >= 0, got {v}")
            # Start from defaults, then apply overrides
            self._weights = dict(self.DEFAULT_WEIGHTS)
            self._weights.update(weights)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        *,
        margin: float,
        is_illegal: bool = False,
        loop_risk: float = 0.0,
        stabilizer_risk: float = 0.0,
        catastrophe_bias: float = 0.0,
        decision_tie_strength: float = 0.0,
    ) -> tuple[float, SignalContributions]:
        """Return (sigma, breakdown).

        When enabled is False (ablation A3 disables the detector),
        returns (0.0, all-zero contributions).

        Parameters
        ----------
        margin:
            Raw margin value from Phase 1 (top-1 minus top-2 score).
            Must be in [-1.0, 1.0].
        is_illegal:
            True if the transition or state violates a legality constraint.
        loop_risk:
            Normalised loop-pressure signal in [0.0, 1.0].
        stabilizer_risk:
            Phase 4 stub. Expected 0.0; must be in [0.0, 1.0].
        catastrophe_bias:
            Additive prior from the catastrophe-label on the FSM edge.
            Must be in [0.0, 0.15].
        decision_tie_strength:
            Normalised top-1 minus top-2 gap used as the decision-tie
            signal. For Phase 2 the caller may pass the same value as
            `margin`. Must be in [0.0, 1.0].

        Returns
        -------
        tuple[float, SignalContributions]
            (sigma, per-signal breakdown).
        """
        # Validate inputs
        if not (-1.0 <= margin <= 1.0):
            raise ValueError(f"margin must be in [-1.0, 1.0], got {margin}")
        if not (0.0 <= loop_risk <= 1.0):
            raise ValueError(f"loop_risk must be in [0.0, 1.0], got {loop_risk}")
        if not (0.0 <= stabilizer_risk <= 1.0):
            raise ValueError(
                f"stabilizer_risk must be in [0.0, 1.0], got {stabilizer_risk}"
            )
        if not (0.0 <= catastrophe_bias <= 0.15):
            raise ValueError(
                f"catastrophe_bias must be in [0.0, 0.15], got {catastrophe_bias}"
            )
        if not (0.0 <= decision_tie_strength <= 1.0):
            raise ValueError(
                f"decision_tie_strength must be in [0.0, 1.0],"
                f" got {decision_tie_strength}"
            )

        zero_contribs = SignalContributions(
            margin_signal=0.0,
            decision_tie_signal=0.0,
            illegal_signal=0.0,
            loop_signal=0.0,
            stabilizer_signal=0.0,
            catastrophe_bias=0.0,
        )

        if not self._enabled:
            return 0.0, zero_contribs

        w = self._weights

        ms = self.signal_from_margin(margin, self._margin_threshold)
        # decision_tie_strength is a pre-normalised signal in [0, 1]; use directly
        # per the required formula. The static helper signal_from_decision_tie is
        # provided for callers that have a raw gap and want to convert it first.
        ill = 1.0 if is_illegal else 0.0

        contribs = SignalContributions(
            margin_signal=ms,
            decision_tie_signal=decision_tie_strength,
            illegal_signal=ill,
            loop_signal=loop_risk,
            stabilizer_signal=stabilizer_risk,
            catastrophe_bias=catastrophe_bias,
        )

        sigma: float = (
            w["margin"] * ms
            + w["decision_tie"] * decision_tie_strength
            + w["illegal"] * ill
            + w["loop"] * loop_risk
            + w["stabilizer"] * stabilizer_risk
        )
        sigma += catastrophe_bias
        sigma = max(0.0, min(1.0, sigma))

        return sigma, contribs

    # ------------------------------------------------------------------
    # Static signal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def signal_from_margin(margin: float, threshold: float) -> float:
        """Monotonic-in-margin signal in [0, 1].

        Phase 2: Below `threshold` the slope is steep (this is the "danger zone"
        where the singular_flag fires). Above threshold the signal continues to
        decay smoothly so σ is monotonically related to margin across the full
        range and AUROC discrimination is preserved.

        margin is clipped to [0.0, 1.0] before computation. Signal at margin=0
        is 1.0; signal at margin=threshold is 0.3; signal at margin=1.0 is 0.0.

        This monotonicity property ensures that sigma's AUROC is always at least
        as good as margin's AUROC even when sigma has no other signals. Previously
        the signal saturated to 0.0 for all margins >= threshold, collapsing
        discrimination for the majority of E0/E1 samples.

        Parameters
        ----------
        margin:
            Raw margin value. Clipped to [0.0, 1.0] before computation.
        threshold:
            The margin value that separates the steep danger zone from the gentle
            decay region. Must be > 0.
        """
        m = max(0.0, min(1.0, float(margin)))
        if m <= threshold:
            # Steep slope in the danger zone: 1.0 -> 0.3 across [0, threshold]
            return 1.0 - 0.7 * (m / threshold) if threshold > 0.0 else 0.0
        # Gentle decay above threshold: 0.3 -> 0.0 across [threshold, 1.0]
        span = 1.0 - threshold
        if span <= 0.0:
            return 0.0
        return 0.3 * (1.0 - m) / span

    @staticmethod
    def signal_from_decision_tie(top1_top2_gap: float, threshold: float) -> float:
        """Return the decision-tie signal in [0, 1].

        Same shape as margin signal but applied to the top-1 minus top-2 gap.
        For Phase 2 the caller can pass the same value as `margin`; the two
        signals exist as separate fields for future flexibility.

        Parameters
        ----------
        top1_top2_gap:
            The gap between the top-1 and top-2 scores.
        threshold:
            Gap at or above which the signal becomes 0.
        """
        if top1_top2_gap <= 0.0:
            return 1.0
        if top1_top2_gap >= threshold:
            return 0.0
        return 1.0 - top1_top2_gap / threshold


def compute_sigma(
    *,
    margin: float,
    is_illegal: bool = False,
    loop_risk: float = 0.0,
    stabilizer_risk: float = 0.0,
    catastrophe_bias: float = 0.0,
    decision_tie_strength: float = 0.0,
    detector: SingularityDetector | None = None,
) -> float:
    """Convenience wrapper: build a default detector if none passed and call .compute.

    Parameters
    ----------
    margin:
        Raw margin value from Phase 1 in [-1.0, 1.0].
    is_illegal:
        True if the transition violates a legality constraint.
    loop_risk:
        Normalised loop-pressure signal in [0.0, 1.0].
    stabilizer_risk:
        Phase 4 stub in [0.0, 1.0].
    catastrophe_bias:
        Additive prior from the catastrophe-label in [0.0, 0.15].
    decision_tie_strength:
        Normalised decision-tie signal in [0.0, 1.0].
    detector:
        Pre-built SingularityDetector instance. A default instance is
        constructed if None is passed.

    Returns
    -------
    float
        sigma in [0.0, 1.0].
    """
    if detector is None:
        detector = SingularityDetector()
    sigma, _ = detector.compute(
        margin=margin,
        is_illegal=is_illegal,
        loop_risk=loop_risk,
        stabilizer_risk=stabilizer_risk,
        catastrophe_bias=catastrophe_bias,
        decision_tie_strength=decision_tie_strength,
    )
    return sigma
