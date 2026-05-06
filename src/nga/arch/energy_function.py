"""Energy function E(x) over agent states.

Phase 5 prescribed form per docs/arch/energy-function-E.md:

    E(x) = alpha * cost(x)
         + beta  * uncertainty(x)
         + gamma * contradiction(x)
         + eta   * loop_pressure(x)
         - kappa * progress(x)

Each term is in [0, 1] (or [-1, 0] for the negated progress term so that
HIGH energy means BAD); the coefficients are non-negative and configurable.

Phase 5 v1 prescribes the formula; later phases may swap in a learnable E_theta
that is regularised toward this prescribed form (q02-energy-function-spec).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["EnergyContributions", "EnergyParameters", "EnergyFunction"]


@dataclass
class EnergyContributions:
    """Per-term breakdown of the energy for a single state.

    Fields
    ------
    cost:
        Weighted cost contribution: alpha * cost(x).
    uncertainty:
        Weighted uncertainty contribution: beta * uncertainty(x).
    contradiction:
        Weighted contradiction contribution: gamma * contradiction(x).
    loop_pressure:
        Weighted loop contribution: eta * loop_pressure(x).
    progress:
        Weighted negated progress contribution: -kappa * progress(x).
    total:
        Scalar sum of all contributions.
    """

    cost: float
    uncertainty: float
    contradiction: float
    loop_pressure: float
    progress: float
    total: float


@dataclass
class EnergyParameters:
    """Coefficient configuration for EnergyFunction.

    All weights must be non-negative. kappa is subtracted (progress is good).

    Fields
    ------
    alpha:
        Weight on cost(x). Default 0.5.
    beta:
        Weight on uncertainty(x). Default 1.0.
    gamma:
        Weight on contradiction(x). Default 1.0.
    eta:
        Weight on loop_pressure(x). Default 0.5.
    kappa:
        Weight on progress(x), subtracted. Default 0.5.
    """

    alpha: float = 0.5
    beta: float = 1.0
    gamma: float = 1.0
    eta: float = 0.5
    kappa: float = 0.5

    def __post_init__(self) -> None:
        for name, value in [
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("gamma", self.gamma),
            ("eta", self.eta),
            ("kappa", self.kappa),
        ]:
            if value < 0.0:
                raise ValueError(f"EnergyParameters.{name} must be >= 0, got {value}")


class EnergyFunction:
    """Computes E(x) from per-state signals.

    Inputs are the SAME signals the singularity detector consumes, plus a
    cost and progress signal that singularity_detector did not need:
      - cost: a non-negative scalar in [0, 1] indicating compute or token cost.
      - uncertainty: 1 - max(distribution); equivalently 1 - confidence.
      - contradiction: stub for Phase 4 (always 0.0 for now).
      - loop_pressure: from TaggerHistory, in [0, 1].
      - progress: a non-negative scalar in [0, 1] indicating task-level progress
        (1.0 = solved, 0.0 = stuck). For E0 use accuracy proxies; for E1 use
        normalised step count toward a terminal state.

    All raw signals are clipped to [0, 1] before weighting.
    """

    def __init__(self, params: EnergyParameters | None = None) -> None:
        """Initialise with optional coefficient configuration.

        Parameters
        ----------
        params:
            EnergyParameters instance. Defaults are used when None is passed.
        """
        self._params: EnergyParameters = params if params is not None else EnergyParameters()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        *,
        cost: float,
        uncertainty: float,
        contradiction: float,
        loop_pressure: float,
        progress: float,
    ) -> EnergyContributions:
        """Return the per-term breakdown plus the scalar total.

        Each raw signal is clipped to [0, 1] before weighting.

        Parameters
        ----------
        cost:
            Compute or token cost proxy in [0, 1].
        uncertainty:
            1 - max(distribution); 1 = maximally uncertain, 0 = certain.
        contradiction:
            Phase 4 stub; pass 0.0.
        loop_pressure:
            Normalised loop pressure from TaggerHistory, in [0, 1].
        progress:
            Task-level progress toward a terminal state, in [0, 1].

        Returns
        -------
        EnergyContributions
            Weighted per-term breakdown and scalar total.
        """
        p = self._params

        c = float(np.clip(cost, 0.0, 1.0))
        u = float(np.clip(uncertainty, 0.0, 1.0))
        k = float(np.clip(contradiction, 0.0, 1.0))
        lp = float(np.clip(loop_pressure, 0.0, 1.0))
        pr = float(np.clip(progress, 0.0, 1.0))

        cost_term = p.alpha * c
        unc_term = p.beta * u
        contra_term = p.gamma * k
        loop_term = p.eta * lp
        progress_term = -p.kappa * pr

        total = cost_term + unc_term + contra_term + loop_term + progress_term

        return EnergyContributions(
            cost=cost_term,
            uncertainty=unc_term,
            contradiction=contra_term,
            loop_pressure=loop_term,
            progress=progress_term,
            total=total,
        )

    def batch_compute(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        """Vectorised energy computation over a batch of states.

        Each key in batch is a (B,) numpy array of the corresponding signal.
        Required keys: "cost", "uncertainty", "contradiction",
        "loop_pressure", "progress".
        Returns a (B,) array of E values.

        Parameters
        ----------
        batch:
            Dictionary mapping signal names to (B,) float arrays. All arrays
            must have the same shape. Values are clipped to [0, 1].

        Returns
        -------
        np.ndarray
            Shape (B,) array of scalar energy values.

        Raises
        ------
        KeyError
            If any required signal key is missing from batch.
        ValueError
            If arrays in batch have inconsistent shapes.
        """
        required_keys = {"cost", "uncertainty", "contradiction", "loop_pressure", "progress"}
        missing = required_keys - set(batch.keys())
        if missing:
            raise KeyError(f"Missing required signal keys: {missing}")

        p = self._params

        c = np.clip(np.asarray(batch["cost"], dtype=np.float64), 0.0, 1.0)
        u = np.clip(np.asarray(batch["uncertainty"], dtype=np.float64), 0.0, 1.0)
        k = np.clip(np.asarray(batch["contradiction"], dtype=np.float64), 0.0, 1.0)
        lp = np.clip(np.asarray(batch["loop_pressure"], dtype=np.float64), 0.0, 1.0)
        pr = np.clip(np.asarray(batch["progress"], dtype=np.float64), 0.0, 1.0)

        # Check consistent shapes
        shapes = {arr.shape for arr in (c, u, k, lp, pr)}
        if len(shapes) > 1:
            raise ValueError(f"Inconsistent array shapes in batch: {shapes}")

        energies: np.ndarray = (
            p.alpha * c
            + p.beta * u
            + p.gamma * k
            + p.eta * lp
            - p.kappa * pr
        )
        return energies
