"""Stratified partition function Z over behavioral strata.

For each stratum lambda in the singularity-types catalog, compute

    Z_lambda = sum_{i: stratum(i) == lambda} exp(-E(x_i) / T)
    Z = sum_lambda Z_lambda
    P(x_i) = exp(-E(x_i) / T) / Z
    P(lambda) = Z_lambda / Z

Phase 5 deferred: the geometric Whitney variant lives in
arch/stratified-partition-function.md as the merged P5 atom. This module
implements only the BEHAVIORAL-STRATUM partition function; geometric Whitney
strata require the full extrusion + Riemannian work that Phase 5 defers.

Numerical stability: all summations use the log-sum-exp identity
    logsumexp(a) = max(a) + log sum_i exp(a_i - max(a))
so no intermediate exp() ever overflows or underflows for reasonable energies
and temperatures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from nga.arch.singularity_types import SingularityType

__all__ = [
    "StratifiedPartition",
    "compute_stratified_partition",
    "expected_energy",
    "free_energy",
    "per_stratum_summary",
]

_NORM_TOL: float = 1e-6


def _logsumexp(a: np.ndarray) -> float:
    """Numerically stable scalar logsumexp of a 1-D array.

    Returns -inf for an empty array (no states in a stratum).
    """
    if a.size == 0:
        return float("-inf")
    m = float(np.max(a))
    return m + float(np.log(np.sum(np.exp(a - m))))


@dataclass
class StratifiedPartition:
    """Result of compute_stratified_partition.

    Fields
    ------
    energies:
        (N,) per-state energy E(x_i) as provided to the function.
    strata:
        Length-N list of SingularityType tags, one per state.
    temperature:
        Inverse-temperature scale T used to compute Boltzmann weights.
    log_Z:
        log of the total partition function; numerically stable.
    log_Z_per_stratum:
        Per-stratum log partition function. Only strata present in
        ``strata`` are included; strata with no states are absent.
    P_per_state:
        (N,) array of normalised point-wise probabilities P(x_i).
    P_per_stratum:
        Per-stratum marginal probability P(lambda) = Z_lambda / Z.
        Only strata present in ``strata`` are included.
    """

    energies: np.ndarray
    strata: list[SingularityType]
    temperature: float
    log_Z: float
    log_Z_per_stratum: dict[SingularityType, float]
    P_per_state: np.ndarray
    P_per_stratum: dict[SingularityType, float]


def compute_stratified_partition(
    energies: np.ndarray,
    strata: list[SingularityType],
    *,
    temperature: float = 1.0,
) -> StratifiedPartition:
    """Compute the Boltzmann partition function stratified by behavioral type.

    Uses a stable log-sum-exp implementation throughout. Validates that both
    the per-state and per-stratum probabilities sum to 1 within 1e-6.

    For each stratum lambda:
        log_Z_lambda = logsumexp(-E_i / T : stratum_i == lambda)
    Total:
        log_Z = logsumexp over lambda of log_Z_lambda
    Per-state probability:
        P(x_i) = exp(-E(x_i) / T - log_Z)
    Per-stratum probability:
        P(lambda) = exp(log_Z_lambda - log_Z)

    Parameters
    ----------
    energies:
        (N,) array of per-state energies. Need not be non-negative; any
        real values are valid.
    strata:
        Length-N list of SingularityType tags, one per state.
    temperature:
        Boltzmann temperature T > 0. Higher T flattens the distribution;
        lower T sharpens it toward minimum-energy states.

    Returns
    -------
    StratifiedPartition
        Fully populated result dataclass.

    Raises
    ------
    ValueError
        If energies and strata have different lengths, if temperature <= 0,
        or if normalization fails beyond the 1e-6 tolerance.
    """
    energies = np.asarray(energies, dtype=np.float64)
    n = len(energies)

    if len(strata) != n:
        raise ValueError(
            f"energies and strata must have the same length; "
            f"got {n} energies and {len(strata)} strata"
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    # Boltzmann exponent for each state: -E_i / T
    log_weights: np.ndarray = -energies / temperature  # (N,)

    # --- per-stratum log partition ---
    unique_strata: list[SingularityType] = sorted(
        set(strata), key=lambda s: s.value
    )

    strata_arr: list[SingularityType] = list(strata)
    log_Z_per_stratum: dict[SingularityType, float] = {}

    for lam in unique_strata:
        mask = np.array([s == lam for s in strata_arr], dtype=bool)
        lz = _logsumexp(log_weights[mask])
        log_Z_per_stratum[lam] = lz

    # --- total log Z ---
    stratum_logzs = np.array(
        [log_Z_per_stratum[lam] for lam in unique_strata], dtype=np.float64
    )
    log_Z = _logsumexp(stratum_logzs)

    # --- per-state probabilities ---
    log_P_per_state: np.ndarray = log_weights - log_Z
    P_per_state: np.ndarray = np.exp(log_P_per_state)

    # --- per-stratum probabilities ---
    P_per_stratum: dict[SingularityType, float] = {
        lam: float(np.exp(log_Z_per_stratum[lam] - log_Z))
        for lam in unique_strata
    }

    # --- validation ---
    sum_P_states = float(np.sum(P_per_state))
    if abs(sum_P_states - 1.0) > _NORM_TOL:
        raise ValueError(
            f"Per-state probabilities do not sum to 1 within {_NORM_TOL}; "
            f"got {sum_P_states:.8f}"
        )

    sum_P_strata = sum(P_per_stratum.values())
    if abs(sum_P_strata - 1.0) > _NORM_TOL:
        raise ValueError(
            f"Per-stratum probabilities do not sum to 1 within {_NORM_TOL}; "
            f"got {sum_P_strata:.8f}"
        )

    return StratifiedPartition(
        energies=energies,
        strata=strata_arr,
        temperature=temperature,
        log_Z=log_Z,
        log_Z_per_stratum=log_Z_per_stratum,
        P_per_state=P_per_state,
        P_per_stratum=P_per_stratum,
    )


def expected_energy(part: StratifiedPartition) -> float:
    """Compute the expected energy E[E(X)] = sum_i P(x_i) * E(x_i).

    Useful as a sanity check: for low temperature, expected_energy should
    approach the minimum energy in the distribution.

    Parameters
    ----------
    part:
        A StratifiedPartition returned by compute_stratified_partition.

    Returns
    -------
    float
        The probability-weighted mean energy.
    """
    return float(np.dot(part.P_per_state, part.energies))


def free_energy(part: StratifiedPartition) -> float:
    """Compute the thermodynamic free energy F = -T * log Z.

    At low temperature, F approaches the minimum energy. The free energy
    is a scalar summary of the partition function useful for comparing
    two different energy landscapes at the same temperature.

    Parameters
    ----------
    part:
        A StratifiedPartition returned by compute_stratified_partition.

    Returns
    -------
    float
        The free energy -T * log Z.
    """
    return -part.temperature * part.log_Z


def per_stratum_summary(
    part: StratifiedPartition,
) -> dict[SingularityType, dict[str, float]]:
    """Return a human-readable summary for each stratum present.

    For each stratum lambda, the summary contains:
      - "n": number of states with that stratum tag.
      - "mean_energy": arithmetic mean of E(x_i) for states in the stratum.
      - "P": marginal probability P(lambda) = Z_lambda / Z.

    Parameters
    ----------
    part:
        A StratifiedPartition returned by compute_stratified_partition.

    Returns
    -------
    dict[SingularityType, dict[str, float]]
        Mapping from stratum tag to summary dict with keys "n",
        "mean_energy", "P".
    """
    result: dict[SingularityType, dict[str, float]] = {}
    strata_arr = part.strata

    for lam in part.P_per_stratum:
        mask = np.array([s == lam for s in strata_arr], dtype=bool)
        n_in_stratum = int(np.sum(mask))
        mean_e = float(np.mean(part.energies[mask])) if n_in_stratum > 0 else 0.0
        result[lam] = {
            "n": float(n_in_stratum),
            "mean_energy": mean_e,
            "P": part.P_per_stratum[lam],
        }

    return result
