"""Energy-weighted loss formula.

Standard cross-entropy is L_CE(x) = -log P(y_true | x). The energy-weighted
version multiplies by the energy at x:

    L_EW(x) = (1 + lambda * E(x)) * L_CE(x)

This upweights "hard" / "singular" / "high-uncertainty" states, training the
classifier to spend capacity where it matters. The +1 baseline ensures the
loss is never zero for low-energy states.

This module exposes the formula as a pure function. Whether it gets used as
the actual training loss depends on the backbone (sklearn cannot consume it;
torch can).

Per docs/arch/energy-weighted-loss.md the weighting function is log-linear:
    w(E) = 1 + lambda * E
which is stable near zero and avoids the exponential blow-up of the
exponential variant for large E values.
"""
from __future__ import annotations

import numpy as np

__all__ = ["energy_weighted_loss", "energy_weighted_batch_loss"]


def energy_weighted_loss(
    log_prob_y_true: float,
    energy: float,
    *,
    lambda_weight: float = 1.0,
) -> float:
    """Compute the energy-weighted loss for a single sample.

    L_EW(x) = (1 + lambda * E(x)) * (-log_prob_y_true)

    The log-linear weight w(E) = 1 + lambda * E is always >= 1 when energy
    >= 0 and lambda >= 0, ensuring the base loss is never reduced below its
    original value for non-negative energies.

    Parameters
    ----------
    log_prob_y_true:
        Log probability of the true class, in (-inf, 0]. For numerical
        stability the caller should clip to a reasonable lower bound such
        as -50 before passing here.
    energy:
        Scalar energy E(x) for this state. Typically non-negative (the
        prescribed form can be negative only when -kappa * progress
        dominates, but the weighting still works correctly in that case).
    lambda_weight:
        Scaling coefficient lambda >= 0 controlling how strongly energy
        amplifies the loss. Default 1.0. Setting to 0 recovers plain
        cross-entropy.

    Returns
    -------
    float
        The scalar energy-weighted loss, non-negative.
    """
    base_loss = -float(log_prob_y_true)
    weight = 1.0 + float(lambda_weight) * float(energy)
    return weight * base_loss


def energy_weighted_batch_loss(
    log_probs_y_true: np.ndarray,
    energies: np.ndarray,
    *,
    lambda_weight: float = 1.0,
) -> float:
    """Compute the mean energy-weighted loss over a batch.

    Applies energy_weighted_loss element-wise and returns the batch mean.

    Parameters
    ----------
    log_probs_y_true:
        (B,) array of per-sample log P(y_true | x). Values should be in
        (-inf, 0]. Clip to a lower bound (e.g. -50) before calling for
        numerical stability.
    energies:
        (B,) array of per-sample energies E(x_i).
    lambda_weight:
        Scaling coefficient lambda >= 0. Default 1.0.

    Returns
    -------
    float
        Scalar mean energy-weighted loss over the batch.

    Raises
    ------
    ValueError
        If log_probs_y_true and energies have different shapes.
    """
    log_probs = np.asarray(log_probs_y_true, dtype=np.float64)
    E = np.asarray(energies, dtype=np.float64)

    if log_probs.shape != E.shape:
        raise ValueError(
            f"log_probs_y_true and energies must have the same shape; "
            f"got {log_probs.shape} and {E.shape}"
        )

    base_losses = -log_probs
    weights = 1.0 + float(lambda_weight) * E
    per_sample: np.ndarray = weights * base_losses
    return float(np.mean(per_sample))
