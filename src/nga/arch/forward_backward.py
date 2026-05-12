"""Classical Baum-Welch forward-backward E-step in log space.

This module provides the soft credit-assignment primitives that the
architecture's Bayesian posterior atoms (``nga.arch.posterior_mask``) couple
to during Phase 7 training. Given an HMM-style sequence with an
initial-state distribution, a transition matrix, and a per-timestep emission
log-likelihood, the forward-backward algorithm computes:

  - ``log_gamma[t, s]``: the marginal posterior over states at every time
    step, ``log P(state_t = s | obs_1:T)``.
  - ``log_xi[t, i, j]``: the marginal posterior over consecutive state
    pairs, ``log P(state_t = i, state_{t+1} = j | obs_1:T)``.
  - ``log_likelihood``: ``log P(obs_1:T)``, the data log-evidence.

The expected pairwise transition counts ``sum_t exp(log_xi[t])`` are exactly
the sufficient statistics that a Beta(alpha, beta) posterior on edge
legality consumes -- see ``nga.arch.posterior_mask.PosteriorMask``.

Why this matters for the architecture
-------------------------------------
Phase 7 wires this E-step into ``energy_minimization_trainer`` so that
posterior_mask updates are driven by *observed* expected counts rather than
by a circular, model-derived "quality" signal that depends on the very
posterior it is updating. This breaks the cold-start bootstrap circle: at
init, when the model has no opinion, observed transitions still produce
real evidence for the Beta posterior, because every (i -> j) pair in a
fully-observed sequence is a positive observation regardless of model
state.

Two regimes are supported:
  - **Fully observed**: the latent IS the observation (no emission
    ambiguity). ``expected_counts_observed`` reduces forward-backward to
    pure pair-counting in O(T).
  - **Latent / typed-anchored**: emissions are non-deterministic (e.g.
    typed latent clusters from ``typed_latent_clustering``). The full
    log-space recursion runs in O(T S^2).

Numerical stability
-------------------
All recursions are in log space and use ``scipy.special.logsumexp``, which
is the standard way to evaluate ``log(sum(exp(x)))`` without underflow when
some entries are extremely negative. A subtle point: in the forward pass
the standard recursion is

    alpha[t, j] = (sum_i alpha[t-1, i] * trans[i, j]) * emit[t, j]

In log space this becomes

    log_alpha[t, j] = logsumexp_i(log_alpha[t-1, i] + log_trans[i, j])
                      + log_emit[t, j]

The reduction is over axis 0 (the *previous* state ``i``) of a (S, S)
broadcast, *not* axis 1; getting that wrong silently swaps ``log_trans``
for its transpose and produces plausible but incorrect posteriors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "ForwardBackwardResult",
    "log_forward",
    "log_backward",
    "log_posterior_states",
    "log_posterior_transitions",
    "forward_backward",
    "expected_counts_observed",
    "bayesian_m_step_beta",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    log_init: np.ndarray | None,
    log_trans: np.ndarray,
    log_emit: np.ndarray,
) -> tuple[int, int]:
    """Validate shapes, returning (T, S). ``log_init`` may be None."""
    if log_emit.ndim != 2:
        raise ValueError(
            f"log_emit must have shape (T, S), got ndim={log_emit.ndim}"
        )
    t_steps, n_states = log_emit.shape
    if t_steps < 1:
        raise ValueError(f"observation sequence must have T >= 1, got T={t_steps}")
    if log_trans.shape != (n_states, n_states):
        raise ValueError(
            f"log_trans must have shape (S, S) = ({n_states}, {n_states}), "
            f"got {log_trans.shape}"
        )
    if log_init is not None:
        if log_init.shape != (n_states,):
            raise ValueError(
                f"log_init must have shape (S,) = ({n_states},), "
                f"got {log_init.shape}"
            )
    return t_steps, n_states


# ---------------------------------------------------------------------------
# Forward / backward primitives
# ---------------------------------------------------------------------------


def log_forward(
    log_init: np.ndarray,
    log_trans: np.ndarray,
    log_emit: np.ndarray,
) -> np.ndarray:
    """Log-space forward recursion.

    Parameters
    ----------
    log_init:
        ``log P(state at t=0)``, shape (S,).
    log_trans:
        ``log P(state_{t+1} = j | state_t = i)``, shape (S, S). Row index
        is the source state ``i``; column index is the destination ``j``.
    log_emit:
        ``log P(obs_t | state_t = s)``, shape (T, S).

    Returns
    -------
    log_alpha: ndarray of shape (T, S)
        ``log_alpha[t, s] = log P(obs_1:t, state_t = s)``.
    """
    log_init = np.asarray(log_init, dtype=np.float64)
    log_trans = np.asarray(log_trans, dtype=np.float64)
    log_emit = np.asarray(log_emit, dtype=np.float64)
    t_steps, n_states = _validate_inputs(log_init, log_trans, log_emit)

    log_alpha = np.full((t_steps, n_states), -np.inf, dtype=np.float64)
    log_alpha[0] = log_init + log_emit[0]

    for t in range(1, t_steps):
        # broadcast: log_alpha[t-1, i] + log_trans[i, j] over i, j.
        # Reduce over i (axis=0 of the (S, S) matrix).
        prev_plus_trans = log_alpha[t - 1, :, None] + log_trans  # (S, S)
        log_alpha[t] = logsumexp(prev_plus_trans, axis=0) + log_emit[t]
    return log_alpha


def log_backward(
    log_trans: np.ndarray,
    log_emit: np.ndarray,
) -> np.ndarray:
    """Log-space backward recursion.

    Parameters
    ----------
    log_trans:
        ``log P(state_{t+1} = j | state_t = i)``, shape (S, S).
    log_emit:
        ``log P(obs_t | state_t = s)``, shape (T, S).

    Returns
    -------
    log_beta: ndarray of shape (T, S)
        ``log_beta[t, s] = log P(obs_{t+1}:T | state_t = s)``. By
        convention ``log_beta[T-1, :] = 0`` (the empty product).
    """
    log_trans = np.asarray(log_trans, dtype=np.float64)
    log_emit = np.asarray(log_emit, dtype=np.float64)
    t_steps, n_states = _validate_inputs(None, log_trans, log_emit)

    log_beta = np.full((t_steps, n_states), -np.inf, dtype=np.float64)
    log_beta[t_steps - 1] = 0.0

    for t in range(t_steps - 2, -1, -1):
        # log_trans[i, j] + log_emit[t+1, j] + log_beta[t+1, j], reduce over j.
        next_term = log_trans + log_emit[t + 1] + log_beta[t + 1]  # (S, S)
        log_beta[t] = logsumexp(next_term, axis=1)
    return log_beta


def log_posterior_states(
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
) -> np.ndarray:
    """Soft state assignments ``log_gamma[t, s] = log P(state_t = s | obs_1:T)``.

    Computed by normalising ``log_alpha + log_beta`` along the state axis.
    """
    log_alpha = np.asarray(log_alpha, dtype=np.float64)
    log_beta = np.asarray(log_beta, dtype=np.float64)
    if log_alpha.shape != log_beta.shape:
        raise ValueError(
            f"log_alpha and log_beta must agree in shape, got "
            f"{log_alpha.shape} vs {log_beta.shape}"
        )
    unnormalised = log_alpha + log_beta  # (T, S)
    log_norm = logsumexp(unnormalised, axis=1, keepdims=True)  # (T, 1)
    return unnormalised - log_norm


def log_posterior_transitions(
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
    log_trans: np.ndarray,
    log_emit: np.ndarray,
) -> np.ndarray:
    """Pairwise posterior over consecutive states.

    Returns ``log_xi`` of shape (T-1, S, S) where

        log_xi[t, i, j] = log P(state_t = i, state_{t+1} = j | obs_1:T).

    Each (T, S, S)-slice over t is a full joint and sums (in linear space)
    to 1.
    """
    log_alpha = np.asarray(log_alpha, dtype=np.float64)
    log_beta = np.asarray(log_beta, dtype=np.float64)
    log_trans = np.asarray(log_trans, dtype=np.float64)
    log_emit = np.asarray(log_emit, dtype=np.float64)
    t_steps, n_states = _validate_inputs(None, log_trans, log_emit)
    if log_alpha.shape != (t_steps, n_states):
        raise ValueError(
            f"log_alpha must have shape (T, S) = ({t_steps}, {n_states}), "
            f"got {log_alpha.shape}"
        )
    if log_beta.shape != (t_steps, n_states):
        raise ValueError(
            f"log_beta must have shape (T, S) = ({t_steps}, {n_states}), "
            f"got {log_beta.shape}"
        )

    if t_steps < 2:
        return np.zeros((0, n_states, n_states), dtype=np.float64)

    # log_xi_unnorm[t, i, j] = log_alpha[t, i] + log_trans[i, j]
    #                         + log_emit[t+1, j] + log_beta[t+1, j]
    # Normalise per-t over (i, j) jointly so each slice is a probability.
    log_alpha_t = log_alpha[:-1, :, None]                 # (T-1, S, 1)
    log_beta_tp1 = log_beta[1:, None, :]                  # (T-1, 1, S)
    log_emit_tp1 = log_emit[1:, None, :]                  # (T-1, 1, S)
    log_xi_unnorm = (
        log_alpha_t + log_trans[None, :, :] + log_emit_tp1 + log_beta_tp1
    )  # (T-1, S, S)
    log_norm = logsumexp(log_xi_unnorm, axis=(1, 2), keepdims=True)
    return log_xi_unnorm - log_norm


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


@dataclass
class ForwardBackwardResult:
    """Result bundle for a single forward-backward pass.

    Attributes
    ----------
    log_gamma:
        Marginal posterior over states, shape (T, S).
    log_xi:
        Marginal posterior over consecutive state pairs, shape (T-1, S, S).
    log_likelihood:
        ``log P(obs_1:T)``, the total data log-evidence.
    expected_transition_counts:
        ``sum_t exp(log_xi[t])``, shape (S, S). These are the sufficient
        statistics that a Beta posterior over edge legality consumes.
    """

    log_gamma: np.ndarray
    log_xi: np.ndarray
    log_likelihood: float
    expected_transition_counts: np.ndarray


def forward_backward(
    log_init: np.ndarray,
    log_trans: np.ndarray,
    log_emit: np.ndarray,
) -> ForwardBackwardResult:
    """Run forward + backward + posteriors and aggregate expected counts.

    See module docstring for the recursion semantics. The aggregated
    ``expected_transition_counts[i, j] = sum_t P(state_t = i,
    state_{t+1} = j | obs_1:T)`` is the natural sufficient statistic for a
    Beta(alpha, beta) edge-legality posterior.
    """
    log_init = np.asarray(log_init, dtype=np.float64)
    log_trans = np.asarray(log_trans, dtype=np.float64)
    log_emit = np.asarray(log_emit, dtype=np.float64)

    log_alpha = log_forward(log_init, log_trans, log_emit)
    log_beta = log_backward(log_trans, log_emit)

    log_gamma = log_posterior_states(log_alpha, log_beta)
    log_xi = log_posterior_transitions(log_alpha, log_beta, log_trans, log_emit)

    # log P(obs_1:T) = logsumexp_s log_alpha[T-1, s].
    log_likelihood = float(logsumexp(log_alpha[-1]))

    if log_xi.shape[0] == 0:
        n_states = log_emit.shape[1]
        expected_counts = np.zeros((n_states, n_states), dtype=np.float64)
    else:
        expected_counts = np.exp(log_xi).sum(axis=0)

    return ForwardBackwardResult(
        log_gamma=log_gamma,
        log_xi=log_xi,
        log_likelihood=log_likelihood,
        expected_transition_counts=expected_counts,
    )


# ---------------------------------------------------------------------------
# Fully-observed special case
# ---------------------------------------------------------------------------


def expected_counts_observed(
    observations: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """Pair-count expected transitions when state is fully observed.

    When the latent IS the observation (delta-emission), forward-backward
    collapses to counting consecutive (i, j) pairs in the observation
    sequence. This is the cold-start path the architecture uses before any
    latent-clustering signal is available.

    Parameters
    ----------
    observations:
        Integer array of shape (T,) with values in [0, n_states).
    n_states:
        Total number of states S.

    Returns
    -------
    counts: ndarray of shape (n_states, n_states), dtype int64
        ``counts[i, j]`` is the number of times the consecutive pair
        ``(i, j)`` appears in ``observations``.
    """
    obs = np.asarray(observations).reshape(-1)
    if obs.dtype.kind not in ("i", "u"):
        if not np.all(obs == obs.astype(np.int64)):
            raise ValueError("observations must be integer state indices")
        obs = obs.astype(np.int64)
    if n_states <= 0:
        raise ValueError(f"n_states must be positive, got {n_states}")
    if obs.size > 0 and (np.any(obs < 0) or np.any(obs >= n_states)):
        raise IndexError(
            f"observation indices must lie in [0, {n_states}), got "
            f"min={int(obs.min())}, max={int(obs.max())}"
        )

    counts = np.zeros((n_states, n_states), dtype=np.int64)
    if obs.size < 2:
        return counts
    src = obs[:-1].astype(np.intp)
    dst = obs[1:].astype(np.intp)
    np.add.at(counts, (src, dst), 1)
    return counts


# ---------------------------------------------------------------------------
# Bayesian M-step
# ---------------------------------------------------------------------------


def bayesian_m_step_beta(
    expected_counts_pos: np.ndarray,
    expected_counts_neg: np.ndarray | None = None,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bayesian M-step for a Beta(alpha, beta) posterior on edge legality.

    The cold-start convention used here is "every observed transition is
    positive evidence for that edge": ``alpha = prior_alpha +
    expected_counts_pos``. Negative evidence is optional; if
    ``expected_counts_neg`` is None, ``beta`` simply equals the prior. This
    matches the architecture's cold-start mode where we have no
    counter-examples until a downstream signal flags an edge as illegal.

    Parameters
    ----------
    expected_counts_pos:
        Non-negative (S, S) array of expected positive observations per
        edge.
    expected_counts_neg:
        Optional non-negative (S, S) array of expected negative
        observations per edge.
    prior_alpha:
        Beta prior alpha. Use 1.0 for the uniform cold start, 0.0 to make
        the posterior coincide with the raw expected counts (improper
        prior; useful for tests).
    prior_beta:
        Beta prior beta. Same conventions as ``prior_alpha``.

    Returns
    -------
    alpha, beta:
        Two (S, S) float arrays giving the updated Beta parameters.
    """
    pos = np.asarray(expected_counts_pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] != pos.shape[1]:
        raise ValueError(
            f"expected_counts_pos must be square (S, S), got {pos.shape}"
        )
    if np.any(pos < 0.0):
        raise ValueError("expected_counts_pos must be non-negative")
    if expected_counts_neg is None:
        neg = np.zeros_like(pos)
    else:
        neg = np.asarray(expected_counts_neg, dtype=np.float64)
        if neg.shape != pos.shape:
            raise ValueError(
                f"expected_counts_neg must match expected_counts_pos shape, "
                f"got {neg.shape} vs {pos.shape}"
            )
        if np.any(neg < 0.0):
            raise ValueError("expected_counts_neg must be non-negative")
    if prior_alpha < 0.0 or prior_beta < 0.0:
        raise ValueError(
            f"prior_alpha and prior_beta must be non-negative, got "
            f"{prior_alpha}, {prior_beta}"
        )

    alpha = float(prior_alpha) + pos
    beta = float(prior_beta) + neg
    return alpha, beta
