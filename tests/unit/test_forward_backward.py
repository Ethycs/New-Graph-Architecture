"""Unit tests for nga.arch.forward_backward.

Covers the log-space forward and backward recursions, the gamma/xi
posterior normalisation, log-likelihood consistency between the forward
and backward passes, the fully-observed pair-counting special case, the
Bayesian M-step for the Beta posterior, and end-to-end recovery on a
synthetic 3-state Markov chain.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from nga.arch.forward_backward import (
    ForwardBackwardResult,
    bayesian_m_step_beta,
    expected_counts_observed,
    forward_backward,
    log_backward,
    log_forward,
    log_posterior_states,
    log_posterior_transitions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_hmm(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small valid HMM in log space with T=4 observations and S=3 states."""
    rng = np.random.default_rng(seed)
    init = rng.dirichlet(np.ones(3))
    trans = np.stack([rng.dirichlet(np.ones(3)) for _ in range(3)], axis=0)
    # Per-time-step normalised emission distribution (each row sums to 1 in
    # linear space). Real HMM emissions are usually conditioned by an
    # observation (so rows are unnormalised likelihoods); using normalised
    # rows here just makes the toy posterior more interpretable.
    emit = np.stack([rng.dirichlet(np.ones(3)) for _ in range(4)], axis=0)
    return np.log(init), np.log(trans), np.log(emit)


# ---------------------------------------------------------------------------
# Forward / backward primitives
# ---------------------------------------------------------------------------


def test_forward_pass_shape() -> None:
    log_init, log_trans, log_emit = _toy_hmm(seed=1)
    log_alpha = log_forward(log_init, log_trans, log_emit)
    assert log_alpha.shape == log_emit.shape  # (T, S)


def test_forward_pass_initial_step() -> None:
    """log_alpha[0, s] = log_init[s] + log_emit[0, s]."""
    log_init, log_trans, log_emit = _toy_hmm(seed=2)
    log_alpha = log_forward(log_init, log_trans, log_emit)
    expected_first = log_init + log_emit[0]
    np.testing.assert_allclose(log_alpha[0], expected_first, rtol=0, atol=1e-12)


def test_backward_pass_terminal_step() -> None:
    """log_beta[T-1, s] = 0 by convention."""
    _, log_trans, log_emit = _toy_hmm(seed=3)
    log_beta = log_backward(log_trans, log_emit)
    np.testing.assert_array_equal(log_beta[-1], np.zeros(log_emit.shape[1]))


def test_log_posterior_states_normalised() -> None:
    log_init, log_trans, log_emit = _toy_hmm(seed=4)
    log_alpha = log_forward(log_init, log_trans, log_emit)
    log_beta = log_backward(log_trans, log_emit)
    log_gamma = log_posterior_states(log_alpha, log_beta)
    sums = np.exp(log_gamma).sum(axis=-1)
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-10)


def test_log_posterior_transitions_normalised() -> None:
    log_init, log_trans, log_emit = _toy_hmm(seed=5)
    log_alpha = log_forward(log_init, log_trans, log_emit)
    log_beta = log_backward(log_trans, log_emit)
    log_xi = log_posterior_transitions(log_alpha, log_beta, log_trans, log_emit)
    # Each (i, j) slice over t is a joint probability and sums to 1.
    sums = np.exp(log_xi).sum(axis=(1, 2))
    np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-10)


def test_log_likelihood_consistent() -> None:
    """logsumexp(log_alpha[-1]) == logsumexp(log_init + log_emit[0] + log_beta[0])."""
    log_init, log_trans, log_emit = _toy_hmm(seed=6)
    log_alpha = log_forward(log_init, log_trans, log_emit)
    log_beta = log_backward(log_trans, log_emit)
    fwd = float(logsumexp(log_alpha[-1]))
    bwd = float(logsumexp(log_init + log_emit[0] + log_beta[0]))
    assert fwd == pytest.approx(bwd, abs=1e-10)


# ---------------------------------------------------------------------------
# Fully-observed pair counting
# ---------------------------------------------------------------------------


def test_expected_counts_observed_simple() -> None:
    obs = np.array([0, 1, 0, 1, 0])
    counts = expected_counts_observed(obs, n_states=2)
    assert counts.shape == (2, 2)
    assert int(counts[0, 1]) == 2
    assert int(counts[1, 0]) == 2
    assert int(counts[0, 0]) == 0
    assert int(counts[1, 1]) == 0


def test_expected_counts_observed_self_loop() -> None:
    obs = np.array([0, 0, 0])
    counts = expected_counts_observed(obs, n_states=2)
    assert int(counts[0, 0]) == 2
    assert counts.sum() == 2


# ---------------------------------------------------------------------------
# Bayesian M-step
# ---------------------------------------------------------------------------


def test_bayesian_m_step_alpha_increases_with_observations() -> None:
    pos_few = np.array([[0.0, 1.0], [0.0, 0.0]])
    pos_many = np.array([[0.0, 5.0], [0.0, 0.0]])
    alpha_few, _ = bayesian_m_step_beta(pos_few)
    alpha_many, _ = bayesian_m_step_beta(pos_many)
    assert alpha_many[0, 1] > alpha_few[0, 1]
    # Untouched edges receive only the prior, identical in both calls.
    assert alpha_few[1, 1] == alpha_many[1, 1]


def test_bayesian_m_step_uninformative_prior_recovers_counts() -> None:
    """With prior_alpha = prior_beta = 0, alpha is exactly the counts."""
    pos = np.array([[0.0, 3.0], [2.0, 1.0]])
    neg = np.array([[1.0, 0.0], [0.0, 4.0]])
    alpha, beta = bayesian_m_step_beta(pos, neg, prior_alpha=0.0, prior_beta=0.0)
    np.testing.assert_array_equal(alpha, pos)
    np.testing.assert_array_equal(beta, neg)


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


def test_forward_backward_result_dataclass_fields_populated() -> None:
    log_init, log_trans, log_emit = _toy_hmm(seed=7)
    result = forward_backward(log_init, log_trans, log_emit)
    assert isinstance(result, ForwardBackwardResult)
    t_steps, n_states = log_emit.shape
    assert result.log_gamma.shape == (t_steps, n_states)
    assert result.log_xi.shape == (t_steps - 1, n_states, n_states)
    assert np.isfinite(result.log_likelihood)
    assert result.expected_transition_counts.shape == (n_states, n_states)
    # Expected counts are non-negative and sum to T - 1 (one xi-mass per gap).
    assert np.all(result.expected_transition_counts >= 0.0)
    assert result.expected_transition_counts.sum() == pytest.approx(
        t_steps - 1, abs=1e-10
    )


# ---------------------------------------------------------------------------
# End-to-end recovery on a synthetic 3-state chain
# ---------------------------------------------------------------------------


def test_three_state_chain_recovery() -> None:
    """Synthesise a 3-state Markov chain and verify expected_transition_counts.

    With a delta-style emission (state directly observed) and a fully
    informative emission likelihood, forward-backward should reproduce the
    raw consecutive-pair counts in the sampled sequence to within float
    tolerance. We then check those counts agree (within 5%) with the
    chain's true expected counts E[N_ij] = (T-1) * pi_i * P_ij over a long
    sequence.
    """
    rng = np.random.default_rng(42)
    n_states = 3
    # A non-trivial transition matrix.
    true_trans = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.2, 0.3, 0.5],
        ]
    )
    # Stationary distribution: solve pi P = pi.
    eigvals, eigvecs = np.linalg.eig(true_trans.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    stationary = np.real(eigvecs[:, idx])
    stationary = stationary / stationary.sum()

    # Sample one long sequence from the chain, starting at the stationary.
    # T must be large enough that the per-cell relative standard error is
    # below the 5% bound on every cell, including the smallest expected
    # count (~0.04 * T for the rarest transition here).
    t_steps = 50000
    states = np.empty(t_steps, dtype=np.int64)
    states[0] = int(rng.choice(n_states, p=stationary))
    for t in range(1, t_steps):
        states[t] = int(rng.choice(n_states, p=true_trans[states[t - 1]]))

    # Build a delta-style emission log-likelihood: log_emit[t, s] = 0 if
    # s == states[t] else -1e9. With this, the posterior collapses onto
    # the observed state at every t.
    log_emit = np.full((t_steps, n_states), -1e9, dtype=np.float64)
    log_emit[np.arange(t_steps), states] = 0.0
    # Use the stationary as the initial-state prior (in log space).
    log_init = np.log(stationary)
    log_trans = np.log(true_trans)

    result = forward_backward(log_init, log_trans, log_emit)

    # Recovered counts should match raw pair counts under delta emission.
    raw_counts = expected_counts_observed(states, n_states=n_states).astype(
        np.float64
    )
    np.testing.assert_allclose(
        result.expected_transition_counts, raw_counts, atol=1e-6
    )

    # And those raw counts should approximate (T-1) * pi_i * P_ij to ~5%.
    expected_counts = (t_steps - 1) * stationary[:, None] * true_trans
    rel_err = np.abs(raw_counts - expected_counts) / np.maximum(
        expected_counts, 1.0
    )
    assert rel_err.max() < 0.05, (
        f"max relative error {rel_err.max():.3f} exceeded 5%"
    )
