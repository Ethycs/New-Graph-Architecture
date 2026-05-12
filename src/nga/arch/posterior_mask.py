"""Bayesian posterior legality mask: Beta(alpha, beta) per directed edge.

This is the Bayesian sibling of nga.arch.graph_legality_mask. Where the
boolean mask treats the adjacency matrix as a hard, oracular constant (an
edge is legal or it is not), the PosteriorMask treats every potential edge
(i -> j) as a Bernoulli random variable whose success probability theta_ij
has a Beta(alpha_ij, beta_ij) posterior. The posterior is updated online
from observed transitions, each weighted by a soft "quality" signal Q in
[0, 1] derived from singularity-detector signals (margin, sigma,
contradiction, loop_risk).

Conjugate update story
----------------------
Beta is the conjugate prior of Bernoulli. If theta ~ Beta(a, b) and we
observe a fractional success x in [0, 1] (interpreted as a soft Bernoulli
trial of mass 1), the posterior is theta | x ~ Beta(a + x, b + 1 - x).
Hard observations recover the standard rule: x=1 increments alpha by 1,
x=0 increments beta by 1. Cold start uses Beta(1, 1), which is the uniform
prior on [0, 1] - no prior knowledge. Posterior mean alpha/(alpha+beta)
starts at 0.5 and migrates toward 1 (legal) or 0 (illegal) as evidence
accumulates.

Drop-in compatibility
---------------------
PosteriorMask.legality_matrix(threshold) returns a boolean (V, V) array
shaped exactly like GraphFSM.legality_matrix, so it is a drop-in for the
hard-mask path. legality_bias() returns a (V, V) log-odds bias suitable
for the smooth (pre-softmax additive) mode.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.special import betaln, digamma

__all__ = ["PosteriorMask"]


_LOG_ODDS_CLIP = 30.0


def _beta_entropy(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Differential entropy of Beta(alpha, beta), element-wise.

    H = ln B(a, b) - (a - 1) psi(a) - (b - 1) psi(b) + (a + b - 2) psi(a + b)
    """
    a = alpha
    b = beta
    return (
        betaln(a, b)
        - (a - 1.0) * digamma(a)
        - (b - 1.0) * digamma(b)
        + (a + b - 2.0) * digamma(a + b)
    )


class PosteriorMask:
    """Beta-distributed posterior over edge legality for a typed graph.

    Each potential edge (i -> j) carries an independent Beta(alpha_ij,
    beta_ij) posterior on the probability that the transition is legal.
    Observations are soft: a transition (src, dst) with quality q in [0, 1]
    contributes q to alpha and (1 - q) to beta, the conjugate update for a
    fractional Bernoulli trial.

    Parameters
    ----------
    n_vertices:
        Number of vertices V; the posterior tensor has shape (V, V).
    prior_alpha:
        Initial alpha for every entry. 1.0 is the uniform cold-start prior.
    prior_beta:
        Initial beta for every entry. 1.0 is the uniform cold-start prior.
    """

    def __init__(
        self,
        n_vertices: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        if n_vertices <= 0:
            raise ValueError(f"n_vertices must be positive, got {n_vertices}")
        if prior_alpha <= 0.0 or prior_beta <= 0.0:
            raise ValueError(
                f"prior_alpha and prior_beta must be positive, got "
                f"{prior_alpha}, {prior_beta}"
            )
        self._n_vertices = int(n_vertices)
        self._alpha = np.full(
            (n_vertices, n_vertices), float(prior_alpha), dtype=np.float64
        )
        self._beta = np.full(
            (n_vertices, n_vertices), float(prior_beta), dtype=np.float64
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_legality_matrix(
        cls,
        legality_matrix: np.ndarray,
        prior_strength: float = 10.0,
    ) -> PosteriorMask:
        """Warm-start from a boolean legality matrix.

        Legal edges receive Beta(prior_strength, 1); illegal edges receive
        Beta(1, prior_strength). With prior_strength=10 this gives posterior
        means of approximately 10/11 ~= 0.909 for legal edges and 1/11 ~=
        0.091 for illegal edges.

        Parameters
        ----------
        legality_matrix:
            Boolean (V, V) array. True at [i, j] means edge i -> j is legal
            under the hand-authored spec.
        prior_strength:
            Pseudo-count placed on the dominant arm. Must be > 0.
        """
        if prior_strength <= 0.0:
            raise ValueError(
                f"prior_strength must be positive, got {prior_strength}"
            )
        legal = np.asarray(legality_matrix, dtype=bool)
        if legal.ndim != 2 or legal.shape[0] != legal.shape[1]:
            raise ValueError(
                f"legality_matrix must be square (V, V), got shape {legal.shape}"
            )
        v = legal.shape[0]
        mask = cls(v, prior_alpha=1.0, prior_beta=1.0)
        # Override priors with warm-start values.
        mask._alpha = np.where(legal, float(prior_strength), 1.0).astype(
            np.float64
        )
        mask._beta = np.where(legal, 1.0, float(prior_strength)).astype(
            np.float64
        )
        return mask

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        """Number of vertices V."""
        return self._n_vertices

    @property
    def alpha(self) -> np.ndarray:
        """Posterior alpha tensor of shape (V, V)."""
        return self._alpha

    @property
    def beta(self) -> np.ndarray:
        """Posterior beta tensor of shape (V, V)."""
        return self._beta

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update(self, src: int, dst: int, quality: float) -> None:
        """Apply a single observation update to edge (src, dst).

        Increments alpha[src, dst] by quality and beta[src, dst] by
        (1 - quality). Quality must lie in [0, 1].
        """
        q = float(quality)
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"quality must be in [0, 1], got {quality}")
        if not (0 <= src < self._n_vertices and 0 <= dst < self._n_vertices):
            raise IndexError(
                f"(src, dst) = ({src}, {dst}) out of range for V={self._n_vertices}"
            )
        self._alpha[src, dst] += q
        self._beta[src, dst] += 1.0 - q

    def update_batch(
        self,
        transitions: np.ndarray,
        qualities: np.ndarray,
    ) -> None:
        """Vectorised batch update.

        Parameters
        ----------
        transitions:
            Integer array of shape (N, 2). Each row is a (src, dst) pair.
        qualities:
            Float array of shape (N,) with values in [0, 1].
        """
        trans = np.asarray(transitions)
        q = np.asarray(qualities, dtype=np.float64)
        if trans.ndim != 2 or trans.shape[1] != 2:
            raise ValueError(
                f"transitions must have shape (N, 2), got {trans.shape}"
            )
        if q.ndim != 1 or q.shape[0] != trans.shape[0]:
            raise ValueError(
                f"qualities must have shape (N,), got {q.shape} for "
                f"N={trans.shape[0]}"
            )
        if q.size == 0:
            return
        if np.any(q < 0.0) or np.any(q > 1.0):
            raise ValueError("qualities must lie in [0, 1]")
        if (
            np.any(trans[:, 0] < 0)
            or np.any(trans[:, 0] >= self._n_vertices)
            or np.any(trans[:, 1] < 0)
            or np.any(trans[:, 1] >= self._n_vertices)
        ):
            raise IndexError("transition indices out of range")

        # np.add.at handles repeated (src, dst) pairs correctly (unbuffered).
        src = trans[:, 0].astype(np.intp)
        dst = trans[:, 1].astype(np.intp)
        np.add.at(self._alpha, (src, dst), q)
        np.add.at(self._beta, (src, dst), 1.0 - q)

    # ------------------------------------------------------------------
    # Posterior summaries
    # ------------------------------------------------------------------

    def posterior_mean(self) -> np.ndarray:
        """Posterior mean alpha / (alpha + beta), shape (V, V)."""
        return self._alpha / (self._alpha + self._beta)

    def posterior_variance(self) -> np.ndarray:
        """Posterior variance ab / ((a+b)^2 (a+b+1)), shape (V, V)."""
        s = self._alpha + self._beta
        return (self._alpha * self._beta) / (s * s * (s + 1.0))

    def posterior_entropy(self) -> np.ndarray:
        """Element-wise differential entropy of Beta(alpha, beta)."""
        return _beta_entropy(self._alpha, self._beta)

    # ------------------------------------------------------------------
    # Drop-in mask interfaces
    # ------------------------------------------------------------------

    def legality_matrix(self, threshold: float = 0.5) -> np.ndarray:
        """Boolean legality matrix from posterior mean STRICTLY ABOVE threshold.

        Drop-in for GraphFSM.legality_matrix. Strict inequality is the
        correct skeptical-prior default: an edge with no evidence (alpha = beta
        = 1, posterior_mean = 0.5) should NOT be presumed legal until net
        positive evidence accumulates. With strict >, the cold-start uniform
        prior at threshold=0.5 marks every unobserved edge illegal until
        observation pushes alpha > beta. Net-positive evidence is the
        Bayesian commitment criterion; ties don't commit.

        Returns
        -------
        np.ndarray of bool, shape (V, V). True iff posterior_mean > threshold.
        """
        return self.posterior_mean() > float(threshold)

    def legality_bias(self) -> np.ndarray:
        """Pre-softmax additive bias: log-odds of posterior mean.

        Returns log(p / (1 - p)), clipped to [-30, +30] for numerical
        stability. At the uniform cold start (p = 0.5) this is 0; as
        evidence accumulates it pushes legal edges toward +30 and illegal
        edges toward -30. Add this to logits in the smooth (soft-mask)
        mode in place of the hard -1e9 fill.
        """
        # Use alpha/beta directly so log-odds are well-defined when alpha or
        # beta are zero at the boundary (shouldn't happen with positive
        # priors, but log(p/(1-p)) = log(alpha/beta) avoids the cancellation
        # of 1 - p when p is near 1).
        with np.errstate(divide="ignore", invalid="ignore"):
            log_odds = np.log(self._alpha) - np.log(self._beta)
        return np.clip(log_odds, -_LOG_ODDS_CLIP, _LOG_ODDS_CLIP)

    # ------------------------------------------------------------------
    # Identity / tracing
    # ------------------------------------------------------------------

    def mask_version_id(self) -> str:
        """Short SHA-256 prefix of (alpha, beta) for trace records.

        Returns the first 16 hex chars of sha256(alpha.tobytes() ||
        beta.tobytes()). Two PosteriorMasks compare equal here iff their
        alpha and beta tensors are bitwise identical.
        """
        h = hashlib.sha256()
        # Ensure deterministic byte layout regardless of contiguity.
        h.update(np.ascontiguousarray(self._alpha).tobytes())
        h.update(np.ascontiguousarray(self._beta).tobytes())
        return h.hexdigest()[:16]
