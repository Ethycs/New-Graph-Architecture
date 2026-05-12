"""Information-geometry primitives: Fisher, KL, and Cramer-Rao on the
statistical manifold of edge legality and policy distributions.

This atom names a correspondence the rest of the architecture has been
exploiting in pieces. Specifically, on the unit interval (the edge-Bernoulli
manifold) the Fisher information

    I(p) = 1 / (p * (1 - p))

induces the Fisher-Rao metric ds^2 = dp^2 / (p(1-p)). Up to a factor of 4 this
is exactly the metric pulled back from the great-circle parametrisation
phi = 2 arcsin(sqrt(p)), so the geodesic distance between Bernoulli(p) and
Bernoulli(q) on the simplex is

    d_FR(p, q) = 2 |arcsin(sqrt(p)) - arcsin(sqrt(q))|.

This is hyperbolic geometry: the same Riemannian structure that
``nga.arch.hyperbolic_embedding`` uses for prototypes via the Poincare ball.
Phase 7 makes the identification explicit so that

  * ``posterior_mask`` (Beta(alpha, beta) per edge) is recognised as a
    posterior on a Fisher-Riemannian manifold whose precision is
    alpha + beta,
  * ``riemannian_gradient`` on the Poincare ball is a special case of
    Fisher-natural-gradient on Bernoulli edges,
  * KL divergence is the canonical loss for matching empirical to model,
    and
  * the Cramer-Rao bound gives a sample-complexity floor on any unbiased
    estimator of edge probabilities.

All routines are pure numpy + scipy.special; numerical stability is achieved
through log-space arithmetic and clipping where appropriate.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import digamma, gammaln

__all__ = [
    "fisher_information_bernoulli",
    "fisher_information_beta_natural",
    "fisher_information_diagonal_mask",
    "kl_categorical",
    "kl_beta",
    "kl_bernoulli",
    "cramer_rao_bound",
    "crb_n_for_target_variance",
    "crb_confidence_for_observations",
    "natural_gradient_diagonal",
    "fisher_rao_distance_bernoulli",
]


# ---------------------------------------------------------------------------
# Fisher information primitives
# ---------------------------------------------------------------------------


def fisher_information_bernoulli(
    p: float | np.ndarray,
) -> float | np.ndarray:
    """Fisher information for Bernoulli(p) at parameter p.

    The score function of Bernoulli(p) is (x - p) / (p(1 - p)); its variance
    (which equals the Fisher information for the regular exponential family)
    is

        I(p) = 1 / (p * (1 - p)).

    The information diverges at the boundaries p = 0 and p = 1 -- this is
    the characteristic Fisher-Rao curvature of the Bernoulli simplex and
    why the induced geometry is hyperbolic.

    Parameters
    ----------
    p:
        Bernoulli parameter(s) in (0, 1). Boundary values are clipped to
        [eps, 1 - eps] before division.

    Returns
    -------
    float or np.ndarray
        I(p), same shape as p.
    """
    eps = 1e-12
    p_arr = np.asarray(p, dtype=float)
    p_clip = np.clip(p_arr, eps, 1.0 - eps)
    out = 1.0 / (p_clip * (1.0 - p_clip))
    if p_arr.ndim == 0:
        return float(out)
    return out


def fisher_information_beta_natural(
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Fisher information for the Bernoulli mean under a Beta(alpha, beta)
    posterior, plus the posterior precision on p.

    For a Beta(alpha, beta) posterior on the Bernoulli mean p, the
    *posterior precision* on p is

        precision(p) = (alpha + beta + 1) / (p_bar * (1 - p_bar))
                     = (alpha + beta) * (alpha + beta + 1) / (alpha * beta)

    where p_bar = alpha / (alpha + beta) is the posterior mean. For the
    diagonal-Fisher / natural-gradient use-case we only need a per-edge
    *concentration* term that scales with the amount of evidence: this is
    the well-known Beta concentration

        kappa = alpha + beta,

    which acts as the precision of the Beta on its mean. We return both:

      * ``fisher_p`` -- the Bernoulli Fisher info I(p_bar) evaluated at the
        posterior mean,
      * ``concentration`` -- alpha + beta, the natural diagonal element used
        in ``fisher_information_diagonal_mask``.

    Parameters
    ----------
    alpha, beta:
        Beta parameters; must be positive (not validated here -- callers
        guarantee this from PosteriorMask).

    Returns
    -------
    (fisher_p, concentration)
        Both same shape as broadcast(alpha, beta).
    """
    a = np.asarray(alpha, dtype=float)
    b = np.asarray(beta, dtype=float)
    p_bar = a / (a + b)
    fisher_p = fisher_information_bernoulli(p_bar)
    concentration = a + b
    if a.ndim == 0 and b.ndim == 0:
        return float(fisher_p), float(concentration)
    return fisher_p, concentration


def fisher_information_diagonal_mask(
    alpha: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """Per-edge diagonal Fisher information for a PosteriorMask.

    For natural-gradient SGD we need a positive scalar per edge that
    represents how much the edge "knows" about its Bernoulli parameter.
    The Beta(alpha, beta) posterior precision on its mean is alpha + beta
    (up to a 1 / variance factor that cancels for the natural-gradient
    rescaling). We therefore use

        F_diag = alpha + beta

    so that the natural-gradient rescaling 1 / F_diag has the right
    intuition: edges with lots of evidence are barely moved per step;
    cold edges are moved a lot.

    Parameters
    ----------
    alpha, beta:
        Same-shape arrays of Beta parameters.

    Returns
    -------
    np.ndarray
        F_diag with the broadcast shape of alpha and beta.
    """
    a = np.asarray(alpha, dtype=float)
    b = np.asarray(beta, dtype=float)
    return a + b


# ---------------------------------------------------------------------------
# KL divergence primitives
# ---------------------------------------------------------------------------


def kl_categorical(
    p: np.ndarray,
    q: np.ndarray,
    axis: int = -1,
    eps: float = 1e-12,
) -> np.ndarray:
    """KL(p || q) for categorical distributions, summed along axis.

    For each pair of distributions along ``axis``,

        KL(p || q) = sum_k p_k * (log p_k - log q_k).

    Numerical stability:

      * p and q are clipped to [eps, 1] before taking logs so log(0) is
        replaced by log(eps) (a finite penalty proportional to log(1/eps)),
      * if p_k = 0 the contribution is 0 (handled implicitly by p_k * ...),
      * we do not normalise p or q -- the caller is responsible.

    Parameters
    ----------
    p, q:
        Same-shape probability arrays.
    axis:
        Axis to sum over.
    eps:
        Floor for clipping.

    Returns
    -------
    np.ndarray
        KL divergences with the reduction axis removed (or scalar for
        1-D inputs along axis -1).
    """
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    p_clip = np.clip(p_arr, eps, 1.0)
    q_clip = np.clip(q_arr, eps, 1.0)
    # p * log(p / q): when p == 0 (after clip we have p == eps but original
    # was 0) the contribution is multiplied by the original 0, so use the
    # unclipped p for the multiplicative weight.
    weight = p_arr
    log_ratio = np.log(p_clip) - np.log(q_clip)
    integrand = weight * log_ratio
    return np.sum(integrand, axis=axis)


def kl_beta(
    alpha_p: float,
    beta_p: float,
    alpha_q: float,
    beta_q: float,
) -> float:
    """Closed-form KL(Beta(alpha_p, beta_p) || Beta(alpha_q, beta_q)).

    Using the standard formula in terms of digamma psi and log-gamma
    lgamma (B(a, b) is the Beta function),

        KL(p || q) = lgamma(alpha_q + beta_q) - lgamma(alpha_p + beta_p)
                   - (lgamma(alpha_q) - lgamma(alpha_p))
                   - (lgamma(beta_q)  - lgamma(beta_p))
                   + (alpha_p - alpha_q) * (psi(alpha_p) - psi(alpha_p + beta_p))
                   + (beta_p  - beta_q)  * (psi(beta_p)  - psi(alpha_p + beta_p)).

    The uniform-on-uniform edge case alpha = beta = 1 is handled exactly:
    psi(1) is finite and lgamma(1) = 0, so the expression evaluates to 0
    when (alpha_p, beta_p) == (alpha_q, beta_q).

    Parameters
    ----------
    alpha_p, beta_p:
        Parameters of the "from" distribution.
    alpha_q, beta_q:
        Parameters of the "to" distribution.

    Returns
    -------
    float
        The KL divergence in nats. Non-negative up to floating-point
        rounding (a tiny negative may appear for nearly-identical params;
        we clip to >= 0).
    """
    ap = float(alpha_p)
    bp = float(beta_p)
    aq = float(alpha_q)
    bq = float(beta_q)
    # log-Beta-function differences in lgamma form.
    log_b_p = gammaln(ap) + gammaln(bp) - gammaln(ap + bp)
    log_b_q = gammaln(aq) + gammaln(bq) - gammaln(aq + bq)
    # KL = log B(q) - log B(p) + (a_p - a_q)(psi(a_p) - psi(a_p+b_p))
    #                          + (b_p - b_q)(psi(b_p) - psi(a_p+b_p))
    psi_apbp = digamma(ap + bp)
    kl = (
        (log_b_q - log_b_p)
        + (ap - aq) * (digamma(ap) - psi_apbp)
        + (bp - bq) * (digamma(bp) - psi_apbp)
    )
    # Floating-point can produce tiny negatives for params that are nearly
    # identical; KL is provably non-negative, so clip.
    return float(max(kl, 0.0))


def kl_bernoulli(
    p: float | np.ndarray,
    q: float | np.ndarray,
    eps: float = 1e-12,
) -> float | np.ndarray:
    """KL(Bernoulli(p) || Bernoulli(q)) elementwise.

    Formula:

        KL(p || q) = p log(p / q) + (1 - p) log((1 - p) / (1 - q)).

    Inputs are clipped to [eps, 1 - eps] for the logarithms. Where p is
    exactly 0 or 1 the corresponding term is 0 (multiplicative).

    Parameters
    ----------
    p, q:
        Bernoulli parameters; broadcastable.
    eps:
        Numerical floor.

    Returns
    -------
    float or np.ndarray
        KL divergence(s).
    """
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    p_log = np.clip(p_arr, eps, 1.0 - eps)
    q_log = np.clip(q_arr, eps, 1.0 - eps)
    term_a = p_arr * (np.log(p_log) - np.log(q_log))
    term_b = (1.0 - p_arr) * (np.log1p(-p_log) - np.log1p(-q_log))
    out = term_a + term_b
    # Clip tiny negatives from rounding.
    out = np.maximum(out, 0.0)
    if p_arr.ndim == 0 and q_arr.ndim == 0:
        return float(out)
    return out


# ---------------------------------------------------------------------------
# Cramer-Rao bound primitives
# ---------------------------------------------------------------------------


def cramer_rao_bound(
    fisher_info: float | np.ndarray,
    n_observations: int,
) -> float | np.ndarray:
    """Cramer-Rao lower bound: Var(theta_hat) >= 1 / (n * I(theta)).

    For an unbiased estimator of theta from n iid observations with Fisher
    information I per observation, the variance of any such estimator is
    at least 1 / (n * I).

    Parameters
    ----------
    fisher_info:
        I(theta) per observation. Must be positive.
    n_observations:
        Number of independent observations. Must be positive.

    Returns
    -------
    float or np.ndarray
        The CRB on the variance of any unbiased estimator.
    """
    if n_observations <= 0:
        raise ValueError(
            f"n_observations must be positive, got {n_observations}"
        )
    fi = np.asarray(fisher_info, dtype=float)
    if np.any(fi <= 0):
        raise ValueError("fisher_info must be positive")
    out = 1.0 / (n_observations * fi)
    if fi.ndim == 0:
        return float(out)
    return out


def crb_n_for_target_variance(
    fisher_info: float | np.ndarray,
    target_var: float,
) -> float | np.ndarray:
    """Required n for the CRB to drop below ``target_var``.

    Solving 1 / (n * I) <= target_var gives n >= 1 / (target_var * I).

    Parameters
    ----------
    fisher_info:
        I(theta) per observation.
    target_var:
        Desired variance ceiling.

    Returns
    -------
    float or np.ndarray
        Required (real-valued) sample size; the caller can ``np.ceil`` to
        get an integer count.
    """
    if target_var <= 0.0:
        raise ValueError(f"target_var must be positive, got {target_var}")
    fi = np.asarray(fisher_info, dtype=float)
    if np.any(fi <= 0):
        raise ValueError("fisher_info must be positive")
    out = 1.0 / (target_var * fi)
    if fi.ndim == 0:
        return float(out)
    return out


def crb_confidence_for_observations(
    fisher_info: float | np.ndarray,
    n_observations: int,
    target_var: float,
) -> float | np.ndarray:
    """Fraction of CRB-required samples we have.

    Returns n / n_required, where n_required = 1 / (target_var * I). Values
    >= 1.0 mean the CRB target is satisfied; smaller values indicate how
    far short of the bound we are.

    Parameters
    ----------
    fisher_info:
        I(theta) per observation.
    n_observations:
        Number of observations to date.
    target_var:
        Variance ceiling we want to clear.

    Returns
    -------
    float or np.ndarray
        n / n_required.
    """
    n_req = crb_n_for_target_variance(fisher_info, target_var)
    fi = np.asarray(fisher_info, dtype=float)
    out = float(n_observations) / np.asarray(n_req, dtype=float)
    if fi.ndim == 0:
        return float(out)
    return out


# ---------------------------------------------------------------------------
# Natural-gradient helpers
# ---------------------------------------------------------------------------


def natural_gradient_diagonal(
    gradient: np.ndarray,
    fisher_diag: np.ndarray,
    damping: float = 1e-6,
) -> np.ndarray:
    """Natural gradient under a diagonal Fisher metric.

    For a diagonal Fisher matrix F = diag(f_i), the natural gradient is

        nat_grad_i = grad_i / (f_i + damping).

    Damping is essential when some f_i can be zero (e.g. an edge that has
    received no observations under the alpha + beta convention with priors
    set to zero -- not the default in PosteriorMask, but defensible). We
    return a fresh array.

    Parameters
    ----------
    gradient:
        Euclidean gradient.
    fisher_diag:
        Diagonal Fisher entries, same shape as gradient.
    damping:
        Tikhonov term added before division. Must be non-negative.

    Returns
    -------
    np.ndarray
        Natural gradient, same shape as input.
    """
    if damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {damping}")
    g = np.asarray(gradient, dtype=float)
    f = np.asarray(fisher_diag, dtype=float)
    return g / (f + damping)


# ---------------------------------------------------------------------------
# Manifold-distance utilities
# ---------------------------------------------------------------------------


def fisher_rao_distance_bernoulli(p: float, q: float) -> float:
    """Fisher-Rao geodesic distance between Bernoulli(p) and Bernoulli(q).

    Under the Fisher metric on the Bernoulli simplex, the change of
    variables phi = 2 * arcsin(sqrt(p)) maps the manifold isometrically to
    a great-circle arc of length pi. The geodesic distance is therefore

        d_FR(p, q) = 2 * |arcsin(sqrt(p)) - arcsin(sqrt(q))|.

    This is the same hyperbolic geometry that ``hyperbolic_embedding`` uses
    on the Poincare ball for prototype vectors -- Phase 7 names this
    correspondence so that legality posteriors and prototype embeddings
    can be analysed under the same Riemannian structure.

    Parameters
    ----------
    p, q:
        Bernoulli parameters in [0, 1]. (Boundary values are admissible
        under arcsin(sqrt(.)) -- arcsin(0) = 0, arcsin(1) = pi/2.)

    Returns
    -------
    float
        Geodesic distance (in radians on the great circle), in [0, pi].
    """
    pf = float(np.clip(p, 0.0, 1.0))
    qf = float(np.clip(q, 0.0, 1.0))
    phi_p = np.arcsin(np.sqrt(pf))
    phi_q = np.arcsin(np.sqrt(qf))
    return float(2.0 * np.abs(phi_p - phi_q))
