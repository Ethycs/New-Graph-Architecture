"""Poincare ball model of hyperbolic d-space, in pure numpy.

The conformal factor is f(x) = 2 / (1 - ||x||^2). Distance between two points
x, y in the open unit ball is

    d(x, y) = acosh( 1 + 2 ||x-y||^2 / ((1-||x||^2) (1-||y||^2)) ).

Riemannian gradient at x for a Euclidean gradient g is:

    grad_R = (1 - ||x||^2)^2 / 4 * g.

The exponential map at the origin reduces to: tanh(||v||) * v / ||v|| for any
direction v. The retraction we use for SGD is the projected Euclidean step
followed by clipping to ||x|| <= 1 - eps. This is a first-order valid
approximation of the exponential map and matches what geoopt's PoincareBall
uses by default for "rgrad" + step.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "EPS",
    "MAX_NORM",
    "project",
    "poincare_distance",
    "exp_map_zero",
    "log_map_zero",
    "riemannian_gradient",
    "rsgd_step",
    "embed_euclidean_to_poincare",
]

EPS = 1e-7
MAX_NORM = 1.0 - 1e-5  # numerical clip for the open-ball constraint


def project(x: np.ndarray) -> np.ndarray:
    """Clip x to lie strictly inside the unit ball: ||x|| <= MAX_NORM.

    Parameters
    ----------
    x:
        Array of shape (..., d). Any leading batch dimensions are supported.

    Returns
    -------
    np.ndarray
        Array of the same shape as x with each vector's norm clamped to
        MAX_NORM.
    """
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    factor = np.where(norms > MAX_NORM, MAX_NORM / np.maximum(norms, EPS), 1.0)
    return x * factor


def poincare_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise hyperbolic distance on the Poincare ball.

    Supported shapes:

    - x and y both (d,) -> scalar float.
    - x and y both (n, d) -> (n,) element-wise distances.
    - x shape (n, d) and y shape (m, d) -> (n, m) all-pairs distances.

    The acosh argument is clipped to >= 1 + EPS and the squared-norm
    denominators are clipped to <= 1 - EPS before division for numerical
    stability.

    Parameters
    ----------
    x:
        Point(s) in the Poincare ball.
    y:
        Point(s) in the Poincare ball.

    Returns
    -------
    np.ndarray
        Hyperbolic distance(s).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # --- scalar case: both are (d,) ---
    if x.ndim == 1 and y.ndim == 1:
        diff = x - y
        diff_sq = float(np.sum(diff ** 2))
        x_sq = float(np.clip(np.sum(x ** 2), 0.0, 1.0 - EPS))
        y_sq = float(np.clip(np.sum(y ** 2), 0.0, 1.0 - EPS))
        denom = (1.0 - x_sq) * (1.0 - y_sq)
        arg = 1.0 + 2.0 * diff_sq / max(denom, EPS)
        arg = max(arg, 1.0 + EPS)
        return np.float64(np.arccosh(arg))

    # --- element-wise case: both (n, d) with identical shapes ---
    if x.ndim == 2 and y.ndim == 2 and x.shape == y.shape:
        diff = x - y
        diff_sq = np.sum(diff ** 2, axis=-1)
        x_sq = np.clip(np.sum(x ** 2, axis=-1), 0.0, 1.0 - EPS)
        y_sq = np.clip(np.sum(y ** 2, axis=-1), 0.0, 1.0 - EPS)
        denom = (1.0 - x_sq) * (1.0 - y_sq)
        arg = 1.0 + 2.0 * diff_sq / np.maximum(denom, EPS)
        arg = np.maximum(arg, 1.0 + EPS)
        return np.arccosh(arg)

    # --- cross case: x is (n, d) and y is (m, d) -> (n, m) ---
    # Also handles the (d,) vs (m, d) and (n, d) vs (d,) mixed cases via
    # ensuring both are at least 2-D.
    if x.ndim == 1:
        x = x[None, :]   # (1, d)
    if y.ndim == 1:
        y = y[None, :]   # (1, d)
    # x: (n, d), y: (m, d)
    x_exp = x[:, None, :]   # (n, 1, d)
    y_exp = y[None, :, :]   # (1, m, d)
    diff = x_exp - y_exp    # (n, m, d)
    diff_sq = np.sum(diff ** 2, axis=-1)  # (n, m)
    x_sq = np.clip(np.sum(x ** 2, axis=-1), 0.0, 1.0 - EPS)  # (n,)
    y_sq = np.clip(np.sum(y ** 2, axis=-1), 0.0, 1.0 - EPS)  # (m,)
    denom = (1.0 - x_sq[:, None]) * (1.0 - y_sq[None, :])  # (n, m)
    arg = 1.0 + 2.0 * diff_sq / np.maximum(denom, EPS)
    arg = np.maximum(arg, 1.0 + EPS)
    result = np.arccosh(arg)  # (n, m)
    # Squeeze trivial leading/trailing dimensions added for the 1-D inputs.
    if result.shape[0] == 1 and result.shape[1] == 1:
        return result[0, 0]
    if result.shape[0] == 1:
        return result[0]  # (m,)
    if result.shape[1] == 1:
        return result[:, 0]  # (n,)
    return result


def exp_map_zero(v: np.ndarray) -> np.ndarray:
    """Exponential map at the origin: v -> tanh(||v||) * v / ||v||.

    For ||v|| -> 0 the limit is v itself (identity to first order). The
    result is automatically inside the open unit ball because tanh < 1.

    Parameters
    ----------
    v:
        Tangent vector(s) at the origin. Shape (..., d).

    Returns
    -------
    np.ndarray
        Point(s) in the Poincare ball, same shape as v.
    """
    v = np.asarray(v, dtype=float)
    norms = np.linalg.norm(v, axis=-1, keepdims=True)  # (..., 1)
    # Avoid division by zero: use v directly where norm is tiny.
    safe_norms = np.maximum(norms, EPS)
    return np.tanh(norms) * v / safe_norms


def log_map_zero(x: np.ndarray) -> np.ndarray:
    """Inverse of exp_map_zero: x -> arctanh(||x||) * x / ||x||.

    Parameters
    ----------
    x:
        Point(s) in the Poincare ball. Shape (..., d).

    Returns
    -------
    np.ndarray
        Tangent vector(s) at the origin, same shape as x.
    """
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)  # (..., 1)
    # Clip norm to stay well inside (0, 1) for arctanh stability.
    clipped = np.clip(norms, 0.0, MAX_NORM)
    safe_norms = np.maximum(norms, EPS)
    return np.arctanh(clipped) * x / safe_norms


def riemannian_gradient(x: np.ndarray, euclidean_grad: np.ndarray) -> np.ndarray:
    """Convert a Euclidean gradient to a Riemannian gradient on the Poincare ball.

    The conversion factor is the squared inverse of the conformal factor:

        grad_R = (1 - ||x||^2)^2 / 4 * grad_E.

    Parameters
    ----------
    x:
        Current point in the Poincare ball. Shape (..., d).
    euclidean_grad:
        Euclidean gradient at x. Shape (..., d).

    Returns
    -------
    np.ndarray
        Riemannian gradient, same shape as euclidean_grad.
    """
    x = np.asarray(x, dtype=float)
    euclidean_grad = np.asarray(euclidean_grad, dtype=float)
    norm_sq = np.sum(x ** 2, axis=-1, keepdims=True)  # (..., 1)
    factor = ((1.0 - norm_sq) ** 2) / 4.0
    return factor * euclidean_grad


def rsgd_step(
    x: np.ndarray, euclidean_grad: np.ndarray, lr: float
) -> np.ndarray:
    """One step of Riemannian SGD with retraction.

    Computes:

        new_x = project( x - lr * riemannian_gradient(x, grad) )

    Returns a fresh array; does not mutate x.

    Parameters
    ----------
    x:
        Current point in the Poincare ball. Shape (..., d).
    euclidean_grad:
        Euclidean gradient at x. Shape (..., d).
    lr:
        Learning rate (positive scalar).

    Returns
    -------
    np.ndarray
        Updated point(s) inside the unit ball, same shape as x.
    """
    rgrad = riemannian_gradient(x, euclidean_grad)
    updated = x - lr * rgrad
    return project(updated)


def embed_euclidean_to_poincare(
    points: np.ndarray, *, scale: float = 0.5
) -> np.ndarray:
    """Project Euclidean points into the Poincare ball via tanh squashing.

    Formula: y = tanh(scale * ||x||) * x / max(||x||, EPS), then project()
    to enforce the strict-interior constraint.

    Smaller scale compresses points closer to the origin; scale=0.5 typically
    uses about half the radial range of the ball.

    Parameters
    ----------
    points:
        Euclidean feature vectors. Shape (n, d_e).
    scale:
        Controls how much of the unit ball gets used. Smaller scale yields
        more compressed, more linear-like embeddings.

    Returns
    -------
    np.ndarray
        Poincare-ball points of shape (n, d_e), each strictly inside the
        unit ball.
    """
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=-1, keepdims=True)  # (n, 1)
    safe_norms = np.maximum(norms, EPS)
    embedded = np.tanh(scale * norms) * points / safe_norms
    return project(embedded)
