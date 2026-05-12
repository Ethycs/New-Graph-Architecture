"""Sigma-weight optimisation via grid search on σ-AUROC.

Phase 18 Track 1 -- cross-grammar weight transfer
=================================================

The :class:`nga.arch.singularity_detector.SingularityDetector` combines
a small set of bounded signals (margin, decision-tie, illegal,
loop_risk, kl_surprise, ...) into the scalar score sigma. The
combination weights are currently a hand-set constant
``DEFAULT_WEIGHTS``; whether those weights are *optimal* on a given
grammar -- and whether weights tuned on one grammar transfer to another
-- is an empirical question.

This module is a sibling of ``singularity_detector.py``: it does NOT
reach into the detector's internals or modify ``DEFAULT_WEIGHTS``.
Instead it consumes the detector's public per-signal contract --
post-transform signal arrays in [0, 1] -- and runs a grid search over
weight combinations to find the dict that maximises
``binary_auroc(sigma_score, error_label)``.

Public surface (Phase 18 Track 1):

  * ``compute_sigma_from_weights(weights, signals) -> np.ndarray`` --
    pure linear blend ``sum_k w_k * signal_k``, clipped to [0, 1].
  * ``sigma_auroc_for_weights(weights, signals, error_labels) -> float``
    -- convenience: blend signals then compute AUROC.
  * ``optimize_sigma_weights(*, signals, error_labels, weight_grid)
    -> dict`` -- grid-search wrapper that returns the best weight dict.

Design choices:

  * Grid search (cartesian product of per-key weight values) keeps the
    optimisation deterministic and bounded in compute. Default grid is
    5 values per key over 5 keys -> 5^5 = 3125 evaluations.
  * Optimisation target is sigma_AUROC rather than e.g. log-loss, which
    matches the metric Phase 2 / Phase 18 actually evaluate.
  * The optimiser is *agnostic* about the source distribution: it sees
    only signals + labels. Whether the resulting weights generalise
    across grammars is the question Phase 18 Track 1 measures.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from nga.arch.failure_margin_auroc import binary_auroc

__all__ = [
    "compute_sigma_from_weights",
    "sigma_auroc_for_weights",
    "optimize_sigma_weights",
    "DEFAULT_OPTIMIZER_GRID",
]


# Default per-key grid. 5^5 = 3125 combinations per call -- bounded and
# reproducible. Callers may pass a custom grid to ``optimize_sigma_weights``
# (e.g. a coarser sweep for quick smoke tests).
DEFAULT_OPTIMIZER_GRID: dict[str, list[float]] = {
    "margin": [0.0, 0.25, 0.5, 1.0, 2.0],
    "decision_tie": [0.0, 0.25, 0.5, 1.0, 2.0],
    "illegal": [0.0, 0.25, 0.5, 1.0, 2.0],
    "loop_risk": [0.0, 0.25, 0.5, 1.0, 2.0],
    "kl_surprise": [0.0, 0.25, 0.5, 1.0, 2.0],
}


def compute_sigma_from_weights(
    weights: dict[str, float],
    signals: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the per-sample sigma score as a weighted sum of signals.

    Output is the linear blend ``sum_k weights[k] * signals[k]`` over
    the keys present in ``weights``. Missing-key signals contribute 0.
    The result is clipped to ``[0.0, 1.0]`` to match
    ``SingularityDetector.compute``'s post-blend clip.

    Parameters
    ----------
    weights:
        Mapping ``signal_name -> non-negative scalar``. Keys not present
        in ``signals`` are silently ignored.
    signals:
        Mapping ``signal_name -> array shape (N,)``. Every value array
        must be 1-D with the same length.

    Returns
    -------
    np.ndarray
        Shape (N,). dtype float64. Each element in [0.0, 1.0].
    """
    if not signals:
        raise ValueError("signals must be a non-empty mapping")

    lengths = {k: np.asarray(v).shape for k, v in signals.items()}
    first_shape = next(iter(lengths.values()))
    if any(s != first_shape for s in lengths.values()):
        raise ValueError(
            f"all signal arrays must share shape; got {lengths!r}"
        )
    if len(first_shape) != 1:
        raise ValueError(
            f"signal arrays must be 1-D; got shape {first_shape!r}"
        )

    n = first_shape[0]
    sigma = np.zeros(n, dtype=np.float64)
    for key, w in weights.items():
        if key not in signals:
            continue
        if w < 0.0:
            raise ValueError(
                f"weight for {key!r} must be >= 0, got {w}"
            )
        arr = np.asarray(signals[key], dtype=np.float64)
        sigma = sigma + float(w) * arr
    np.clip(sigma, 0.0, 1.0, out=sigma)
    return sigma


def sigma_auroc_for_weights(
    weights: dict[str, float],
    signals: dict[str, np.ndarray],
    error_labels: np.ndarray,
) -> float:
    """Return ``binary_auroc(sigma_blend, error_labels)`` for ``weights``.

    Convenience wrapper combining :func:`compute_sigma_from_weights`
    with :func:`nga.arch.failure_margin_auroc.binary_auroc`.

    Parameters
    ----------
    weights:
        Mapping ``signal_name -> non-negative scalar``.
    signals:
        Mapping ``signal_name -> array shape (N,)``.
    error_labels:
        Shape (N,) of bool / int. True / 1 means the prediction at that
        sample was incorrect.

    Returns
    -------
    float
        AUROC in [0, 1]. Returns 0.5 (with warning) for degenerate
        single-class label sets, matching ``binary_auroc`` semantics.
    """
    sigma = compute_sigma_from_weights(weights, signals)
    labels = np.asarray(error_labels)
    if labels.shape != sigma.shape:
        raise ValueError(
            f"error_labels shape {labels.shape!r} does not match "
            f"signals shape {sigma.shape!r}"
        )
    return float(binary_auroc(sigma, labels))


def optimize_sigma_weights(
    *,
    signals: dict[str, np.ndarray],
    error_labels: np.ndarray,
    weight_grid: dict[str, list[float]] | None = None,
) -> dict[str, float]:
    """Return the weight dict that maximises sigma-AUROC by grid search.

    For every combination of weights in the cartesian product of
    ``weight_grid`` (one value per key), compute the sigma blend over
    ``signals`` and evaluate AUROC against ``error_labels``. Return the
    weight dict with the highest AUROC. Ties are broken by the first
    combination encountered (cartesian-product order is deterministic).

    Parameters
    ----------
    signals:
        Mapping ``signal_name -> array shape (N,)``. Each signal must
        already be in [0, 1] (the per-signal contract used by
        ``SingularityDetector.compute``). Keys absent from
        ``weight_grid`` are still allowed in ``signals`` but are not
        searched over (their weight stays implicitly 0).
    error_labels:
        Shape (N,) of bool / int. True / 1 = error at that sample.
    weight_grid:
        Optional override of :data:`DEFAULT_OPTIMIZER_GRID`. Keys
        present here drive the search; their values are the discrete
        weight options. If ``None``, the default 5^5 grid is used over
        the canonical signal set.

    Returns
    -------
    dict[str, float]
        ``{signal_name: best_weight}`` over the grid keys.
    """
    if not signals:
        raise ValueError("signals must be a non-empty mapping")

    grid = dict(weight_grid) if weight_grid is not None else dict(
        DEFAULT_OPTIMIZER_GRID
    )
    if not grid:
        raise ValueError("weight_grid must be a non-empty mapping")

    # Restrict the search to signal keys that are actually available.
    grid_keys = [k for k in grid if k in signals]
    if not grid_keys:
        raise ValueError(
            "weight_grid keys do not intersect signal keys; "
            f"grid={list(grid)!r} signals={list(signals)!r}"
        )

    labels = np.asarray(error_labels)
    if labels.ndim != 1:
        raise ValueError("error_labels must be a 1-D array")

    # Cartesian product over per-key weight values.
    per_key_values = [grid[k] for k in grid_keys]
    best_auroc = -1.0
    best_weights: dict[str, float] | None = None

    for combo in product(*per_key_values):
        weights = {k: float(v) for k, v in zip(grid_keys, combo)}
        # All-zero weights produce a constant sigma; skip rather than
        # bias the result with the warned 0.5 fallback.
        if all(w == 0.0 for w in weights.values()):
            continue
        try:
            auroc = sigma_auroc_for_weights(weights, signals, labels)
        except Exception:
            continue
        if auroc > best_auroc:
            best_auroc = auroc
            best_weights = weights

    if best_weights is None:
        # Fallback: every combo failed. Return all-zero baseline.
        return {k: 0.0 for k in grid_keys}
    return best_weights
