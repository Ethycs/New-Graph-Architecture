"""AUROC of singularity scores against ground-truth errors.

Two callables: one computes the AUROC of margin (-margin, since lower margin
should predict failure), the other of sigma(x). The Phase 2 acceptance test
asserts sigma_auroc >= margin_auroc + 0.03 on E0/E1 outputs.
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["margin_auroc", "sigma_auroc", "binary_auroc"]


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC where higher score = more likely positive class.

    Parameters
    ----------
    scores:
        Shape (N,) of float. Higher score means the model predicts the
        positive class more strongly.
    labels:
        Shape (N,) of bool or int (0/1). True / 1 = positive (error).

    Returns
    -------
    float
        AUROC in [0.0, 1.0]. Returns 0.5 with a UserWarning for degenerate
        inputs where all labels are the same class.

    Notes
    -----
    Implementation uses the rank-sum (Mann-Whitney U) formula:

        AUC = (sum_of_ranks_of_positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

    Tied scores are handled by averaging ranks (mid-rank convention), which
    matches scipy.stats.roc_auc_score and sklearn behaviour. scipy.stats.rankdata
    is used when scipy is available; otherwise a manual numpy-based fallback
    computes the same result.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)

    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("scores and labels must be 1-D arrays")
    if scores.shape != labels.shape:
        raise ValueError(
            f"scores and labels must have the same length:"
            f" {scores.shape} vs {labels.shape}"
        )

    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())

    if n_pos == 0 or n_neg == 0:
        warnings.warn(
            "binary_auroc: degenerate input - all labels are the same class."
            " Returning 0.5.",
            UserWarning,
            stacklevel=2,
        )
        return 0.5

    # Attempt to use scipy for rank computation (average ties)
    try:
        from scipy.stats import rankdata  # type: ignore[import]

        ranks = rankdata(scores, method="average")
    except ImportError:
        ranks = _rankdata_manual(scores)

    # Mann-Whitney U / Wilcoxon rank-sum formula
    rank_sum_pos = float(ranks[labels].sum())
    auroc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    # Clip defensively to [0, 1] for floating-point edge cases
    return float(np.clip(auroc, 0.0, 1.0))


def _rankdata_manual(scores: np.ndarray) -> np.ndarray:
    """Compute average ranks for scores without scipy.

    Parameters
    ----------
    scores:
        1-D array of floats.

    Returns
    -------
    np.ndarray
        1-D array of average ranks (1-based) with ties averaged.
    """
    n = len(scores)
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    # Resolve ties: for each group of equal values, set rank = mean of group
    # Work on the sorted array and write back
    sorted_scores = scores[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # Elements order[i:j] all have the same score
        if j > i + 1:
            mean_rank = (i + 1 + j) / 2.0  # average of 1-based positions i+1..j
            ranks[order[i:j]] = mean_rank
        i = j

    return ranks


def margin_auroc(margins: np.ndarray, errors: np.ndarray) -> float:
    """AUROC using -margin as the prediction score.

    Lower margin predicts failure, so the score passed to binary_auroc
    is negated: a smaller (more negative) margin becomes a larger
    (more positive) score, correctly ranking failures above successes.

    Parameters
    ----------
    margins:
        Shape (N,) of float margin values from Phase 1.
    errors:
        Shape (N,) of bool/int. errors[i] = True iff y_hat[i] != y_true[i].

    Returns
    -------
    float
        AUROC in [0.0, 1.0].
    """
    margins = np.asarray(margins, dtype=float)
    errors = np.asarray(errors, dtype=bool)
    return binary_auroc(-margins, errors)


def sigma_auroc(sigmas: np.ndarray, errors: np.ndarray) -> float:
    """AUROC using sigma(x) as the prediction score.

    Higher sigma predicts failure, so scores are passed directly to
    binary_auroc without negation.

    Parameters
    ----------
    sigmas:
        Shape (N,) of float sigma values in [0, 1] from the
        SingularityDetector.
    errors:
        Shape (N,) of bool/int. errors[i] = True iff y_hat[i] != y_true[i].

    Returns
    -------
    float
        AUROC in [0.0, 1.0].
    """
    sigmas = np.asarray(sigmas, dtype=float)
    errors = np.asarray(errors, dtype=bool)
    return binary_auroc(sigmas, errors)
