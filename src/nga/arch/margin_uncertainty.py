"""Top-1 minus top-2 margin: the primary singularity signal in Phase 1.

The output is normalized to roughly the [-1, 1] range so it fits the margin
field bounds in drivers/results_jsonl.py.
"""
from __future__ import annotations

import numpy as np

__all__ = ["compute_margin", "low_margin_mask"]


def compute_margin(distribution: np.ndarray) -> float:
    """Top-1 minus top-2 of a single softmax distribution.

    Argument:
        distribution: shape (V,), non-negative, sums to ~1.0.

    Returns:
        A scalar in [-1, 1]. For a well-calibrated distribution, the value lies
        in [0, 1] (top-1 is always >= top-2). The negative range is reserved for
        callers that pass an unsorted score difference.

    Raises:
        ValueError: If distribution is empty or has fewer than 2 elements.
    """
    arr = np.asarray(distribution, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"distribution must be 1-D, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("distribution must be non-empty")
    if arr.size < 2:
        raise ValueError(
            f"distribution must have at least 2 elements to compute top-2 margin, "
            f"got {arr.size}"
        )
    # np.partition returns elements such that the element at index -2 is the
    # second-largest value and the element at index -1 is the largest value.
    top2 = np.partition(arr, -2)[-2:]
    top1_val = float(top2[-1])
    top2_val = float(top2[-2])
    return top1_val - top2_val


def low_margin_mask(margins: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask: True where margin < threshold.

    Used by the E0 runner to compute low_margin_accuracy and to set the
    per-sample singular_flag.

    Args:
        margins: Shape (B,) array of margin values (one per sample).
        threshold: Scalar threshold; samples with margin < threshold are flagged.

    Returns:
        Shape (B,) boolean array. True indicates a low-margin (potentially
        singular) sample.
    """
    margins_arr = np.asarray(margins, dtype=float)
    return margins_arr < threshold
