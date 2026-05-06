"""Distance-based loss for hyperbolic embeddings.

For Phase 3 a simple "pull true class, push others" loss is enough:

    L(x, y_true, prototypes) = d(x, p_y_true)^2
                              + lambda * mean_{j != y_true} max(0, m - d(x, p_j))^2

where m is a margin. This is a hyperbolic analog of the contrastive triplet
loss; the squared distance keeps gradients bounded near small distances.
"""
from __future__ import annotations

import numpy as np

from nga.arch.hyperbolic_embedding import poincare_distance

__all__ = ["distance_loss", "predict_by_nearest_prototype"]


def distance_loss(
    *,
    point: np.ndarray,        # shape (d,) or (B, d)
    prototypes: np.ndarray,   # shape (V, d)
    label_idx: int | np.ndarray,
    margin: float = 1.0,
    weight_negative: float = 0.5,
) -> float:
    """Single-sample or batch hinge-like distance loss.

    For a single point (shape (d,)) and scalar label_idx:

        L = d(point, p_true)^2
            + weight_negative * mean_{j != true} max(0, margin - d(point, p_j))^2

    For a batch (shape (B, d)) and label_idx of shape (B,), returns the
    mean loss over the batch.

    Parameters
    ----------
    point:
        Query embedding(s). Shape (d,) or (B, d). Must be inside the Poincare
        ball.
    prototypes:
        Prototype matrix of shape (V, d). Each row inside the Poincare ball.
    label_idx:
        Index (or array of indices) into the first axis of prototypes giving
        the true class for each point.
    margin:
        Margin m for the negative-pair hinge term. Negative pairs with
        distance >= m contribute zero loss.
    weight_negative:
        Multiplier lambda for the negative (push) term.

    Returns
    -------
    float
        Scalar loss value.
    """
    point = np.asarray(point, dtype=float)
    prototypes = np.asarray(prototypes, dtype=float)
    n_proto = prototypes.shape[0]

    # --- single-sample path ---
    if point.ndim == 1:
        label_idx = int(label_idx)  # type: ignore[arg-type]
        # Distance to the true prototype.
        d_pos = float(poincare_distance(point, prototypes[label_idx]))
        loss_pos = d_pos ** 2

        # Hinge push for all wrong prototypes.
        loss_neg = 0.0
        neg_count = 0
        for j in range(n_proto):
            if j == label_idx:
                continue
            d_neg = float(poincare_distance(point, prototypes[j]))
            hinge = max(0.0, margin - d_neg)
            loss_neg += hinge ** 2
            neg_count += 1

        if neg_count > 0:
            loss_neg /= neg_count
        return float(loss_pos + weight_negative * loss_neg)

    # --- batch path ---
    label_idx_arr = np.asarray(label_idx, dtype=int)
    batch_size = point.shape[0]

    # All pairwise distances: shape (B, V).
    all_dists = poincare_distance(point, prototypes)

    # Positive distances: select the true-class distance per sample.
    d_pos_vec = all_dists[np.arange(batch_size), label_idx_arr]  # (B,)
    loss_pos_batch = d_pos_vec ** 2  # (B,)

    # Negative hinge: for each sample, zero out the true-class column then
    # compute hinge over remaining V-1 prototypes.
    hinge_mat = np.maximum(0.0, margin - all_dists) ** 2  # (B, V)
    # Zero out the true-class entry for each row.
    mask = np.ones((batch_size, n_proto), dtype=float)
    mask[np.arange(batch_size), label_idx_arr] = 0.0
    hinge_mat = hinge_mat * mask

    # Mean over the V-1 negative prototypes per sample.
    neg_sum = hinge_mat.sum(axis=1)  # (B,)
    n_neg = max(n_proto - 1, 1)
    loss_neg_batch = neg_sum / n_neg  # (B,)

    total_per_sample = loss_pos_batch + weight_negative * loss_neg_batch
    return float(total_per_sample.mean())


def predict_by_nearest_prototype(
    points: np.ndarray,        # shape (N, d)
    prototypes: np.ndarray,    # shape (V, d)
) -> tuple[np.ndarray, np.ndarray]:
    """Return the index of the nearest prototype and its distance for each point.

    Parameters
    ----------
    points:
        Query embeddings of shape (N, d). Each row must be inside the Poincare
        ball.
    prototypes:
        Prototype matrix of shape (V, d). Each row inside the Poincare ball.

    Returns
    -------
    argmin_idx : np.ndarray
        Integer array of shape (N,) containing the index into prototypes of
        the nearest prototype for each point.
    distance_to_argmin : np.ndarray
        Float array of shape (N,) with the hyperbolic distance to that nearest
        prototype.
    """
    points = np.asarray(points, dtype=float)
    prototypes = np.asarray(prototypes, dtype=float)

    # All-pairs distances: shape (N, V).
    all_dists = poincare_distance(points, prototypes)

    argmin_idx = np.argmin(all_dists, axis=1).astype(int)           # (N,)
    distance_to_argmin = all_dists[np.arange(len(points)), argmin_idx]  # (N,)

    return argmin_idx, distance_to_argmin
