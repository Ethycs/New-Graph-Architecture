"""Per-vertex prototype anchors p_v in hyperbolic space.

Phase 3 construction: prototypes are learned via Riemannian SGD to (a) sit
near the centroid of training points labelled with their vertex, and (b)
respect the hyperbolic distance structure implied by FSM adjacency.

The prototype set is a (V, d) matrix indexed by vertex_ids order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nga.arch.graph_fsm import GraphFSM
from nga.arch.hyperbolic_embedding import (
    embed_euclidean_to_poincare,
    poincare_distance,
    project,
    rsgd_step,
)

__all__ = ["PrototypeBundle", "init_prototypes_from_class_means", "fit_prototypes"]

_EPS = 1e-7
_PERTURB_SCALE = 1e-3  # scale for near-origin perturbation of empty-class prototypes


@dataclass
class PrototypeBundle:
    """Container for a set of per-vertex hyperbolic prototype vectors.

    Attributes
    ----------
    vertex_ids:
        Ordered list of vertex id strings; row i of prototypes corresponds
        to vertex_ids[i].
    prototypes:
        Float array of shape (V, d) with each row strictly inside the
        Poincare unit ball.
    dimension:
        Embedding dimension d.
    """

    vertex_ids: list[str]
    prototypes: np.ndarray  # shape (V, d), inside the unit ball
    dimension: int


def init_prototypes_from_class_means(
    *,
    fsm: GraphFSM,
    features: np.ndarray,        # shape (N, d_e); MUST equal target dim d
    labels: list[str],           # length N; values from fsm.vertex_ids
    target_dim: int,
) -> PrototypeBundle:
    """Initialize per-vertex prototypes as the (Poincare-projected) Euclidean
    mean of points labelled with that vertex.

    If a vertex has zero training samples, its prototype is placed near the
    origin with a small deterministic perturbation seeded by the vertex index.

    Parameters
    ----------
    fsm:
        The runtime graph FSM that defines the vertex ordering.
    features:
        Euclidean (or already-Poincare-projected) feature vectors of shape
        (N, d_e). d_e MUST equal target_dim or a ValueError is raised.
    labels:
        List of vertex id strings of length N, one per feature row.
    target_dim:
        Expected embedding dimension d. Caller is responsible for projecting
        features to this dimension before passing.

    Returns
    -------
    PrototypeBundle
        Initialized bundle with prototypes inside the Poincare ball.

    Raises
    ------
    ValueError
        If features.shape[1] != target_dim.
    """
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[1] != target_dim:
        raise ValueError(
            f"features.shape[1] must equal target_dim={target_dim}, "
            f"got features.shape={features.shape}"
        )

    vertex_ids = fsm.vertex_ids
    n_vertices = len(vertex_ids)
    vertex_index = fsm.vertex_index

    # Accumulate sums and counts per vertex.
    sums = np.zeros((n_vertices, target_dim), dtype=float)
    counts = np.zeros(n_vertices, dtype=int)
    for feat, label in zip(features, labels):
        idx = vertex_index[label]
        sums[idx] += feat
        counts[idx] += 1

    prototypes = np.zeros((n_vertices, target_dim), dtype=float)
    for v_idx in range(n_vertices):
        if counts[v_idx] > 0:
            mean_vec = sums[v_idx] / counts[v_idx]
            # Project the Euclidean mean into the Poincare ball.
            prototypes[v_idx] = embed_euclidean_to_poincare(
                mean_vec[None, :], scale=0.5
            )[0]
        else:
            # Place near origin with a deterministic tiny perturbation.
            rng = np.random.default_rng(v_idx)
            direction = rng.normal(size=target_dim)
            direction /= np.maximum(np.linalg.norm(direction), _EPS)
            prototypes[v_idx] = project(
                (direction * _PERTURB_SCALE)[None, :]
            )[0]

    return PrototypeBundle(
        vertex_ids=vertex_ids,
        prototypes=prototypes,
        dimension=target_dim,
    )


def fit_prototypes(
    bundle: PrototypeBundle,
    *,
    features: np.ndarray,
    labels: list[str],
    n_steps: int = 50,
    lr: float = 0.1,
    seed: int = 0,
) -> PrototypeBundle:
    """Tighten prototypes via Riemannian SGD.

    For each step, iterates over all (feature, label) pairs and computes:

        loss = poincare_distance(x, prototype[label])^2
               - poincare_distance(x, prototype[neg_label])^2

    where neg_label is one randomly sampled wrong vertex. The Euclidean
    gradient of the loss with respect to each prototype is approximated
    by finite differences and passed to rsgd_step.

    Features are fixed (already inside the Poincare ball - the caller
    supplies them). Only the prototype matrix is updated.

    Parameters
    ----------
    bundle:
        Starting prototype bundle. Not mutated.
    features:
        Poincare-ball points of shape (N, d). Fixed during optimization.
    labels:
        Vertex id string per feature row, length N.
    n_steps:
        Number of full passes over the dataset.
    lr:
        Learning rate for rsgd_step.
    seed:
        Random seed for negative sample selection.

    Returns
    -------
    PrototypeBundle
        A new bundle with updated prototypes; the input bundle is unchanged.
    """
    features = np.asarray(features, dtype=float)
    vertex_ids = bundle.vertex_ids
    n_vertices = len(vertex_ids)
    vertex_index: dict[str, int] = {vid: i for i, vid in enumerate(vertex_ids)}

    # Work on a copy of the prototype matrix so we do not mutate the input.
    protos = bundle.prototypes.copy()

    rng = np.random.default_rng(seed)
    n_samples = len(labels)
    label_indices = np.array([vertex_index[lab] for lab in labels], dtype=int)

    for _step in range(n_steps):
        # Shuffle order each step for better convergence.
        perm = rng.permutation(n_samples)

        for sample_idx in perm:
            x = features[sample_idx]       # shape (d,)
            pos_idx = label_indices[sample_idx]

            # Sample one negative vertex (different from positive).
            neg_candidates = [i for i in range(n_vertices) if i != pos_idx]
            if not neg_candidates:
                continue
            neg_idx = int(rng.choice(neg_candidates))

            # --- Positive gradient ---
            # d_pos = poincare_distance(x, protos[pos_idx])
            # Euclidean gradient of d_pos^2 w.r.t. protos[pos_idx], approximated
            # by central differences along each dimension.
            h = 1e-5
            d = protos.shape[1]
            grad_pos = np.zeros(d, dtype=float)
            for dim in range(d):
                p_plus = protos[pos_idx].copy(); p_plus[dim] += h
                p_minus = protos[pos_idx].copy(); p_minus[dim] -= h
                d_plus = float(poincare_distance(x, p_plus))
                d_minus = float(poincare_distance(x, p_minus))
                grad_pos[dim] = (d_plus**2 - d_minus**2) / (2.0 * h)

            # --- Negative gradient ---
            # We want to push the negative prototype away: subtract gradient of
            # poincare_distance^2 w.r.t. the negative prototype (negative sign
            # means we increase the distance).
            grad_neg = np.zeros(d, dtype=float)
            for dim in range(d):
                p_plus = protos[neg_idx].copy(); p_plus[dim] += h
                p_minus = protos[neg_idx].copy(); p_minus[dim] -= h
                d_plus = float(poincare_distance(x, p_plus))
                d_minus = float(poincare_distance(x, p_minus))
                grad_neg[dim] = (d_plus**2 - d_minus**2) / (2.0 * h)

            # Update positive prototype (pull toward x).
            protos[pos_idx] = rsgd_step(protos[pos_idx], grad_pos, lr)
            # Update negative prototype (push away from x: negate the gradient).
            protos[neg_idx] = rsgd_step(protos[neg_idx], -grad_neg, lr)

    return PrototypeBundle(
        vertex_ids=vertex_ids,
        prototypes=protos,
        dimension=bundle.dimension,
    )
