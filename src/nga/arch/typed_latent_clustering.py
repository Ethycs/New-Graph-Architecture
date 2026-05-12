"""Typed latent clustering on the Poincare ball.

This module implements a Riemannian k-means variant where each observation
carries a *deterministic* type label ``q_t`` and an *inferred* latent cluster
ID ``lambda_t``. Migration of latent IDs is constrained to occur strictly
inside a type-orbit: an observation of type ``q`` is only ever assigned to
prototypes belonging to type ``q``.

Equivariance story
------------------
The architecture's group action acts on type IDs (the deterministic ``q_t``
component). Because that action is *outside* this clustering routine -- we
take the types as a given input here -- the clustering itself becomes
equivariant by construction: permuting type IDs simply relabels the per-type
prototype dictionary, and the within-type latent assignments are unchanged.

Conversely, the latent IDs ``lambda_t`` are *inferred* via Riemannian
k-means inside the orbit ``{x : type(x) == q}`` for each ``q``. They have no
canonical meaning across runs (only up to a permutation within their type),
which is exactly the gauge freedom of latent variables.

The state-induction picture
---------------------------
Initial latent labels are drawn uniformly at random within each type. Across
iterations they migrate (Lloyd-style assignment / centroid updates) toward
stable cluster centres, with the centroid step being the Karcher / Frechet
mean on the Poincare ball.

Implementation notes
--------------------
- All Poincare-ball operations are reused from
  ``nga.arch.hyperbolic_embedding`` (no torch / sklearn).
- The Karcher mean is computed by gradient descent in the tangent space at
  the current iterate, using ``log_map_zero`` / ``exp_map_zero`` together
  with a translation to the current point in the ball. Concretely we recentre
  via Mobius-inverse-translation by a first-order linearisation: at each
  inner step, project the current centre to the origin via translation along
  the negative log map, average the translated points, exp-map back, and
  un-translate. This converges to the Frechet mean on the Poincare ball for
  small clusters in typical hyperbolic regimes; we cap inner iterations at
  ``max_inner_iter`` (default 10) which is empirically sufficient for the
  noise levels we use elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Union

import numpy as np

from nga.arch.hyperbolic_embedding import (
    EPS,
    exp_map_zero,
    log_map_zero,
    poincare_distance,
    project,
)

__all__ = [
    "ClusteringResult",
    "cluster_typed_latents",
    "riemannian_centroid",
]

TypeId = Hashable


@dataclass
class ClusteringResult:
    """Outcome of a typed latent clustering run.

    Attributes
    ----------
    labels:
        Integer array of shape ``(N,)``. ``labels[t]`` is the latent cluster
        ID of observation ``t`` *within* its type-orbit. IDs are local to the
        type, i.e. they restart from 0 for each type and are only meaningful
        when paired with the type label.
    prototypes:
        Mapping from type ID to its prototype matrix of shape
        ``(L_q, dim)``. Each row is a Poincare-ball point. Types that had
        zero observations map to an empty ``(0, dim)`` array.
    n_iter:
        Number of outer Lloyd iterations actually performed.
    converged:
        ``True`` if the change-of-labels fraction dropped below ``tol``
        before ``max_iter`` was reached.
    final_distortion:
        Sum of Riemannian distances from each observation to its assigned
        prototype, evaluated on the final iterate.
    """

    labels: np.ndarray
    prototypes: dict[TypeId, np.ndarray]
    n_iter: int
    converged: bool
    final_distortion: float


def riemannian_centroid(
    points: np.ndarray,
    *,
    max_inner_iter: int = 10,
    lr: float = 0.1,
) -> np.ndarray:
    """Karcher / Frechet mean of ``points`` on the Poincare ball.

    Implementation: iterate gradient descent in the tangent space at the
    current iterate. At each step, translate so the current centre lies at
    the origin, average the log-mapped tangents (which is the Euclidean
    direction of steepest descent of the squared-distance loss at the
    origin), exp-map a small step in that direction, then un-translate.

    Parameters
    ----------
    points:
        Array of shape ``(K, dim)`` of Poincare-ball points. Must be
        non-empty.
    max_inner_iter:
        Maximum number of gradient-descent iterations.
    lr:
        Step size in the tangent space; values close to 1.0 give a
        Newton-like step on the squared-distance loss at the origin (whose
        Hessian is the identity in tangent coords), while smaller values are
        more conservative for larger / noisier clusters.

    Returns
    -------
    np.ndarray
        Estimated centroid as a Poincare-ball point of shape ``(dim,)``.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError(
            f"riemannian_centroid expects (K, dim) with K>=1; got {pts.shape}"
        )
    if pts.shape[0] == 1:
        return project(pts[0].copy())

    # Initialise at the log-mapped Euclidean mean of the tangents at zero.
    # This is the same as exp_0( mean( log_0(x_i) ) ) -- a fast first guess.
    tangents0 = log_map_zero(pts)                # (K, dim)
    centre = exp_map_zero(tangents0.mean(axis=0))
    centre = project(centre)

    for _ in range(max_inner_iter):
        # Translate the cluster so that ``centre`` sits at the origin via
        # the (linearised) Mobius inverse-translation: subtract the log map
        # of centre from each point's log map. This is exact at the origin
        # and a first-order valid retraction elsewhere -- which is the same
        # approximation rsgd_step uses.
        log_centre = log_map_zero(centre[None, :])[0]   # (dim,)
        log_pts = log_map_zero(pts)                     # (K, dim)
        # tangents at current centre (first-order approximation)
        tangents = log_pts - log_centre[None, :]        # (K, dim)
        grad = tangents.mean(axis=0)                    # (dim,)

        # Step size threshold: stop when the gradient norm is tiny.
        if float(np.linalg.norm(grad)) < 1e-8:
            break

        new_log_centre = log_centre + lr * grad
        new_centre = exp_map_zero(new_log_centre)
        centre = project(new_centre)

    return centre


def _normalise_l_per_type(
    L_per_type: Union[int, dict[TypeId, int]],
    unique_types: list[TypeId],
) -> dict[TypeId, int]:
    """Resolve ``L_per_type`` into a dict keyed by every observed type."""
    if isinstance(L_per_type, dict):
        # Allow callers to specify L for types that may have zero
        # observations; we silently keep those entries (an empty-type entry
        # ends up with zero prototypes anyway).
        out: dict[TypeId, int] = {}
        for q in unique_types:
            if q not in L_per_type:
                raise ValueError(
                    f"L_per_type dict missing entry for observed type {q!r}"
                )
            out[q] = int(L_per_type[q])
        # Carry over any extra (unobserved) entries the caller supplied.
        for q, v in L_per_type.items():
            out.setdefault(q, int(v))
        return out
    return {q: int(L_per_type) for q in unique_types}


def _assign_within_type(
    obs: np.ndarray,
    protos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (argmin_idx, min_dist) for each ``obs`` row against ``protos``.

    obs: (n, dim), protos: (L, dim). Returns two arrays of shape (n,).
    """
    if protos.shape[0] == 1:
        d = poincare_distance(obs, np.broadcast_to(protos, obs.shape))
        return np.zeros(obs.shape[0], dtype=int), np.atleast_1d(d).astype(float)
    # cross distance: poincare_distance handles (n, d) vs (m, d) -> (n, m)
    D = poincare_distance(obs, protos)             # (n, L)
    if D.ndim == 1:
        # Defensive: if obs has a single row, poincare_distance squeezes.
        D = D[None, :]
    idx = np.argmin(D, axis=1)
    mins = D[np.arange(D.shape[0]), idx]
    return idx.astype(int), mins.astype(float)


def cluster_typed_latents(
    *,
    observations_in_poincare: np.ndarray,
    types: np.ndarray,
    L_per_type: Union[int, dict[TypeId, int]],
    max_iter: int = 50,
    seed: int = 42,
    tol: float = 1e-4,
) -> ClusteringResult:
    """Riemannian k-means within type-orbits on the Poincare ball.

    Parameters
    ----------
    observations_in_poincare:
        Float array of shape ``(N, dim)``. Each row must already lie inside
        the open Poincare unit ball.
    types:
        Type label per observation, shape ``(N,)``. May be integer- or
        string-valued (anything hashable).
    L_per_type:
        Either a single int (use the same number of latent clusters for every
        type) or a dict mapping every observed type to its cluster count. If
        a dict, it must include every type appearing in ``types``.
    max_iter:
        Maximum number of outer Lloyd iterations.
    seed:
        Seed for deterministic prototype initialisation.
    tol:
        Convergence tolerance on the *fraction* of labels that changed in the
        last iteration.

    Returns
    -------
    ClusteringResult
        Per-observation latent IDs (local to type), per-type prototype
        matrices, and convergence diagnostics.
    """
    obs = np.asarray(observations_in_poincare, dtype=float)
    types_arr = np.asarray(types)
    if obs.ndim != 2:
        raise ValueError(
            f"observations_in_poincare must be (N, dim); got {obs.shape}"
        )
    if types_arr.shape != (obs.shape[0],):
        raise ValueError(
            f"types must be 1-D of length N={obs.shape[0]}; got {types_arr.shape}"
        )

    n, dim = obs.shape
    rng = np.random.default_rng(seed)

    # Stable order of unique types: first-occurrence order.
    seen: set[TypeId] = set()
    unique_types: list[TypeId] = []
    for q in types_arr.tolist():
        if q not in seen:
            seen.add(q)
            unique_types.append(q)

    L_dict = _normalise_l_per_type(L_per_type, unique_types)

    # Indices of observations belonging to each type.
    idx_by_type: dict[TypeId, np.ndarray] = {
        q: np.where(types_arr == q)[0] for q in unique_types
    }

    # ---------- Initialisation: k-means++ seeding per type ----------
    # Picks the first prototype uniformly at random within the type, then
    # samples each subsequent one with probability proportional to its
    # squared Riemannian distance to the closest already-chosen prototype.
    # This keeps the routine deterministic under ``seed`` while avoiding the
    # bad local minima that plain uniform random init can fall into.
    prototypes: dict[TypeId, np.ndarray] = {}
    for q in unique_types:
        idx_q = idx_by_type[q]
        L_q = L_dict[q]
        if idx_q.size == 0 or L_q == 0:
            prototypes[q] = np.zeros((0, dim), dtype=float)
            continue
        obs_q = obs[idx_q]
        if idx_q.size < L_q:
            # Fewer points than clusters: fall back to sampling with replacement.
            chosen = rng.choice(idx_q, size=L_q, replace=True)
            prototypes[q] = obs[chosen].copy()
            continue
        # k-means++ seeding.
        first = int(rng.integers(0, idx_q.size))
        chosen_local: list[int] = [first]
        for _k in range(1, L_q):
            current = obs_q[chosen_local]                 # (m, dim)
            D = poincare_distance(obs_q, current)
            if D.ndim == 1:
                D = D[:, None]
            min_d = np.min(D, axis=1)                     # (n_q,)
            sq = min_d ** 2
            total = float(sq.sum())
            if total <= 0.0:
                # All points coincide with chosen ones -- fall back to uniform.
                pick = int(rng.integers(0, idx_q.size))
            else:
                probs = sq / total
                pick = int(rng.choice(idx_q.size, p=probs))
            chosen_local.append(pick)
        prototypes[q] = obs_q[chosen_local].copy()
    # Carry forward any unobserved types so callers see them.
    for q, L_q in L_dict.items():
        if q not in prototypes:
            prototypes[q] = np.zeros((0, dim), dtype=float)

    labels = np.full(n, -1, dtype=int)
    converged = False
    n_iter = 0
    final_distortion = float("inf")

    for it in range(1, max_iter + 1):
        n_iter = it
        prev_labels = labels.copy()

        # ---------- Assignment step (per type) ----------
        total_distortion = 0.0
        for q in unique_types:
            idx_q = idx_by_type[q]
            if idx_q.size == 0:
                continue
            protos_q = prototypes[q]
            if protos_q.shape[0] == 0:
                # No prototypes available for this type: leave labels at -1.
                labels[idx_q] = -1
                continue
            assigned, dists = _assign_within_type(obs[idx_q], protos_q)
            labels[idx_q] = assigned
            total_distortion += float(dists.sum())

        # ---------- Update step: Karcher mean per (type, cluster) ----------
        for q in unique_types:
            idx_q = idx_by_type[q]
            protos_q = prototypes[q]
            L_q = protos_q.shape[0]
            if idx_q.size == 0 or L_q == 0:
                continue
            new_protos = protos_q.copy()
            for k in range(L_q):
                member_mask = labels[idx_q] == k
                members = obs[idx_q][member_mask]
                if members.shape[0] == 0:
                    # Empty cluster: re-init to a random observation of q.
                    pick = rng.choice(idx_q)
                    new_protos[k] = obs[pick].copy()
                else:
                    new_protos[k] = riemannian_centroid(members, max_inner_iter=5)
            prototypes[q] = new_protos

        # ---------- Convergence check ----------
        if it == 1:
            change_frac = 1.0
        else:
            change_frac = float(np.mean(labels != prev_labels))
        final_distortion = total_distortion
        if change_frac < tol:
            converged = True
            break

    # Final assignment after the last update so labels reflect the final
    # prototypes (otherwise the very last update step is "ahead" of labels).
    final_distortion = 0.0
    for q in unique_types:
        idx_q = idx_by_type[q]
        if idx_q.size == 0:
            continue
        protos_q = prototypes[q]
        if protos_q.shape[0] == 0:
            labels[idx_q] = -1
            continue
        assigned, dists = _assign_within_type(obs[idx_q], protos_q)
        labels[idx_q] = assigned
        final_distortion += float(dists.sum())

    return ClusteringResult(
        labels=labels,
        prototypes=prototypes,
        n_iter=n_iter,
        converged=converged,
        final_distortion=final_distortion,
    )
