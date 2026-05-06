"""Orbit-pair attention: O(|V/H|^2) scoring instead of O(|V|^2).

Standard scaled-dot-product attention computes a V x V score matrix.
Orbit-pair attention pools tokens into orbit groups (size |V/H|), computes
a |V/H| x |V/H| score matrix, then broadcasts back to V x V via orbit
membership. Identical in expressivity when the group action is a true
symmetry of the underlying task; cheaper by a factor of (|V| / |V/H|)^2.

Lossy-compression note
----------------------
When the input (Q, K, V) is NOT symmetric under the group action - i.e.
vertices in the same orbit carry genuinely different query/key/value
embeddings - orbit-pair attention produces a smoothed approximation. The
mean-pooling step inside each orbit discards within-orbit variance, so the
output is an orbit-averaged version of the standard result. Users should only
rely on orbit-pair attention as a drop-in replacement when the group action is
a verified symmetry of the data distribution; otherwise treat it as a
regularised (compressed) approximation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nga.arch.orbit_quotient_space import OrbitDecomposition


@dataclass
class AttentionFlops:
    """FLOPs summary for standard vs orbit-pair attention.

    Attributes
    ----------
    standard_flops:
        Total FLOPs for the standard V x V attention path.
    orbit_pair_flops:
        Total FLOPs for the orbit-pair O x O attention path (including
        the mean-pool and broadcast steps).
    reduction_ratio:
        (standard - orbit_pair) / standard, in [0, 1].
        0 means no savings (trivial group, O == V).
        1 means complete compression (single orbit, O == 1).
    """

    standard_flops: int
    orbit_pair_flops: int
    reduction_ratio: float  # in [0, 1]


# ---------------------------------------------------------------------------
# Standard attention
# ---------------------------------------------------------------------------


def standard_attention(
    queries: np.ndarray,  # shape (V, d)
    keys: np.ndarray,     # shape (V, d)
    values: np.ndarray,   # shape (V, d_v)
) -> np.ndarray:
    """Plain scaled-dot-product attention; returns shape (V, d_v).

    Computes softmax(Q K^T / sqrt(d)) @ V where d is the key/query
    dimension (queries.shape[1]).

    Parameters
    ----------
    queries:
        Shape (V, d). Query matrix.
    keys:
        Shape (V, d). Key matrix.
    values:
        Shape (V, d_v). Value matrix.

    Returns
    -------
    np.ndarray
        Shape (V, d_v). Attention-weighted values.
    """
    d = queries.shape[1]
    scale = float(d) ** 0.5

    # Score matrix: (V, V)
    scores = queries @ keys.T / scale

    # Numerically stable softmax along key axis.
    scores -= scores.max(axis=1, keepdims=True)
    attn = np.exp(scores)
    attn /= attn.sum(axis=1, keepdims=True)

    # Weighted sum of values: (V, d_v)
    return attn @ values


# ---------------------------------------------------------------------------
# Orbit-pair attention
# ---------------------------------------------------------------------------


def orbit_pair_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    decomposition: OrbitDecomposition,
) -> np.ndarray:
    """Orbit-pair attention: O x O scoring broadcast back to V x V.

    Algorithm
    ---------
    1. Mean-pool Q, K, V within each orbit to get orbit-level tensors
       Q_o (O, d), K_o (O, d), V_o (O, d_v).
    2. Compute scaled-dot-product attention over the O x O matrix.
    3. Broadcast the orbit-level output back to every member vertex of
       each orbit, producing a (V, d_v) result.

    Lossy-compression caveat
    ~~~~~~~~~~~~~~~~~~~~~~~~
    When Q, K, or V differ within an orbit (asymmetric input), step 1
    discards that within-orbit variance. The output is then a smoothed
    approximation of standard_attention. This is acceptable when the group
    action is a verified symmetry of the task; otherwise callers should
    prefer standard_attention or use orbit-pair attention as explicit
    regularisation.

    Parameters
    ----------
    queries:
        Shape (V, d). Query matrix.
    keys:
        Shape (V, d). Key matrix.
    values:
        Shape (V, d_v). Value matrix.
    decomposition:
        Orbit decomposition produced by decompose_orbits.

    Returns
    -------
    np.ndarray
        Shape (V, d_v). Attention output replicated within each orbit.
    """
    V = decomposition.vertex_count
    O = decomposition.n_orbits
    d = queries.shape[1]
    d_v = values.shape[1]

    # Step 1: mean-pool within each orbit -> orbit-level tensors (O, d)/(O, d_v).
    Q_o = np.zeros((O, d), dtype=queries.dtype)
    K_o = np.zeros((O, d), dtype=keys.dtype)
    V_o = np.zeros((O, d_v), dtype=values.dtype)

    for o_idx, orbit in enumerate(decomposition.orbits):
        idx = np.array(orbit, dtype=int)
        Q_o[o_idx] = queries[idx].mean(axis=0)
        K_o[o_idx] = keys[idx].mean(axis=0)
        V_o[o_idx] = values[idx].mean(axis=0)

    # Step 2: scaled-dot-product attention at orbit level -> (O, d_v).
    #
    # Orbit-size softmax weighting
    # ----------------------------
    # An orbit representative stands in for orbit_size[j] tokens; without
    # log-orbit-size weighting, softmax under-counts large orbits relative to
    # fixed singletons. Adding log(orbit_size[j]) to the score along the KEY
    # axis (j) is equivalent to multiplying the unnormalised softmax weight by
    # orbit_size[j], restoring the per-token contribution that mean-pooling
    # erased. For the single-orbit case log(orbit_size) is a constant added to
    # every column and cancels under softmax (so existing single-orbit parity
    # is preserved); for the multi-orbit case it is the term that makes
    # orbit-pair attention numerically equivalent to standard attention on a
    # symmetric input.
    scale = float(d) ** 0.5
    scores = Q_o @ K_o.T / scale  # (O, O)

    orbit_sizes = np.array(
        [len(orbit) for orbit in decomposition.orbits], dtype=scores.dtype
    )  # shape (O,)
    # Broadcast over the KEY axis (j): every query row sees +log(orbit_size[j])
    # on column j, i.e. weights orbit j by orbit_size[j] before softmax.
    scores = scores + np.log(orbit_sizes)[np.newaxis, :]

    scores -= scores.max(axis=1, keepdims=True)
    attn = np.exp(scores)
    attn /= attn.sum(axis=1, keepdims=True)
    orbit_output = attn @ V_o  # (O, d_v)

    # Step 3: broadcast orbit output to every member vertex.
    out = np.empty((V, d_v), dtype=values.dtype)
    for o_idx, orbit in enumerate(decomposition.orbits):
        for v in orbit:
            out[v] = orbit_output[o_idx]

    return out


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------


def attention_flops(decomposition: OrbitDecomposition, *, d: int = 32) -> AttentionFlops:
    """FLOPs estimate for standard vs orbit-pair attention paths.

    Standard path (V x V)
    ~~~~~~~~~~~~~~~~~~~~~
    - Q K^T multiply:  2 * V^2 * d  FLOPs
    - Softmax:             V^2      FLOPs  (approx; ignores exp as 1 FLOP)
    - P @ V multiply:  2 * V^2 * d  FLOPs
    Total: 4 * V^2 * d + V^2

    Orbit-pair path (O x O)
    ~~~~~~~~~~~~~~~~~~~~~~~
    - Mean-pool Q, K, V: each pool over orbit members costs O(V) work,
      so: 3 * V * d  FLOPs  (one pass through all V rows for each of Q, K, V)
    - Q_o K_o^T:      2 * O^2 * d  FLOPs
    - Softmax:             O^2      FLOPs
    - P_o @ V_o:      2 * O^2 * d  FLOPs
    - Broadcast:      V * d_v ~ V * d  FLOPs
    Total: 3*V*d + 4*O^2*d + O^2 + V*d  =  4*V*d + 4*O^2*d + O^2

    Reduction ratio: (standard - orbit_pair) / standard, in [0, 1].
    - Trivial group (O == V):  numerator ~ 0, ratio -> 0.0.
    - Single orbit  (O == 1):  orbit_pair is tiny, ratio -> 1.0.

    Parameters
    ----------
    decomposition:
        The orbit decomposition produced by decompose_orbits.
    d:
        Query/key/value embedding dimension (default 32).

    Returns
    -------
    AttentionFlops
        Populated with integer FLOPs counts and the reduction ratio.
    """
    V = decomposition.vertex_count
    O = decomposition.n_orbits

    standard = 4 * V * V * d + V * V
    orbit_pair = 4 * V * d + 4 * O * O * d + O * O

    saved = standard - orbit_pair
    ratio = float(saved) / float(standard) if standard > 0 else 0.0
    ratio = max(0.0, min(1.0, ratio))  # clamp to [0, 1]

    return AttentionFlops(
        standard_flops=standard,
        orbit_pair_flops=orbit_pair,
        reduction_ratio=ratio,
    )
