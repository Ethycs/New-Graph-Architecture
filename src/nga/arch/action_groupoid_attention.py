"""Action-groupoid attention.

Standard attention works on V tokens (vertices). The forgetful orbit_pair
attention works on |V/H| tokens (orbit reps), losing cycle structure.

Action-groupoid attention works on the dart set (size 2*E) augmented with
the monodromy generators rho, tau as edge-types. Conceptually this is
attention on the *action groupoid*: vertices and the morphisms between
them. Closed walks in the groupoid correspond to monodromy-group elements,
so the cycle structure is preserved.

For Phase 4 v2 we expose two attention paths:

  1. groupoid_attention(Q_darts, K_darts, V_darts, monodromy_data):
       attention over the dart set with attention scores BIASED by the
       monodromy class (darts in the same rho-orbit attend more strongly
       than darts in different orbits). This is the "cycle-aware" version.

  2. orbit_set_attention (the existing forgetful version, re-exported)

The user can pick either at runtime via Config.attention_kind.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nga.arch.monodromy_group import MonodromyData

__all__ = [
    "GroupoidAttentionFlops",
    "groupoid_attention",
    "attention_flops_comparison",
]


@dataclass
class GroupoidAttentionFlops:
    """FLOPs summary comparing standard, groupoid, and orbit-set attention paths.

    Attributes
    ----------
    standard_flops:
        FLOPs for standard V x V vertex attention.
    groupoid_flops:
        FLOPs for groupoid attention over 2*E darts with cycle bias.
    orbit_set_flops:
        FLOPs for forgetful orbit-set attention over |V/H| orbit reps.
    groupoid_reduction:
        Fractional FLOPs saved by groupoid vs standard; negative means groupoid
        costs more (expected when dart count exceeds vertex count).
    orbit_set_reduction:
        Fractional FLOPs saved by orbit-set vs standard; in [0, 1].
    cycle_preservation:
        True iff groupoid attention actually preserves cycle structure, i.e.
        n_darts > n_orbits (there is non-trivial cycle structure beyond the
        orbit decomposition).
    """

    standard_flops: int
    groupoid_flops: int       # 2*E tokens, with cycle bias
    orbit_set_flops: int      # |V/H| tokens, forgetful
    groupoid_reduction: float  # vs standard
    orbit_set_reduction: float  # vs standard
    cycle_preservation: bool  # True iff groupoid actually preserves cycle structure


def _softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax.

    Parameters
    ----------
    scores:
        Shape (m, n) of float.

    Returns
    -------
    np.ndarray
        Shape (m, n) of float. Each row sums to 1.
    """
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def groupoid_attention(
    queries: np.ndarray,           # shape (n_darts, d)
    keys: np.ndarray,              # shape (n_darts, d)
    values: np.ndarray,            # shape (n_darts, d_v)
    monodromy: MonodromyData,
    *,
    cycle_bias_strength: float = 1.0,
) -> np.ndarray:
    """Attention over darts with monodromy-class-aware bias.

    Score(i, j) = Q_i K_j^T / sqrt(d) + cycle_bias_strength * 1[mc_i == mc_j]

    where mc_i = monodromy.monodromy_class[i]. Darts in the same monodromy
    class get a +cycle_bias_strength bonus before softmax. This preserves
    cycle structure: closed walks bind tokens that share a cycle signature.

    Parameters
    ----------
    queries:
        Shape (n_darts, d). Query matrix over the dart set.
    keys:
        Shape (n_darts, d). Key matrix over the dart set.
    values:
        Shape (n_darts, d_v). Value matrix over the dart set.
    monodromy:
        Monodromy data produced by compute_monodromy. Provides
        monodromy_class labels used for cycle-aware biasing.
    cycle_bias_strength:
        Additive bonus applied to score(i, j) whenever dart i and dart j
        share the same monodromy class. Default 1.0. Set to 0.0 to recover
        standard scaled-dot-product attention on darts.

    Returns
    -------
    np.ndarray
        Shape (n_darts, d_v). Attention-weighted values for each dart.
    """
    n_darts, d = queries.shape

    # Base attention scores: (n_darts, n_darts).
    scale = float(d) ** 0.5
    scores = queries @ keys.T / scale

    # Cycle-aware bias: +cycle_bias_strength for same monodromy class.
    if cycle_bias_strength != 0.0:
        mc = monodromy.monodromy_class  # shape (n_darts,)
        # Broadcast comparison: same_class[i, j] = True iff mc[i] == mc[j].
        same_class = (mc[:, None] == mc[None, :])  # shape (n_darts, n_darts)
        scores = scores + cycle_bias_strength * same_class.astype(scores.dtype)

    attn = _softmax_rows(scores)  # shape (n_darts, n_darts)
    return attn @ values          # shape (n_darts, d_v)


def attention_flops_comparison(
    n_vertices: int,
    n_darts: int,
    n_orbits: int,
    *,
    d: int = 32,
) -> GroupoidAttentionFlops:
    """FLOPs for standard (V^2), groupoid (n_darts^2), and orbit-set (|V/H|^2)
    attention paths. Reports cycle_preservation = True iff n_darts > n_orbits
    (i.e. there is a non-trivial cycle structure to preserve).

    FLOPs model (same formula for all three paths, scaled by token count N):
      - Q K^T:    2 * N^2 * d
      - softmax:      N^2
      - P @ V:    2 * N^2 * d
      Total:      4 * N^2 * d + N^2

    For groupoid attention the token count is n_darts (2*E darts).
    For standard attention the token count is n_vertices (V).
    For orbit-set attention the token count is n_orbits (|V/H|).

    The cycle_bias computation adds an O(n_darts^2) boolean broadcast,
    which is dominated by the attention terms and omitted from the FLOPs
    count for simplicity.

    Parameters
    ----------
    n_vertices:
        Number of vertices V in the graph.
    n_darts:
        Number of darts 2*E in the dart set.
    n_orbits:
        Number of orbits |V/H| under the group action (forgetful path).
    d:
        Query/key/value embedding dimension.

    Returns
    -------
    GroupoidAttentionFlops
        Populated with integer FLOPs counts, fractional reductions, and the
        cycle_preservation flag.
    """
    def _flops(n: int) -> int:
        return 4 * n * n * d + n * n

    standard = _flops(n_vertices)
    groupoid = _flops(n_darts)
    orbit_set = _flops(n_orbits)

    groupoid_reduction = (float(standard - groupoid) / float(standard)) if standard > 0 else 0.0
    orbit_set_reduction = (float(standard - orbit_set) / float(standard)) if standard > 0 else 0.0

    # Clamp orbit_set_reduction to [0, 1]; groupoid may be negative (darts > vertices).
    orbit_set_reduction = max(0.0, min(1.0, orbit_set_reduction))

    return GroupoidAttentionFlops(
        standard_flops=standard,
        groupoid_flops=groupoid,
        orbit_set_flops=orbit_set,
        groupoid_reduction=groupoid_reduction,
        orbit_set_reduction=orbit_set_reduction,
        cycle_preservation=n_darts > n_orbits,
    )
