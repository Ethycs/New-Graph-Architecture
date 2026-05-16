"""KL-canonical regime signatures.

Information-geometric K-choice for the labelled hypergraph. Each regime is
characterised by a categorical output distribution ``p_lambda(y | x)``
estimated from corpus observations. The KL divergence

    D(p_i || p_j) = sum_y p_i(y) * log(p_i(y) / p_j(y))

is a coordinate-free measure of distance between regime distributions:
invariant under reparametrisation of the substrate, dependent only on the
distributions themselves. Two regimes whose pairwise KL falls below a
principled threshold are statistically equivalent and can be canonicalised
into one.

This atom computes:

  - per-regime KL signatures (row of the per-regime KL matrix)
  - threshold-based equivalence clustering
  - principled threshold suggestion (gap-detection or Cramer-Rao floor)

Dependencies: numpy + stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "KLRegimeSignature",
]


_EPS = 1.0e-12


def _safe_log(x: np.ndarray) -> np.ndarray:
    """Elementwise log with a small floor to keep 0 * log(0) = 0."""
    return np.log(np.clip(x, _EPS, None))


def _normalise_rows(p: np.ndarray) -> np.ndarray:
    """Normalise each row of ``p`` so that it sums to 1 (with epsilon floor)."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"Expected 2-D conditional matrix; got shape {p.shape}")
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return p / row_sums


@dataclass
class KLRegimeSignature:
    """Compute and cluster regimes by KL-canonical signatures."""

    @staticmethod
    def compute(
        per_regime_conditionals: dict[str, np.ndarray],
        *,
        symmetric: bool = True,
    ) -> dict[str, np.ndarray]:
        """Return per-regime KL signature vectors.

        Parameters
        ----------
        per_regime_conditionals:
            ``regime_id -> p_lambda(y)`` over a fixed shared support. Each
            value must be a 1-D array of non-negative reals; rows are
            normalised internally.
        symmetric:
            If True (default), use the symmetric KL ``D(p||q) + D(q||p)``
            so the signature distances are a proper metric on the order of
            magnitudes that matter for clustering. If False, use directed
            ``D(self || other)``.

        Returns
        -------
        dict[regime_id, np.ndarray]
            For each regime, a 1-D array of length ``len(per_regime_conditionals)``
            giving the (symmetric or directed) KL distance from this regime
            to every other regime, ordered by sorted regime ID. Self-distance
            is 0.
        """
        if not per_regime_conditionals:
            return {}
        ids = sorted(per_regime_conditionals)
        k = len(ids)
        # Stack into (K, V) matrix in sorted-ID order; normalise rows.
        vec_len = None
        rows: list[np.ndarray] = []
        for rid in ids:
            row = np.asarray(per_regime_conditionals[rid], dtype=float).flatten()
            if vec_len is None:
                vec_len = row.shape[0]
            elif row.shape[0] != vec_len:
                raise ValueError(
                    "All conditionals must share the same support length; "
                    f"got {vec_len} and {row.shape[0]}"
                )
            rows.append(row)
        p = _normalise_rows(np.vstack(rows))

        # Pairwise directed KL: D[i, j] = sum_y p_i(y) * log(p_i(y)/p_j(y))
        log_p = _safe_log(p)
        # log_ratio[i, j, y] = log p_i(y) - log p_j(y)
        # KL[i, j] = sum_y p_i(y) * log_ratio[i, j, y]
        # Compute via broadcasting:
        # log_diff: shape (K, K, V)
        log_diff = log_p[:, None, :] - log_p[None, :, :]
        kl_directed = np.einsum("iv,ijv->ij", p, log_diff)
        # Clip tiny negatives from float noise.
        kl_directed = np.clip(kl_directed, 0.0, None)

        if symmetric:
            kl_matrix = kl_directed + kl_directed.T
        else:
            kl_matrix = kl_directed

        return {rid: kl_matrix[i].copy() for i, rid in enumerate(ids)}

    @staticmethod
    def cluster(
        signatures: dict[str, np.ndarray],
        threshold: float,
    ) -> dict[str, str]:
        """Equivalence-class regimes whose pairwise KL is below ``threshold``.

        Two regimes are placed in the same canonical class if their
        signature-distance entry (computed against each other) is at most
        ``threshold``. The canonical representative of each class is the
        regime whose ID sorts first.

        Parameters
        ----------
        signatures:
            Output of ``compute``. ``signatures[r][k]`` is the KL distance
            from regime ``r`` to the ``k``-th regime in sorted-ID order.
        threshold:
            Non-negative KL distance below which two regimes are identified.
            ``0.0`` means "only identify regimes with identical distributions";
            larger values produce coarser clusters.

        Returns
        -------
        dict[regime_id, canonical_id]
            Maps every input regime to the ID of its canonical-class
            representative.
        """
        if threshold < 0.0:
            raise ValueError(f"threshold must be non-negative; got {threshold}")
        ids = sorted(signatures)
        if not ids:
            return {}
        # Build adjacency: i ~ j iff signatures[ids[i]][j] <= threshold.
        n = len(ids)
        idx = {rid: i for i, rid in enumerate(ids)}
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            # Keep the smaller representative (sorted-ID order ⇒ canonical).
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

        for i, rid in enumerate(ids):
            sig = signatures[rid]
            if sig.shape[0] != n:
                raise ValueError(
                    f"signature for {rid!r} has length {sig.shape[0]}; expected {n}"
                )
            for j in range(n):
                if i == j:
                    continue
                if float(sig[j]) <= threshold:
                    union(i, j)

        result: dict[str, str] = {}
        for rid in ids:
            root = find(idx[rid])
            result[rid] = ids[root]
        return result

    @staticmethod
    def suggest_threshold(
        signatures: dict[str, np.ndarray],
        method: str = "gap",
    ) -> float:
        """Suggest a principled clustering threshold.

        Two methods are supported:

        - ``"gap"`` (default): sort the off-diagonal KL values in ascending
          order; find the largest relative gap; return the midpoint of that
          gap. This separates a "noise band" of small KLs (within-cluster
          variation) from a "structure band" of large KLs (between-cluster).
        - ``"median"``: half the median off-diagonal KL value. Conservative;
          tends to under-cluster.

        With fewer than three regimes there is no gap to find; returns
        ``0.0`` (no clustering).
        """
        ids = sorted(signatures)
        if len(ids) < 3:
            return 0.0
        n = len(ids)
        # Collect off-diagonal pairs.
        off: list[float] = []
        for i, rid in enumerate(ids):
            sig = signatures[rid]
            for j in range(n):
                if i == j:
                    continue
                off.append(float(sig[j]))
        if not off:
            return 0.0
        off_arr = np.sort(np.asarray(off, dtype=float))
        if method == "median":
            return float(np.median(off_arr) * 0.5)
        if method == "gap":
            # Largest absolute gap between consecutive sorted values.
            diffs = np.diff(off_arr)
            if diffs.size == 0:
                return 0.0
            gap_idx = int(np.argmax(diffs))
            lo, hi = float(off_arr[gap_idx]), float(off_arr[gap_idx + 1])
            return 0.5 * (lo + hi)
        raise ValueError(f"Unknown method {method!r}; expected 'gap' or 'median'")
