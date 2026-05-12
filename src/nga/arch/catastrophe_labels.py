"""Catastrophe-theoretic labels on FSM edges.

Catastrophe theory (Thom, Arnold) classifies the local geometry of smooth
critical points of a real-valued potential E(x) by codimension. The first
two of the ADE-indexed elementary catastrophes are the only ones this wave
detects numerically:

    fold  (A_2): smooth quadratic critical point. Local normal form
                 E(x) = (x - a)**2. Diagnostic: the second derivative is
                 non-zero at the critical point, so the curve has a clean
                 minimum or maximum.

    cusp  (A_3): degenerate cubic-quartic critical point. Local normal form
                 E(x) = x**4 + a x**2 + b x. Diagnostic: the second
                 derivative vanishes at the critical point but the third
                 (or fourth) derivative does not, so the curvature flattens
                 and the trajectory inflects rather than bottoms out.

    none        : neither diagnostic fires. The energy curve is locally
                  smooth and free of catastrophe-class structure (e.g.
                  linear, or a region away from any critical point).

`SWALLOWTAIL` (A_4) and `BUTTERFLY` (A_5) are kept as named enum members so
that downstream code can label edges that come from external taxonomies, but
the numerical detector in this wave never returns either. They will be
filled in alongside the stratified partition function (q02-energy-function-spec).

Detection scheme
----------------
We approximate derivatives with central finite differences on equally-spaced
energy samples E_0, E_1, E_2, E_3, E_4 around a putative critical point at
the centre index 2. With spacing h = 1 (the absolute spacing cancels for
classification purposes since we threshold a magnitude against `eps`):

    d2E ~  E_1 - 2 E_2 + E_3
    d3E ~ (-E_0 + 2 E_1 - 2 E_3 + E_4) / 2

Classification at the centre sample:

    |d2E| > eps  -> FOLD
    |d3E| > eps  -> CUSP
    otherwise    -> NONE

For longer paths we locate the index whose first-difference magnitude is
smallest (the closest interior point to a critical point) and run the
5-sample window classifier there.

The legacy stub helpers `get_edge_label` and `edge_priors` are preserved
unchanged for callers that look up labels stored on the FSM edge spec; the
new `classify_*` functions are the real numerical detector.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from nga.arch.graph_fsm import GraphFSM

__all__ = [
    "CatastropheLabel",
    "get_edge_label",
    "edge_priors",
    "classify_curvature_1d",
    "classify_edge_catastrophe",
    "edge_catastrophe_label",
]


class CatastropheLabel(str, Enum):
    NONE = "none"
    FOLD = "fold"
    CUSP = "cusp"
    SWALLOWTAIL = "swallowtail"
    BUTTERFLY = "butterfly"


# --- numerical detector -------------------------------------------------------


def _finite_diff_d2_d3(samples: np.ndarray) -> tuple[float, float]:
    """Central-difference 2nd and 3rd derivatives at the centre of a 5-sample window.

    Spacing h is treated as 1; we threshold magnitudes, so the absolute scale
    of h merely rescales `eps`. Caller is responsible for handing in an array
    of length exactly 5.
    """
    e0, e1, e2, e3, e4 = (float(samples[i]) for i in range(5))
    d2 = e1 - 2.0 * e2 + e3
    d3 = (-e0 + 2.0 * e1 - 2.0 * e3 + e4) / 2.0
    return d2, d3


def classify_curvature_1d(
    energy_samples: np.ndarray, *, eps: float = 1e-3
) -> CatastropheLabel:
    """Classify the catastrophe type at the centre of a 5-sample energy window.

    Parameters
    ----------
    energy_samples:
        1-D array of equally-spaced energy values around a putative critical
        point. Must have length >= 5; the 5 samples centred on the array
        midpoint are used. For an array of length exactly 5, that is the
        whole array.
    eps:
        Magnitude threshold for the finite-difference derivatives. Set this
        higher than the noise floor of your energy estimator and lower than
        the curvature you care about.

    Returns
    -------
    CatastropheLabel
        FOLD if |d2E| > eps, else CUSP if |d3E| > eps, else NONE.

    Raises
    ------
    ValueError
        If `energy_samples` has fewer than 5 elements or is not 1-D.
    """
    arr = np.asarray(energy_samples, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"classify_curvature_1d expects a 1-D array, got shape {arr.shape}"
        )
    if arr.shape[0] < 5:
        raise ValueError(
            "classify_curvature_1d needs at least 5 samples for a centred "
            f"finite-difference window, got {arr.shape[0]}"
        )

    centre = arr.shape[0] // 2
    window = arr[centre - 2 : centre + 3]
    d2, d3 = _finite_diff_d2_d3(window)

    if abs(d2) > eps:
        return CatastropheLabel.FOLD
    if abs(d3) > eps:
        return CatastropheLabel.CUSP
    return CatastropheLabel.NONE


def classify_edge_catastrophe(
    energy_along_path: np.ndarray, *, eps: float = 1e-3
) -> CatastropheLabel:
    """Classify the catastrophe along a sampled energy path of length >= 5.

    The path is scanned for the interior index whose absolute first-difference
    is smallest — i.e. the point closest to a local critical point — and the
    5-sample window centred there is fed to `classify_curvature_1d`. If the
    minimum lies near an endpoint we shift inward so the window stays in
    bounds.

    Parameters
    ----------
    energy_along_path:
        1-D array of energies sampled along an edge / trajectory. Must have
        length >= 5.
    eps:
        Forwarded to `classify_curvature_1d`.

    Returns
    -------
    CatastropheLabel
        Result of the centred 5-sample classifier at the most-critical index.

    Raises
    ------
    ValueError
        If `energy_along_path` has fewer than 5 elements or is not 1-D.
    """
    arr = np.asarray(energy_along_path, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"classify_edge_catastrophe expects a 1-D array, got shape {arr.shape}"
        )
    n = arr.shape[0]
    if n < 5:
        raise ValueError(
            "classify_edge_catastrophe needs a path of length >= 5 for the "
            f"local 5-sample window, got length {n}"
        )

    grad = np.abs(np.diff(arr))
    # Use the midpoint of the smallest-gradient interval as the critical index.
    crit = int(np.argmin(grad))
    # Clamp so [crit-2, crit+3] stays in bounds.
    crit = max(2, min(crit, n - 3))
    window = arr[crit - 2 : crit + 3]
    return classify_curvature_1d(window, eps=eps)


def edge_catastrophe_label(
    src_energy: float,
    dst_energy: float,
    mid_samples: np.ndarray,
    *,
    eps: float = 1e-3,
) -> CatastropheLabel:
    """Classify an edge given endpoint energies and intermediate samples.

    Convenience wrapper that concatenates `[src_energy, *mid_samples, dst_energy]`
    and forwards to `classify_edge_catastrophe`. The combined path length must
    be >= 5, i.e. there must be at least 3 mid-samples.

    Parameters
    ----------
    src_energy:
        Energy at the source endpoint.
    dst_energy:
        Energy at the destination endpoint.
    mid_samples:
        1-D array of energies sampled strictly between source and destination.
    eps:
        Forwarded to the underlying classifier.

    Returns
    -------
    CatastropheLabel
        Result of `classify_edge_catastrophe` on the concatenated path.

    Raises
    ------
    ValueError
        If `mid_samples` is not 1-D or the combined path is shorter than 5.
    """
    mids = np.asarray(mid_samples, dtype=float)
    if mids.ndim != 1:
        raise ValueError(
            f"edge_catastrophe_label expects 1-D mid_samples, got shape {mids.shape}"
        )
    path = np.concatenate(([float(src_energy)], mids, [float(dst_energy)]))
    return classify_edge_catastrophe(path, eps=eps)


# --- legacy stub-prior surface (preserved) ------------------------------------


def get_edge_label(fsm: GraphFSM, src: str, dst: str) -> CatastropheLabel:
    """Look up the catastrophe label for an edge.

    Phase 2: returns NONE unless the FSM Edge model carries a recognised label
    in its `label` field (e.g. "fold"). Edges with unknown labels return NONE
    silently. If the transition is not legal, also returns NONE.

    Parameters
    ----------
    fsm:
        A loaded GraphFSM instance whose underlying spec provides the edge list.
    src:
        Source vertex id.
    dst:
        Destination vertex id.

    Returns
    -------
    CatastropheLabel
        The label for the matching edge, or NONE when the edge is absent,
        un-labelled, or carries an unrecognised label string.
    """
    if not fsm.is_legal_transition(src, dst):
        return CatastropheLabel.NONE
    for edge in fsm._spec.edges:
        if edge.source == src and edge.target == dst:
            label = getattr(edge, "label", None)
            if label is None:
                return CatastropheLabel.NONE
            try:
                return CatastropheLabel(label)
            except ValueError:
                return CatastropheLabel.NONE
    return CatastropheLabel.NONE


def edge_priors(label: CatastropheLabel) -> dict[str, float]:
    """Default sigma-weighting priors for an edge with the given catastrophe label.

    Phase 2 stub: FOLD/CUSP/SWALLOWTAIL/BUTTERFLY edges contribute a small
    additive bias to the singularity score; NONE contributes zero. Real
    numeric priors are an open question (q02-energy-function-spec).

    Parameters
    ----------
    label:
        The catastrophe label of the edge.

    Returns
    -------
    dict[str, float]
        A mapping with at least the key "bias" (additive singularity prior).
    """
    if label == CatastropheLabel.NONE:
        return {"bias": 0.0}
    if label == CatastropheLabel.FOLD:
        return {"bias": 0.05}
    if label == CatastropheLabel.CUSP:
        return {"bias": 0.10}
    if label == CatastropheLabel.SWALLOWTAIL:
        return {"bias": 0.15}
    if label == CatastropheLabel.BUTTERFLY:
        return {"bias": 0.20}
    return {"bias": 0.0}
