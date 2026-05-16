"""SAE adapter: sparse feature decomposition with named/residual split.

A thin interface over a sparse feature decomposer (e.g. a pretrained sparse
autoencoder) that produces, for each activation vector, a sparse code over
some feature dictionary, together with a partition of the active features
into:

  - ``named``: features the analyst has a human label for (looked up from
              an external dictionary).
  - ``residual``: features that are active but unlabelled — mathematically
              canonical, semantically opaque.

This module ships two default implementations and a Protocol that real
SAE wrappers must satisfy:

  - ``IdentitySAEAdapter``: substrate-as-its-own-basis. Each substrate
    dimension is a feature; the top-k absolute-value dimensions are
    reported as active. Used when no SAE has been trained yet.
  - ``MockLabelledSAEAdapter``: wraps an arbitrary dimension dictionary
    (``feature_id -> label_or_None``). Used for tests and for substrates
    where a partial labelling exists.

A future ``PretrainedSAEAdapter`` (Phase 28) will wrap a downloaded SAE
checkpoint; this file does not implement it — the interface is what
matters here.

Dependencies: numpy + stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

__all__ = [
    "SparseFeatureCode",
    "SAEAdapter",
    "IdentitySAEAdapter",
    "MockLabelledSAEAdapter",
]


@dataclass
class SparseFeatureCode:
    """One activation's sparse feature decomposition.

    Fields
    ------
    active_features:
        Integer IDs of the active features (positions in the feature
        dictionary). ``len(active_features) == len(activations)``.
    activations:
        Floating-point activation magnitudes for each active feature,
        same order as ``active_features``.
    total_features:
        Size of the full feature dictionary (so callers can compute
        sparsity as ``len(active_features) / total_features``).
    named:
        Feature IDs (as strings) that have human labels. Equal to or a
        subset of ``active_features`` cast to strings.
    residual:
        Feature IDs (as strings) that are active but unlabelled.
    feature_labels:
        For each named feature, the human-readable label
        (``feature_id_as_str -> label``).
    """

    active_features: list[int]
    activations: np.ndarray
    total_features: int
    named: list[str] = field(default_factory=list)
    residual: list[str] = field(default_factory=list)
    feature_labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.active_features) != len(self.activations):
            raise ValueError(
                "active_features and activations must have the same length; "
                f"got {len(self.active_features)} vs {len(self.activations)}"
            )
        self.activations = np.asarray(self.activations, dtype=float)

    def sparsity(self) -> float:
        """Fraction ``len(active_features) / total_features`` in [0, 1]."""
        if self.total_features <= 0:
            return 0.0
        return len(self.active_features) / float(self.total_features)


class SAEAdapter(Protocol):
    """Protocol every SAE wrapper must satisfy.

    Implementations decompose a single ``activations`` vector into a sparse
    code, looking up which features have labels from an internal dictionary.
    """

    def encode(self, activations: np.ndarray) -> SparseFeatureCode:
        """Return the sparse code for one activation vector."""
        ...

    def feature_labels(self) -> dict[int, str]:
        """Return the full ``feature_id -> label`` dictionary.

        Only labelled features appear; unlabelled features have no entry.
        """
        ...


class IdentitySAEAdapter:
    """The substrate is its own basis. No features have labels.

    Useful as the fallback when no SAE has been trained on the substrate
    yet. Reports the top-``k`` absolute-value substrate dimensions as
    active features and emits an empty label dictionary, so every active
    feature is residual by construction.
    """

    def __init__(self, dim: int, top_k: int | None = None) -> None:
        """
        Parameters
        ----------
        dim:
            Substrate dimension (e.g. 768 for GPT-2 small).
        top_k:
            How many top-magnitude dimensions to report as active. ``None``
            means "all of them" — every dimension is active. A value like
            ``32`` gives sparse codes by absolute-value selection.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive when set; got {top_k}")
        self._dim = int(dim)
        self._top_k = top_k

    def encode(self, activations: np.ndarray) -> SparseFeatureCode:
        arr = np.asarray(activations, dtype=float).flatten()
        if arr.shape[0] != self._dim:
            raise ValueError(
                f"activations must have length {self._dim}; got {arr.shape[0]}"
            )
        if self._top_k is None or self._top_k >= self._dim:
            idx = np.arange(self._dim)
        else:
            idx = np.argpartition(np.abs(arr), self._dim - self._top_k)[-self._top_k :]
            idx = np.sort(idx)
        active = [int(i) for i in idx]
        vals = arr[idx]
        return SparseFeatureCode(
            active_features=active,
            activations=vals,
            total_features=self._dim,
            named=[],
            residual=[str(i) for i in active],
            feature_labels={},
        )

    def feature_labels(self) -> dict[int, str]:
        return {}


class MockLabelledSAEAdapter:
    """Wraps an explicit label dictionary; otherwise behaves like Identity.

    Parameters
    ----------
    dim:
        Substrate dimension.
    labels:
        ``feature_id -> label`` mapping for the labelled features. Features
        absent from this dict are considered unlabelled and routed into the
        residual list when active.
    top_k:
        Same as ``IdentitySAEAdapter`` — how many top-magnitude dimensions
        to report as active per encode call.
    """

    def __init__(
        self,
        dim: int,
        labels: dict[int, str],
        top_k: int | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive when set; got {top_k}")
        for fid in labels:
            if not (0 <= int(fid) < dim):
                raise ValueError(
                    f"label feature_id {fid} outside [0, {dim})"
                )
        self._dim = int(dim)
        self._top_k = top_k
        self._labels: dict[int, str] = {int(k): str(v) for k, v in labels.items()}

    def encode(self, activations: np.ndarray) -> SparseFeatureCode:
        arr = np.asarray(activations, dtype=float).flatten()
        if arr.shape[0] != self._dim:
            raise ValueError(
                f"activations must have length {self._dim}; got {arr.shape[0]}"
            )
        if self._top_k is None or self._top_k >= self._dim:
            idx = np.arange(self._dim)
        else:
            idx = np.argpartition(np.abs(arr), self._dim - self._top_k)[-self._top_k :]
            idx = np.sort(idx)
        active = [int(i) for i in idx]
        vals = arr[idx]
        named: list[str] = []
        residual: list[str] = []
        feature_labels: dict[str, str] = {}
        for fid in active:
            sfid = str(fid)
            if fid in self._labels:
                named.append(sfid)
                feature_labels[sfid] = self._labels[fid]
            else:
                residual.append(sfid)
        return SparseFeatureCode(
            active_features=active,
            activations=vals,
            total_features=self._dim,
            named=named,
            residual=residual,
            feature_labels=feature_labels,
        )

    def feature_labels(self) -> dict[int, str]:
        return dict(self._labels)
