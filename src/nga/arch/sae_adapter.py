"""SAE adapter: sparse feature decomposition with named/residual split.

A thin interface over a sparse feature decomposer (e.g. a pretrained sparse
autoencoder) that produces, for each activation vector, a sparse code over
some feature dictionary, together with a partition of the active features
into:

  - ``named``: features the analyst has a human label for (looked up from
              an external dictionary).
  - ``residual``: features that are active but unlabelled — mathematically
              canonical, semantically opaque.

This module ships three default implementations and a Protocol that real
SAE wrappers must satisfy:

  - ``IdentitySAEAdapter``: substrate-as-its-own-basis. Each substrate
    dimension is a feature; the top-k absolute-value dimensions are
    reported as active. Used when no SAE has been trained yet.
  - ``MockLabelledSAEAdapter``: wraps an arbitrary dimension dictionary
    (``feature_id -> label_or_None``). Used for tests and for substrates
    where a partial labelling exists.
  - ``PretrainedSAEAdapter`` (Phase 28): wraps a trained SAE checkpoint
    on disk (encoder weights + decoder weights + optional labels JSON).
    Used to plug a real trained SAE into the labelled-hypergraph build.

Dependencies: numpy + stdlib only. The PretrainedSAEAdapter loads a
checkpoint stored as a numpy ``.npz`` (no torch dependency at load time;
training is in ``scripts/phase28_train_sae.py`` and does require torch).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

__all__ = [
    "SparseFeatureCode",
    "SAEAdapter",
    "IdentitySAEAdapter",
    "MockLabelledSAEAdapter",
    "PretrainedSAEAdapter",
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


class PretrainedSAEAdapter:
    """Wraps a trained SAE checkpoint on disk.

    Checkpoint format (numpy ``.npz``):

      W_enc       (n_features, d_in)   encoder weight matrix
      b_enc       (n_features,)        encoder bias
      W_dec       (d_in, n_features)   decoder weight matrix (optional)
      b_dec       (d_in,)              decoder bias / pre-encoder subtraction (optional)

    SAE forward (encoder-only path):

      x = activations - b_dec     # pre-encoder subtraction; b_dec acts as
                                  # the SAE's "data-center"
      z = relu(W_enc @ x + b_enc) # raw sparse code

    Activation selection: features with ``z > activation_threshold`` are
    reported as active. Optionally also clamp to top-``top_k`` magnitudes.

    Labels JSON (companion file, optional): ``{"<feature_id>": "<label>", ...}``
    -- keys are stringified ints. Features not in the labels dict become
    residual (unlabelled but active).

    The adapter loads at construction time and caches encoder weights on
    CPU; encoding one activation is one matrix-vector product.
    """

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        labels_path: str | Path | None = None,
        activation_threshold: float = 0.0,
        top_k: int | None = None,
    ) -> "PretrainedSAEAdapter":
        """Load an SAE from a ``.npz`` checkpoint.

        Parameters
        ----------
        checkpoint_path:
            Path to the ``.npz`` file with at minimum ``W_enc`` and
            ``b_enc`` arrays. ``W_dec`` and ``b_dec`` are optional;
            ``b_dec`` defaults to zeros (so the activation is encoded
            without pre-encoder subtraction).
        labels_path:
            Optional path to a JSON file with ``{feature_id_str: label}``
            entries. If absent or set to None, the adapter has no
            labelled features and every active feature is residual.
        activation_threshold:
            Features whose post-ReLU activation exceeds this value count
            as active. Default 0.0 (any positive activation).
        top_k:
            Optional cap on the number of active features per encode.
            When set, the top-k by activation magnitude are kept (after
            the threshold filter).
        """
        ckpt = np.load(Path(checkpoint_path))
        if "W_enc" not in ckpt or "b_enc" not in ckpt:
            raise ValueError(
                f"checkpoint at {checkpoint_path} missing W_enc / b_enc"
            )
        W_enc = np.asarray(ckpt["W_enc"], dtype=np.float64)
        b_enc = np.asarray(ckpt["b_enc"], dtype=np.float64)
        if W_enc.ndim != 2:
            raise ValueError(f"W_enc must be 2D; got shape {W_enc.shape}")
        n_features, d_in = W_enc.shape
        if b_enc.shape != (n_features,):
            raise ValueError(
                f"b_enc shape {b_enc.shape} != (n_features,) = ({n_features},)"
            )
        b_dec = np.asarray(ckpt["b_dec"], dtype=np.float64) if "b_dec" in ckpt else np.zeros(d_in)
        if b_dec.shape != (d_in,):
            raise ValueError(
                f"b_dec shape {b_dec.shape} != (d_in,) = ({d_in},)"
            )

        labels: dict[int, str] = {}
        if labels_path is not None:
            raw = json.loads(Path(labels_path).read_text())
            for k, v in raw.items():
                fid = int(k)
                if not (0 <= fid < n_features):
                    raise ValueError(
                        f"label feature_id {fid} outside [0, {n_features})"
                    )
                labels[fid] = str(v)

        return cls(
            W_enc=W_enc,
            b_enc=b_enc,
            b_dec=b_dec,
            labels=labels,
            activation_threshold=float(activation_threshold),
            top_k=top_k,
        )

    def __init__(
        self,
        *,
        W_enc: np.ndarray,
        b_enc: np.ndarray,
        b_dec: np.ndarray,
        labels: dict[int, str] | None = None,
        activation_threshold: float = 0.0,
        top_k: int | None = None,
    ) -> None:
        if W_enc.ndim != 2:
            raise ValueError(f"W_enc must be 2D; got shape {W_enc.shape}")
        n_features, d_in = W_enc.shape
        if b_enc.shape != (n_features,):
            raise ValueError(
                f"b_enc shape {b_enc.shape} != ({n_features},)"
            )
        if b_dec.shape != (d_in,):
            raise ValueError(f"b_dec shape {b_dec.shape} != ({d_in},)")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive when set; got {top_k}")
        self._W_enc = np.asarray(W_enc, dtype=np.float64)
        self._b_enc = np.asarray(b_enc, dtype=np.float64)
        self._b_dec = np.asarray(b_dec, dtype=np.float64)
        self._n_features = int(n_features)
        self._d_in = int(d_in)
        self._labels: dict[int, str] = dict(labels or {})
        self._activation_threshold = float(activation_threshold)
        self._top_k = top_k

    @property
    def d_in(self) -> int:
        return self._d_in

    @property
    def n_features(self) -> int:
        return self._n_features

    def encode(self, activations: np.ndarray) -> SparseFeatureCode:
        arr = np.asarray(activations, dtype=np.float64).flatten()
        if arr.shape[0] != self._d_in:
            raise ValueError(
                f"activations must have length {self._d_in}; got {arr.shape[0]}"
            )
        # SAE encoder: z = relu(W_enc @ (h - b_dec) + b_enc).
        x = arr - self._b_dec
        pre = self._W_enc @ x + self._b_enc
        z = np.maximum(pre, 0.0)
        # Active features: above threshold, optionally top-k.
        active_mask = z > self._activation_threshold
        active_idx = np.where(active_mask)[0]
        if self._top_k is not None and active_idx.size > self._top_k:
            # Keep top-k by magnitude.
            magnitudes = z[active_idx]
            keep = np.argpartition(magnitudes, active_idx.size - self._top_k)[-self._top_k:]
            active_idx = np.sort(active_idx[keep])
        vals = z[active_idx]
        active = [int(i) for i in active_idx]
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
            total_features=self._n_features,
            named=named,
            residual=residual,
            feature_labels=feature_labels,
        )

    def feature_labels(self) -> dict[int, str]:
        return dict(self._labels)
