"""Labelled hypergraph with residual: regime structure that owns its semantic gap.

A regime is not a single opaque label. It is a triple:

  - canonical signature   - a coordinate-free fingerprint of the regime's
                            output conditional distribution (typically a
                            row of the per-regime KL-distance matrix).
                            Defines regime identity intrinsically.
  - named                 - typed coordinates (FSM state, dominant SAE
                            features, ...) that we can describe in human
                            terms.
  - residual              - feature axes that are mathematically canonical
                            and behaviourally present but lack a human-natural
                            name.

A transition is not a binary edge. It is a hyperedge carrying the
feature-delta across the boundary (which features turned on, which turned
off, optional geometric boundary, optional Beta(alpha, beta) posterior).

The current PCG-X discrete graph is the strict projection of the labelled
hypergraph: forget named/residual/canonical_signature/feature_delta and you
get the (V, E) tuple back unchanged. Backward compatibility is automatic.

Dependencies: numpy + stdlib only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "Regime",
    "Hyperedge",
    "LabelledHypergraph",
]


@dataclass
class Regime:
    """One regime in the labelled hypergraph.

    Fields
    ------
    regime_id:
        Stable string identifier (e.g. ``"regime_7"`` or ``"L10_regime_7"``).
    canonical_signature:
        Coordinate-free fingerprint of the regime's output conditional. A
        1-D ``np.ndarray`` whose entries are typically the per-regime KL
        distances to every other regime, ordered by regime ID. Zero-length
        is permitted (no signature computed yet); higher-dim arrays are
        rejected.
    named:
        Human-readable coordinates active in this regime. Keys are axis
        names by convention (e.g. ``"fsm_state"``, ``"sae_dominant"``);
        values are the labels at this regime.
    residual:
        Feature IDs (e.g. SAE feature indices as strings) that fire in this
        regime but lack a human label.
    p_lambda:
        Optional stratified-partition probability ``P(lambda) = Z_lambda / Z``.
        ``None`` when the partition has not been computed.
    support_count:
        Number of training-corpus steps that landed in this regime.
    """

    regime_id: str
    canonical_signature: np.ndarray = field(default_factory=lambda: np.zeros(0))
    named: dict[str, str] = field(default_factory=dict)
    residual: list[str] = field(default_factory=list)
    p_lambda: float | None = None
    support_count: int = 0

    def __post_init__(self) -> None:
        sig = np.asarray(self.canonical_signature, dtype=float)
        if sig.ndim > 1:
            raise ValueError(
                "canonical_signature must be 0-d or 1-d; "
                f"got ndim={sig.ndim}, shape={sig.shape}"
            )
        self.canonical_signature = sig

    def signature_hash(self, length: int = 12) -> str:
        """Short hex hash of the canonical signature. Stable across runs."""
        arr = np.ascontiguousarray(self.canonical_signature, dtype=np.float64)
        h = hashlib.sha256(arr.tobytes()).hexdigest()
        return h[:length]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "regime_id": self.regime_id,
            "canonical_signature": self.canonical_signature.tolist(),
            "named": dict(self.named),
            "residual": list(self.residual),
            "p_lambda": self.p_lambda,
            "support_count": int(self.support_count),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Regime:
        """Inverse of ``to_dict``."""
        return cls(
            regime_id=str(data["regime_id"]),
            canonical_signature=np.asarray(
                data.get("canonical_signature", []), dtype=float
            ),
            named=dict(data.get("named", {})),
            residual=list(data.get("residual", [])),
            p_lambda=data.get("p_lambda"),
            support_count=int(data.get("support_count", 0)),
        )


@dataclass
class Hyperedge:
    """One hyperedge in the labelled hypergraph.

    Generalises a binary graph edge by carrying the feature-delta across
    the boundary. Reducing a ``Hyperedge`` to ``(src_regime_id, dst_regime_id)``
    recovers the discrete-graph edge unchanged.

    Fields
    ------
    src_regime_id, dst_regime_id:
        Endpoint regime identifiers.
    feature_delta:
        Map ``feature_id -> signed magnitude`` describing which features
        changed across the transition. Positive values denote features that
        turned on (gained activation in ``dst`` relative to ``src``);
        negative values denote features that turned off.
    boundary_geometry:
        Optional dict carrying a marching-cubes-fitted boundary description
        (e.g. ``{"normal": [...], "offset": float}`` for a hyperplane).
        ``None`` when no geometric reconstruction is available.
    sigma_at_crossing:
        Optional aggregate sigma observed at the empirical crossing event,
        or the average over recorded crossings.
    beta_posterior:
        Optional ``(alpha, beta)`` tuple — the Beta posterior parameters on
        this transition's legality, lifted from the existing posterior_mask
        atom when available.
    traversal_count:
        Number of corpus steps observed traversing this hyperedge.
    """

    src_regime_id: str
    dst_regime_id: str
    feature_delta: dict[str, float] = field(default_factory=dict)
    boundary_geometry: dict[str, Any] | None = None
    sigma_at_crossing: float | None = None
    beta_posterior: tuple[float, float] | None = None
    traversal_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "src_regime_id": self.src_regime_id,
            "dst_regime_id": self.dst_regime_id,
            "feature_delta": {k: float(v) for k, v in self.feature_delta.items()},
            "boundary_geometry": self.boundary_geometry,
            "sigma_at_crossing": self.sigma_at_crossing,
            "beta_posterior": (
                list(self.beta_posterior) if self.beta_posterior is not None else None
            ),
            "traversal_count": int(self.traversal_count),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hyperedge:
        """Inverse of ``to_dict``."""
        beta = data.get("beta_posterior")
        beta_tuple: tuple[float, float] | None
        if beta is None:
            beta_tuple = None
        else:
            beta_tuple = (float(beta[0]), float(beta[1]))
        return cls(
            src_regime_id=str(data["src_regime_id"]),
            dst_regime_id=str(data["dst_regime_id"]),
            feature_delta={
                str(k): float(v) for k, v in data.get("feature_delta", {}).items()
            },
            boundary_geometry=data.get("boundary_geometry"),
            sigma_at_crossing=data.get("sigma_at_crossing"),
            beta_posterior=beta_tuple,
            traversal_count=int(data.get("traversal_count", 0)),
        )


@dataclass
class LabelledHypergraph:
    """The labelled hypergraph that lifts PCG-X's discrete regime graph.

    Holds a mutable ``regimes`` dict (keyed by ``regime_id``) and a
    ``hyperedges`` list. Metadata records the substrate, layer harvested,
    SAE adapter identity, and KL threshold used to canonicalise regimes —
    enough for an analyst to reproduce the structure from raw activations.

    Construct via the explicit ``__init__`` or the ``from_pcg_graph`` class
    method. Project back to the current PCG-X discrete graph via
    ``as_discrete_graph``. Round-trip to JSON via ``to_json`` /
    ``from_json``.
    """

    regimes: dict[str, Regime] = field(default_factory=dict)
    hyperedges: list[Hyperedge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pcg_graph(
        cls,
        regime_ids: list[str],
        edges: list[tuple[str, str]] | None = None,
        *,
        edge_counts: dict[tuple[str, str], int] | None = None,
        canonical_signatures: dict[str, np.ndarray] | None = None,
        named_labels: dict[str, dict[str, str]] | None = None,
        residual_features: dict[str, list[str]] | None = None,
        feature_deltas: dict[tuple[str, str], dict[str, float]] | None = None,
        p_lambda: dict[str, float] | None = None,
        support_counts: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LabelledHypergraph:
        """Construct a ``LabelledHypergraph`` from PCG-X-style inputs.

        Each optional input degrades gracefully when missing: omitted
        canonical signatures default to length-0 arrays; omitted named
        labels default to empty dicts; omitted residual features default
        to empty lists. The result is a well-formed hypergraph regardless
        of how much enrichment is supplied.

        Parameters
        ----------
        regime_ids:
            All regime identifiers, in canonical order. Duplicates rejected.
        edges:
            Observed transitions as ``(src, dst)`` pairs. ``None`` means no
            edges (regimes-only).
        edge_counts:
            Optional per-edge traversal counts. Edges absent from this map
            default to count 1 (one observation).
        canonical_signatures:
            Optional per-regime KL-canonical signature vectors.
        named_labels:
            Optional per-regime ``axis_name -> label`` dicts.
        residual_features:
            Optional per-regime list of residual feature IDs.
        feature_deltas:
            Optional per-edge feature-delta dicts.
        p_lambda:
            Optional per-regime stratified-partition probability.
        support_counts:
            Optional per-regime corpus support count.
        metadata:
            Optional free-form metadata dict.
        """
        if len(regime_ids) != len(set(regime_ids)):
            raise ValueError("regime_ids must be unique")
        regimes: dict[str, Regime] = {}
        for rid in regime_ids:
            sig = (
                canonical_signatures.get(rid)
                if canonical_signatures is not None
                else None
            )
            named = named_labels.get(rid, {}) if named_labels is not None else {}
            residual = (
                residual_features.get(rid, [])
                if residual_features is not None
                else []
            )
            plam = p_lambda.get(rid) if p_lambda is not None else None
            count = support_counts.get(rid, 0) if support_counts is not None else 0
            regimes[rid] = Regime(
                regime_id=rid,
                canonical_signature=(
                    np.asarray(sig, dtype=float) if sig is not None else np.zeros(0)
                ),
                named=dict(named),
                residual=list(residual),
                p_lambda=plam,
                support_count=int(count),
            )

        hyperedges: list[Hyperedge] = []
        if edges is not None:
            id_set = set(regime_ids)
            for src, dst in edges:
                if src not in id_set or dst not in id_set:
                    raise ValueError(
                        f"Edge ({src!r}, {dst!r}) references unknown regime"
                    )
                delta = (
                    feature_deltas.get((src, dst), {})
                    if feature_deltas is not None
                    else {}
                )
                count = (
                    edge_counts.get((src, dst), 1)
                    if edge_counts is not None
                    else 1
                )
                hyperedges.append(
                    Hyperedge(
                        src_regime_id=src,
                        dst_regime_id=dst,
                        feature_delta={str(k): float(v) for k, v in delta.items()},
                        traversal_count=int(count),
                    )
                )

        return cls(
            regimes=regimes,
            hyperedges=hyperedges,
            metadata=dict(metadata) if metadata is not None else {},
        )

    # ------------------------------------------------------------------
    # Projection back to the discrete graph
    # ------------------------------------------------------------------

    def as_discrete_graph(self) -> tuple[list[str], list[tuple[str, str]]]:
        """Project to ``(regime_ids, edges)`` — the current PCG-X output shape.

        Forgets every enrichment (signatures, named/residual, deltas,
        geometry, posteriors). The returned tuple is bitwise equal to
        what ``from_pcg_graph`` was originally called with for ``regime_ids``
        and ``edges``, modulo ordering.
        """
        regime_ids = list(self.regimes.keys())
        edges = [(e.src_regime_id, e.dst_regime_id) for e in self.hyperedges]
        return regime_ids, edges

    # ------------------------------------------------------------------
    # Interpretability calibration
    # ------------------------------------------------------------------

    def interpretation_coverage(self) -> dict[str, float]:
        """Per-regime ratio ``len(named) / (len(named) + len(residual))``.

        Returns 1.0 for fully-named regimes, 0.0 for residual-only regimes.
        Regimes with neither named nor residual entries return ``nan``
        (no features to interpret either way). This is the explicit
        calibration of how much of each regime's structure is human-readable.
        """
        coverage: dict[str, float] = {}
        for rid, regime in self.regimes.items():
            named_n = len(regime.named)
            residual_n = len(regime.residual)
            total = named_n + residual_n
            if total == 0:
                coverage[rid] = float("nan")
            else:
                coverage[rid] = named_n / total
        return coverage

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, *, sort_keys: bool = True, indent: int | None = None) -> str:
        """Serialise to a canonical JSON string.

        With ``sort_keys=True`` (default) the same hypergraph always emits
        the same string — used for hash-based identity in
        ``LabelledHypergraph.hash``.
        """
        payload = {
            "regimes": [self.regimes[k].to_dict() for k in sorted(self.regimes)],
            "hyperedges": [
                e.to_dict()
                for e in sorted(
                    self.hyperedges,
                    key=lambda x: (x.src_regime_id, x.dst_regime_id),
                )
            ],
            "metadata": self.metadata,
        }
        return json.dumps(payload, sort_keys=sort_keys, indent=indent)

    @classmethod
    def from_json(cls, payload: str) -> LabelledHypergraph:
        """Inverse of ``to_json``."""
        data = json.loads(payload)
        regimes = {
            r["regime_id"]: Regime.from_dict(r) for r in data.get("regimes", [])
        }
        hyperedges = [Hyperedge.from_dict(e) for e in data.get("hyperedges", [])]
        metadata = dict(data.get("metadata", {}))
        return cls(regimes=regimes, hyperedges=hyperedges, metadata=metadata)

    def hash(self, length: int = 12) -> str:
        """Short hex hash of the canonical JSON form. Stable across runs."""
        canon = self.to_json(sort_keys=True, indent=None)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:length]
