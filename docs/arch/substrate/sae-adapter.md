# SAE Adapter

**Cluster:** arch
**Status:** implemented (Phase 26 — interface + identity/mock defaults; Phase 28 will land a real pretrained SAE)
**Tags:** #sae #features #polysemanticity #interpretability #substrate

## What

A thin interface over a sparse feature decomposer (e.g. a pretrained sparse autoencoder) that produces, per activation vector, a sparse code over a feature dictionary together with a partition of the active features into:

- **named** — features that have a human label looked up from an external dictionary.
- **residual** — features that fire but lack a label.

Ships two default implementations:

- `IdentitySAEAdapter` — substrate is its own basis. Each dimension is a feature; no labels exist; every active feature is residual. Used when no SAE has been trained on the substrate yet.
- `MockLabelledSAEAdapter` — wraps an arbitrary `feature_id -> label` dictionary; otherwise identical to identity behaviour. Used for tests and for substrates with a partial labelling.

The Protocol `SAEAdapter` defines the contract any wrapper must satisfy. A future `PretrainedSAEAdapter` (Phase 28) will wrap a downloaded SAE checkpoint (Anthropic's released GPT-2 SAE or equivalent); the contract above is what it must respect.

## Why

Polysemanticity is the load-bearing obstruction at the semantic layer of regime interpretation: a single substrate dimension typically encodes multiple unrelated concepts under superposition. SAEs decompose the activation into an overcomplete sparse basis where features become monosemantic at high rates. Plugging an SAE into PCG-X via this interface is what populates the `named` / `residual` split on every regime. See [`docs/interpretability-push.md`](../../interpretability-push.md) for the conceptual motivation.

The Protocol-and-defaults design is deliberate: the system never *requires* an SAE to be wired in. Without one, `IdentitySAEAdapter` makes every active dimension residual, which honestly says "we have no monosemantic labels yet" rather than fabricating fake structure. The system's interpretability claim is calibrated by what each regime's `named` cardinality actually is.

## Interface

```python
@dataclass
class SparseFeatureCode:
    active_features: list[int]
    activations: np.ndarray
    total_features: int
    named: list[str]
    residual: list[str]
    feature_labels: dict[str, str]

    def sparsity(self) -> float: ...

class SAEAdapter(Protocol):
    def encode(self, activations: np.ndarray) -> SparseFeatureCode: ...
    def feature_labels(self) -> dict[int, str]: ...

class IdentitySAEAdapter:
    def __init__(self, dim: int, top_k: int | None = None) -> None: ...

class MockLabelledSAEAdapter:
    def __init__(self, dim: int, labels: dict[int, str], top_k: int | None = None) -> None: ...
```

Both default implementations accept `top_k` for selecting the top-magnitude features as active (substrate-as-its-own-basis sparsification). `top_k=None` means "all dimensions are active."

## Build steps

For each regime, after the discrete graph has been extracted:

1. Compute the regime's mean activation centroid over its support set.
2. `adapter.encode(centroid) -> SparseFeatureCode`.
3. Populate `Regime.named = {axis: label for axis, label in code.feature_labels.items()}` plus any other named coordinates (e.g. `fsm_state`).
4. Populate `Regime.residual = code.residual`.

For each hyperedge:

1. Compute source and destination centroids.
2. Encode both, take the difference between their activation profiles by feature ID.
3. Threshold to keep only meaningfully-changed features; this is `feature_delta`.

## Edge cases

- Activation dim mismatch raises `ValueError`.
- `top_k >= dim` falls back to "all features active."
- Labels for feature IDs outside `[0, dim)` are rejected at construction.

## Links

- **See also:**
  - [Labelled Hypergraph](../graph/labelled-hypergraph.md) — primary consumer.
  - [KL Regime Signature](../graph/kl-regime-signature.md) — sibling K-choice machinery.
  - [Frozen Encoder Backbone](frozen-encoder-backbone.md) — supplies the activations that get encoded.
- **Drives:** the `named` / `residual` decomposition on every `Regime`; the `feature_delta` on every `Hyperedge`.

## Reference

- Proposal: [`docs/proposals/labelled-hypergraph.md`](../../proposals/labelled-hypergraph.md).
- Conceptual motivation: [`docs/interpretability-push.md`](../../interpretability-push.md) §"Is the obstruction polysemanticity?"
