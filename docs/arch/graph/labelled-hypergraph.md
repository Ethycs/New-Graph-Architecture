# Labelled Hypergraph

**Cluster:** arch
**Status:** implemented (Phase 26)
**Tags:** #regime-graph #pcg-x #interpretability #semantic-gap #information-geometry #sae

## What

The labelled hypergraph lifts PCG-X's discrete regime graph $(V, E)$ to a structure that explicitly carries the **canonical statistical identity** of each regime (a KL-distance signature), its **named coordinates** (FSM state, dominant SAE features, anything we can describe in human terms), and its **residual feature support** (mathematically canonical features that fire but lack a human label). Each transition is a **hyperedge** carrying the feature-delta crossed at that boundary.

```
Regime:
    regime_id              str
    canonical_signature    np.ndarray   # KL-distance fingerprint
    named                  dict[str,str]  # human-labelled coordinates
    residual               list[str]      # unnamed-but-canonical features
    p_lambda               float | None   # stratified-partition probability
    support_count          int

Hyperedge:
    src_regime_id, dst_regime_id   str
    feature_delta                  dict[str, float]  # signed per-feature change
    boundary_geometry              dict | None       # marching-cubes hyperplane
    sigma_at_crossing              float | None
    beta_posterior                 (alpha, beta) | None
    traversal_count                int
```

The current PCG-X output is the strict projection of this structure: forget `named`, `residual`, `canonical_signature`, `feature_delta`, `boundary_geometry`, `beta_posterior` and you recover the `(regime_ids, edges)` tuple PCG-X already emits. Backward compatibility is automatic; the discrete graph is what `LabelledHypergraph.as_discrete_graph()` returns.

## Why

The previous discrete regime graph conflated three different things into a single opaque label. A regime "regime_7" silently bundled:

1. The regime's intrinsic statistical identity (what conditional distribution it produces over outputs).
2. The human-meaningful label we attached ("the model is tracking an open bracket").
3. The mathematically-canonical-but-unnamed feature support (whatever SAE features are active and we have not labelled).

These are different and need to stay separate. The labelled hypergraph owns each one explicitly: canonical identity is data (the KL signature), named label is human projection (`named` dict), residual is honest opacity (`residual` list). Interpretability per regime is then a *measurable property* of the data — `len(named) / (len(named) + len(residual))` — not a rhetorical claim.

This is the data-structure side of the project's master theorem (Whitney stratification + the stratified partition function in `Mathematics.md`). Each regime ↔ one stratum (or behavioural-quotient of several after the KL-quotient); each hyperedge ↔ one stratum-boundary transition; σ-as-smooth-scalar-field fires at $\Sigma$ crossings. See [`docs/interpretability-push.md`](../../interpretability-push.md) for the conceptual story.

## Interface

**Three dataclasses** in `nga.arch.labelled_hypergraph`:

- `Regime` — one node. `signature_hash(length=12)` returns a stable short hex hash for joining to decision traces. `to_dict()` / `from_dict()` for serialisation.
- `Hyperedge` — one edge. `to_dict()` / `from_dict()` for serialisation.
- `LabelledHypergraph` — the whole structure. Key methods:

```python
LabelledHypergraph.from_pcg_graph(
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
    metadata: dict | None = None,
) -> LabelledHypergraph

graph.as_discrete_graph() -> (list[str], list[tuple[str, str]])
graph.interpretation_coverage() -> dict[str, float]    # named / (named + residual) per regime
graph.to_json(sort_keys=True) -> str
LabelledHypergraph.from_json(payload: str) -> LabelledHypergraph
graph.hash(length=12) -> str
```

Each optional input degrades gracefully: omit `canonical_signatures` and every regime gets a length-0 signature; omit `named_labels` and every regime gets an empty `named` dict. The hypergraph is well-formed with any subset of enrichment supplied.

## Build steps

1. Construct via `from_pcg_graph` with whatever enrichment is available. The minimum input is `regime_ids`.
2. Compute canonical signatures with [`KLRegimeSignature.compute`](kl-regime-signature.md) from per-regime empirical conditionals.
3. Encode each regime's activation centroid through an [`SAEAdapter`](../substrate/sae-adapter.md) to split active features into `named` / `residual`.
4. Compute `feature_delta_at_transition` by encoding source and destination centroids and reporting feature-wise differences above threshold.
5. Optionally lift Beta posteriors from `posterior_mask` onto hyperedges.
6. Serialise with `to_json()` for a per-run `hypergraph.json` artefact next to `decision_trace.jsonl`.

## Decision-trace schema

Schema 1.2 of `decision_trace.jsonl` adds four additive fields that record the per-step labelled-hypergraph view:

- `regime_named_label: dict[str, str] | None`
- `regime_residual_features: list[str] | None`
- `regime_kl_signature_hash: str | None`
- `feature_delta_at_transition: dict[str, float] | None`

All default to `None` for backward compatibility with v1.0 and v1.1 runners. A v1.2 runner populating these fields joins to the run-level `hypergraph.json` via `regime_kl_signature_hash`.

## Links

- **See also:**
  - [KL Regime Signature](kl-regime-signature.md) — coordinate-free KL signatures and threshold-based equivalence clustering.
  - [SAE Adapter](../substrate/sae-adapter.md) — sparse feature decomposition with named/residual split.
  - [Decision Trace JSONL](../../drivers/decision-trace-jsonl.md) — schema 1.2 records.
  - [Stratified Partition Function](../energy/stratified-partition-function.md) — `p_lambda` per regime lifts from here.
  - [Bisimulation Quotient](bisimulation-quotient.md) — the prior discrete-graph quotient; superseded by KL-quotient under information geometry.
- **Drives:** Phase 27 marching-simplices reconstruction, Phase 28 SAE plug-in, geometric audit trace.
- **Driven by:** the PCG-X regime extractor (`e28_pcg_extractor`, `e30_pcg_extractor_pretrained`).

## Reference

- Proposal: [`docs/proposals/labelled-hypergraph.md`](../../proposals/labelled-hypergraph.md).
- Conceptual story: [`docs/interpretability-push.md`](../../interpretability-push.md).
- Master theorem: `Mathematics.md` §"Whitney Stratification and Group-Action Orbit Types" (line 1287+).
