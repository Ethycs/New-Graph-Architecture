# Labelled Hypergraph with Residual: Owning the Semantic Gap in the Data Structure

**Status:** accepted-2026-05-14 — Phase 26 atoms + schema shipped (uncommitted on `main` at audit time; sources, atom docs, and acceptance-bar tests all present)
**Phase target:** Phase 26 (atoms + schema), prerequisite to Phase 27 (geometric reconstruction) and Phase 28 (SAE plug-in)
**Date posted:** 2026-05-14
**Author:** TPN research team

## Outcome — 2026-05-14

Accepted and built. The six acceptance bars (A1–A6) are exercised by:

- [tests/unit/test_labelled_hypergraph.py](../../tests/unit/test_labelled_hypergraph.py) (A1 round-trip, A2 projection, A6 hypergraph construction)
- [tests/unit/test_kl_regime_signature.py](../../tests/unit/test_kl_regime_signature.py) (A4 KL signature)
- [tests/unit/test_sae_adapter.py](../../tests/unit/test_sae_adapter.py) (A5 named/residual split)
- [tests/unit/test_decision_trace_jsonl_v12.py](../../tests/unit/test_decision_trace_jsonl_v12.py) (A3 schema compat)

Atoms shipped:

- [src/nga/arch/labelled_hypergraph.py](../../src/nga/arch/labelled_hypergraph.py) + [docs/arch/graph/labelled-hypergraph.md](../arch/graph/labelled-hypergraph.md)
- [src/nga/arch/kl_regime_signature.py](../../src/nga/arch/kl_regime_signature.py) + [docs/arch/graph/kl-regime-signature.md](../arch/graph/kl-regime-signature.md)
- [src/nga/arch/sae_adapter.py](../../src/nga/arch/sae_adapter.py) + [docs/arch/substrate/sae-adapter.md](../arch/substrate/sae-adapter.md)
- Schema bump to v1.2 in [src/nga/drivers/decision_trace_jsonl.py](../../src/nga/drivers/decision_trace_jsonl.py) (`DECISION_TRACE_SCHEMA_VERSION = "1.2"`) with the four new fields (`regime_named_label`, `regime_residual_features`, `regime_kl_signature_hash`, `feature_delta_at_transition`).

Unblocks Phase 27 (marching simplices + $d_{\text{eff}}$) and Phase 28 (real SAE plug-in).

## Abstract

We propose lifting the PCG-X regime graph from a discrete graph (regimes as opaque labels, transitions as binary edges) to a **labelled hypergraph with residual**: each regime carries (i) a canonical statistical fingerprint (KL-distance signature over its output conditional), (ii) a named label part listing the human-interpretable features active in it, and (iii) a residual list naming the mathematically-canonical-but-unnamed features. Each transition becomes a hyperedge carrying the feature-delta across the boundary. The structure makes the semantic gap an explicit, auditable property of the data — not an unspoken caveat. The motivation is the conversation thread documented in `docs/interpretability-push.md`: combining the project's master theorem (Whitney stratification + stratified partition function) with information-geometric K-choice and SAE-based monosemantic features yields a complete, drawable, audit-grade map of substrate computation, but only if the regime graph's representation can carry "what is named" and "what remains canonical but unnamed" separately. The current discrete graph cannot.

## Background

`docs/interpretability-push.md` records the move from "PCG-X extracts a discrete control graph" to "PCG-X is the discrete shadow of a Whitney stratification." The push surfaced three obstructions stacked on each other:

1. **Geometric → behavioural quotient** (Layer 1). The substrate's intrinsic stratification is finer than PCG-X's K = V extraction. Information geometry's KL-divergence quotient resolves this in principle.
2. **Polysemanticity** (Layer 2, partial). Substrate features encode multiple concepts under superposition; SAEs decompose them into monosemantic basis. Resolvable engineering.
3. **Semantic labelling** (Layer 2, irreducible). Even monosemantic features need human-supplied names; refinement does not shrink this.

The discrete regime graph as currently extracted cannot represent any of (1)-(3) cleanly. A regime is a single opaque label; transitions are binary edges; whatever the regime "is about" mathematically is collapsed into its name. This proposal lifts the data structure to make the three layers explicit.

## Central commitment

A regime is **not** a single opaque label. It is a triple:

- **Canonical signature** — a KL-divergence-based fingerprint of the regime's conditional output distribution. Invariant under reparametrisation; defines regime identity intrinsically.
- **Named label** — typed coordinates (FSM state, dominant SAE features, etc.) that *we can describe in human terms*.
- **Residual** — feature axes (e.g. SAE feature IDs) that are *mathematically canonical and behaviourally present* but lack a human-natural name.

A transition is **not** a binary edge. It is a hyperedge carrying the feature-delta — which features turned on, which turned off, and (when marching-cubes reconstruction is available) the geometric boundary hyperplane crossed.

The system's interpretability claim becomes calibrated by what each regime's `named` vs `residual` cardinality is. Two regimes with rich `named` and small `residual` are highly interpreted; two regimes with sparse `named` and large `residual` are honestly under-interpreted, and the structure says so on its face.

## What works immediately (no experiment needed)

These are licensed by composition and do not require validation:

1. **KL divergence is a coordinate-free measure of distribution distance.** Standard information geometry. Two regimes with conditional distributions $p_{\lambda_1}$ and $p_{\lambda_2}$ have a well-defined $\mathrm{KL}(p_{\lambda_1} \| p_{\lambda_2})$ regardless of how the substrate was parametrised.
2. **The hypergraph reduces to the current discrete graph by projection.** Forget `feature_delta`, `boundary_geometry`, `named`, `residual`, and keep only `(src_regime_id, dst_regime_id)`. The current PCG-X output is the strict projection of the proposed structure; backward compatibility is automatic.
3. **The schema extension is additive.** Adding fields with `None` defaults to `DecisionTraceRecord` does not break existing readers; the existing version 1.1 bump pattern is the precedent.
4. **The data structure composes with the stratified partition function.** $Z = \sum_\lambda \int_{S_\lambda / H} e^{-E/T} d\mu_\lambda$ over regimes is unchanged; the hypergraph just gives the sum indices their full structure.
5. **The data structure composes with the master theorem.** Each `Regime` ↔ one stratum (or behavioural-quotient of several); each `Hyperedge` ↔ one stratum-boundary transition; the named/residual decomposition is the explicit split of the substrate's local coordinates into human-readable and not.

## What requires experimental validation

1. Whether KL-fingerprint clustering (with a principled threshold) recovers the existing K = V partition on synthetic grammars, or surfaces a finer / coarser principled K.
2. Whether the named/residual split with a real SAE (e.g. Anthropic's released GPT-2 SAE features) produces a regime structure where named-feature cardinality scales sub-linearly with substrate width (the polysemanticity claim).
3. Whether hyperedge `feature_delta` is consistent across runs at fixed seed (canonical) vs across seeds (substrate-specific). Identifiability vs reproducibility.
4. Whether the schema extension's storage cost stays bounded (per-step JSONL grows by ~3-5 small dicts).
5. Whether downstream ControlPolicy decisions can use the named/residual split — does recovery via "match named features of nearest legal regime" outperform discrete BFS on the synthetic adversarial benchmark?

These are post-build validations; the proposed work delivers the atoms and schema, not the validations.

## Methods

The deliverable is six items:

### 1. `src/nga/arch/labelled_hypergraph.py`

Three dataclasses with pydantic-compatible serialisation:

- `Regime`: `regime_id: str`, `canonical_signature: np.ndarray`, `named: dict[str, str]`, `residual: list[str]`, `p_lambda: float | None`, `support_count: int`.
- `Hyperedge`: `src_regime_id: str`, `dst_regime_id: str`, `feature_delta: dict[str, float]`, `boundary_geometry: dict | None`, `sigma_at_crossing: float | None`, `beta_posterior: tuple[float, float] | None`, `traversal_count: int`.
- `LabelledHypergraph`: `regimes: dict[str, Regime]`, `hyperedges: list[Hyperedge]`, `metadata: dict`. Methods: `from_pcg_graph(...)` constructor, `as_discrete_graph()` projection (returns `(V, E)` tuple matching current PCG-X output), `to_json()` / `from_json()` round-trip.

The constructor accepts a PCG-X discrete graph + per-regime activation samples + an optional SAE adapter + an optional KL-signature computer, and produces the full labelled-hypergraph. Each optional input degrades gracefully: missing SAE means `named` and `residual` are empty; missing KL computer means `canonical_signature` is the zero vector.

### 2. `src/nga/arch/kl_regime_signature.py`

A small, testable computer for the KL-canonical fingerprint:

- `KLRegimeSignature.compute(per_regime_conditionals: dict[str, np.ndarray]) -> dict[str, np.ndarray]` — given the empirical output distribution per regime over a fixed support, returns one signature vector per regime (the per-regime KL-distances to every other regime, ordered by regime ID).
- `KLRegimeSignature.cluster(signatures, threshold) -> dict[str, str]` — equivalence-class regimes whose pairwise KL is below threshold, return regime → canonical-cluster mapping.
- `KLRegimeSignature.suggest_threshold(signatures, method='cramer_rao' | 'gap') -> float` — pick a principled threshold by the substrate's noise floor (Cramér-Rao on the per-regime estimator) or by a gap in the empirical KL distribution.

### 3. `src/nga/arch/sae_adapter.py`

A clean interface over a sparse feature decomposer with two implementations:

- `SAEAdapter` (Protocol): `encode(activations: np.ndarray) -> SparseFeatureCode`, `feature_labels() -> dict[int, str | None]`. Implementations register which features have human-supplied labels.
- `IdentitySAEAdapter` — default no-op. Each substrate dimension is its own feature; no labels. Useful for tests and for substrate-as-its-own-basis fallback.
- `MockLabelledSAEAdapter` — wraps an arbitrary mapping `feature_id -> label`. For unit tests and for substrates where we have a partial label dictionary.
- (Future) `PretrainedSAEAdapter` — wraps a downloaded SAE from Anthropic / OpenAI release. Phase 28 deliverable; not in this proposal's scope.

`SparseFeatureCode` is a dataclass: `active_features: list[int]`, `activations: np.ndarray`, `total_features: int`.

### 4. Schema extension to `src/nga/drivers/decision_trace_jsonl.py`

Bump to schema 1.2. New fields, all defaulting to None for backward compatibility:

- `regime_named_label: dict[str, str] | None` — the regime's named coordinates at this step (e.g. `{"fsm_state": "q_open", "sae_dominant": "f_142_eiffel"}`).
- `regime_residual_features: list[str] | None` — the regime's residual feature support at this step.
- `regime_kl_signature_hash: str | None` — short hash of the regime's canonical signature, joining traces to a hypergraph snapshot file.
- `feature_delta_at_transition: dict[str, float] | None` — features that turned on / off across this step's transition.

The existing `output_node_tuple` and `axis_node_ids` fields remain unchanged; the new fields complement them by carrying the named/residual decomposition the node-tuple coordinates do not.

### 5. `docs/arch/labelled-hypergraph.md` (atom doc)

Standard atom-doc format: one-line definition, interface contract, when to use, when not to, links to consumers.

### 6. `docs/model-class.md` update

Add a sixth commitment to the TPN's load-bearing-list:

> **6. Regime-graph carries named labels + residual, separately.** The system's regime graph is a hypergraph where each regime explicitly records its canonical statistical identity (KL fingerprint), its human-named label part (typed coordinates), and its residual feature support (mathematically canonical, semantically unnamed). The data structure owns the semantic gap rather than hiding it; the interpretability claim is calibrated by what is in `named` vs `residual` at each regime.

Plus a `LabelledHypergraph` entry in the formal definition's tuple.

## Validation protocol

Acceptance for the atom-level build (this proposal):

- **A1.** Round-trip test: a `LabelledHypergraph` serialised to JSON and back is bitwise equal under canonical encoding.
- **A2.** Projection test: `LabelledHypergraph.as_discrete_graph()` on a hypergraph built from a known PCG-X output recovers the original discrete graph exactly.
- **A3.** Schema-compat test: every existing decision-trace consumer reads a v1.2 row with the new fields populated; a v1.1 row reads in v1.2 with new fields defaulted to None.
- **A4.** KL-signature test: on a hand-constructed two-regime case with distributions $p_1 = (0.9, 0.1)$ and $p_2 = (0.1, 0.9)$, the signature distance matches the analytic $\mathrm{KL}(p_1 \| p_2)$ to numerical tolerance.
- **A5.** SAE-adapter test: `MockLabelledSAEAdapter` with a partial label dictionary correctly splits an activation into named features (those in the dictionary) and residual features (those not).
- **A6.** Hypergraph construction test: building a hypergraph from a Phase 24 / E30 regime-graph fixture (existing artefact) produces a well-formed structure with expected regime count, edge count, and per-regime KL signatures.

Each acceptance bar is a single test under `tests/unit/`. The atom census test must register the four new atoms.

## Pre-registered out-of-scope

This proposal does not include:

- Real SAE training or downloading (Phase 28).
- Wiring `LabelledHypergraph` into E28 / E30 runners (separate proposal once the structure is validated).
- Marching-cubes geometric reconstruction (Phase 27, separate proposal in `interpretability-push.md`).
- ControlPolicy upgrades to use the named/residual split (Phase 29+).

The proposal is a scaffolding step. Its success is "the new atoms exist, are tested, and can be composed with the existing PCG-X output without regressions." Its value is realised by the proposals that follow.

## Sudden large implications (if accepted)

- The TPN's load-bearing commitment count goes from 5 to 6. The architectural contract changes.
- `decision_trace.jsonl` becomes a richer audit artefact and forward-compatibility is preserved by additive schema.
- Every downstream interpretability claim becomes calibrated per regime by `len(named)` vs `len(residual)` — no more uncalibrated "PCG-X is interpretable" assertions.
- The path to publishable "complete structural interpretability + honest semantic gap accounting" goes from "narrative we tell" to "structure the code emits."

## Risks and limitations

- **Schema bloat.** Each decision-trace row grows by ~3-5 fields. Storage and parse cost increase by ~10-20% per row. Acceptable.
- **Empty residual on synthetic grammars.** With no SAE plugged in, `residual` is always empty and the structure looks redundant. This is expected; the residual fills in once Phase 28 lands an SAE. The structure must exist *before* that to receive it.
- **KL signature is O(K²) in regime count.** For K = 40 (current Phase 24 max) this is 1600 entries — trivial. For very fine future K (state × token), may need compression. Out of scope.
- **The `named` dictionary's keys are not standardised** across substrates. We use `fsm_state`, `sae_dominant`, `sae_topk_supports` as conventions but do not formally define them. Convention emerges from use; lock down once it has settled.

## Timeline

- Atoms + schema: 1 day (this work).
- Atom doc + model-class update: 1 hour.
- Tests passing on the existing atom census: 1-2 hours debug + verify.
- Total: ~1 working day for the build.

## Decision criterion

Accept and merge if:

- All six acceptance bars (A1-A6) pass.
- Existing test suite is green (no regression).
- Atom census recognises the four new atoms (`labelled_hypergraph`, `kl_regime_signature`, `sae_adapter`, schema bump).
- `docs/model-class.md` reflects the sixth commitment.

Reject if:

- Any acceptance bar fails after honest debugging.
- The schema bump breaks any existing trace consumer.
- The structure cannot be projected back to the current PCG-X discrete graph without information loss (A2).

## Connection to the wider plan

| Phase | Deliverable | Depends on |
|---|---|---|
| **Phase 26 (this proposal)** | Labelled-hypergraph atom + schema extension | — |
| Phase 27 (in `interpretability-push.md`) | $d_{\text{eff}}$ measurement + marching simplices + per-stratum jet fit | Phase 26's hypergraph as the carrier |
| Phase 28 (sketched in `interpretability-push.md`) | SAE plug-in fills `named`/`residual` from real monosemantic features | Phase 26's `sae_adapter` interface |
| Phase 29+ | ControlPolicy uses named-feature recovery; geometric audit trace | All of the above |

Phase 26 unlocks Phases 27-29. Building it first is the bottleneck removal.
