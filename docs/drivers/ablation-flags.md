# Ablation Flag Set

**Cluster:** driver
**Status:** spec
**Tags:** #driver #contract #ablation-matrix #boolean-knobs

## What

A set of boolean feature flags controlling which architectural components are enabled or disabled. Each flag corresponds to a major architectural piece (graph masking, typed scores, singularity detection, IDF weighting, hyperbolic geometry, group quotients, frozen reservoirs, trace history). The 10 ablation variants `A0`–`A9` are predefined tuples of these flags: `A0` = all enabled (full system), `A1` = no graph mask, etc. [CLI Runner](./cli-runner.md) loads the chosen tuple via [Config (YAML)](./config.md).ablation and applies the same flags identically to both the arch builder and the exp runners, ensuring ablation comparisons are valid.

## Why

Ablation studies isolate the contribution of each component. If the architecture disables one flag (e.g., `graph_mask_enabled=false`) but the experiment evaluates as if it were enabled, the ablation is invalid. A shared boolean flag matrix ensures both halves remove the same component. The flags are the operational mechanism for ablation: each flag maps to a specific piece of code that can be toggled on/off. This is the keystone of the ablation sweep: it lets you systematically answer "does this component help?"

## Interface

**File path (runtime):**
- Source: `configs/ablations.yaml` (single shared file at repo root) defining all of `A0`…`A9`.
- Selection: [Config (YAML)](./config.md).ablation = `"A0"`…`"A9"` selects one tuple per run.
- Snapshot: [CLI Runner](./cli-runner.md) writes the resolved tuple to `runs/<run_id>/ablation_snapshot.yaml`.

**Schema (YAML, top of file):**
```yaml
schema_version: "1.0"
ablations:
  A0:
    name: "Full system"
    graph_mask_enabled: true
    typed_scores_enabled: true
    singularity_detector_enabled: true
    idf_weighting_enabled: true
    hyperbolic_geometry_enabled: true
    group_quotient_enabled: true
    reservoir_frozen: true
    trace_history_enabled: true
  A1:
    name: "No graph mask"
    graph_mask_enabled: false
    typed_scores_enabled: true
    singularity_detector_enabled: true
    idf_weighting_enabled: true
    hyperbolic_geometry_enabled: true
    group_quotient_enabled: true
    reservoir_frozen: true
    trace_history_enabled: true
  # ... A2 through A9 (each disables exactly one flag relative to A0)
```

**Top-level field schema:**

| Field | Type | Required | Example | Semantic note |
|---|---|---|---|---|
| `schema_version` | str | required | `"1.0"` | Bumped when flags are added/removed/renamed. |
| `ablations` | object<str, AblationTuple> | required | `{A0: …, A1: …}` | Keys MUST be of the form `A<int>`; `[A0..A9]` are reserved for the ten standard variants. |

**AblationTuple schema:**

| Field | Type | Required | Default | Example | Semantic note |
|---|---|---|---|---|---|
| `name` | str | required | — | `"No graph mask"` | Human-readable label, logged with every metric. |
| `graph_mask_enabled` | bool | required | `true` | `true` | Apply legality mask from [Graph FSM Spec](./graph-fsm-spec.md) during inference. |
| `typed_scores_enabled` | bool | required | `true` | `true` | Emit typed labels Y; require grammar constraints. |
| `singularity_detector_enabled` | bool | required | `true` | `true` | Compute σ(x) flags; route them into control. |
| `idf_weighting_enabled` | bool | required | `true` | `true` | Reweight loss by per-stratum IDF using `Config.idf_alpha`. |
| `hyperbolic_geometry_enabled` | bool | required | `true` | `true` | Embed in $\mathbb{H}^d$; otherwise Euclidean. |
| `group_quotient_enabled` | bool | required | `true` | `true` | Use orbit-quotient attention; requires `Config.group_spec_path`. |
| `reservoir_frozen` | bool | required | `true` | `true` | Freeze encoder; train only typed readout. (A0 default = `true` — frozen reservoir is the baseline.) |
| `trace_history_enabled` | bool | required | `true` | `true` | Maintain past-action memory across steps. |

**Flag → effect-when-disabled:**

| Flag | Effect if disabled |
|---|---|
| `graph_mask_enabled` | Classifier may predict illegal transitions; `transition_legal` in [results.jsonl](./results-jsonl.md) is still computed for diagnostic purposes. |
| `typed_scores_enabled` | Output is raw logits without grammar type structure. |
| `singularity_detector_enabled` | σ(x) is not computed; `singular_flag` in [results.jsonl](./results-jsonl.md) is always `false`. |
| `idf_weighting_enabled` | Uniform per-sample loss weighting; rare strata may be underfit. |
| `hyperbolic_geometry_enabled` | Embed in Euclidean space of the same `embedding_dim`; lose hierarchy signal. |
| `group_quotient_enabled` | Attention is dense $O(n^2)$; orbit aggregation skipped. |
| `reservoir_frozen` | Full backprop through encoder; more parameters trained. |
| `trace_history_enabled` | Stateless inference; agent ignores past actions. |

**Ablation → research question (A0–A9):**

| Ablation | Disabled flag | Research question |
|---|---|---|
| `A0` | (none) | Full architecture baseline. |
| `A1` | `graph_mask_enabled` | Does legality masking improve accuracy and eliminate invalid transitions? |
| `A2` | `typed_scores_enabled` | Does explicit type structure help over flat classification? |
| `A3` | `singularity_detector_enabled` | Do low-margin/contradiction flags improve calibration over raw confidence? |
| `A4` | (singularity signals computed but ignored in control) | Do σ(x) signals help when used for routing, vs. only logged? |
| `A5` | `idf_weighting_enabled` | Does weighting rare strata improve rare-failure recall? |
| `A6` | `hyperbolic_geometry_enabled` | Does negative curvature improve transfer or compress dimensions? |
| `A7` | `group_quotient_enabled` | Does symmetry-aware attention reduce compute without hurting performance? |
| `A8` | `reservoir_frozen` | Does frozen-encoder + readout-only match full end-to-end training? |
| `A9` | `trace_history_enabled` | Does memory of past actions improve control and loop detection? |

Note on `A3` vs. `A4`: `A3` flips `singularity_detector_enabled=false` (no detector at all). `A4` keeps the detector running and logged, but routes control as if it were absent. This isolates "compute σ(x)" from "use σ(x)".

## Variant axes (orthogonal to A0–A9)

Two design questions that initially looked like decisions are kept open as orthogonal **variant axes**: every cell of the A-matrix can be run under each variant tuple, and the comparison is the experiment.

### Axis V1: `reservoir_frozen` framing

The A0 vs A8 pair already covers `reservoir_frozen ∈ {true, false}` on the ablation axis (A0 = frozen baseline, A8 = unfrozen deviation). What changes here is **reporting convention**: every E7-relevant report shows the matched pair `(A0_frozen, A8_unfrozen)` side-by-side rather than treating A8 as a one-flag deviation. No new flag is added; the existing `reservoir_frozen: true` default in A0 is preserved.

Cost: zero new code, zero new runs. Reporting templates list both rows.

Resolves: build-order consideration #1 ("`reservoir_frozen` default for A0").

### Axis V2: `fsm_source ∈ {hand, compiled}`

A new variant axis on [Config (YAML)](./config.md). `fsm_source = hand` (default) loads the hand-authored YAML at `Config.graph_fsm_path`. `fsm_source = compiled` runs [Grammar Compiler](../arch/substrate/grammar-compiler.md) over `Config.grammar_spec_path` and uses its emitted FSM. Both producers must emit identical-by-schema artifacts; an acceptance test (`test_compiler_output_matches_hand_authored`) round-trips both through [Graph FSM Spec](./graph-fsm-spec.md) and asserts equivalence. The compiler stays in its current Phase 5 position; while it is absent, only `hand` is selectable.

Cost: one new field on Config (V2); one acceptance test; doubles the matrix on E experiments that genuinely depend on the FSM (E1, E2, E9). Cheap experiments (E0–E5) run the full `(A × V2)` grid; expensive experiments (E2, E9) run only the resolved variant after the cheap grid identifies a winner.

Resolves: build-order consideration #4 ("Grammar compiler ordering").

### Variant grid summary

| Axis | Values | Default | First experiment that exercises it |
|---|---|---|---|
| `ablation` | A0 … A9 | A0 | every E |
| `reservoir_frozen` framing (V1) | `(A0_frozen, A8_unfrozen)` matched pair | A0 baseline + A8 deviation, reported as a pair | E7 |
| `fsm_source` (V2) | `hand` \| `compiled` | `hand` until compiler lands | E1 (cheap grid); locked for E2/E9 |

For cheap experiments, the test matrix size is `|A| × |V1| × |V2| = 10 × 1 × 2 = 20` cells per E run (V1 collapses because A0/A8 are already in `|A|`). For expensive experiments, only the variant resolved by the cheap grid is run.

**Versioning:**
- `schema_version` required. Current: `"1.0"`.
- Adding a new flag bumps to `"1.1"`; old ablations get the flag with the documented A0 default.
- Adding a new ablation (`A10`, `A11`, …) is non-breaking when no flag is added.
- Renaming a flag is a major bump.

**Producers:** human design under `configs/ablations.yaml`.
**Consumers:** [CLI Runner](./cli-runner.md) (loader and snapshot writer); [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md) and every arch atom that respects a flag (e.g., [Graph Legality Mask](../arch/graph/graph-legality-mask.md), [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md), [Hyperbolic Embedding](../arch/hyperbolic/hyperbolic-embedding.md), [Orbit-Pair Attention](../arch/group/orbit-pair-attention.md), [Frozen Encoder Backbone](../arch/substrate/frozen-encoder-backbone.md)); [Ablation Matrix](../exp/ablation-matrix.md); every E-experiment reads the resolved tuple from the run snapshot.

## Build steps

1. Define `AblationConfig` dataclass with the eight boolean fields above; defaults match `A0`. Validators reject unknown flags.
2. Author `configs/ablations.yaml` with `A0`…`A9` as defined above.
3. Implement `ablation_loader.py`: `load(path, ablation_id) → AblationConfig`; rejects unsupported `schema_version`; raises on missing ablation id.
4. Wire arch components to consume `AblationConfig`:
   - [Graph Legality Mask](../arch/graph/graph-legality-mask.md) → `graph_mask_enabled`.
   - [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md) → `typed_scores_enabled`.
   - [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md) → `singularity_detector_enabled` (and `A4` separately gates downstream routing).
   - [Hyperbolic Distance Loss](../arch/hyperbolic/hyperbolic-distance-loss.md) → `idf_weighting_enabled`.
   - [Hyperbolic Embedding](../arch/hyperbolic/hyperbolic-embedding.md) → `hyperbolic_geometry_enabled` (Euclidean fallback path).
   - [Orbit-Pair Attention](../arch/group/orbit-pair-attention.md), [Orbit Quotient Space](../arch/group/orbit-quotient-space.md) → `group_quotient_enabled`.
   - [Frozen Encoder Backbone](../arch/substrate/frozen-encoder-backbone.md), [Typed Readout Layer](../arch/typed/typed-readout-layer.md) → `reservoir_frozen`.
   - (trace memory module — TBD) → `trace_history_enabled`.
5. Wire [CLI Runner](./cli-runner.md) to: accept `--ablation`; load the tuple; pass to both the arch builder and the eval runner; snapshot to `runs/<run_id>/ablation_snapshot.yaml`.
6. Wire [Ablation Matrix](../exp/ablation-matrix.md) to drive the sweep across `A0`…`A9`.
7. Add a regression test: load every ablation, verify exactly one flag flip relative to A0 for `A1`–`A3`, `A5`–`A9`, and the documented A4 special case.

## Links

- **See also:** [Config (YAML)](./config.md) (selects ablation), [CLI Runner](./cli-runner.md) (loads and applies), [Ablation Matrix](../exp/ablation-matrix.md) (sweep runner)
- **Drives:** [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md), [Graph Legality Mask](../arch/graph/graph-legality-mask.md), [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md), [Hyperbolic Embedding](../arch/hyperbolic/hyperbolic-embedding.md), [Hyperbolic Distance Loss](../arch/hyperbolic/hyperbolic-distance-loss.md), [Orbit-Pair Attention](../arch/group/orbit-pair-attention.md), [Frozen Encoder Backbone](../arch/substrate/frozen-encoder-backbone.md), [Typed Readout Layer](../arch/typed/typed-readout-layer.md)
- **Driven by:** human design
- **Open:** (q09-ablation-coverage — are A0–A9 sufficient?), (q10-interaction-effects — test combinations or keep orthogonal?)
