# Typed Score Record Contract

**Cluster:** driver
**Status:** spec
**Tags:** #driver #contract #wire-format #classifier-output

## What

The typed score record is the JSON-line wire format for classifier outputs $Y$. Each record binds a predicted type label with confidence, a distance array (for hyperbolic or other geometric alignment), optional embedding coordinates, and the precomputed margin. Architecture emits one record per sample after inference; experiments consume these records to evaluate margins, singularities, and transition legality. This is the bridge between model predictions and validation/metrics.

## Why

Without a strict wire format, architecture and experiment measure different things. The architecture might emit softmax probabilities; the experiment expects margin gaps. The architecture might include raw logits; the experiment ignores them or misinterprets them. A shared schema ensures that confidence calibration, margin singularity detection, and transition legality checks all see the same data. Precision in this contract is load-bearing: the entire [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md), [Margin Uncertainty](../arch/typed/margin-uncertainty.md), and (transition-validator — exp) depend on interpreting $Y$ correctly.

## Interface

**File path (runtime):** `runs/<run_id>/scores.jsonl` — one JSON Lines record per inference sample. The arch component [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md) appends a record per sample; [Metric Collectors](../exp/metric-collectors.md) and downstream analyzers stream-read.

**Schema (JSON per line):**
```json
{
  "schema_version": "2.0",
  "sample_id": "test_0042",
  "label": "Navigate",
  "confidence": 0.87,
  "margin": 0.12,
  "dist": [0.12, 0.34, 0.05, 1.41, 0.88, 0.31, 0.77],
  "softmax_distribution": {"Navigate": 0.87, "Verify": 0.08, "Write": 0.05},
  "z_H": [0.5, -0.3, 0.21, 0.04, -0.11, 0.18, 0.07, 0.42],
  "regime_id": 3,
  "control_verdict": "NORMAL",
  "experiment": "E28",
  "ablation": "A0",
  "seed": 42,
  "step": 1000,
  "timestamp": "2026-05-04T10:23:45Z"
}
```

**Field schema:**

| Field | Type | Units | Required | Example | Semantic note |
|---|---|---|---|---|---|
| `schema_version` | str | semver | required | `"1.0"` | Match against reader's supported set; abort on unknown major. |
| `sample_id` | str | — | required | `"test_0042"` | Stable identifier; joins to [results.jsonl](./results-jsonl.md) on the same key. |
| `label` | str | — | required | `"Navigate"` | $\hat y$ = predicted type. Must be a vertex `id` in [Graph FSM Spec](./graph-fsm-spec.md). |
| `confidence` | float | probability ∈ [0,1] | required | `0.87` | $\max(\mathrm{softmax})$ over labels at temperature $T$. Energy-based heads MUST normalize to the same range and document it in `schema_version`. |
| `margin` | float | probability ∈ [-1,1] | required | `0.12` | `confidence(top1) - confidence(top2)` in the same units as `confidence`. Same definition as `margin` in [results.jsonl](./results-jsonl.md). |
| `dist` | array<float> | distance | required | `[0.12, 0.34, …]` | Length = `vertex_count` from [Graph FSM Spec](./graph-fsm-spec.md). Order: vertices in the FSM's `vertices` list. Lower = closer to prototype. |
| `softmax_distribution` | object<str,float> | probability | optional | `{"Navigate":0.87,…}` | Full distribution; values sum to 1.0. Keys ⊆ FSM vertex ids. Used by (calibration-auditor — exp). |
| `z_H` | array<float> | hyperbolic coord | optional | `[0.5,-0.3,…]` | Length = `embedding_dim` from [Config (YAML)](./config.md). Required when `hyperbolic_geometry_enabled`; otherwise may be omitted or carry Euclidean coordinates. |
| `regime_id` | int \| null | — | optional (v2.0+) | `3` | PCG-X regime label from the bisimulation quotient. Integer index into the regime graph; `-1` (or `null`) marks an FSM state that never appeared as a regime cell during training (`regime_unknown`). Mirrors the `_UNKNOWN_REGIME` sentinel in [`src/nga/exp/e28_pcg_extractor.py`](../../src/nga/exp/e28_pcg_extractor.py). Producers: [E28](../exp/e28-pcg-extractor.md), [E30](../exp/e30-pcg-extractor-pretrained.md). Omit on records emitted by non-PCG-X runners (E0–E9). |
| `control_verdict` | str (enum) \| null | — | optional (v2.0+) | `"NORMAL"` | The substrate-agnostic control decision from [Control Policy](../arch/substrate/control-policy.md): one of `"NORMAL"`, `"RECOVERY"`, `"ABSTAIN"`. Same enum as `control.verdict` in [decision-trace-jsonl](./decision-trace-jsonl.md); kept flat here to avoid forcing per-sample-record consumers to read the parallel trace stream. Omit on non-PCG-X runners. |
| `experiment` | str (enum) | — | required | `"E28"` | One of `E0`…`E9`, `E24`/`E25`/`E28`/`E30`. Same value as in [metrics.jsonl](./metrics-jsonl.md), [results.jsonl](./results-jsonl.md). |
| `ablation` | str (enum) | — | required | `"A0"` | One of `A0`…`A9` per [Ablation Flag Set](./ablation-flags.md). |
| `seed` | int | — | required | `42` | Run seed; mirrors [Config (YAML)](./config.md).seed. |
| `step` | int | training step | required | `1000` | Training step at which inference was run; matches [metrics.jsonl](./metrics-jsonl.md). |
| `timestamp` | str (ISO 8601) | UTC | optional | `"2026-05-04T10:23:45Z"` | Wall-clock; useful for latency analysis. |

**Versioning:**
- `schema_version` is required. Current: `"2.0"`. Records with `"1.0"` remain readable; downstream consumers MUST tolerate missing `regime_id` / `control_verdict` on legacy records.
- Optional fields may be omitted on write; readers default them to `None` and skip dependent analyses.
- **v2.0 additions** (additive, no break for v1.0 readers that ignore unknown fields):
  - `regime_id` — PCG-X regime label (int).
  - `control_verdict` — `NORMAL`/`RECOVERY`/`ABSTAIN` from the substrate-agnostic [Control Policy](../arch/substrate/control-policy.md).
  - `experiment` enum extended to include `E24`, `E25`, `E28`, `E30`.
- If `confidence` semantics change (e.g., switching to energy-based normalization), bump the *minor* on the active major (`"2.1"`) and document the formula. Readers SHOULD warn on `"2.0"` data being mixed with `"2.1"` in the same aggregate.
- The order and length of `dist` is fixed by the FSM at write time; if the FSM changes, the records become stale and a major version bump is required.

**Producers:** [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md), [Typed Score Record (arch)](../arch/typed/typed-score-record.md). Concretely, `arch/typed_field_pipeline.py` packages logits + embedding + margin and `arch/typed_score_record.py` serializes one record per sample.
**Consumers:** [Metric Collectors](../exp/metric-collectors.md) (drives margin/AUROC aggregations), [Evidence-Level Tracker](../exp/evidence-tracker.md), [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md), [Margin Uncertainty](../arch/typed/margin-uncertainty.md), [Confusion Graph](../arch/typed/confusion-graph.md), and (calibration-auditor, geometry-analyzer — exp components implemented under `exp/metric_collectors.py`).

## Build steps

1. Define a `TypedScoreRecord` Pydantic/dataclass with the field schema above; validators: `0 ≤ confidence ≤ 1`, `len(dist) == grammar.vertex_count`, optional fields default to `None`, sum-to-1 check on `softmax_distribution`.
2. Write `score_writer.py` with append-only `write(record, path)`; called by [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md) immediately after classification with `path = runs/<run_id>/scores.jsonl`.
3. Write `score_reader.py` with streaming `read(path) → Iterator[TypedScoreRecord]`; checks `schema_version` and raises on unsupported major.
4. Wire [Typed Score Record (arch)](../arch/typed/typed-score-record.md) to populate every required field, including `margin` (don't make consumers recompute) and `experiment`/`ablation`/`seed`/`step` from the runtime context provided by [CLI Runner](./cli-runner.md).
5. Wire [Metric Collectors](../exp/metric-collectors.md) to ingest `scores.jsonl` and emit derived metrics into [metrics.jsonl](./metrics-jsonl.md).
6. Add unit tests for: round-trip serialization, malformed records (missing required fields), version mismatch, and `dist` length mismatch with grammar.
7. Document the canonical `confidence` formula in code comments next to the schema definition.

## Links

- **See also:** [metrics.jsonl](./metrics-jsonl.md) (where aggregated statistics go), [results.jsonl](./results-jsonl.md) (per-sample predictions; shares `sample_id`, `margin`, `experiment`, `ablation`, `seed`, `step`), [Config (YAML)](./config.md) (provides `embedding_dim` consumed by `z_H`)
- **Drives:** [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md), [Margin Uncertainty](../arch/typed/margin-uncertainty.md), [Confusion Graph](../arch/typed/confusion-graph.md), [Metric Collectors](../exp/metric-collectors.md)
- **Driven by:** [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md), [Typed Score Record (arch)](../arch/typed/typed-score-record.md)
- **Open:** (q02-confidence-normalization — softmax max always, or allow custom energy-based?)
