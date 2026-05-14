# results.jsonl

**Cluster:** driver
**Status:** spec
**Tags:** #driver #contract #append-only #per-sample-results

## What

An append-only JSON Lines stream of per-sample predictions and ground-truth labels. Each record contains the sample identifier, ground truth, prediction, margin, singularity flag, transition-legality flag, scalar cost, and run/ablation context. Accumulated across evaluation runs, this stream is the complete audit trail of model behavior: it enables post-hoc analysis of failure modes, confusion patterns, singularity effectiveness, and transfer degradation. Architecture emits one record per inference; experiments consume to compute failure metrics and diagnostic breakdowns.

## Why

Aggregate metrics hide important details. To debug why the architecture fails on a particular sample, or why singularity detection succeeds or fails, you need the full per-sample ground truth and prediction. If architecture and experiment use different label conventions or confidence definitions, confusion will arise. A shared append-only format ensures both can audit the same data. This is critical for [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md) validation (did the singularity flag actually predict the failure?) and for (confusion-analyzer — exp) (which label pairs are confused most often?).

## Interface

**File path (runtime):** `runs/<run_id>/results.jsonl`. `run_id = <experiment>_<ablation>_seed<seed>`. Format: JSON Lines, append-only.

**Schema (JSON per line):**
```json
{
  "schema_version": "1.0",
  "run_id": "E2_A0_seed42",
  "sample_id": "test_0123",
  "y_true": "Navigate",
  "y_hat": "Navigate",
  "margin": 0.15,
  "singular_flag": false,
  "transition_legal": true,
  "cost": 0.08,
  "experiment": "E2",
  "ablation": "A0",
  "split": "test",
  "seed": 42,
  "step": 1000,
  "timestamp": "2026-05-04T10:23:45Z"
}
```

**Field schema (required core):**

| Field | Type | Units | Required | Example | Semantic note |
|---|---|---|---|---|---|
| `schema_version` | str | semver | required | `"1.0"` | Bumped when fields change. |
| `run_id` | str | — | required | `"E2_A0_seed42"` | Same `run_id` as [metrics.jsonl](./metrics-jsonl.md). |
| `sample_id` | str | — | required | `"test_0123"` | Joins to [Typed Score Record Contract](./typed-score-record.md).sample_id. Stable across re-runs of the same dataset. |
| `y_true` | str | — | required | `"Navigate"` | Ground-truth label; must be a vertex `id` in [Graph FSM Spec](./graph-fsm-spec.md). |
| `y_hat` | str | — | required | `"Navigate"` | Predicted label; same domain as `y_true`. Equals [Typed Score Record Contract](./typed-score-record.md).label for the same `sample_id`. |
| `margin` | float | probability ∈ [-1,1] | required | `0.15` | `confidence(top1) - confidence(top2)`. Same definition and units as [Typed Score Record Contract](./typed-score-record.md).margin. |
| `singular_flag` | bool | — | required | `false` | True iff [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md) flagged this sample (low margin, contradiction, loop, or illegal pressure). |
| `transition_legal` | bool | — | required | `true` | True iff `(q_t, y_hat)` is in the edge set of [Graph FSM Spec](./graph-fsm-spec.md). When `graph_mask_enabled=false` (ablation A1), still computed for diagnostic purposes. |
| `cost` | float ≥ 0 | loss-units | required | `0.08` | Per-sample loss (cross-entropy by default; document custom losses in `schema_version` notes). |
| `experiment` | str (enum) | — | required | `"E2"` | One of `E0`…`E9`. |
| `ablation` | str (enum) | — | required | `"A0"` | One of `A0`…`A9` per [Ablation Flag Set](./ablation-flags.md). |
| `split` | str (enum) | — | required | `"test"` | One of `train`, `val`, `test`. |
| `seed` | int | — | required | `42` | Mirrors [Config (YAML)](./config.md).seed. |
| `step` | int | training step | required | `1000` | Training step at evaluation; matches [metrics.jsonl](./metrics-jsonl.md).step. |
| `timestamp` | str (ISO 8601) | UTC | optional | `"2026-05-04T10:23:45Z"` | Wall-clock; useful for latency analysis. |

**Optional extended fields (diagnostic):**

| Field | Type | Units | Required | Example | Semantic note |
|---|---|---|---|---|---|
| `softmax_distribution` | object<str,float> | probability | optional | `{"Navigate":0.78, "Verify":0.15, "Write":0.07}` | Sums to 1.0. Same shape as in [Typed Score Record Contract](./typed-score-record.md). Consumed by (calibration-auditor — exp). |
| `z_H` | array<float> | hyperbolic coord | optional | `[0.5,-0.3,…]` | Length = [Config (YAML)](./config.md).embedding_dim. Same field as in [Typed Score Record Contract](./typed-score-record.md). |
| `confusion_type` | str | — | optional | `"5↔9"` | Free-form confusion-pair label from [Confusion Graph](../arch/typed/confusion-graph.md). |
| `loop_detected` | bool | — | optional | `false` | True if the agent entered a repetition loop on this sample. |
| `token_cost` | int | tokens | optional | `42` | LLM tokens spent (E9, agent tasks). |
| `time_ms` | float | ms | optional | `12.5` | Wall-clock latency for this sample. |

**Versioning:**
- `schema_version` required. Current: `"1.0"`.
- Adding optional fields is non-breaking.
- Renaming/typing changes (e.g. promoting `softmax_distribution` to required) require a major bump.
- Readers MUST tolerate missing optional fields.

**Producers:** [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md) (after each inference) plus [Graph FSM (arch)](../arch/graph/graph-fsm.md) (computes `transition_legal` from current state and `y_hat`). Both write through `results_writer.py`. [CLI Runner](./cli-runner.md) supplies `run_id`, `experiment`, `ablation`, `seed`, `step`.

**Consumers:** [Evidence-Level Tracker](../exp/evidence-tracker.md) (per-ablation confusion matrices and singularity stats), [Confusion Graph](../arch/typed/confusion-graph.md) (rebuilds error multigraph), [Failure-vs-Margin AUROC](../arch/singularity/failure-margin-auroc.md), [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md) validation, and (calibration-auditor, geometry-analyzer, failure-predictor — exp components living under `exp/metric_collectors.py`).

## Build steps

1. Define `ResultRecord` dataclass with the field schema above; optional fields default to `None`; validators check `y_true`/`y_hat` are valid FSM vertex ids and `margin ∈ [-1, 1]`.
2. Implement `results_writer.py` with append-only `write(record, path)`; the writer is invoked once per inference call from [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md).
3. Implement `results_reader.py` with streaming reader; rejects unsupported major version, tolerates missing optional fields.
4. Wire [Graph FSM (arch)](../arch/graph/graph-fsm.md) to compute `transition_legal` against [Graph FSM Spec](./graph-fsm-spec.md) edges and pass it to the writer.
5. Wire [Evidence-Level Tracker](../exp/evidence-tracker.md) to read every `runs/*/results.jsonl`, build per-ablation confusion matrices, and compute singularity-vs-error joint statistics.
6. Implement (confusion-analyzer — exp): groups by `(y_true, y_hat)`; reports rates and top-k confusions; writes derived metrics into [metrics.jsonl](./metrics-jsonl.md) (`confusion_graph_density`, etc.).
7. Implement (singularity-validator — exp): correlates `singular_flag` with `y_true ≠ y_hat` and emits `auroc_failure` to [metrics.jsonl](./metrics-jsonl.md).
8. Add a regression test: write a known set of records, read them back, recompute aggregate accuracy, confirm match.

## Links

- **See also:** [Typed Score Record Contract](./typed-score-record.md) (sibling per-sample stream; shares `sample_id`, `margin`, `experiment`, `ablation`, `seed`, `step`), [metrics.jsonl](./metrics-jsonl.md) (shares `run_id`), [Config (YAML)](./config.md), [Graph FSM Spec](./graph-fsm-spec.md) (defines the label set)
- **Drives:** [Evidence-Level Tracker](../exp/evidence-tracker.md), [Confusion Graph](../arch/typed/confusion-graph.md), [Failure-vs-Margin AUROC](../arch/singularity/failure-margin-auroc.md)
- **Driven by:** [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md), [Graph FSM (arch)](../arch/graph/graph-fsm.md)
- **Open:** (q06-extended-fields — which optional fields should be mandatory vs. diagnostic extensions?)
