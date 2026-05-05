# metrics.jsonl

**Cluster:** driver
**Status:** spec
**Tags:** #driver #contract #append-only #metrics-stream

## What

An append-only JSON Lines stream of scalar metrics collected during training and evaluation. Each record carries the experiment identifier, ablation variant, data split, metric name, scalar value, training step, seed, and run identifier. Trainers append records as training progresses; [Evidence-Level Tracker](../exp/evidence-tracker.md) reads and aggregates them to produce summaries and charts. This stream is the single source of truth for all scalar performance measurements.

## Why

Metrics must be logged consistently by both architecture training loop and experiment evaluation loop, with identical field semantics. If the architecture computes `illegal_transition_rate` as (# invalid / # total) but the experiment computes it as (# valid / # total), ablation comparisons are bogus. A shared append-only format ensures atomicity (no partial writes), immutability (old records cannot be modified), and traceability (every metric has timestamp, step, seed, run_id context). This is critical for reproducibility and for evidence aggregation across multiple runs.

## Interface

**File path (runtime):** `runs/<run_id>/metrics.jsonl`. `run_id = <experiment>_<ablation>_seed<seed>`. Format: JSON Lines, append-only.

**Schema (JSON per line):**
```json
{
  "schema_version": "1.0",
  "run_id": "E2_A0_seed42",
  "experiment": "E2",
  "ablation": "A0",
  "split": "train",
  "metric_name": "accuracy",
  "value": 0.876,
  "step": 100,
  "seed": 42,
  "timestamp": "2026-05-04T10:23:45Z"
}
```

**Field schema:**

| Field | Type | Units | Required | Example | Semantic note |
|---|---|---|---|---|---|
| `schema_version` | str | semver | required | `"1.0"` | Bumped when fields change; readers compare on load. |
| `run_id` | str | — | required | `"E2_A0_seed42"` | `<experiment>_<ablation>_seed<seed>`; stable join key with [results.jsonl](./results-jsonl.md). |
| `experiment` | str (enum) | — | required | `"E2"` | One of `E0`…`E9`. |
| `ablation` | str (enum) | — | required | `"A0"` | One of `A0`…`A9` per [Ablation Flag Set](./ablation-flags.md). |
| `split` | str (enum) | — | required | `"train"` | One of `train`, `val`, `test`, `singular_region`, `boundary_region` (the last two for diagnostic subsets). |
| `metric_name` | str (enum) | — | required | `"accuracy"` | From the canonical name table below; readers warn (not error) on unknown names. |
| `value` | float | metric-specific | required | `0.876` | Units depend on `metric_name`; ratio for accuracy/rates, count for steps, ms for latency, etc. |
| `step` | int | training step | required | `100` | Training step or epoch; `0` for pre-training baselines, `-1` for post-hoc summaries. |
| `seed` | int | — | required | `42` | Mirrors [Config (YAML)](./config.md).seed. |
| `timestamp` | str (ISO 8601) | UTC | optional | `"2026-05-04T10:23:45Z"` | Wall-clock when the metric was recorded. |

**Canonical `metric_name` values (extend cautiously):**

| Group | Names | Unit |
|---|---|---|
| Classification | `accuracy`, `low_margin_accuracy`, `singular_region_accuracy`, `margin`, `confidence` | ratio (or probability for `margin`/`confidence`) |
| Graph | `illegal_transition_rate`, `confusion_graph_density`, `transition_coverage` | ratio |
| Singularity | `auroc_failure`, `singular_region_precision`, `singular_region_recall` | ratio |
| Efficiency | `sample_efficiency`, `compute_flops`, `latency_ms`, `token_cost` | samples, flops, ms, tokens |
| Transfer | `transfer_gap`, `domain_divergence` | ratio, KL/Wasserstein |
| Dynamics | `loop_rate`, `episode_length`, `success_rate` | ratio, steps, ratio |

**Versioning:**
- `schema_version` required. Current: `"1.0"`.
- Adding new `metric_name` values is non-breaking; readers warn but do not fail on unknown names.
- Adding new fields with defaults is non-breaking.
- Type or required-set changes require a major bump and explicit reader migration.

**Producers:** [Metric Collectors](../exp/metric-collectors.md) (single shared writer); called from arch trainers (post-epoch and post-validation) and from [CLI Runner](./cli-runner.md) at evaluation time. Concretely:
- (trainer — arch component, lives in `arch/typed_field_pipeline.py`) appends classification/graph/efficiency metrics each epoch.
- E0–E9 runners (`exp/e[0-9]*.md` codified in `exp/metric_collectors.py`) append final evaluation metrics.

**Consumers:** [Evidence-Level Tracker](../exp/evidence-tracker.md) (groups by `(experiment, ablation, metric_name)` for summaries), [Ablation Matrix](../exp/ablation-matrix.md) (cross-tabulates ablations), and any (metrics-plotter — exp) producing learning curves and bar charts.

## Build steps

1. Define `MetricsRecord` dataclass with the field schema above; validators for `metric_name` (canonical or warn), `split`, and numeric types.
2. Implement `metrics_writer.py` with atomic-append `write(record, path)`; exposes a `MetricCollector` context that fills `run_id`, `experiment`, `ablation`, `seed` from the runtime context produced by [CLI Runner](./cli-runner.md).
3. Implement `metrics_reader.py` with streaming `read(path) → Iterator[MetricsRecord]`; rejects unsupported major version.
4. Wire arch trainers (entry from [Typed Field Pipeline](../arch/typed-field-pipeline.md)) to call the collector after each epoch and validation step.
5. Wire E0–E9 evaluation paths (under `exp/`) and [Metric Collectors](../exp/metric-collectors.md) to call the collector after each ablation variant.
6. Wire [Evidence-Level Tracker](../exp/evidence-tracker.md) to read every `runs/*/metrics.jsonl`, group by `(experiment, ablation, metric_name)`, and report mean/std/min/max.
7. Document the canonical `metric_name` table in code as the single source of truth.

## Links

- **See also:** [results.jsonl](./results-jsonl.md) (per-sample audit; shares `run_id`, `experiment`, `ablation`, `seed`, `step`), [Config (YAML)](./config.md) (carries `seed` and ablation context), [Ablation Flag Set](./ablation-flags.md)
- **Drives:** [Evidence-Level Tracker](../exp/evidence-tracker.md), [Ablation Matrix](../exp/ablation-matrix.md)
- **Driven by:** [Typed Field Pipeline](../arch/typed-field-pipeline.md), [Metric Collectors](../exp/metric-collectors.md), [CLI Runner](./cli-runner.md)
- **Open:** (q05-metric-aggregation — pre-aggregate per ablation, or leave raw?)
