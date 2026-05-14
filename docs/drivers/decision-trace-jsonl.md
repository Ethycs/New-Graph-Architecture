# Decision Trace JSONL

**Cluster:** drv
**Status:** implemented
**Tags:** #decision-trace #audit #jsonl #v11 #node-tuple

## What

Per-step interpretability stream. Each record exposes the full chain of signals that drove a single prediction: typed scores → mask → margin → stratum tag → σ signals → energy → partition → control decision → final action. A run with $N$ steps produces $N$ decision-trace rows; an analyst can read one row and reconstruct exactly why the agent did what it did at that step. Companion to `results.jsonl` (model-output stream) and `metrics.jsonl` (aggregate stream); `decision_trace.jsonl` is the WHY stream.

Schema v1.1 makes the "all-output-must-be-a-node" commitment first-class on the wire by adding an explicit `output_node_tuple` plus the transition that produced it. Existing v1.0 fields (`sigma_total`, `energy_breakdown`, `mask`, `control`, ...) remain present and required as before — they become *derived diagnostics* explaining the choice, not the primary identity of "what was emitted." v1.1 is purely additive.

## Why

The TPN's third commitment is "audit by construction." Without a decision-trace artefact, the architecture's interpretability claim is rhetorical — you can describe what σ, mask, energy etc. do, but you cannot reconstruct a specific prediction's pedigree. The decision trace is the executable form of "every decision is auditable": one row per step, every load-bearing intermediate exposed, the output identity tied to an explicit node-tuple on the typed product graph. Schema v1.1's `output_node_tuple` is the wire-level enforcement of the "no raw floats cross interfaces" commitment.

## Interface

Each row is a JSON object on its own line. Required fields (v1.0):

- `step` (int) — step index within the run.
- `experiment` (str), `ablation` (str), `seed` (int) — provenance.
- `top1_state` (str) — the chosen vertex.
- `mask` (object) — `{post_argmax: str, illegal_top1: bool, mask_version_id: str}`.
- `margin` (float) — top1 − top2 logit margin (pre-sigmoid).
- `stratum_tag` (str), `stratum_bitmask` (int).
- `sigma_total` (float), `sigma_signals` (object) — per-component decomposition.
- `energy` (object) — `{total: float, breakdown: {cost, uncertainty, contradiction, loop, progress}}`.
- `partition` (object) — `{Z_lambda: float, P_lambda: float, stratum: str}`.
- `control` (object) — `{verdict: "NORMAL" | "RECOVERY" | "ABSTAIN", reason: str}`.
- `final_action` (str | int) — what the runner emitted downstream.

Schema v1.1 adds (all default to `None` for back-compat):

- `output_node_tuple` (list[str] | null) — the node-tuple identity on the typed product graph.
- `edge_traversed` (object | null) — `{src: NodeTuple, dst: NodeTuple, posterior_summary: {alpha, beta}}`.
- `axis_node_ids` (object | null) — per-typed-axis vertex IDs at this step.

File path convention: `runs/{experiment}_{ablation}_seed{seed}/decision_trace.jsonl`.

## Build steps

- Use [`drivers/jsonl_writer.py::JsonlWriter`](../../src/nga/drivers/jsonl_writer.py) for atomic line-buffered writes.
- Construct a v1.1 row by composing the v1.0 dict with the additive fields; let `None` defaults flow through for runners that haven't been upgraded.
- Reuse existing atoms: [Singularity Detector](../arch/singularity/singularity-detector.md) for σ, [Behavioral Stratum Tagger](../arch/singularity/behavioral-stratum-tagger.md) for stratum, [Energy Function](../arch/energy/energy-function-E.md) for energy, [Stratified Partition Function](../arch/energy/stratified-partition-function.md) for partition, [Control Policy](../arch/substrate/control-policy.md) for the verdict, [Axis Quantizer](../arch/hyperbolic/axis-quantizer.md) for typed-axis vertex IDs.
- Validate post-write: `tests/unit/test_decision_trace_jsonl_v11.py` parses the file and asserts every row has the v1.1 fields present (even if null).

## Links

- **See also:** [Results JSONL](./results-jsonl.md), [Metrics JSONL](./metrics-jsonl.md), [Typed Score Record](./typed-score-record.md), [Config](./config.md).
- **Driven by:** every TPN runner over the typed FSM (E0–E23) and the PCG-X runner over the regime graph (E28, since Phase 23e). Same schema both ways; on the regime side the state names use the `regime_K` / `regime_unknown` convention and `mask.enabled` is `false` because PCG-X does not apply a hard mask to the prediction. The trace is substrate-agnostic.
- **Drives:** post-hoc analysis tools (`aggregate.py`, `failure_margin_auroc`), the structural-AUROC computation, the audit-trail interpretability primitive proposed in `docs/proposals/graph-extraction.md`.
- **Math:** the trace is a section of the typed product graph indexed by step; `output_node_tuple` is the node identity in [Product Graph](../arch/graph/product-graph.md).
- **Open:** [[open.qNN-trace-versioning]] — under what conditions does a future schema bump break wire compatibility vs stay additive?
