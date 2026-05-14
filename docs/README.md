# Master Plan: Building the Architecture and the Experiment Harness

A Zettelkasten of ~100 atomic notes covering the system in [Architecture.md](../Architecture.md) and the harness in [Experiments.md](../Experiments.md). Mathematics anchors live in [Mathematics.md](../Mathematics.md). The `drivers/` cluster is the spine — those are the shared contracts both halves consume.

## Where to start

- New to the project? Read in this order: [Drivers](drivers/_index.md) → [Architecture](arch/_index.md) → [Experiments](exp/_index.md).
- Want to run something? Start with [Minimal Next Experiment](exp/minimal-next-experiment.md) and [CLI Runner](drivers/cli-runner.md).
- Want the Phase 20–25 synthesis? See "Recent work" below.
- Want to know what's unresolved? See [Open Questions](open/_index.md).

## Map of the docs

| Cluster | What it covers | Count |
|---|---|---|
| [arch/](arch/_index.md) | architecture atoms in 7 themed sub-buckets (graph, typed, hyperbolic, singularity, group, energy, substrate) | 46 |
| [exp/](exp/_index.md) | experiment harness — E0–E9 + E24/E25/E28/E30 runners, dataset adapters, ablation matrix, infrastructure | 24 |
| [drivers/](drivers/_index.md) | shared contracts — config, typed-score & decision-trace records, graph FSM spec, metrics/results JSONL, CLI runner, ablation flags | 8 |
| [open/](open/_index.md) | open questions and pending decisions | 12 |
| [proposals/](proposals/_index.md) | pre-registered research proposals (entry: `graph-extraction.md`, now implemented) | 1 |
| [setup/](setup/language-framework.md) | environment setup notes — language framework and test fixtures | 2 |

## Recent work — Phase 20–25 synthesis

These root-level docs are the readable arc through the most recent work. They lean on the cluster atoms above for the technical details.

- [results.md](results.md) — paper-shaped synthesis of Phases 0–24: load-bearing findings, σ role taxonomy, PCG-X outcomes.
- [insights.md](insights.md) — engineering reflection on why TPN on a frozen pretrained substrate is useful; deployment shape.
- [interpretability-push.md](interpretability-push.md) — the conceptual move toward complete structural interpretability via Whitney stratification + σ as singular-set distance.
- [model-class.md](model-class.md) — one-line definition of the TPN model class, the load-bearing commitments, and the substrate-choice philosophy.
- [build-order.md](build-order.md) — dependency DAG: drivers → arch atoms → experiment runners.

## Note conventions

- Each atom has: What / Why / Interface / Build steps / Links sections.
- Status: `spec` (drafted) / `stub` / `implemented` / `tested` / `deprecated`.
- Links use plain Markdown so they render in any viewer.
- Math is referenced, never duplicated — see [Mathematics.md](../Mathematics.md).

## Drivers are the spine

| Driver | What it carries | Producers (arch) | Consumers (exp) |
|---|---|---|---|
| [config.md](drivers/config.md) | YAML — single source of truth | model builder reads | runner reads |
| [typed-score-record.md](drivers/typed-score-record.md) | wire format of Y (schema 2.0: also `regime_id`, `control_verdict`) | classifier emits | metric collector reads |
| [graph-fsm-spec.md](drivers/graph-fsm-spec.md) | (V, E, w_v, g_v, m_v) file format | mask layer loads | runner validates transitions |
| [metrics-jsonl.md](drivers/metrics-jsonl.md) | append stream of run metrics | trainer appends | evidence-tracker aggregates |
| [results-jsonl.md](drivers/results-jsonl.md) | append stream of per-sample preds | inference appends | ablation comparator reads |
| [decision-trace-jsonl.md](drivers/decision-trace-jsonl.md) | per-step WHY stream (typed scores → mask → σ → energy → control verdict → node-tuple); v1.1 | runners emit | post-hoc analysis + audit |
| [cli-runner.md](drivers/cli-runner.md) | `python run.py --experiment Ek --ablation Aj ...` | exposes flags | drives experiments |
| [ablation-flags.md](drivers/ablation-flags.md) | the bool knobs A0–A9 are tuples of | code respects | runner sets |

## Build order

Drivers first → arch atoms (depth-first by dependency, walking the 7 sub-buckets) → exp runners (E0 → E1 → E4 → E5 → ramp, then the PCG-X arc E24 → E25 → E28 → E30). See [build-order.md](build-order.md) (written by the Opus polish stage).
