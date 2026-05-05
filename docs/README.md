# Master Plan: Building the Architecture and the Experiment Harness

A Zettelkasten of ~70 atomic notes covering the system in [Architecture.md](../Architecture.md) and the harness in [Experiments.md](../Experiments.md). Mathematics anchors live in [Mathematics.md](../Mathematics.md). The `drivers/` cluster is the spine — those are the shared contracts both halves consume.

## Where to start

- New to the project? Read in this order: [Drivers](drivers/_index.md) → [Architecture](arch/_index.md) → [Experiments](exp/_index.md).
- Want to run something? Start with [Minimal Next Experiment](exp/minimal-next-experiment.md) and [CLI Runner](drivers/cli-runner.md).
- Want to know what's unresolved? See [Open Questions](open/_index.md).

## Map of the docs

| Cluster | What it covers | Count |
|---|---|---|
| [arch/](arch/_index.md) | architecture atoms — typed scoring, graph FSM, hyperbolic embedding, singularity, group quotient, energy, reservoir, grammar | 30 |
| [exp/](exp/_index.md) | experiment harness — E0–E9 runners, ablation matrix, dataset adapters, metric collectors | 20 |
| [drivers/](drivers/_index.md) | shared contracts — config, typed-score record, graph FSM spec, metrics/results JSONL, CLI runner, ablation flags | 7 |
| [open/](open/_index.md) | open questions and pending decisions | 12 |

## Note conventions

- Each note has: What / Why / Interface / Build steps / Links sections.
- Status: `spec` (drafted) / `stub` / `implemented` / `tested`.
- Links use plain Markdown so they render in any viewer.
- Math is referenced, never duplicated — see [Mathematics.md](../Mathematics.md).

## Drivers are the spine

| Driver | What it carries | Producers (arch) | Consumers (exp) |
|---|---|---|---|
| [config.md](drivers/config.md) | YAML — single source of truth | model builder reads | runner reads |
| [typed-score-record.md](drivers/typed-score-record.md) | wire format of Y | classifier emits | metric collector reads |
| [graph-fsm-spec.md](drivers/graph-fsm-spec.md) | (V, E, w_v, g_v, m_v) file format | mask layer loads | runner validates transitions |
| [metrics-jsonl.md](drivers/metrics-jsonl.md) | append stream of run metrics | trainer appends | evidence-tracker aggregates |
| [results-jsonl.md](drivers/results-jsonl.md) | append stream of per-sample preds | inference appends | ablation comparator reads |
| [cli-runner.md](drivers/cli-runner.md) | `python run.py --experiment Ek --ablation Aj ...` | exposes flags | drives experiments |
| [ablation-flags.md](drivers/ablation-flags.md) | the bool knobs A0–A9 are tuples of | code respects | runner sets |

## Build order

Drivers first → arch atoms (depth-first by dependency) → exp runners (E0 → E1 → E4 → E5 → ramp). See [build-order.md](build-order.md) (written by the Opus polish stage).
