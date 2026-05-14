# E2 — Real BabyAI / MiniGrid

**Cluster:** exp
**Status:** spec
**Tags:** #real-benchmark #reproducibility #primary-target

## What

Install BabyAI/MiniGrid, adapt observation-to-typed-feature converters, run 5-model comparison on selected levels: flat policy, concept-bottleneck, graph FSM, graph+singularity, hyperbolic graph. Primary metrics are success rate, sample efficiency, illegal-action rate, and recovery from ambiguity. This is the main reproducibility target and the cleanest controlled testbed before moving to less grammar-constrained domains.

## Why

Real BabyAI tests the architecture on actual compositional language-grounded instruction following. Grid-world states and actions form a clean transition graph, and instructions are compositional (navigate, pickup, open door). If the typed-graph controller requires fewer samples to reach stable performance, this validates the sample-efficiency claim. Failure cases (ambiguous instructions, conflicting subgoals) should be flagged by the singularity detector. This is the bridge between toy benchmarks and real agent applications.

## Interface

**Reads:**
- BabyAI/MiniGrid environment API
- instruction encoder (language → typed fields)
- observation encoder (image → grid features)
- pre-trained or initialized policy networks

**Writes:**
- [`metrics.jsonl`](../drivers/metrics-jsonl.md): per-level per-model {success_rate, sample_count_to_threshold, illegal_rate, recovery_rate}
- [`results.jsonl`](../drivers/results-jsonl.md): episode traces with action predictions, margins, singularity flags
- learning curves: sample efficiency by model variant
- failure logs: ambiguous cases flagged by singularity detector

## Build steps

1. Install minigrid (`pip install minigrid`); load a curated BabyAI-like level subset.
2. Build observation encoder: image → {grid-position, grid-occupancy, goal-vector}.
3. Build instruction encoder: text → {action-primitive, object-type, location-constraint}.
4. Train 5 model variants on each level, varying only the architecture (flat/typed/graph/singularity/hyperbolic).
5. For each variant, track samples until success_rate ≥ 0.9.
6. Measure illegal actions, recoveries from ambiguous states, singularity-detector AUROC.
7. Emit results; plot learning curves.

## Links

- **See also:** [E1 — Synthetic BabyAI Grid Run](./e1-synthetic-babyai.md), [E9 — Full Agent Trace Benchmark](./e9-full-trace-benchmark.md), [Dataset Adapter — Real MiniGrid/BabyAI Wrapper](./dataset-minigrid-wrapper.md)
- **Drives:** validates graph FSM on real grammar domain
- **Driven by:** (minigrid-instruction-loader — external), [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md), [Graph Legality Mask](../arch/graph/graph-legality-mask.md)
- **Math:** success_rate = (episode_return ≥ threshold).mean() over n_eval episodes
- **Open:** (level-selection, instruction-ambiguity-measurement — open questions)
