# Evidence-Level Tracker

**Cluster:** exp
**Status:** spec
**Tags:** #evidence #claim-tracking #reproducibility #meta-evaluation

## What

Meta-experiment logger tracking evidence level per major claim across all runs. Claims: "graph masks reduce illegal actions," "singularity detector predicts failures," "hyperbolic geometry compresses hierarchy," etc. Per claim, track status: {unsupported, toy test, synthetic benchmark, real benchmark}. Aggregates results from E0–E9. Emits evidence.jsonl: rows of {claim_id, claim_text, evidence_level, supporting_experiments, confidence}. This is how we truthfully assess what has been proven vs speculated.

## Why

Honest science requires tracking what is actually supported by evidence vs what is aspirational. The evidence tracker prevents cargo-cult claims and forces us to distinguish toy benchmarks from real-world validation. It also enables selective re-running: if a claim was only supported by synthetic data, we know to prioritize the real benchmark. This is meta-infrastructure but load-bearing for research credibility.

## Interface

**Reads:**
- all experiment results (metrics.jsonl from E0–E9)
- claim registry (JSON: {claim_id, claim_text, required_evidence_level})

**Writes:**
- evidence.jsonl: {claim_id, claim_text, level, experiments, confidence, open_questions}
- evidence_summary.md: human-readable summary table

## Build steps

1. Define claim registry: e.g., {id: "graph_reduces_illegal", text: "graph mask eliminates invalid transitions", required_level: "real_benchmark"}.
2. For each claim, identify supporting experiments: e.g., graph-reduces-illegal is supported by E1, E2, ablation-matrix.
3. Aggregate results: if E1 shows zero illegal rate, claim is at least "synthetic_benchmark" level.
4. If E2 also shows zero illegal rate, claim advances to "real_benchmark" level.
5. If ablation shows removal of graph mask increases illegal rate, claim is "load_bearing."
6. Emit evidence.jsonl: per-claim, list supporting experiments and current level.
7. Regenerate evidence_summary.md after each major experiment run.

## Links

- **See also:** [Ablation Matrix A0–A9](./ablation-matrix.md), [E0 — MNIST Typed-State Sanity Run](./e0-mnist.md), [E1 — Synthetic BabyAI Grid Run](./e1-synthetic-babyai.md), [E2 — Real BabyAI / MiniGrid](./e2-real-babyai.md), [E9 — Full Agent Trace Benchmark](./e9-full-trace-benchmark.md)
- **Drives:** research credibility and re-prioritization
- **Driven by:** [Metric Collectors](./metric-collectors.md) output (metrics.jsonl from all experiments)
- **Math:** confidence = (n_supporting_at_required_level) / (n_required_conditions); level_priority = {unsupported < toy < synthetic < real}
- **Open:** (claim-registry, confidence-computation — open questions)
