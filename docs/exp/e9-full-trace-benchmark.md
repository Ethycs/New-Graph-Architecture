# E9 — Full Agent Trace Benchmark

**Cluster:** exp
**Status:** spec
**Tags:** #full-architecture #multi-domain #agent-traces #integration

## What

Run the complete architecture on multi-step agent traces across 3 domains (BabyAI, ALFWorld, ScienceWorld). Full pipeline: observation → encoder h_t → typed classifier Y_t → hyperbolic embedding z_H → distribution P(Y_t, z_t) → orbit quotient Γ/H → graph legality E → next-state scores → singularity detector σ → action φ(x_t). Measure success rate, invalid-transition rate, average steps-per-episode, loop rate, failure-prediction AUROC, and transfer metrics.

## Why

Experiments E0–E8 test individual components in isolation or on toy benchmarks. E9 is the full integration: does the complete architecture actually improve agent efficiency on real multi-step tasks? This is the load-bearing claim test. If any component is ineffective or if the assembly breaks down, E9 will reveal it. Success is demonstrating that the full system reaches target success rate and illegal-transition rate with fewer samples and lower loop rate than baselines.

## Interface

**Reads:**
- BabyAI environment (or offline traces)
- ALFWorld simulator + TextWorld state
- ScienceWorld environment
- typed feature extractors for each domain
- task graphs and transition legality
- pre-trained hyperbolic embeddings from E3
- singularity detector from E4

**Writes:**
- [`metrics.jsonl`](../drivers/metrics-jsonl.md): per-domain per-model {success_rate, illegal_rate, avg_steps, loop_rate, failure_auroc, transfer}
- [`results.jsonl`](../drivers/results-jsonl.md): episode traces with actions, predictions, margins, singularities
- summary table: all metrics across domains and models
- failure analysis: cases caught by singularity detector, cases missed

## Build steps

1. Load or initialize all three environments (BabyAI, ALFWorld, ScienceWorld).
2. Build domain-specific typed feature extractors for each environment.
3. Load pre-trained graph and hyperbolic embeddings from E3.
4. Assemble full pipeline: observation → h_t → Y_t → z_H → P(Y_t, z_t) → Γ/H → E → scores → σ → action.
5. Run n_episodes episodes per domain per model; track success, illegal actions, steps, loops.
6. Measure singularity-detector AUROC on failure cases.
7. Test transfer: train on source domain; zero-shot evaluate on target domain.
8. Emit metrics and episode traces.

## Links

- **See also:** [E1 — Synthetic BabyAI Grid Run](./e1-synthetic-babyai.md), [E2 — Real BabyAI / MiniGrid](./e2-real-babyai.md), [E3 — Hyperbolic vs Euclidean Embedding Sweep](./e3-hyperbolic-vs-euclidean.md), [E4 — Singularity Detector Validation](./e4-singularity-auroc.md), [Ablation Matrix A0–A9](./ablation-matrix.md)
- **Drives:** integration test for full architecture
- **Driven by:** (minigrid-instruction-loader, alfworld-loader, scienceworld-loader — external), [Dataset Adapter — Research-Agent Trace Replay](./dataset-trace-replay.md)
- **Math:** loop_rate = (action_repeats ≥ 3).mean(); success_rate = (episode_return ≥ threshold).mean()
- **Open:** (success-threshold-per-domain — open question), [How is "loop risk" scored in σ(x)?](../open/q12-loop-risk-detection.md)
