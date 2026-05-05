# Experiment Atoms

The exp cluster contains 20 atomic notes covering the experiment harness: E0–E9 runner experiments, ablation studies, dataset adapters, and infrastructure (metric collectors and evidence tracking). Read these if you want to run, reproduce, or extend the experiments.

## Notes

### E0–E9 Runners

- [E0 — MNIST Typed-State Sanity Run](./e0-mnist.md) — Minimal end-to-end sanity check: classifier → typed labels → confusion graph on sklearn digits.
- [E1 — Synthetic BabyAI Grid Run](./e1-synthetic-babyai.md) — Tests graph-legality masking on in-process synthetic grid world; no external dependencies.
- [E2 — Real BabyAI / MiniGrid](./e2-real-babyai.md) — 5-model comparison on real compositional language-grounded grid tasks; primary target.
- [E3 — Hyperbolic vs Euclidean Embedding Sweep](./e3-hyperbolic-vs-euclidean.md) — Sweeps dimensions {2,4,8,16,32} to test hyperbolic compression claim.
- [E4 — Singularity Detector Validation](./e4-singularity-auroc.md) — Tests σ(x) AUROC against actual failures; proves detector is load-bearing.
- [E5 — IDF Rare-Stratum Weighting Ablation](./e5-idf-ablation.md) — IDF weighting vs uniform; measures rare-state recall improvement.
- [E6 — Group-Quotient Attention](./e6-group-quotient-attention.md) — Orbit-based attention at $O(|V/H|^2)$ cost vs dense $O(n^2)$ baseline.
- [E7 — Reservoir Readout vs End-to-End](./e7-reservoir-vs-end2end.md) — Frozen encoder + linear readout vs full backprop; tests parameter efficiency.
- [E8 — Transfer Experiment](./e8-transfer-experiment.md) — Zero-shot transfer across BabyAI task families; tests whether graph abstracts domain shift.
- [E9 — Full Agent Trace Benchmark](./e9-full-trace-benchmark.md) — Full architecture on BabyAI, ALFWorld, ScienceWorld; integration test.

### Ablation

- [Ablation Matrix A0–A9](./ablation-matrix.md) — Full 10-variant ablation sweep; A0 = full system, A1–A9 each disable one component.

### Dataset Adapters

- [Dataset Adapter — MNIST Typed](./dataset-mnist-typed.md) — sklearn digits: typed labels, margin, confusion graph from logistic regression.
- [Dataset Adapter — Synthetic BabyAI Grid](./dataset-synthetic-babyai-grid.md) — In-process 7-state × 5-task generator; no external dependencies.
- [Dataset Adapter — Real MiniGrid/BabyAI Wrapper](./dataset-minigrid-wrapper.md) — gym.Env wrapper producing typed features from real MiniGrid environments.
- [Dataset Adapter — ALFWorld Loader](./dataset-alfworld-loader.md) — TextWorld + ALFRED loader for embodied household tasks; used by E9.
- [Dataset Adapter — ScienceWorld Loader](./dataset-scienceworld-loader.md) — Loader for elementary-science reasoning tasks; furthest from original design space.
- [Dataset Adapter — Research-Agent Trace Replay](./dataset-trace-replay.md) — Replay offline traces from LLM-based agents; deployment-nearest dataset.

### Infrastructure

- [Metric Collectors](./metric-collectors.md) — Standardized collectors for all metrics; emit to metrics.jsonl.
- [Evidence-Level Tracker](./evidence-tracker.md) — Meta-logger tracking evidence level per major claim across all runs.
- [Minimal Next Experiment](./minimal-next-experiment.md) — The single most actionable next step: 5-model BabyAI comparison. Do this first.

## See also

- [Architecture index](../arch/_index.md)
- [Drivers index](../drivers/_index.md)
- [Open questions](../open/_index.md)
- [Root README](../README.md)
