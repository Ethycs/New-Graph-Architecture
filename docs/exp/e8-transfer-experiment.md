# E8 — Transfer Experiment

**Cluster:** exp
**Status:** spec
**Tags:** #transfer-learning #domain-generalization #structured-abstraction

## What

Train the 5 model variants (flat, typed, graph, graph+singularity, hyperbolic) on a source task family (e.g., BabyAI Goto tasks), then evaluate zero-shot on a related but distinct target family (e.g., Pickup tasks). Measure target risk R_t, transfer gap R_t - R_s, and structured-domain divergence Div(P_s^φ, P_t^φ). Hypothesis: graph and hyperbolic variants have smaller transfer gaps than flat or typed-only baselines, suggesting that geometric abstraction generalizes better.

## Why

Transfer is a hard test of whether the architecture captures something general. A flat policy learns task-specific action patterns; a typed classifier learns semantic field mappings; a graph FSM learns legal state transitions; a hyperbolic graph learns hierarchical task structure. If hyperbolic geometry or the graph FSM truly capture domain-agnostic task structure, they should transfer better to novel domains. This tests whether the structured abstraction has learned something deeper than the training task.

## Interface

**Reads:**
- source task family (e.g., Goto: observation → target_location)
- target task family (e.g., Pickup: observation → target_object)
- task graphs for both families
- trained models from E2 or E1 (all 5 variants)

**Writes:**
- [`metrics.jsonl`](../drivers/metrics-jsonl.md): source_accuracy, target_accuracy, transfer_gap (for each of 5 models)
- [`results.jsonl`](../drivers/results-jsonl.md): per-target-sample predictions, confidence, singularity flags
- transfer matrix: source_family × target_family → accuracy
- domain-divergence plot: P_source vs P_target in embedding space

## Build steps

1. Train all 5 model variants on source family; measure R_s = source_accuracy.
2. Freeze all model weights; evaluate zero-shot on target family; measure R_t.
3. Compute transfer gap: gap_i = R_t[i] - R_s[i] for each model i.
4. Estimate structured-domain divergence: compute graph node embeddings (hyperbolic or Euclidean); measure KL or Wasserstein distance between source and target distributions.
5. Plot gap vs model variant; visualize embedding divergence.
6. Emit metrics and domain-divergence plot.

## Links

- **See also:** [E2 — Real BabyAI / MiniGrid](./e2-real-babyai.md), [E3 — Hyperbolic vs Euclidean Embedding Sweep](./e3-hyperbolic-vs-euclidean.md), [E9 — Full Agent Trace Benchmark](./e9-full-trace-benchmark.md)
- **Drives:** validates structured abstraction for domain generalization
- **Driven by:** (task-family-loader — external)
- **Math:** transfer_gap = R_t - R_s; Div(P_s^φ, P_t^φ) = KL(P_t^φ || mean(P_s^φ, P_t^φ))
- **Open:** [What defines "successful transfer" in E8?](../open/q05-transfer-success-criterion.md), (divergence-metric — open question)
