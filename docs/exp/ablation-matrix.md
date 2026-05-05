# Ablation Matrix A0–A9

**Cluster:** exp
**Status:** spec
**Tags:** #ablation #load-bearing #component-validation

## What

Run the full architecture on E2 (BabyAI) or E1 (synthetic) with 10 variants: A0 (full system), then A1–A9 each disabling one component (graph mask, typed grammar, uncertainty distribution, singularity detector, IDF weighting, Euclidean instead of hyperbolic, no group quotienting, end-to-end instead of reservoir, no trace history). Measure accuracy, parameter count, FLOPs, sample efficiency, illegal-transition rate, and failure-prediction AUROC. A component is load-bearing iff its removal hurts ≥2 metrics by ≥3%.

## Why

The efficiency claims are only credible if each component is demonstrably load-bearing. The ablation matrix is the definitive proof: if removing a piece leaves performance and efficiency unchanged, it is cargo cult. If every removal hurts something (compute, accuracy, sample efficiency, robustness, or interpretability), the architecture is load-bearing. This experiment directly answers: which components matter and how much?

## Interface

**Reads:**
- task traces from E1 or E2
- full trained model (A0)
- 9 model variants (A1–A9) with each component disabled
- test set with ground-truth labels

**Writes:**
- [`metrics.jsonl`](../drivers/metrics-jsonl.md): accuracy, param_count, flops, samples_to_threshold, illegal_rate, failure_auroc (for each A0–A9)
- [`results.jsonl`](../drivers/results-jsonl.md): ablation table as JSON
- ablation summary: matrix of (component × metric) with percent deltas from A0

## Build steps

1. Train full system A0 on all training data; measure and record all metrics.
2. For A1–A9: revert each component and retrain (or load pre-trained variants).
   - A1: remove graph mask (classifier only).
   - A2: remove typed grammar (flat features).
   - A3: remove uncertainty distribution (deterministic prediction).
   - A4: remove singularity detector (no low-margin flagging).
   - A5: remove IDF weighting (uniform sample weights).
   - A6: Euclidean instead of hyperbolic embedding.
   - A7: no group quotienting (dense attention).
   - A8: end-to-end training instead of reservoir readout.
   - A9: no trace history (single-step prediction only).
3. Evaluate each on test set; measure all metrics.
4. Compute delta from A0: (A_i - A0) / A0 for each metric and variant.
5. Emit ablation table; highlight components with delta ≥ 3% on ≥2 metrics.

## Links

- **See also:** [E1 — Synthetic BabyAI Grid Run](./e1-synthetic-babyai.md), [E2 — Real BabyAI / MiniGrid](./e2-real-babyai.md), [E3 — Hyperbolic vs Euclidean](./e3-hyperbolic-vs-euclidean.md), [E4 — Singularity Detector Validation](./e4-singularity-auroc.md), [E5 — IDF Rare-Stratum Weighting](./e5-idf-ablation.md), [E6 — Group-Quotient Attention](./e6-group-quotient-attention.md), [E7 — Reservoir Readout vs End-to-End](./e7-reservoir-vs-end2end.md)
- **Drives:** proves load-bearing claims
- **Driven by:** (task-trace-loader — external), [Graph Legality Mask](../arch/graph-legality-mask.md), [Typed Field Pipeline](../arch/typed-field-pipeline.md), [Hyperbolic Embedding](../arch/hyperbolic-embedding.md), [Singularity Detector σ(x)](../arch/singularity-detector.md)
- **Math:** delta_i,m = (metric_m[A_i] - metric_m[A0]) / metric_m[A0]; load_bearing iff Σ(delta ≥ 3%) ≥ 2
- **Open:** (ablation-threshold — open question)
