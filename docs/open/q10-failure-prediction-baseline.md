# Does singularity score σ(x) beat margin alone for failure prediction?

**Cluster:** open
**Status:** open
**Tags:** #singularity #failure-prediction #margins #e4-load-bearing

## What

The composite singularity score $\sigma(x) = w_1 \cdot m^{-1} + w_2 \cdot \mathrm{contradiction} + w_3 \cdot \mathrm{loop\_pressure} + w_4 \cdot \mathrm{illegal\_pressure}$ combines multiple signals. Is $\sigma$ strictly better than margin alone for predicting agent failures, or does margin capture most of the signal with less complexity?

This is the **E4 load-bearing question**: does singularity theory actually improve failure prediction, or is it theoretical ornamentation?

## Why

If margin alone achieves > 85% AUROC for failure detection, the composite $\sigma$ adds theoretical complexity without empirical gain. If $\sigma$ consistently outperforms margin by > 5 AUROC points, singularity theory is justified as a core design principle.

The answer determines whether [Singularity Detector σ(x)](../arch/singularity-detector.md) is kept, refined, or replaced with margin-only uncertainty.

## Interface

**Affected zettels:** [Singularity Detector σ(x)](../arch/singularity-detector.md), [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md), [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md)

**Decision criteria:**
- Measure AUROC (failure detection) for: margin alone, entropy alone, full $\sigma(x)$.
- Compute: precision@top-k, recall@fixed-FPR, cost saved by early intervention.
- Ablate $\sigma$ components; attribute improvement to each term.
- Target: $\sigma$ beats margin by ≥ 5 AUROC points, or margin is sufficient and $\sigma$ is retired.

## Build steps

- Run [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md) on [E2 — Real BabyAI / MiniGrid](../exp/e2-real-babyai.md), [E8 — Transfer Experiment](../exp/e8-transfer-experiment.md), and [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md).
- Define failure: illegal transition, timeout, or low-quality action selected.
- Compute: AUROC(margin), AUROC($\sigma$), component ablations.
- Fit weights $w_i$ on validation set via logistic regression.
- Report confusion matrix: high-margin-high-$\sigma$, high-margin-low-$\sigma$, etc.
- If $\sigma$ ≥ margin + 5 AUROC, keep; else simplify to margin.

## Links

- **See also:** [Does σ(x) get weighted into loss, detection-only, or scheduled?](./q07-singularity-loss-weighting.md), [How is the energy function E(x) specified or learned?](./q02-energy-function-spec.md)
- **Affects:** [Singularity Detector σ(x)](../arch/singularity-detector.md), [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md), [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md)
- **Math:** [Experiments.md §E4-Singularity-Detector](../Experiments.md#e4--singularity-detector)
