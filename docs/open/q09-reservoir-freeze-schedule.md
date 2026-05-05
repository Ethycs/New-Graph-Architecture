# What is the encoder freeze schedule in reservoir readout?

**Cluster:** open
**Status:** open
**Tags:** #reservoir #encoder #freezing #fine-tuning

## What

In [E7 — Reservoir Readout vs End-to-End](../exp/e7-reservoir-vs-end2end.md), the encoder/LLM is frozen and only the readout head is trained. Should the encoder be frozen from epoch 0, frozen after a warm-up period, or selectively unfrozen on rare strata?

The choice trades sample efficiency (frozen is faster) against feature adaptation (unfrozen learns task-specific representations).

## Why

- **Frozen from epoch 0**: Minimal compute, fast convergence, maximum parameter efficiency (motivation for reservoir computing).
- **Warm-up then freeze**: Readout stabilizes on fixed features, then locked. Balances early plasticity and late stability.
- **Selective unfreeze**: Rare strata get fine-tuned encoder features. Expensive but may improve rare-event recall.

This determines whether the full system is truly "reservoir-like" (pre-trained → frozen → linear readout) or more like fine-tuning with parameter budgets.

## Interface

**Affected zettels:** [Frozen Encoder Backbone](../arch/frozen-encoder-backbone.md), [E7 — Reservoir Readout vs End-to-End](../exp/e7-reservoir-vs-end2end.md), [E5 — IDF Rare-Stratum Weighting Ablation](../exp/e5-idf-ablation.md)

**Decision criteria:**
- Compare: frozen-from-start vs. warm-up-then-freeze vs. selective-unfreeze.
- Measure: samples to convergence, final accuracy, rare-event recall, trainable parameters.
- Check: does selective unfreeze recover compression gain?
- Target: frozen-from-start is viable (parameter efficiency claim holds).

## Build steps

- Implement three schedule variants: (A) always frozen, (B) frozen after epoch N, (C) frozen globally except on orbits with high IDF.
- Train on [E2 — Real BabyAI / MiniGrid](../exp/e2-real-babyai.md) with each; measure: convergence speed, accuracy, rare-recall, parameter count.
- Log which strata/orbits are unfrozen in variant C.
- If A (always frozen) meets accuracy target, claim reservoir property; else adopt B or C and relax claim.
- Document freeze-schedule hyperparameters.

## Links

- **See also:** [How are IDF weights updated at runtime?](./q03-idf-runtime-schedule.md)
- **Affects:** [Frozen Encoder Backbone](../arch/frozen-encoder-backbone.md), [E7 — Reservoir Readout vs End-to-End](../exp/e7-reservoir-vs-end2end.md), [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md)
- **Math:** [Experiments.md §E7-Reservoir-Readout-Experiment](../Experiments.md#e7--reservoir-readout-experiment)
