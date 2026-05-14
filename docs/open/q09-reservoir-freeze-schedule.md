# What is the encoder freeze schedule in reservoir readout?

**Cluster:** open
**Status:** resolved-2026-05-14 — no frozen-encoder commitment
**Tags:** #reservoir #encoder #freezing #fine-tuning

## Status update — 2026-05-14 (commit 8d67c17)

Resolved by policy change. Commit `8d67c17` — "Drop frozen-encoder commitment from the TPN architectural contract" — formally retires the requirement that the encoder be frozen. The substrate is now a free design variable.

Empirical context that motivated the change:

- Phase 20 Wave-B (frozen encoder) — frozen-encoder substrate carries non-trivial-but-incomplete FSM information; cluster purity ≈ 0.79–0.85 but strict Hamming bar not met. See [research_log.md §Phase 20 Wave-B](../../research_log.md).
- Phase 20 Wave-C (trained from scratch) — substrate jumps materially over the frozen baseline; strict bar still misses but the gap is clustering, not extraction. See [research_log.md §Phase 20 Wave-C](../../research_log.md).
- Phase 24 (frozen pretrained GPT-2) — frozen pretrained substrate **decisively beats** the from-scratch transformer and matches the Wave-C state-conditioned MLP. See [research_log2.md §Phase 24](../../research_log2.md).

Net: the "reservoir" framing (frozen-from-epoch-0, train only readout) is one of several supported configurations, not an architectural commitment. The TPN contract no longer asserts it; downstream notes ([Frozen Encoder Backbone](../arch/substrate/frozen-encoder-backbone.md), [E7 — Reservoir Readout vs End-to-End](../exp/e7-reservoir-vs-end2end.md)) should be re-read in light of this.

## What

In [E7 — Reservoir Readout vs End-to-End](../exp/e7-reservoir-vs-end2end.md), the encoder/LLM is frozen and only the readout head is trained. Should the encoder be frozen from epoch 0, frozen after a warm-up period, or selectively unfrozen on rare strata?

The choice trades sample efficiency (frozen is faster) against feature adaptation (unfrozen learns task-specific representations).

## Why

- **Frozen from epoch 0**: Minimal compute, fast convergence, maximum parameter efficiency (motivation for reservoir computing).
- **Warm-up then freeze**: Readout stabilizes on fixed features, then locked. Balances early plasticity and late stability.
- **Selective unfreeze**: Rare strata get fine-tuned encoder features. Expensive but may improve rare-event recall.

This determines whether the full system is truly "reservoir-like" (pre-trained → frozen → linear readout) or more like fine-tuning with parameter budgets.

## Interface

**Affected zettels:** [Frozen Encoder Backbone](../arch/substrate/frozen-encoder-backbone.md), [E7 — Reservoir Readout vs End-to-End](../exp/e7-reservoir-vs-end2end.md), [E5 — IDF Rare-Stratum Weighting Ablation](../exp/e5-idf-ablation.md)

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
- **Affects:** [Frozen Encoder Backbone](../arch/substrate/frozen-encoder-backbone.md), [E7 — Reservoir Readout vs End-to-End](../exp/e7-reservoir-vs-end2end.md), [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md)
- **Math:** [Experiments.md §E7-Reservoir-Readout-Experiment](../Experiments.md#e7--reservoir-readout-experiment)
