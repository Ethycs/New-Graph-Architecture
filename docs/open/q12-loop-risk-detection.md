# How is "loop risk" scored in σ(x)?

**Cluster:** open
**Status:** resolved-2026-05-12 — variant (A) N-gram on regime graph
**Tags:** #loop-risk #singularity #detection #infinite-loops

## Status update — 2026-05-12 (Phase 23e)

De-facto answered: **variant (A) — Markov N-gram history**, applied at the regime-graph level rather than the raw action level. Phase 23e wires loop-risk into the PCG-X regime trace as: "predicted regime revisited within the last 5 same-program steps." See [research_log2.md §Phase 23e](../../research_log2.md) for the implementation in `_emit_regime_decision_trace`.

In the 5-grammar sweep, loop-risk was rarely the dominant σ contributor; the illegal signal (`(current_regime, predicted_regime) ∉ edge set`) is the cleanest separator of train vs. held-out, and σ is bimodal so the loop-risk term mainly adds to the ABSTAIN tail. Variants (B) autocorrelation, (C) learned head, and (D) ensemble were not implemented — the simple N-gram on the regime graph hit the cheapest-thing-that-works bar.

## What

The singularity score $\sigma(x)$ includes a loop-risk term to detect when an agent is about to enter or is trapped in a nonproductive cycle (e.g., endless retrieval, repeated tool calls). Is loop risk scored via:
- **(A)** Markov N-gram history: did the last N actions repeat?
- **(B)** Autocorrelation: are recent state visits correlated with earlier ones?
- **(C)** Learned head: a separate classifier trained to detect loop-prone regions?
- **(D)** Combination with fallback logic?

The choice trades simplicity (A, B) against adaptivity (C).

## Why

Loop detection is critical for agent safety. A poorly tuned detector either triggers false alarms (stops productive cycles) or misses real loops (agent hangs). The scoring method determines responsiveness and generalization to unseen loop patterns.

This component contributes to [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md) and is tested in [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md) and [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md).

## Interface

**Affected zettels:** [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md), [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md), [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md)

**Decision criteria:**
- Measure: loop detection recall (catch actual loops) and false-alarm rate (avoid stopping valid repetition).
- Compare: N-gram, autocorrelation, learned, hybrid.
- Check: does choice generalize to unseen loop patterns?
- Target: recall > 80%, false-alarm rate < 5%.

## Build steps

- Define ground truth: label agent trajectories with loop/no-loop (e.g., consecutive identical states, repeated action sequences, or manual annotation).
- Implement: (A) N-gram checker (repeat threshold), (B) autocorrelation on state visit counts, (C) supervised classifier on trajectory features, (D) ensemble.
- Run [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md) with each method.
- Measure: ROC, precision, recall, latency per loop check.
- If N-gram or autocorrelation achieves target, use it (simpler); else adopt learned classifier.
- Document threshold or model weights.

## Links

- **See also:** [Does σ(x) get weighted into loss, detection-only, or scheduled?](./q07-singularity-loss-weighting.md), [Does singularity score σ(x) beat margin alone?](./q10-failure-prediction-baseline.md)
- **Affects:** [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md), [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md), [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md)
- **Math:** [Architecture.md §Catastrophe-Theoretic Enrichment](../Architecture.md#catastrophe-theoretic-enrichment)
