# What defines "successful transfer" in E8?

**Cluster:** open
**Status:** open
**Tags:** #transfer #ablation #metrics #domain-shift

## What

[E8 — Transfer Experiment](../exp/e8-transfer-experiment.md) compares transfer performance of flat policy, concept bottleneck, graph FSM, and hyperbolic graph across task families. How do we measure success: absolute target accuracy above X%, relative improvement over flat baseline, or relative improvement over supervised target-task performance?

This shapes what "the graph abstracts domain shift" means operationally.

## Why

- **Absolute**: "We achieve 80% on target" is concrete but ignores source difficulty.
- **Relative-to-flat**: "Graph outperforms flat by 5 points" isolates the abstraction gain but conflates source and target difficulty.
- **Relative-to-oracle**: "We close 60% of the source→supervised gap" controls for target task difficulty but requires an oracle.

The choice affects claimed contribution size, ablation design, and comparison fairness across different task pairs.

## Interface

**Affected zettels:** [E8 — Transfer Experiment](../exp/e8-transfer-experiment.md), [Orbit Quotient Space](../arch/group/orbit-quotient-space.md), [Hyperbolic Embedding](../arch/hyperbolic/hyperbolic-embedding.md)

**Decision criteria:**
- Measure source and target performance (flat, graph, oracle).
- Compute three metrics: $R_t$, $R_t - R_{t,\mathrm{flat}}$, $\frac{R_{\mathrm{oracle}} - R_t}{R_{\mathrm{oracle}} - R_{t,\mathrm{flat}}}$.
- Decide which metric is primary (usually relative-to-flat as proxy for abstraction quality).
- Commit to threshold (e.g., "ΔR ≥ 5 points counts as successful").

## Build steps

- Run [E2 — Real BabyAI / MiniGrid](../exp/e2-real-babyai.md) (source) → different BabyAI levels or ALFWorld (target).
- Train: flat policy, graph FSM, hyperbolic graph.
- Compute: $R_s$ (source), $R_t^{\mathrm{flat}}$ (target, flat), $R_t^{\mathrm{graph}}$ (target, graph), $R_{t,\mathrm{oracle}}$ (supervised target).
- Report all three transfer metrics; highlight chosen primary metric.
- If graph transfer beats flat by ≥ 5 points consistently across pairs, call transfer successful.

## Links

- **See also:** [Which BabyAI level set should be the canonical E2 benchmark?](./q04-babyai-dataset-choice.md), [Is the group action H specified or discovered?](./q06-group-action-discovery.md)
- **Affects:** [E8 — Transfer Experiment](../exp/e8-transfer-experiment.md), [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md)
- **Math:** [Experiments.md §E8-Transfer-Experiment](../Experiments.md#e8--transfer-experiment)
