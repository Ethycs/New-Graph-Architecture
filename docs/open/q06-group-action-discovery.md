# Is the group action H specified or discovered?

**Cluster:** open
**Status:** open
**Tags:** #group-action #quotient #symmetry #discovery

## What

The quotient-attention mechanism ([Orbit-Pair Attention](../arch/orbit-pair-attention.md)) compresses an object set $V$ into orbits under a group action $\Gamma$, reducing all-to-all attention from $O(n^2)$ to $O(|V/\Gamma|^2)$. Is the group action $H$ given a priori (e.g., "these objects are equivalent by grammar"), or should it be discovered from data (e.g., via symmetry detection or representation clustering)?

## Why

- **Specified**: Fast, interpretable, guaranteed correct. Requires domain knowledge.
- **Discovered**: Flexible, learns unanticipated symmetries. Expensive, risk of false equivalence.

The choice affects whether [E6 — Group-Quotient Attention](../exp/e6-group-quotient-attention.md) tests a hard constraint or a soft clustering. It also gates whether the full architecture is self-contained (discovers its own structure) or requires human grammar annotation.

## Interface

**Affected zettels:** [Group Action on Graph](../arch/group-action-on-graph.md), [E6 — Group-Quotient Attention](../exp/e6-group-quotient-attention.md), [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md)

**Decision criteria:**
- Can grammar naturally define equivalence classes (e.g., "all RetrieveTool actions are equivalent")?
- Is there evidence of unspecified symmetry in agent/task data (e.g., tool redundancy)?
- What is cost of discovery (clustering, orbit detection, validation)?
- Target: if specified group exists, use it; else, test learned quotient as ablation.

## Build steps

- Identify task-specific equivalences (e.g., grammar rules, tool classes).
- Implement two paths: (A) hand-specified orbits from grammar, (B) learned orbits via spectral clustering or symmetry detection on state/action embeddings.
- Run [E6 — Group-Quotient Attention](../exp/e6-group-quotient-attention.md) with both; measure: attention FLOPs, latency, accuracy, orbit purity.
- If specified group outperforms learned by margin, lock it; else pursue learned discovery as future work.
- Document group for reproducibility.

## Links

- **See also:** [When does an orbit re-expand from quotient form?](./q08-quotient-reexpansion-threshold.md), [Which BabyAI level set should be the canonical E2 benchmark?](./q04-babyai-dataset-choice.md)
- **Affects:** [Group Action on Graph](../arch/group-action-on-graph.md), [E6 — Group-Quotient Attention](../exp/e6-group-quotient-attention.md), [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md)
- **Math:** [Architecture.md §Quotients, Monodromy, Boundary Memory](../Architecture.md#quotients-monodromy-boundary-memory)
