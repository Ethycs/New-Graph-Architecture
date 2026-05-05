# Does hyperbolic dimension reduction actually materialize at scale?

**Cluster:** open
**Status:** open
**Tags:** #hyperbolic #efficiency #tradeoff #overhead

## What

Hyperbolic geometry promises to encode hierarchy in fewer dimensions than Euclidean space. But hyperbolic distance computation, gyrovector operations, and Möbius transformations incur per-operation overhead. Does the promised dimensionality reduction actually outweigh the constant-factor overhead when accuracy and latency are jointly optimized?

Or is the benefit illusory at practical task-graph sizes (e.g., 100–10k nodes)?

## Why

If hyperbolic geometry requires $d=16$ to match Euclidean $d=8$ in accuracy, but each hyperbolic operation costs 4× a Euclidean operation, then $16 \times 4 = 64 > 8$, and Euclidean is faster. The claim that "hyperbolic compresses hierarchy" only holds if the true effective cost per sample is lower.

This gates the design choice of [Hyperbolic Embedding](../arch/hyperbolic-embedding.md). If this doesn't hold, revert to Euclidean embedding with learned hierarchy.

## Interface

**Affected zettels:** [Hyperbolic Embedding](../arch/hyperbolic-embedding.md), (euclidean-scoring — arch component), [E3 — Hyperbolic vs Euclidean Embedding Sweep](../exp/e3-hyperbolic-vs-euclidean.md), [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md)

**Decision criteria:**
- Measure: accuracy vs. dimension for Euclidean and hyperbolic on task graphs of varying size.
- Compute: ops/sample for each, total wall-clock latency.
- Find crossover: at what task-graph size does hyperbolic win?
- Target: hyperbolic ≥ Euclidean latency at all practical sizes, or Euclidean is primary.

## Build steps

- Generate or collect task graphs at scales: 64, 256, 1024, 4096 nodes.
- Train classifiers: Euclidean MLP and Poincaré-ball MLP on each graph.
- Measure: test accuracy for each $d$ and metric.
- Time: encoder forward pass, distance computation, softmax, end-to-end.
- Plot: accuracy(d) and latency(d) for both; compute cost-adjusted dimension: $d_{\mathrm{eff}} = d \times (t_{\mathrm{hyp}} / t_{\mathrm{Euclidean}})$.
- If $d_{\mathrm{eff}}^{\mathrm{hyp}} < d^{\mathrm{Euclidean}}$ at target scales, keep hyperbolic; else revert.

## Links

- **See also:** [What hyperbolic dimension is needed for the task graph?](./q01-hyperbolic-dim.md)
- **Affects:** [Hyperbolic Embedding](../arch/hyperbolic-embedding.md), (euclidean-scoring — arch component), [E3 — Hyperbolic vs Euclidean Embedding Sweep](../exp/e3-hyperbolic-vs-euclidean.md)
- **Math:** [Architecture.md §Hyperbolic Task Space and Geometry](../Architecture.md#hyperbolic-task-space-and-geometry)
