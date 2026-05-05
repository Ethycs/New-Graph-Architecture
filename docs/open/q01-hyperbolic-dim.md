# What hyperbolic dimension is needed for the task graph?

**Cluster:** open
**Status:** open
**Tags:** #hyperbolic #geometry #efficiency

## What

What intrinsic dimension $d$ of hyperbolic space $\mathbb{H}^d$ is sufficient to embed a research-agent task graph without losing hierarchy information? Does $d = 4$ to $8$ suffice, or does required dimension scale with graph size, branching factor, or task complexity?

The question is empirical and architectural: at what dimension does hyperbolic compression break even against the overhead of Poincaré-ball arithmetic?

## Why

Hyperbolic geometry promises efficient hierarchy encoding—volume grows exponentially, so tree-like branching separates cleanly. But hyperbolic distance computation, inversion, and gyrovector operations are expensive compared to Euclidean arithmetic. If the dimension needed is large (e.g., $d > 20$) or scales with graph size, the speedup evaporates.

The answer gates [E3 — Hyperbolic vs Euclidean Embedding Sweep](../exp/e3-hyperbolic-vs-euclidean.md) and affects whether [Hyperbolic Embedding](../arch/hyperbolic-embedding.md) is viable for real agents.

## Interface

**Affected zettels:** [Hyperbolic Embedding](../arch/hyperbolic-embedding.md), [Typed Field Pipeline](../arch/typed-field-pipeline.md), [E3 — Hyperbolic vs Euclidean Embedding Sweep](../exp/e3-hyperbolic-vs-euclidean.md)

**Decision criteria:** 
- Measure embedding distortion (geodesic vs. graph distance) at d = 2, 4, 8, 16.
- Compute per-operation latency vs. Euclidean baseline.
- Determine if d grows with $|V|$ (nodes), branching, or depth.
- Target: equal or better accuracy with dim reduction ≥ 2×.

## Build steps

- Generate synthetic hierarchical task graphs at multiple sizes (16, 64, 256, 1024 nodes).
- Embed each in $\mathbb{H}^d$ for $d \in \{2,4,8,16\}$ using Poincaré ball with Adam on curvature parameter.
- Measure distortion: $\sum_{(u,v)} |d_{\mathbb{H}}(z_u, z_v) - d_G(u,v)|^2 / |E|$.
- Run singularity-detection classifier on latent $z \in \mathbb{H}^d$ vs. flat baseline; report accuracy-latency tradeoff.
- Sweep hyperbolic curvature $c \in (0, 1]$ to find optimal curvature per $d$.

## Links

- **See also:** [How are IDF weights updated at runtime?](./q03-idf-runtime-schedule.md), [Does hyperbolic dimension reduction actually materialize at scale?](./q11-hyperbolic-vs-euclidean-tradeoff.md)
- **Affects:** [Hyperbolic Embedding](../arch/hyperbolic-embedding.md), [E3 — Hyperbolic vs Euclidean Embedding Sweep](../exp/e3-hyperbolic-vs-euclidean.md)
- **Math:** [Architecture.md §Hyperbolic Task Space](../Architecture.md#hyperbolic-task-space-and-geometry)
