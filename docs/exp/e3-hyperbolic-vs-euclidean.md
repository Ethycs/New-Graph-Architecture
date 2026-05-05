# E3 — Hyperbolic vs Euclidean Embedding Sweep

**Cluster:** exp
**Status:** spec
**Tags:** #hyperbolic-geometry #hierarchy #dimension-efficiency

## What

For a curated BabyAI level or synthetic grid-task benchmark, embed task nodes into Euclidean vs hyperbolic spaces (via Poincaré / Lorentz model) with varying dimensions {2, 4, 8, 16, 32}. Classify states by distance to nearest node prototype or edge-tube. Measure accuracy, task-completion rate, and curvature sensitivity. Hypothesis: hyperbolic geometry achieves parity accuracy at half the Euclidean dimension, compressing hierarchical task structure.

## Why

The typed graph may be inherently hierarchical: missions → subgoals → objects → primitive actions. Hyperbolic space naturally embeds trees and hierarchies with exponential compression. If the task graph has tree-like structure, hyperbolic embedding should require fewer learned parameters and transfer better across similar domains. This tests whether geometry is load-bearing beyond the discrete graph itself.

## Interface

**Reads:**
- task graph: edges from E1 or E2
- node labels and node features
- Poincaré/Lorentz embedding code (geoopt or torch)
- test trajectories

**Writes:**
- [`metrics.jsonl`](../drivers/metrics-jsonl.md): accuracy by (geometry, dimension) pair, success_rate, param_count
- [`results.jsonl`](../drivers/results-jsonl.md): per-sample distance-to-node, curvature, radius values
- visualization: embedding in 2D Poincaré disk, distance matrix heatmap
- sweep table: accuracy vs dimension for Euclidean and hyperbolic

## Build steps

1. Parse task graph from E1 or E2; label nodes by task type and role (mission, subgoal, primitive).
2. Initialize Euclidean prototypes: {2,4,8,16,32}-dimensional random; optimize via contrastive loss.
3. Initialize hyperbolic prototypes: same dimensions; optimize via Riemannian SGD (geoopt).
4. For each state in test set, compute distance to all prototypes in both geometries; classify by nearest.
5. Measure accuracy, task success, and illegal-transition rate.
6. Compute geometry sensitivity: accuracy change per curvature parameter.
7. Emit metrics; plot 2D Poincaré embedding for visualization.

## Links

- **See also:** [E2 — Real BabyAI / MiniGrid](./e2-real-babyai.md), [E1 — Synthetic BabyAI Grid Run](./e1-synthetic-babyai.md), [Hyperbolic Embedding](../arch/hyperbolic-embedding.md)
- **Drives:** efficiency claim for hyperbolic structure
- **Driven by:** (task-graph-loader — external), [Hyperbolic Embedding](../arch/hyperbolic-embedding.md)
- **Math:** Poincaré distance = arcosh(1 + 2||u-v||²/((1-||u||²)(1-||v||²))); accuracy vs dim
- **Open:** (curvature-tuning, hierarchy-detection — open questions)
