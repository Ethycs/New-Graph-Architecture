# Weighted Plumbing Graph

**Cluster:** arch
**Status:** spec
**Tags:** #weighted-graph #singularity-resolution #intersection-form

## What
A weighted plumbing graph is a variant FSM enrichment where each vertex and edge carries an intersection-form weight and genus annotation. The weights must satisfy negative-definiteness constraints to enable realization as a surface singularity resolution graph. Unlike bare weighted task graphs, plumbing graphs are designed to lift task structure into algebraic geometry via the realization functor.

## Why
Not every abstract weighted graph can be realized as a singularity. Plumbing weights encode the dual graph of an exceptional divisor in a resolution; negative-definite forms guarantee the graph comes from a normal surface singularity. This ensures that when we realize a task graph as a geometric object, the realization is coherent and meaningful. Without plumbing structure, a graph is disconnected from singularity theory.

## Interface
- **Input:** (from [Graph FSM](graph-fsm.md)) typed graph with basic weights; optionally genus and self-intersection data.
- **Output:** (to [Graph Realization Functor](graph-realization-functor.md)) weighted plumbing graph $(G, w_v, g_v, Q)$ where $w_v$ is self-intersection, $g_v$ is genus per vertex, and $Q$ is the intersection matrix (negative definite).

## Build steps
- For each vertex $v$, assign self-intersection weight $w_v \in \mathbb{Z}_{\leq -2}$ (or -1 for rational curves). Compute genus $g_v$ (typically 0 for trees, $\geq 0$ for cyclic components).
- Assemble the intersection matrix $Q_{ij}$ where $Q_{ii} = w_i$ (self-intersection) and $Q_{ij} = e_{ij}$ (number of transverse intersection points between divisors $i, j$).
- Check negative-definiteness: compute eigenvalues of $Q$; all must be negative. If not, refine graph by subdividing edges or adjusting weights.
- Store plumbing data as a labeled graph with matrix payload.
- Validate that the graph is connected and has the right topology for a singularity resolution (tree, tree with cycles allowed per resolution structure).

## Links
- **See also:** [Graph FSM](graph-fsm.md), [Graph Realization Functor](graph-realization-functor.md), [Dynkin / ADE Classification](dynkin-ade-classification.md)
- **Drives:** (singularity-resolution-test — planned experiment)
- **Driven by:** [Graph FSM](graph-fsm.md)
- **Math:** [Architecture.md §Weighted plumbing graph to surface singularity](../../Architecture.md#three-routes-to-singularities) — realization theorems for negative-definite plumbing graphs.
- **Open:** (genus-inference — infer from task graph semantics, or hand-assign?)
