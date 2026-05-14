# Singularity Extraction Functor

**Cluster:** arch
**Status:** spec
**Tags:** #extraction-functor #singularity-to-graph #forgetful-functor

## What
The extraction functor $\Gamma: \text{Sing} \to \text{Graph}_{\text{enriched}}$ maps a singularity (or a discriminant, resolution, Milnor fiber) back to a typed graph. It extracts the combinatorial skeleton: the dual resolution graph, intersection graph, or monodromy graph. This is the inverse direction, a forgetful map that trades geometric richness for combinatorial clarity.

## Why
Singularities contain more information than graphs (equations, analytic structure, deformation families). But that extra data can obscure structure. Extraction recovers the graph—the essential combinatorial object—and discards analytic detail. This enables lossy compression and comparison: two non-isomorphic singularities may have the same resolution graph (as noted in Architecture.md §Round trip B). Extraction is essential for diagnosis and stratification.

## Interface
- **Input:** (from [Graph Realization Functor](graph-realization-functor.md)) singularity object $X_G$ (equations, resolution data, Milnor fiber, or monodromy).
- **Output:** (to [Graph FSM](graph-fsm.md) and [Confusion Graph](../typed/confusion-graph.md)) typed graph $\Gamma(X_G) = (V, E, w_v, g_v, m_v)$ with weights, labels, and monodromy annotations.

## Build steps
- If input is a surface singularity: extract the dual resolution graph from the exceptional divisor configuration (vertices = divisors, edges = intersection points, weights = self-intersections).
- If input is a quiver variety: extract the underlying quiver (vertices = stable dimensions, edges = maps).
- If input is a Milnor fiber or discriminant: extract the vanishing-cycle graph (vertices = homology cycles, edges = their intersections) or monodromy graph (vertices = connected regions, edges = permutations).
- For each vertex/edge, recover or compute the original labels $g_v$ (semantic type) and annotations $m_v$ (monodromy order, etc.).
- Return the enriched graph and annotate it with markers indicating what was lost during extraction (e.g., "du Val type A3, non-unique analytic structure").

## Links
- **See also:** [Graph Realization Functor](graph-realization-functor.md), [Graph FSM](graph-fsm.md), [Confusion Graph](../typed/confusion-graph.md)
- **Drives:** (round-trip-consistency-test — planned experiment)
- **Driven by:** [Graph Realization Functor](graph-realization-functor.md)
- **Math:** [Architecture.md §Round trip B: singularity → graph → singularity](../../Architecture.md#the-two-round-trips) — extraction forgets analytic structure.
- **Open:** (extraction-canonical — canonical extraction, or resolution-dependent?)
