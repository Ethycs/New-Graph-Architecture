# Graph Realization Functor

**Cluster:** arch
**Status:** spec
**Tags:** #realization-functor #singularity-geometry #graph-to-variety

## What
The realization functor $R: \text{Graph}_{\text{enriched}} \to \text{Sing}$ maps a typed/weighted graph to an algebraic-geometric object: a singularity, hypersurface, quiver variety, or plumbed 4-manifold. It assigns the graph geometric meaning by realizing it as a dual resolution graph, a graph hypersurface, or a moduli space of representations. This is the one-way lift from combinatorics to geometry.

## Why
A graph is combinatorial shadow; a singularity is geometric event. Without realization, graphs remain abstract. With it, you inherit singularity-theoretic tools: monodromy, discriminants, vanishing cycles, deformation theory, Milnor fibers. Realization makes the graph testable: you can compute Hodge diamonds, intersection numbers, and topological invariants from the singularity. It also makes agent task structure geometrically tangible.

## Interface
- **Input:** (from [Graph FSM](./graph-fsm.md), [Weighted Plumbing Graph](./weighted-plumbing-graph.md), or [Dynkin / ADE Classification](./dynkin-ade-classification.md)) enriched graph $(G, w_v, g_v, \text{intersection form})$.
- **Output:** (to [Singularity Extraction Functor](./singularity-extraction-functor.md) and diagnostics) singularity object $X_G$ (encoded as defining equations, resolution data, or discriminant), with associated Milnor fiber, monodromy action, or boundary.

## Build steps
- Branch on graph type: (a) plumbing graph → surface singularity via plumbing construction; (b) ADE graph → du Val singularity with explicit equation; (c) generic graph → graph hypersurface via Kirchhoff polynomial.
- (a) For plumbing: use negative-definite intersection matrix $Q$ to construct the plumbed 4-manifold; contract exceptional divisors to yield singular surface $(X_G, 0)$.
- (b) For ADE: use canonical singularity equations (e.g., $A_n$: $x^2 + y^2 + z^{n+1} = 0$) derived from Dynkin type.
- (c) For generic: compute Kirchhoff polynomial $\Psi_G$ from graph (spanning trees, edge labeling); define hypersurface $X_G = \{\Psi_G = 0\}$.
- Compute resolution data: exceptional divisors, blowup centers, coordinate changes.
- Store singularity as equation, divisor class group, or explicit resolution map.

## Links
- **See also:** [Weighted Plumbing Graph](./weighted-plumbing-graph.md), [Dynkin / ADE Classification](./dynkin-ade-classification.md), [Singularity Extraction Functor](./singularity-extraction-functor.md)
- **Drives:** (realization-consistency-test — planned experiment)
- **Driven by:** [Dynkin / ADE Classification](./dynkin-ade-classification.md)
- **Math:** [Architecture.md §Three Routes to Singularities](../Architecture.md#three-routes-to-singularities) — realization via plumbing, ADE, and graph hypersurface.
- **Open:** (which-realization — which route is most interpretable for agents?)
