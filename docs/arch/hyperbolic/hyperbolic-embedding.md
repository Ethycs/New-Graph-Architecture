# Hyperbolic Embedding

**Cluster:** arch
**Status:** spec
**Tags:** #geometry #embedding #hierarchy

## What
Embed the discrete task graph into $\mathbb{H}^d$ (Poincaré or Lorentz model) such that tree-likeness and ancestor-descendant hierarchy are preserved as hyperbolic distance. The negative curvature of hyperbolic space gives natural exponential separation and intrinsic hierarchy without explicit supervision.

## Why
Hyperbolic geometry permits exponential growth of degrees of freedom, which matches the branching structure of decision trees and task graphs. A tree embedded in $\mathbb{H}^d$ has distances proportional to tree depth, whereas Euclidean embedding forces metric distortion. This enables implicit hierarchy capture in a single low-dimensional representation, reducing the burden of explicit stratum labels and improving generalization to out-of-distribution task graphs.

## Interface
**Inputs:**
- Typed graph $G_{\text{typed}}$ with edge weights $w_e$ and node types $\tau_v$.
- Hyperbolicity target $\delta$ (tree-likeness tolerance).

**Outputs:**
- Embedding map $\phi: V(G) \to \mathbb{H}^d$, assigning each node a hyperbolic point.
- Curvature parameter $K < 0$ for the model.
- Distance matrix $D_{\text{hyp}}(v, w) = \frac{1}{|K|^{1/2}} \cosh^{-1}(\ldots)$ respecting graph structure.

## Build steps
- Choose model: Poincaré ball (visualization-friendly) or Lorentz hyperboloid (computation-stable).
- Initialize nodes uniformly or by BFS radius from root; place high-degree hubs near origin.
- Run Riemannian gradient descent on edge-preservation loss: $\mathcal{L} = \sum_{(u,v) \in E} (D_{\text{hyp}}(u,v) - w_{uv})^2 + \lambda \sum_{(u,w) \notin E} \max(0, w_{uv} - D_{\text{hyp}}(u,w))$.
- Validate via [Gromov δ Diagnostic](gromov-hyperbolicity-diagnostic.md) to ensure $\delta < \delta_{\max}$.
- Store $\phi$ as normalized coordinates; compute geodesic distance table for downstream use.

## Links
- **See also:** [Graph Prototype Vectors](graph-prototype-vectors.md), [Graph Extrusion](graph-extrusion.md)
- **Drives:** (hyperbolic-validation, hierarchy-recovery — planned experiments)
- **Driven by:** (typed-graph-contract — external driver)
- **Math:** [Mathematics.md §Hyperbolic Geometry](../../Mathematics.md#hyperbolic)
- **Open:** (curvature-selection, embedding-initialization — open questions)
