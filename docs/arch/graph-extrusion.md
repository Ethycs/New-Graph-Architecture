# Graph Extrusion

**Cluster:** arch
**Status:** spec
**Tags:** #geometry #stratification #topology

## What
Convert the discrete graph into a continuous stratified geometric object: each node becomes a cell (open ball, simplex, or manifold patch) in $\mathbb{H}^d$, and each edge becomes a geodesic tube. Extrusion lifts the graph from a 1-skeleton into a higher-dimensional stratified space, enabling continuous deformation and singularity detection at boundaries.

## Why
A discrete graph has no meaningful interior; every point is a vertex or edge. Extrusion creates a stratified manifold with-boundary where singularities appear as violations of Whitney conditions at stratum interfaces. This geometry houses the [Singularity Detector σ(x)](./singularity-detector.md), which evaluates whether observation clusters respect stratum boundaries. Without extrusion, singularities remain abstract; with it, they are geometric codimension-1 events in an ambient space.

## Interface
**Inputs:**
- Hyperbolic embedding $\phi: V(G) \to \mathbb{H}^d$.
- Edge weights $w_e$ and geodesic distances $D_{\text{hyp}}(u, v)$.
- Tubular neighborhood radius $\epsilon$ (per-edge).

**Outputs:**
- Stratified space $X = \bigcup_{v} C_v \cup \bigcup_{e} T_e$ where $C_v$ are cell regions and $T_e$ are geodesic tubes.
- Stratum descriptor: for each point in $X$, its dimension and incidence type.
- [Stratified Partition Function](./stratified-partition-function.md) coordinates.

## Build steps
- Assign each node $v$ a ball $C_v = B_{\epsilon_v}(\phi(v))$ in $\mathbb{H}^d$; choose $\epsilon_v$ proportional to node degree or local curvature.
- For each edge $e = (u,v)$, construct tube $T_e$ of radius $\epsilon_e$ around the geodesic from $\phi(u)$ to $\phi(v)$; ensure tubes overlap with endpoint balls.
- Compute union $X$ and its stratification by dimension: 0-skeleton (nodes), 1-skeleton (edges), 2-strata (tube interiors and ball interiors).
- Mark boundary points: where $C_v$ and $T_e$ meet, or where two tubes touch.
- Tag each observation with stratum type; verify [Stratified Partition Function](./stratified-partition-function.md) conditions.

## Links
- **See also:** [Hyperbolic Embedding](./hyperbolic-embedding.md), [Stratified Partition Function](./stratified-partition-function.md)
- **Drives:** (stratum-respecting-trajectories, boundary-crossing-detection — planned experiments)
- **Driven by:** (typed-graph-contract — external driver)
- **Math:** [Mathematics.md §Stratified Spaces](../Mathematics.md#stratification)
- **Open:** (optimal-extrusion-radius — open question)
