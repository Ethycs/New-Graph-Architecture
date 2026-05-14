# Stratified Partition Function

**Cluster:** arch
**Status:** spec
**Tags:** #stratification #energy #probability #inference

## What

The merged Phase 5 module that does two things in one pass over the extruded state manifold $X$:

1. **Whitney stratification.** Carves $X = \bigsqcup_\lambda S_\lambda$ into smooth strata satisfying Whitney conditions $a$ and $b$, using the cell/tube decomposition produced by [Graph Extrusion](../hyperbolic/graph-extrusion.md). Each stratum $S_\lambda$ is a connected smooth piece of fixed dimension and intrinsic type (cell interior, edge interior, boundary, etc.).
2. **Stratified partition function.** Computes the per-stratum partition $Z_\lambda = \int_{S_\lambda} e^{-E(x)/T}\,dx$ from the energy field $E$ produced by [Energy Function E(x)](energy-function-E.md), then $Z = \sum_\lambda Z_\lambda$ and $P(\lambda) = Z_\lambda / Z$. The point-wise probability is $P(x) = e^{-E(x)/T} / Z$.

The two computations live in one atom because they co-depend: the partition function is meaningless without a stratification to sum over, and the geometric stratification is unused machinery without the partition function consuming it. They share the same iteration over $X$ and the same numerical-stability tricks (log-sum-exp, max subtraction).

## Why

Behavioral tags from [Behavioral Stratum Tagger](../singularity/behavioral-stratum-tagger.md) tell you *what kind* of state you are in. The stratified partition function tells you *how likely* each kind is, given the energy landscape. Together they give a calibrated distribution $P(\lambda)$ over behavioral regimes — the actual object the agent uses for routing, beam search, and likelihood-based training.

Splitting these into separate atoms duplicates the $X$-traversal, leaves Whitney stratification with no consumer of its own, and creates a coordination surface (which atom owns numerical stability? which atom owns the $\lambda$ index?) that has no payoff. Merging them resolves [build-order consideration #3](../../build-order.md#build-order-considerations-surfaced-during-driver-polish): the geometric Whitney content lives only where it is actually consumed, and Phase 2 uses the lightweight categorical [Behavioral Stratum Tagger](../singularity/behavioral-stratum-tagger.md) instead.

## Interface

**Inputs:**
- Extruded manifold $X = \bigcup_v C_v \cup \bigcup_e T_e$ from [Graph Extrusion](../hyperbolic/graph-extrusion.md).
- Hyperbolic embedding map from [Hyperbolic Embedding](../hyperbolic/hyperbolic-embedding.md).
- Energy field $E: X \to \mathbb{R}$ from [Energy Function E(x)](energy-function-E.md).
- Inverse temperature $T \in (0, \infty)$ from [Config](../../drivers/config.md)`.temperature` (may be annealed).
- Optional pre-computed cache of stratum bases.

**Outputs:**
- Stratification index $\lambda(x)$: which stratum each point belongs to.
- Per-stratum partition $Z_\lambda$ and total $Z$.
- Per-stratum probability $P(\lambda) = Z_\lambda / Z$.
- Point-wise probability $P(x) = e^{-E(x)/T} / Z$.
- Log-partition $\log Z$ (numerically stable).
- Whitney-condition residual per stratum boundary (diagnostic).

**Where it writes:** scalar $Z$, per-stratum $Z_\lambda$ and $P(\lambda)$, and per-step $P(x)$ go into [metrics.jsonl](../../drivers/metrics-jsonl.md) and [results.jsonl](../../drivers/results-jsonl.md) respectively.

## Build steps

1. **Stratify $X$.** Enumerate strata: 0-cells (node interiors), 0-boundaries (node boundary spheres), 1-cells (edge tube interiors), 1-boundaries (cell-tube interfaces), higher-dim cells if extrusion produces them. Each stratum gets a stable integer label.
2. **Project observations.** For each $x$, find the nearest stratum and record $\lambda(x)$ + Whitney residual.
3. **Compute energies.** Evaluate $E(x)$ via [Energy Function E(x)](energy-function-E.md) on a sample (or quadrature grid) within each stratum.
4. **Per-stratum partition $Z_\lambda$.** Stable log-sum-exp: $\log Z_\lambda = \log\sum_{x_i \in S_\lambda} e^{-E(x_i)/T}$, after subtracting $\max_i E(x_i)$ for numerical stability.
5. **Aggregate to total.** $\log Z = \mathrm{logsumexp}_\lambda(\log Z_\lambda)$.
6. **Probabilities.** $P(\lambda) = e^{\log Z_\lambda - \log Z}$; $P(x) = e^{-E(x)/T - \log Z}$.
7. **Validate Whitney conditions** $a$ and $b$ at boundaries: limit of tangent spaces of higher-dim strata is contained in tangent space of adjacent lower-dim stratum. Flag violations to a diagnostic stream.
8. **Validate normalization.** Assert $\sum_\lambda P(\lambda) = 1$ within $10^{-6}$. Flag failures hard (this is the canary for numerical bugs).

## Build order note

This atom replaces both the old `whitney-stratification-layer` (P2/P3) and the old `partition-function-Z` (P5). It lands once, in Phase 5, after [Graph Extrusion](../hyperbolic/graph-extrusion.md) provides the manifold and [Energy Function E(x)](energy-function-E.md) provides the field. Phase 2 uses the simpler [Behavioral Stratum Tagger](../singularity/behavioral-stratum-tagger.md), which is decoupled from this atom.

## Links

- **See also:** [Energy Function E(x)](energy-function-E.md), [Energy-Weighted Loss](energy-weighted-loss.md), [Graph Extrusion](../hyperbolic/graph-extrusion.md), [Behavioral Stratum Tagger](../singularity/behavioral-stratum-tagger.md)
- **Drives:** [Energy-Weighted Loss](energy-weighted-loss.md), (inference-sampling — planned experiment), [E7 — Reservoir vs end-to-end](../../exp/e7-reservoir-vs-end2end.md)
- **Driven by:** [Energy Function E(x)](energy-function-E.md), [Graph Extrusion](../hyperbolic/graph-extrusion.md), [Hyperbolic Embedding](../hyperbolic/hyperbolic-embedding.md), [Config](../../drivers/config.md)
- **Math:** [Mathematics.md §Whitney Stratification](../../Mathematics.md#whitney-stratification-and-group-action-orbit-types), [Mathematics.md §Boltzmann Distribution](../../Mathematics.md#boltzmann)
- **Open:** [q02 — energy function spec](../../open/q02-energy-function-spec.md), (temperature-annealing-schedule — open question)
