# Typed Latent Clustering

**Cluster:** arch
**Status:** implemented
**Tags:** #riemannian-k-means #poincare #type-orbit #self-supervised #phase-19b

## What

Riemannian k-means in the Poincaré ball, with the constraint that latent-cluster migration is restricted to a **type orbit**. Each observation carries a deterministic type label $q_t$ (from the typed FSM) and an inferred latent cluster ID $\lambda_t$. The update rule is: an observation of type $q$ may only be reassigned to a cluster whose own type is $q$. This makes clustering compatible with the architecture's type system — clusters discovered within type A never absorb observations of type B, even if Euclidean geometry would prefer it.

## Why

Phase 19B's load-bearing result depended on this atom: with no disease labels in training, the architecture clustered 4920 patients in the Poincaré ball at K=41 and recovered cluster_purity 0.878 / ARI 0.819 / NMI 0.966 against the ground-truth medical taxonomy. Type-orbit restriction was the structural lever — without it, hyperbolic k-means would have collapsed structurally distinct symptom families into Euclidean-shortest-path clusters that don't respect the FSM's type system. The atom is also the prospective backbone of `docs/proposals/graph-extraction.md`'s vertex-discovery step: cluster hidden states of an arbitrary trained network, then run Phase A on the discovered transitions.

## Interface

- **Input:** Poincaré-ball points (shape $(N, d)$, each row $\|x\| < 1$), optional per-point type labels $q$ (shape $(N,)$).
- **Output:** cluster assignments $\lambda$ (shape $(N,)$, integer in $[0, K)$), cluster centroids in the Poincaré ball (shape $(K, d)$), mean distortion $\frac{1}{N} \sum_i d_{\mathbb{D}}(x_i, c_{\lambda_i})$.
- **Distance:** Poincaré ball metric $d_{\mathbb{D}}(x, y) = \mathrm{arccosh}(1 + 2 \|x - y\|^2 / ((1 - \|x\|^2)(1 - \|y\|^2)))$ with boundary clipping for numerical stability.
- **Centroid update:** Frechet mean in the Poincaré ball, computed by iterated exponential / logarithmic maps.

## Build steps

- Initialise centroids by k-means++ in the tangent space at the origin (the Euclidean approximation is good near the origin).
- Assign step: $\lambda_i = \arg\min_k d_{\mathbb{D}}(x_i, c_k)$, but if type labels are present, restrict $k$ to clusters with matching type.
- Update step: compute Frechet mean per cluster via iterated logarithmic / exponential maps; reproject after each iteration to keep the centroid in the ball.
- Distortion: $\frac{1}{N} \sum_i d_{\mathbb{D}}(x_i, c_{\lambda_i})$ — monotonically decreasing under correct implementation.
- Type-orbit invariant: assert that every cluster's type label is constant across its members.

## Links

- **See also:** [Hyperbolic Embedding](../hyperbolic/hyperbolic-embedding.md), [Hyperbolic Distance Loss](../hyperbolic/hyperbolic-distance-loss.md), [Graph Prototype Vectors](../hyperbolic/graph-prototype-vectors.md).
- **Drives:** Phase 19B / E23 (the self-supervised diagnostic discovery runner); the prospective Phase 20 graph-extraction runner.
- **Driven by:** the typed FSM (for type labels) and any encoder that emits Poincaré-ball points.
- **Math:** Riemannian k-means on the Poincaré disk model; the Frechet mean is the Riemannian centre of mass; type-orbit restriction is the equivariance condition under the trivial group acting per-type.
- **Open:** how to choose K Bayesian-nonparametrically (currently $K$ is a hyperparameter; the graph-extraction proposal calls for stick-breaking DP-mixture).
