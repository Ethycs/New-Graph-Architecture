# Hyperbolic Distance Loss

**Cluster:** arch
**Status:** spec
**Tags:** #loss #training #embedding

## What
Training objective that pulls observations toward their typed state prototype in $\mathbb{H}^d$ and pushes non-matching observations away, measured in hyperbolic distance. The loss balances attraction to correct prototype, repulsion from incorrect ones, and geometric regularization of the embedding structure.

## Why
Hyperbolic distance naturally captures hierarchy and margin; a small distance in $\mathbb{H}^d$ means nodes are "semantically close" in the tree structure. This loss ensures that observations cluster around their type prototype, creating tight decision boundaries for [Singularity Detector σ(x)](./singularity-detector.md). Without this loss, prototypes and embeddings drift, and singularities (contradictions, low-margin states) become indistinguishable from normal states.

## Interface
**Inputs:**
- Observation batch $\{(v_i, \mathbf{o}_i, y_i)\}$ where $v_i$ is node index, $\mathbf{o}_i \in \mathbb{H}^d$ is observed point, $y_i \in \{0, 1\}$ is error label.
- Prototype map $\mathbf{p}: V \to \mathbb{H}^d$.
- Margin parameter $m > 0$ and temperature $\tau$.

**Outputs:**
- Scalar loss $\mathcal{L}_{\text{hyp}}$.
- Gradient flow for embedding $\phi$ and prototypes $\mathbf{p}$.

## Build steps
- For each observation $(v_i, \mathbf{o}_i, y_i)$, compute distances to prototype and to nearest off-type prototype: $d^+ = D_{\text{hyp}}(\mathbf{o}_i, \mathbf{p}_{v_i})$ and $d^- = \min_{v' \neq v_i} D_{\text{hyp}}(\mathbf{o}_i, \mathbf{p}_{v'})$.
- Attraction term: $\mathcal{L}_+ = \mathbb{E}[(d^+)^2]$ over all observations.
- Contrastive term: $\mathcal{L}_- = \mathbb{E}[\max(0, m - d^-)^2]$; push off-type prototypes beyond margin.
- Error weighting: upweight loss for $y_i = 1$ (error observations) by factor $w_{\text{err}}$.
- Regularization: $\lambda_{\text{embed}} \|\text{grad}_\phi D_{\text{hyp}}\|^2$ to smooth embedding; $\lambda_{\text{proto}} \sum_v \|\mathbf{p}_v - \phi(v)\|^2$ to anchor prototypes.
- Total: $\mathcal{L}_{\text{hyp}} = \mathcal{L}_+ + \alpha \mathcal{L}_- + \lambda_{\text{embed}} + \lambda_{\text{proto}}$.

## Links
- **See also:** [Graph Prototype Vectors](./graph-prototype-vectors.md), [Hyperbolic Embedding](./hyperbolic-embedding.md)
- **Drives:** (margin-calibration, prototype-separation — planned experiments)
- **Driven by:** (observation-log — external input)
- **Math:** [Mathematics.md §Riemannian Optimization](../Mathematics.md#riemannian)
- **Open:** (margin-schedule — open question)
