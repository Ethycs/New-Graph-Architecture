# Energy Function E(x)

**Cluster:** arch
**Status:** spec
**Tags:** #energy #pressure #landscape

## What

The energy function $E(x) = \alpha \cdot \text{cost}(x) + \beta \cdot \text{uncertainty}(x) + \gamma \cdot \text{contradiction}(x) + \eta \cdot \text{loop}(x) - \kappa \cdot \text{progress}(x)$ assigns a scalar pressure to each state in a retrieval graph. High energy indicates difficult, contradictory, or loopy regions; low energy marks confident, progress-rich regions. Coefficients $\alpha, \beta, \gamma, \eta, \kappa$ are hyperparameters or learnable weights. The energy landscape is the scalar field $E: V \to \mathbb{R}$.

## Why

Energy functions provide a unified, interpretable metric for training prioritization and inference search. Instead of treating all states equally, energy guides the model toward high-progress, low-contradiction regions and away from loops and redundancy. This enables energy-weighted loss (up-weighting singular/high-energy states during training) and energy-based sampling for inference (Boltzmann-like distributions over next steps). Without energy, the model has no principled way to trade off competing objectives (cost vs. uncertainty vs. progress).

## Interface

**Inputs:**
- State $x \in V$ with features: cost (latency), uncertainty (entropy), contradiction (logic conflict), loop count, progress (distance to solution).
- Hyperparameters: $\alpha, \beta, \gamma, \eta, \kappa \in \mathbb{R}_{\geq 0}$.

**Outputs:**
- Scalar energy $E(x) \in \mathbb{R}$.
- Energy landscape: vector of energies for all states.
- Derivatives $\nabla_{\text{features}} E$ for feature-aware optimization.

## Build steps

1. **Compute scalar features:** For each state $x$, measure cost (e.g., inference latency, number of API calls), uncertainty (Shannon entropy of next-token distribution), contradiction (count of conflicts in logical constraints), loop depth (nesting level in repeat structures), and progress (distance metric to solution, e.g., edit distance or task completion percentage).
2. **Normalize features:** Standardize each feature to zero mean, unit variance across the training graph to prevent scale domination.
3. **Define coefficients:** Set or learn $\alpha, \beta, \gamma, \eta, \kappa$. Start with uniform weights; tune via ablation or meta-learning.
4. **Compute energy:** For each state, compute the weighted sum $E(x) = \alpha \cdot c(x) + \beta \cdot u(x) + \gamma \cdot \text{contr}(x) + \eta \cdot \text{loop}(x) - \kappa \cdot p(x)$.
5. **Validate landscape:** Plot or histogram the energy distribution. Check that high-energy states are indeed problematic (e.g., high error rate in downstream experiments).
6. **Integrate with loss:** Pass energies to [Energy-Weighted Loss](energy-weighted-loss.md) and partition function [Stratified Partition Function](stratified-partition-function.md).

## Links

- **See also:** [Stratified Partition Function](stratified-partition-function.md), [Energy-Weighted Loss](energy-weighted-loss.md)
- **Drives:** [Stratified Partition Function](stratified-partition-function.md), [Energy-Weighted Loss](energy-weighted-loss.md), (energy-landscape-viz — planned experiment)
- **Driven by:** (feature-extraction-spec — external input)
- **Math:** [Mathematics.md §Energy Models](../../Mathematics.md#energy)
- **Open:** (feature-weighting-ablation — open question)
