# Energy-Weighted Loss

**Cluster:** arch
**Status:** spec
**Tags:** #loss #training #weighting

## What

Energy-weighted loss reweights the training objective to prioritize high-energy (singular, difficult, contradictory) states and de-prioritize low-energy (easy, generic) states. If the target state is $x_*$ with energy $E(x_*)$, the loss is $\ell_{\text{ew}}(x_*) = w(E(x_*)) \cdot \ell_{\text{base}}(x_*)$, where $w(E)$ is an increasing function (e.g., $w(E) = e^{E/T'}$ or $w(E) = \max(1, E/E_0)$). This ensures the model learns to navigate difficult regions and is not dragged down by trivial examples.

## Why

Naive cross-entropy loss treats all states equally, wasting capacity on easy states that the model already predicts well. High-energy states (singular, contradictory, loop-prone) are both harder to learn and more important for robust retrieval. By up-weighting them, the model allocates learning capacity proportionally to difficulty. This reduces overfitting to generic states, improves singular-region handling, and stabilizes training by discouraging the model from collapsing to a simple, low-energy attractor.

## Interface

**Inputs:**
- Base loss: $\ell_{\text{base}}(x_*) \in \mathbb{R}_{\geq 0}$ (cross-entropy or other standard loss).
- Energy landscape: $\mathbf{E} \in \mathbb{R}^{|V|}$.
- Target state: $x_* \in V$.
- Weighting function: $w: \mathbb{R} \to \mathbb{R}_{> 0}$ (e.g., exponential, piecewise linear).
- Hyperparameter: temperature or threshold $T' \in \mathbb{R}_{> 0}$.

**Outputs:**
- Weighted loss: scalar $\ell_{\text{ew}} = w(E(x_*)) \cdot \ell_{\text{base}}(x_*) \in \mathbb{R}_{\geq 0}$.
- Per-batch loss: mean over batch samples.

## Build steps

1. **Compute base loss:** For each training example $(x_{\text{prev}}, x_*)$, compute standard loss $\ell_{\text{base}}(x_*) = -\log P_{\theta}(x_* | x_{\text{prev}})$ or similar.
2. **Retrieve energies:** Look up $E(x_*)$ for each target state from the pre-computed energy landscape.
3. **Define weighting function:** Choose $w(E)$. Common options:
   - Exponential: $w(E) = e^{\lambda E}$ for $\lambda > 0$.
   - Hinge: $w(E) = \max(1, E / E_{\text{ref}})$ to avoid overly small weights.
   - Log-linear: $w(E) = 1 + \beta E$ for stability near zero.
4. **Compute weights:** $\mathbf{w} = w(\mathbf{E})$ for all states.
5. **Apply to loss:** $\ell_{\text{ew}}(x_*) = w(E(x_*)) \cdot \ell_{\text{base}}(x_*)$ for each example.
6. **Aggregate:** Compute batch mean $\frac{1}{B} \sum_{b} \ell_{\text{ew}}(x_{*, b})$ and backpropagate.

## Links

- **See also:** [Energy Function E(x)](./energy-function-E.md), [Stratified Partition Function](./stratified-partition-function.md)
- **Drives:** (weighted-loss-ablation — planned experiment)
- **Driven by:** [Energy Function E(x)](./energy-function-E.md)
- **Math:** [Mathematics.md §Importance Weighting](../Mathematics.md#importance-weighting)
- **Open:** (weighting-function-ablation — open question)
