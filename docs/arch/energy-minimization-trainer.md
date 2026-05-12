# Energy-Minimisation Trainer

**Cluster:** arch
**Status:** implemented
**Tags:** #energy-based #riemannian-sgd #beta-dirichlet #single-loss

## What

A single-loss trainer that minimises observed-trajectory energy. Instead of optimising a classification cross-entropy, the loss IS the energy at the observed (input → predicted node-tuple) point. A single `step` performs three intertwined updates: (i) Riemannian SGD on prototype positions in the Poincaré ball; (ii) Bayesian Beta-Dirichlet M-step on the [Posterior Mask](./posterior-mask.md); (iii) gradient update on the typed-readout heads. This is the classical (sklearn / numpy) trainer; the torch sibling is [Torch Energy Trainer](./torch-energy-trainer.md).

## Why

The architecture's third commitment is that energy is the single canonical loss. A trainer that optimises cross-entropy on top of an energy-based architecture creates two competing optimisation targets and disconnects the gradient signal from the energy landscape that σ, the mask, and the partition function all reference. Energy-minimisation training is the *single-loss* commitment made concrete: every learnable quantity (prototypes, posterior, heads) moves along the gradient of one scalar.

## Interface

- **Input:** `EnergyMinimizationTrainer(prototypes, posterior_mask, typed_readout, lr, alpha_cost, beta_uncertainty, gamma_contradiction, eta_loop, kappa_progress)` — the energy coefficients are the only training-time hyperparameters beyond `lr`.
- **`step(x, true_next_state)`:** compute the energy at the observed transition; backpropagate via numpy gradients to prototypes, soft-update the posterior via Beta-Dirichlet M-step, and gradient-step the readout heads.
- **`fit(corpus, epochs)`:** loop over the corpus, calling `step` per sample.
- **Output:** updated prototypes (Riemannian-projected back into the Poincaré ball), updated PosteriorMask, updated readout heads.

## Build steps

- Compute energy $E(x, j) = \alpha \cdot \text{cost}(j) + \beta \cdot d_{\mathbb{D}}(\phi(x), p_j) + \gamma \cdot \text{contradiction}(x, j) + \eta \cdot \text{loop_pressure}(j) - \kappa \cdot \text{progress}(j)$.
- Prototype gradient: $\nabla_{p_j} E = \beta \cdot \nabla_{p_j} d_{\mathbb{D}}(\phi(x), p_j)$; project the update through the Riemannian exponential map.
- Posterior update: increment $\alpha_{ij}$ by quality $q$ (a function of $E$ at the observed transition); increment $\beta_{ij}$ by $1 - q$.
- Readout update: gradient on the per-type LogisticRegression head's parameters.
- Reproject prototypes after each step to keep $\|p_j\| < 1 - \epsilon$.

## Links

- **See also:** [Torch Energy Trainer](./torch-energy-trainer.md), [Energy Function](./energy-function-E.md), [Posterior Mask](./posterior-mask.md), [Hyperbolic Embedding](./hyperbolic-embedding.md).
- **Drives:** E14 ListOps and several Phase 7 / 8 runners (classical substrate).
- **Driven by:** the energy coefficients and the [Posterior Mask](./posterior-mask.md) state at step entry.
- **Math:** Riemannian SGD on the Poincaré ball; closed-form Beta posterior update; coordinate descent across (prototypes, posterior, heads).
- **Open:** the relative weighting of the three updates inside a single step — currently they run in fixed sequence per `step`; whether interleaving order matters in practice is unmeasured.
