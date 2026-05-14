# Information Geometry

**Cluster:** arch
**Status:** implemented
**Tags:** #fisher #kl #cramer-rao #natural-gradient #beta-bernoulli #phase-7

## What

Information-geometric primitives — Fisher information, KL divergence, and Cramér-Rao bounds — on the statistical manifolds the architecture inhabits. Two manifolds are load-bearing: the unit interval (the edge-Bernoulli manifold), where Fisher information is $I(\theta) = \alpha + \beta$ and the natural metric is the Beta-conjugate diagonal, and the open Poincaré ball (the prototype manifold), where Fisher coincides with the hyperbolic metric tensor. Together they let the architecture set learning rates by natural gradient, measure KL between posterior states, and use Cramér-Rao as a sample-sufficiency stopping rule.

## Why

The architecture has been using information geometry implicitly since Phase 5 — Beta posteriors, hyperbolic prototypes, KL surprise as a σ component. Phase 7 named the framework that was already there: Fisher information is the natural unit of "how much evidence has this edge accumulated," and Cramér-Rao is the natural unit of "is this estimate certain enough to commit." Without an explicit info-geometry atom, every other atom has to re-derive these quantities ad hoc; with it, they share a single set of primitives and the natural-gradient learning rates and sample-sufficiency tests become uniform across the codebase.

## Interface

- **Fisher information** on Beta-Bernoulli edges: `fisher_beta(alpha, beta) -> alpha + beta` (element-wise, shape $(V, V)$).
- **Natural gradient** on the Beta posterior: $\tilde \nabla = (1 / (\alpha + \beta)) \cdot \nabla$.
- **KL between Beta posteriors** $\mathrm{KL}(\text{Beta}(a_1, b_1) \| \text{Beta}(a_2, b_2))$ in closed form via $\log \mathrm{B}$ and digamma.
- **Cramér-Rao bound**: given a target variance, the minimum sample count is $N \geq 1 / (I(\theta) \cdot \text{target_var})$; the atom exposes `crb_min_samples(I, target_var)`.
- **`kl_surprise(prev_mean, new_mean)`**: scalar surprise signal consumed by [Singularity Detector](../singularity/singularity-detector.md).
- **Fisher on the Poincaré ball**: the inverse-conformal-factor diagonal $4 / (1 - \|x\|^2)^2$ on the unit ball; consumed by Riemannian SGD.

## Build steps

- Implement closed-form Fisher / natural-gradient / KL on Beta via `scipy.special.betaln`, `digamma`.
- Implement the unit-ball Fisher in numpy with boundary-safe clipping (the metric blows up at $\|x\| = 1$; clip to $1 - \epsilon$).
- Provide `crb_satisfied_fraction(I_per_edge, target_var)` for runner-level reporting.
- Expose a `KLSurprise` adapter that takes a `PosteriorMask` snapshot before and after an update.

## Links

- **See also:** [Posterior Mask](../energy/posterior-mask.md), [Hyperbolic Embedding](hyperbolic-embedding.md), [Singularity Detector](../singularity/singularity-detector.md).
- **Drives:** the natural-gradient learning rates in [Energy Minimisation Trainer](../energy/energy-minimization-trainer.md); the σ_kl_surprise component; the `crb_satisfied_fraction` reporting in Phase 7 runners.
- **Driven by:** the current PosteriorMask state and the prototype embeddings.
- **Math:** Fisher-Rao geometry on exponential families (Amari); KL is the canonical loss on the statistical manifold; Cramér-Rao bounds give the inverse-Fisher minimum variance.
- **Open:** when conjugacy breaks (recursive TPN, see `docs/proposals/graph-extraction.md`), what variational approximation to natural gradient should the architecture default to?
