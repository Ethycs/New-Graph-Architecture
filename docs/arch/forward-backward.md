# Forward-Backward (Phase A E-step)

**Cluster:** arch
**Status:** implemented
**Tags:** #phase-a #baum-welch #em #log-space #soft-credit

## What

Classical Baum-Welch forward-backward in log space. Given an HMM-style sequence with an initial-state distribution $\pi_0$, a transition matrix $A$, and a per-timestep emission log-likelihood matrix $B$ of shape $(T, V)$, the atom computes the per-timestep state posteriors $\gamma_t(i) = P(z_t = i \mid x_{1:T})$ and the per-transition edge expected counts $\xi_t(i, j) = P(z_t = i, z_{t+1} = j \mid x_{1:T})$ that the Bayesian Beta-Dirichlet M-step in [Posterior Mask](./posterior-mask.md) consumes. The whole computation is in log space so underflow is impossible on long sequences.

## Why

The architecture's Bayesian posterior on edge legality needs *soft* transition counts: instead of "edge $(i, j)$ was used at timestep $t$" (a hard observation, which is what `update(s, d, 1.0)` would carry), it needs $\xi_t(i, j)$ — the probability that the unobserved underlying state walked $i \to j$ at time $t$ given the entire observation sequence. Without this, the posterior either over-commits (every observed edge is treated as certain) or sits at uniform forever (if no edge can be observed directly). Forward-backward is the E-step of the EM loop that fits the posterior to the observed corpus; without it, Phase A has no soft credit signal to drive the M-step.

## Interface

- **Input:** initial log-distribution `log_pi0` of shape $(V,)$; transition log-matrix `log_A` of shape $(V, V)$; emission log-likelihood matrix `log_B` of shape $(T, V)$.
- **Output:** `gamma` of shape $(T, V)$ (per-timestep posteriors in log space and / or normalised) and `xi` of shape $(T-1, V, V)$ (transition posteriors). Also the log-marginal $\log P(x_{1:T})$ for monitoring EM convergence.
- **Driven by:** the runner's tokenised observation stream and the current `posterior_mask` log-odds bias.
- **Drives:** [Posterior Mask](./posterior-mask.md)'s Beta-Dirichlet M-step (via `update_batch` on $\xi_t(i, j)$ summed across $t$).

## Build steps

- Forward pass: $\log \alpha_t(j) = \mathrm{logsumexp}_i [\log \alpha_{t-1}(i) + \log A_{ij}] + \log B_t(j)$ with $\log \alpha_1(j) = \log \pi_0(j) + \log B_1(j)$.
- Backward pass: $\log \beta_t(i) = \mathrm{logsumexp}_j [\log A_{ij} + \log B_{t+1}(j) + \log \beta_{t+1}(j)]$, with $\log \beta_T(i) = 0$.
- $\gamma_t(i) = \log \alpha_t(i) + \log \beta_t(i) - \mathrm{logsumexp}_k [\log \alpha_t(k) + \log \beta_t(k)]$.
- $\xi_t(i, j) = \log \alpha_t(i) + \log A_{ij} + \log B_{t+1}(j) + \log \beta_{t+1}(j) - \log P(x_{1:T})$.
- Log-marginal $\log P(x_{1:T}) = \mathrm{logsumexp}_i \log \alpha_T(i)$.
- Use `scipy.special.logsumexp` for stable summation in log space.

## Links

- **See also:** [Posterior Mask](./posterior-mask.md), [Information Geometry](./information-geometry.md), [Graph FSM](./graph-fsm.md).
- **Drives:** every runner with `phase_a_*` metrics; the deterministic FSM-recovery story.
- **Driven by:** the typed FSM (`log_A` derived from `legality_bias()` or the current hard mask).
- **Math:** classical EM for HMMs (Baum-Welch); the M-step on `_alpha, _beta` is the Beta-Dirichlet natural-gradient closed-form update.
- **Open:** what to do when emission likelihoods $\log B_t$ are themselves uncertain — currently they come from the trained classifier head; the proper Bayesian treatment puts a posterior on the heads too.
