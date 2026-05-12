# Posterior Mask

**Cluster:** arch
**Status:** implemented
**Tags:** #bayesian #beta-bernoulli #legality #mask #phase-a

## What

The Bayesian sibling of [Graph Legality Mask](./graph-legality-mask.md). Each directed edge $(i \to j)$ in the typed graph carries an independent Beta$(\alpha_{ij}, \beta_{ij})$ posterior over its legality probability $\theta_{ij}$. Observations are soft: a transition $(s, d)$ with quality $q \in [0, 1]$ contributes $q$ to $\alpha$ and $(1 - q)$ to $\beta$ — the conjugate update for a fractional Bernoulli trial. Cold start uses Beta(1, 1) (uniform). The class exposes both a drop-in `legality_matrix(threshold)` boolean (strict `>`, the skeptical-prior default) and a `legality_bias()` log-odds tensor for smooth additive masking.

## Why

The hard `GraphFSM.legality_matrix` is oracular: an edge is legal or it is not. Real protocols are uncertain — some edges become legal only after evidence accumulates, and the system needs to **learn** the protocol while operating it. PosteriorMask is the architecture's mechanism for that: it carries Bayesian uncertainty per edge, supports online conjugate updates, and falls back to the hard-mask contract via `legality_matrix(threshold)`. Without it, Phase A has nothing to write into and the mask cannot grow with observation.

## Interface

- **Constructor:** `PosteriorMask(n_vertices, prior_alpha=1.0, prior_beta=1.0)`.
- **Warm start:** `from_legality_matrix(legal, prior_strength=10.0)` — promotes a boolean legality matrix to a Beta posterior.
- **Updates:** `update(src, dst, quality)` and `update_batch(transitions, qualities)` — the conjugate Beta update.
- **Summaries:** `posterior_mean()`, `posterior_variance()`, `posterior_entropy()` — all of shape $(V, V)$.
- **Drop-in mask:** `legality_matrix(threshold=0.5)` returns a boolean $(V, V)$ array using STRICT `>` (skeptical prior); `legality_bias()` returns log-odds clipped to $[-30, +30]$.
- **Identity:** `mask_version_id()` returns a 16-hex-char SHA-256 prefix over $(\alpha, \beta)$ for trace records.

## Build steps

- Beta(1, 1) cold start: `_alpha = _beta = np.full((V, V), 1.0)`.
- `update(src, dst, q)`: increment `_alpha[src, dst] += q`, `_beta[src, dst] += 1 - q`; validate $q \in [0, 1]$.
- `update_batch`: use `np.add.at` to handle repeated $(src, dst)$ pairs unbuffered.
- `legality_matrix(threshold)`: return `posterior_mean() > threshold` — strict `>` is the Bayesian commitment criterion (ties don't commit).
- `legality_bias()`: $\log(\alpha) - \log(\beta)$, clipped at $\pm 30$ for numerical stability.
- `mask_version_id`: SHA-256 of `np.ascontiguousarray(alpha).tobytes() || beta.tobytes()`, first 16 hex chars.

## Links

- **See also:** [Graph Legality Mask](./graph-legality-mask.md), [Forward Backward](./forward-backward.md), [Information Geometry](./information-geometry.md).
- **Drives:** every TPN runner that exercises Phase A; the decision trace's `mask_version_id` field.
- **Driven by:** [Forward Backward](./forward-backward.md) (the E-step that produces soft transition counts).
- **Math:** Beta-Bernoulli conjugacy; Fisher information on the Beta family is $\alpha + \beta$; the Beta-Dirichlet M-step is the natural-gradient closed-form update.
- **Open:** [[open.qNN-posterior-prior-strength]] — how to choose `prior_strength` on warm-start from a hand-authored legality matrix.
