# Bayesian-Nonparametric K

**Cluster:** arch
**Status:** implemented
**Tags:** #graph-extraction #phase-20 #k-selection #model-selection

## What

Choose the number of clusters $K$ from data. Given an embedding matrix $X$ of shape $(N, d)$ and a candidate range `K_range`, `estimate_k` fits a $K$-means clustering at each candidate and scores it under one of three criteria — held-out negative log-likelihood (the principled "predict unseen samples" criterion), BIC (the classical penalised-likelihood model-selection workhorse), or the chord-distance elbow heuristic (the cheap sanity-check that runs in milliseconds). Returns a `KEstimationResult` with `K_star`, per-K scores, labels at $K_\star$, centroids at $K_\star$, and the criterion name.

## Why

Phase 19B's clustering result used $K = 41$ because we knew there were 41 diseases in the corpus. The graph-extraction proposal (`docs/proposals/graph-extraction.md`) calls for $K$ to come from the data: cluster the hidden states of an arbitrary trained network, choose the K that balances fit against complexity, then run Phase A on the resulting transitions. Without a K-selection atom, every Phase 20 extraction would need a hand-tuned K, contradicting the proposal's universality claim. Three criteria are exposed because they answer different questions: held-out NLL is what we want philosophically; BIC is what we want when held-out data is scarce; the elbow is the heuristic that runs first and tells us roughly where to look.

## Interface

- **`estimate_k(X, K_range, *, criterion='holdout_nll', seed=0, holdout_fraction=0.2, max_iter=50) -> KEstimationResult`** — the only public entry point.
- **`KEstimationResult`** — frozen dataclass with `K_star`, `K_range`, `scores`, `labels`, `centroids`, `criterion`.
- **Criteria:** `"holdout_nll"` (lower is better; the proposal's primary choice), `"bic"` (lower is better; isotropic-Gaussian-of-fixed-variance model), `"elbow"` (within-cluster sum of squared distances; argmax chord-distance is the chosen $K$).

## Build steps

- **k-means++ init** — sample the first centroid uniformly; each subsequent centroid is drawn with probability proportional to its squared distance to the nearest existing centroid.
- **Lloyd iteration** — alternate `assign(X | centroids)` and `update_centroids(X, labels)` until the centroid shift is below tolerance or `max_iter` is reached.
- **Held-out NLL** — split the corpus into train / holdout via a seeded permutation; fit centroids on train; score holdout under an equal-weight isotropic-Gaussian mixture (variance fixed at 1.0 so only the means are free parameters).
- **BIC** — $\text{BIC} = -2 \log L + p \log N$ with $p = K \cdot d$ (means only). Fit on the full corpus.
- **Elbow** — within-cluster sum of squared distances; pick the index whose perpendicular distance to the chord between the (K, score) curve's endpoints is largest.
- **Refit at K_star on the full corpus** so the returned labels / centroids reflect the entire dataset, not just the training split.

## Links

- **See also:** [Hidden State Harvester](./hidden-state-harvester.md), [Typed Latent Clustering](./typed-latent-clustering.md), [Forward Backward](./forward-backward.md).
- **Drives:** the graph-extraction runner (E24, in development); the Phase 20 Tier 1 / Tier 2 / Tier 3 validation tiers in the proposal.
- **Driven by:** the harvested hidden-state matrix produced by [Hidden State Harvester](./hidden-state-harvester.md).
- **Math:** k-means++ initialisation (Arthur & Vassilvitskii 2007); BIC (Schwarz 1978); held-out predictive likelihood under a Gaussian-mixture surrogate is the proper-scoring-rule equivalent of cross-validation; the elbow heuristic is the chord-distance variant of the knee-finding family.
- **Open:** [[open.qNN-dp-mixture]] — replace held-out NLL with a proper Bayesian-nonparametric stick-breaking DP-mixture if held-out NLL turns out to be unstable. The proposal calls this out as the v2 path. The Riemannian-metric extension (k-means in the Poincaré ball) is deferred to a Riemannian variant atom.
