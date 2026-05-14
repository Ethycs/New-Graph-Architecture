# Gromov δ Diagnostic

**Cluster:** arch
**Status:** spec
**Tags:** #validation #geometry #metric

## What
Compute the Gromov hyperbolicity constant $\delta$ of the realized graph embedding: the supremum over all 4-tuples of points of the maximum distance excess when comparing path lengths and tree-like distances. $\delta$-thin-triangle test quantifies deviation from tree-like structure; small $\delta$ validates the hyperbolic embedding.

## Why
Hyperbolic geometry assumes underlying tree structure. If the true graph is far from tree-like (many cycles, strong cliques), the embedding will distort metrics and the [Singularity Detector σ(x)](../singularity/singularity-detector.md) will suffer from baseline contradictions. The $\delta$ diagnostic flags this early: high $\delta$ signals that the typed graph is not fundamentally hierarchical, so neither hyperbolic embedding nor singularity detection should be trusted without repair (e.g., cycle contraction, re-typing).

## Interface
**Inputs:**
- Hyperbolic embedding $\phi: V(G) \to \mathbb{H}^d$ with distance matrix $D_{\text{hyp}}$.
- Graph edges $E(G)$.
- Sample size $n_{\text{samples}}$ (for large graphs, sample 4-tuples uniformly).

**Outputs:**
- Hyperbolicity constant $\delta_{\text{est}}$ and confidence interval.
- Flag: pass/warn/fail relative to threshold $\delta_{\max}$.
- Per-edge hyperbolicity residual: how much does each edge violate thin-triangle conditions.

## Build steps
- Sample (or enumerate) 4-tuples $(a, b, c, d)$ uniformly from $V(G)$.
- For each 4-tuple, compute three pairwise-distance sums in the embedding: $S_1 = D(a,b) + D(c,d)$, $S_2 = D(a,c) + D(b,d)$, $S_3 = D(a,d) + D(b,c)$.
- Let $M_{\max} = \max(S_1, S_2)$ and $M_{\min}$ be the middle value. Thin-triangle excess is $M_{\max} - M_{\min}$; record $\delta_{abcd} = \frac{1}{2}(M_{\max} - M_{\min})$.
- Estimate global $\delta = \max_{\text{all 4-tuples}} \delta_{abcd}$.
- If $\delta > \delta_{\max}$ (e.g., 0.5), warn that the embedding is not sufficiently hyperbolic; consider re-typing or graph refinement.

## Links
- **See also:** [Hyperbolic Embedding](hyperbolic-embedding.md), [Graph Prototype Vectors](graph-prototype-vectors.md)
- **Drives:** (embedding-quality-report — planned experiment)
- **Driven by:** (typed-graph-contract — external driver)
- **Math:** [Mathematics.md §Hyperbolic Metric Spaces](../../Mathematics.md#hyperbolic-metric)
- **Open:** (delta-threshold-setting — open question)
