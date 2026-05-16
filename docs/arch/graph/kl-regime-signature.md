# KL Regime Signature

**Cluster:** arch
**Status:** implemented (Phase 26)
**Tags:** #information-geometry #regime-graph #pcg-x #K-choice

## What

Coordinate-free signatures and equivalence-clustering for regimes in the labelled hypergraph. Each regime is characterised by its empirical conditional distribution over outputs $p_\lambda(y \mid x)$; the KL divergence

$$D(p_i \| p_j) = \sum_y p_i(y) \log \frac{p_i(y)}{p_j(y)}$$

is the canonical distance between regimes — invariant under reparametrisation of the substrate, dependent only on the distributions themselves. Two regimes whose pairwise KL falls below a principled threshold are statistically equivalent and can be canonicalised.

## Why

PCG-X at K = V extracts ~10-40 regimes by argmax-of-next-state-head — a behavioural quotient of the substrate's Whitney stratification. Different extraction seeds / projection heads can produce different partitions of equivalent statistical content. KL-signature clustering is the information-geometric move that makes regime identity intrinsic: regimes get canonical equivalence classes that do not depend on the extractor's coordinate choice. This is the K-choice resolution from [`docs/interpretability-push.md`](../../interpretability-push.md).

## Interface

```python
KLRegimeSignature.compute(
    per_regime_conditionals: dict[str, np.ndarray],
    *,
    symmetric: bool = True,
) -> dict[str, np.ndarray]

KLRegimeSignature.cluster(
    signatures: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, str]            # regime_id -> canonical_class_id

KLRegimeSignature.suggest_threshold(
    signatures: dict[str, np.ndarray],
    method: str = "gap",       # "gap" or "median"
) -> float
```

- `compute` returns one signature vector per regime; entry `[k]` is the (optionally symmetric) KL distance to the `k`-th regime in sorted-ID order. Self-distance is zero.
- `cluster` builds the equivalence graph where regimes within `threshold` are joined, then runs union-find. Canonical-class IDs are the sorted-first member of each class.
- `suggest_threshold` picks a principled cut. `"gap"` finds the largest gap in the sorted off-diagonal KL distribution and returns its midpoint — separating the "noise band" (within-cluster) from the "structure band" (between-cluster). `"median"` returns half the median; conservative.

## Threshold convention

- `0.0` strictly equates only distributions that match bitwise.
- `~1e-3` typical for tightly-aligned regimes on the same grammar.
- `>1` indicates substantially different output distributions.

The gap-detection threshold is dataset-dependent; rerun `suggest_threshold` per extraction.

## Build steps

1. Aggregate per-regime empirical output distributions over a fixed support (e.g. softmax over the projection head's class set).
2. Call `compute` to get the signature matrix.
3. Either pass a hand-chosen threshold to `cluster`, or call `suggest_threshold` first and then `cluster`.
4. Feed the signature vectors into `LabelledHypergraph.from_pcg_graph(..., canonical_signatures=...)`.

## Edge cases

- Fewer than 3 regimes ⇒ `suggest_threshold` returns 0.0 (no gap to find).
- Empty conditionals ⇒ `compute` returns an empty dict.
- Zero-sum rows are normalised internally to avoid divide-by-zero; this is best-effort and produces zero distance to a uniform distribution.
- Numerical noise can produce tiny negative KL values; clipped to 0 internally.

## Links

- **See also:**
  - [Labelled Hypergraph](labelled-hypergraph.md) — consumer.
  - [Information Geometry](../hyperbolic/information-geometry.md) — the project's Fisher-Rao infrastructure on the typed-classifier simplex.
  - [Bisimulation Quotient](bisimulation-quotient.md) — the prior discrete behavioural quotient.
- **Drives:** the `canonical_signature` field of every `Regime`.

## Reference

- Proposal: [`docs/proposals/labelled-hypergraph.md`](../../proposals/labelled-hypergraph.md).
