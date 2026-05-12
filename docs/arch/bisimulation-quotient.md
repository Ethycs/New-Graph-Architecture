# Bisimulation Quotient

**Cluster:** arch
**Status:** implemented
**Tags:** #phase-21 #quotient #bisimulation #myhill-nerode #graph-extraction

## What

Greedy agglomerative quotient of an over-clustered transition graph by **behavioural equivalence**. Given a per-step labelling at $K$ over-clusters and the observed $(K, K)$ transition counts, the atom repeatedly merges the two clusters whose merge criterion is most similar until exactly `target_K` equivalence classes remain. Five criteria are supported: `transition` (L2 on outgoing row-distributions), `emission` (L2 on per-cluster token distributions), `transition_plus_incoming` (adds the column distribution — closer to formal bisimulation), `full` (transition + incoming + emission averaged), and `random` (baseline). Returns a `QuotientResult` with the new labelling, merged transition counts, optional merged emission counts, the original-to-merged cluster map, and the merge history.

## Why

Wave C established that a trained-from-scratch encoder produces **sub-clusters within FSM states**: K-selection picks $K > V$ because the substrate's representation is finer-grained than the gold FSM. Forcing $K = V$ destroys cluster purity (some sub-clusters survive, others get force-merged arbitrarily) and Hamming sits at 0.24 — 5× the strict bar. The Myhill–Nerode reframing fixes the target: **two latent clusters should be the same state iff they are observationally equivalent** (same outgoing transitions, same incoming transitions, same emissions, same downstream consequences). The extraction pipeline therefore becomes:

```
overcluster → estimate transition + emission laws → quotient by behavioural
equivalence → minimal graph + posterior edge confidence
```

This atom implements the quotient step. Without it, the architecture is stuck at "the substrate is finer-grained than the FSM and there is no way to recover the FSM exactly." With it, over-clustering becomes the *correct intermediate representation* — the substrate keeps its richness, and the quotient produces the FSM at the right resolution.

## Interface

- **`quotient_by_bisimulation(labels, transition_counts, *, target_K, criterion='full', emission_counts=None, use_incoming=True, seed=0) -> QuotientResult`**
- **`QuotientResult`** — frozen dataclass with `labels`, `transition_counts`, `emission_counts`, `cluster_map`, `merge_history`, `criterion`.
- **`MERGE_CRITERIA`** — module-level constant enumerating `("transition", "emission", "transition_plus_incoming", "full", "random")`.

## Build steps

- Row-normalise the current `transition_counts` (and `counts.T` for incoming, and `emission_counts` for emissions) with Laplace smoothing so empty rows don't divide by zero.
- Maintain an `alive` bitmask over current equivalence-class IDs and a `cluster_map` from original cluster ID to its current equivalence class.
- Iterate: until `alive.sum() == target_K`, find the pair $(i, j)$ of alive clusters whose merge-distance under the chosen criterion is minimal; merge $j$ into $i$ (sum rows + columns, dissolve $j$); update `cluster_map`.
- After the loop, renumber alive IDs into `[0, target_K)` and rebuild the reduced count matrices.
- Return the `QuotientResult` with the renumbered per-step labels (`cluster_map[original_labels]`).

## Links

- **See also:** [E24 Graph Extraction](../exp/e24-graph-extraction.md), [Forward Backward](./forward-backward.md), [Posterior Mask](./posterior-mask.md), [Bayesian Nonparametric K](./bayesian-nonparametric-k.md).
- **Drives:** Phase 21 (the overcluster-then-quotient sweep), and the universal-graph-extraction proposal's reframing from "exact graph recovery" to "minimal quotient under observational equivalence."
- **Driven by:** the over-clustered output of [Bayesian Nonparametric K](./bayesian-nonparametric-k.md) + [Hidden State Harvester](./hidden-state-harvester.md).
- **Math:** Myhill–Nerode equivalence on finite labelled transition systems; greedy agglomerative clustering on the cluster-similarity graph. Formal bisimulation is the symmetric closure under graph reversal — captured here by `transition_plus_incoming` and `full`.
- **Open:** [[open.qNN-quotient-stopping]] — instead of `target_K` fixed up front, stop when the minimum pair distance crosses a threshold (Bayesian-nonparametric / split-merge variant).
