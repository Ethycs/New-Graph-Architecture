# σ-Weight Optimiser

**Cluster:** arch
**Status:** implemented
**Tags:** #sigma #grid-search #cross-grammar-transfer #phase-18

## What

A grid-search optimiser over the linear σ-aggregation weights $\{w_{\text{margin}}, w_{\text{decision_tie}}, w_{\text{illegal}}, w_{\text{kl_surprise}}, w_{\text{loop_risk}}\}$ that maximises σ-AUROC on a source corpus. The optimiser was built for Phase 18 Track 1's cross-grammar σ-weight transfer experiment (Python big → JSON) and exposes both the tuned weight vector and the AUROC achieved at the optimum.

## Why

The [Singularity Detector](./singularity-detector.md) combines five components into a single σ score via a linear sum with hand-set weights. Hand-set weights are arbitrary; the question of which mixture maximises σ-AUROC on a given grammar is empirical, not a-priori. The σ-weight optimiser is the atom that lets the architecture *tune* its σ ensemble and lets the project ask transferability questions: do weights tuned on grammar X work on grammar Y? Phase 18 Track 1 found that linear σ aggregation tends to degenerate on margin-saturated grammars (the optimum on Python big collapsed to `{margin: 2.0, others: 0.0}`), making cross-grammar transfer near-trivial in either direction — the architectural finding that linear σ aggregation has a structural ceiling.

## Interface

- **Input:** a corpus of $(x, \text{is_failure})$ pairs (binary failure label per sample); a `SingularityDetector` configured with the five-component decomposition; a grid spec (list of candidate values per weight).
- **Method:** `optimize(grid_spec) -> (best_weights, best_auroc)` — evaluates σ-AUROC at every grid cell, returns the argmax.
- **Output:** a weight vector $\in \mathbb{R}^5$ and the σ-AUROC at that vector.
- **Drives:** the E20 runner (`exp/e20_sigma_transfer.py`), which loads the tuned weights and evaluates them on a target grammar.

## Build steps

- Build the grid as the Cartesian product of per-weight candidate lists.
- For each cell, instantiate the SingularityDetector with those weights and compute σ-AUROC against the corpus's failure labels.
- Return the argmax and the corresponding AUROC.
- Cache the per-cell evaluation in a JSONL log to support a follow-up "non-linear σ aggregation" experiment (Phase 18 future-work bullet (b)).

## Links

- **See also:** [Singularity Detector](./singularity-detector.md), [Failure Margin AUROC](./failure-margin-auroc.md).
- **Drives:** E20 (Python big → JSON σ-weight transfer; Phase 18 Track 1).
- **Driven by:** the σ failure-AUROC objective on the source corpus.
- **Math:** the σ ensemble is $\sigma = \sum_k w_k \cdot s_k$; the AUROC objective is non-differentiable in the weights, so grid search is appropriate at this dimension; for higher dimensions a non-linear combiner (a small MLP) would replace this atom.
- **Open:** [[open.qNN-non-linear-sigma]] — replace linear aggregation with a non-linear σ combiner so cross-grammar transfer can exploit complementarity.
