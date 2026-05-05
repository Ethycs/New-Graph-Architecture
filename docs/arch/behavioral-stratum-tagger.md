# Behavioral Stratum Tagger

**Cluster:** arch
**Status:** spec
**Tags:** #stratification #behavior #classifier-output

## What

A module that assigns each state $x$ a **behavioral stratum tag** drawn from the discrete catalog defined in [Singularity Types Catalog](./singularity-types.md): one of `nominal`, `low-margin`, `decision-tie`, `contradiction`, `illegal`, `loop-risk`, `stabilizer-jump`. The tag is a categorical label, not a geometric region. Computation is a small rule-based classifier over signals already present in the [Typed Score Record](./typed-score-record.md) and [results.jsonl](../drivers/results-jsonl.md): margin, second-best gap, history, and the legality bit.

This is the Phase 2 "stratification" the rest of the system actually consumes. It does **not** require Whitney conditions, smooth manifolds, or hyperbolic embedding; those live in [Stratified Partition Function](./stratified-partition-function.md) (Phase 5).

## Why

The [Singularity Detector σ(x)](./singularity-detector.md) needs to know *which kind* of singularity the state is, not just that one is present. Routing decisions, the energy function's contradiction term, and the IDF rare-stratum weighting all key off the tag — they care about category, not topology. Forcing geometry into Phase 2 would block on graph-extrusion (Phase 3); decoupling the categorical tag from the geometric stratification lets Phase 2 close as soon as the signals it depends on exist.

A second reason: the catalog is fixed and small (≤ 7 entries). A discrete tagger is easier to test, log, and debug than a continuous geometric stratification. The geometric content earns its place in Phase 5 where the partition function actually sums over strata; until then it would be unused machinery.

## Interface

**Inputs:**
- [Typed Score Record](../drivers/typed-score-record.md) for the current step (margin, top-1/top-2, distribution, predicted state).
- Boolean `transition_legal` from [Graph Legality Mask](./graph-legality-mask.md).
- Recent history slice (last $H$ predicted states) for loop-risk detection.
- Optional [Stabilizer Signature](./stabilizer-signature.md) (Phase 4); when absent, the `stabilizer-jump` rule is a no-op.

**Outputs:**
- Stratum tag $\lambda(x) \in$ catalog from [Singularity Types Catalog](./singularity-types.md).
- Per-tag confidence $\in [0,1]$ (rule-fired strength).
- Multi-tag bitmask: a state can satisfy several rules simultaneously; $\lambda(x)$ is the highest-priority tag, the bitmask records all that fired.

**Where it writes:** `behavioral_stratum` and `stratum_bitmask` fields on [results.jsonl](../drivers/results-jsonl.md).

## Build steps

1. Define the rule set: one predicate per catalog entry. Rules are pure functions of (record, history, fsm). Document priority order (illegal > stabilizer-jump > contradiction > decision-tie > low-margin > loop-risk > nominal).
2. Implement `tagger.py` with `tag(record, history, fsm) -> (tag, bitmask, confidence)`.
3. Add the two output fields to [results.jsonl](../drivers/results-jsonl.md) schema and bump `schema_version`.
4. Wire [Singularity Detector σ(x)](./singularity-detector.md) to consume `behavioral_stratum` instead of computing categories internally.
5. Add a unit test per rule (one positive + one negative case each) and one end-to-end test that runs E0 and asserts every record has a non-null tag.

## Links

- **See also:** [Singularity Types Catalog](./singularity-types.md), [Singularity Detector σ(x)](./singularity-detector.md), [Stratified Partition Function](./stratified-partition-function.md)
- **Drives:** [Singularity Detector σ(x)](./singularity-detector.md), [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md), [E5 — IDF Rare-Stratum Weighting Ablation](../exp/e5-idf-ablation.md)
- **Driven by:** [Typed Score Record](../drivers/typed-score-record.md), [Graph Legality Mask](./graph-legality-mask.md)
- **Math:** [Mathematics.md §Whitney Stratification](../Mathematics.md#whitney-stratification-and-group-action-orbit-types) (the geometric notion this atom deliberately does not implement; see [Stratified Partition Function](./stratified-partition-function.md) for the Phase 5 atom that does).
- **Open:** [q07 — singularity loss weighting](../open/q07-singularity-loss-weighting.md), [q12 — loop-risk detection](../open/q12-loop-risk-detection.md)
