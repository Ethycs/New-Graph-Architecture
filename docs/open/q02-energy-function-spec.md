# How is the energy function E(x) specified or learned?

**Cluster:** open
**Status:** open
**Tags:** #energy #loss #partition-function #catastrophe

## What

Is the energy function $E(x)$ in the partition-function formulation prescribed from catastrophe theory (e.g., a fold, cusp, or swallowtail potential), learned end-to-end via gradient descent, or inferred from empirical margin and score differences?

The partition function $Z = \sum_q \exp(-\beta E(q | x))$ governs the distribution over next states. The choice of $E$ determines whether the system is a disciplined singular-geometry model or a learned softmax with extra structure.

## Why

This choice cascades through:
- **Interpretability**: A prescribed catastrophe potential is transparent and debuggable; a learned $E$ is black-box.
- **Generalization**: Catastrophe-theory structure may transfer; generic learning may not.
- **Stability**: Learned $E$ may not respect domain physics or constraints.
- **Computational cost**: Prescribed forms compute instantly; learning adds optimization overhead.

The answer determines feasibility of [Catastrophe Labels](../arch/catastrophe-labels.md) and success criteria for [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md).

## Interface

**Affected zettels:** [Catastrophe Labels](../arch/catastrophe-labels.md), [Stratified Partition Function](../arch/stratified-partition-function.md), [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md), [E5 — IDF Rare-Stratum Weighting Ablation](../exp/e5-idf-ablation.md)

**Decision criteria:**
- Compare test loss and transfer accuracy: prescribed vs. learned vs. hybrid.
- Measure interpretability: can we extract agent dynamics from $E$?
- Check constraint satisfaction: does learned $E$ respect graph legality?
- Target: non-transferring learner forced to become prescribed + margin correction.

## Build steps

- Implement three variants: (A) prescribed fold/cusp, (B) learnable MLP $E_\theta$, (C) hybrid (prescribed structure + learned perturbation).
- Train each on [E2 — Real BabyAI / MiniGrid](../exp/e2-real-babyai.md) and [E8 — Transfer Experiment](../exp/e8-transfer-experiment.md).
- Measure: test accuracy, transfer gap, interpretability (can we extract singularity set?), runtime.
- Ablate: remove learned $E$ from full system, compare to margin alone.
- If prescribed $E$ outperforms learned, adopt catastrophe layer; else fall back to margin-based loss.

## Links

- **See also:** [Does σ(x) get weighted into loss, detection-only, or scheduled?](./q07-singularity-loss-weighting.md), [Does singularity score σ(x) beat margin alone?](./q10-failure-prediction-baseline.md)
- **Affects:** [Catastrophe Labels](../arch/catastrophe-labels.md), [Stratified Partition Function](../arch/stratified-partition-function.md), [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md)
- **Math:** [Architecture.md §Catastrophe-Theoretic Enrichment](../Architecture.md#catastrophe-theoretic-enrichment)
