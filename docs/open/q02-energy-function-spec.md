# How is the energy function E(x) specified or learned?

**Cluster:** open
**Status:** conditionally-answered-2026-05-08 — hybrid via predictive projection + partition probe
**Tags:** #energy #loss #partition-function #catastrophe

## Status update — 2026-05-08 (Phases 22a + 23)

The "prescribed catastrophe potential vs learned MLP vs hybrid" trichotomy has been overtaken by the actual implementation. The current answer is **hybrid, but framed as a predictive projection on top of a learned partition function**, not as a catastrophe potential.

- **Phase 7** named the architecture's Bayesian + Riemannian + hyperbolic apparatus as information geometry by construction (Fisher / KL / Beta-on-Fisher-Rao). See [research_log.md §Phase 7](../../research_log.md).
- **Phase 22a — partition-function probe.** [e27_extraction_partition_probe.py](../../src/nga/exp/e27_extraction_partition_probe.py) adds a partition-function probe head whose target is the uniform-over-legal-successors distribution under the gold FSM legality matrix; gradient through the probe reshapes the encoder's equivalence relation toward FSM-state alignment. Mechanism validated (cluster purity 0.78 → 0.85), strict Hamming bar still missed. See [research_log2.md §Phase 22a](../../research_log2.md).
- **Phase 23 — PCG-X predictive projection.** [predictive_projection.py](../../src/nga/arch/predictive_projection.py) adds `h → z` plus probe heads (next-state, entropy-regression, failure, optional adversarial-token). The "energy" is now the predictive head's softmax temperature × score; `argmax(next_state_logits)` partitions states without an explicit prescribed potential. See [research_log2.md §Phase 23](../../research_log2.md).

So the actual answer is:
- (A) prescribed fold/cusp potential — **not adopted**; no catastrophe potential is wired in.
- (B) learned MLP — **partially**; the learned predictive projection plays the role.
- (C) hybrid — **closest match**; the partition function $Z$ is structurally fixed (sum over legal successors / observed transitions), and the *score* feeding into $Z$ is learned via the projection's probe heads.

Open sub-questions that remain:
- **Self-supervised $Z$.** Phase 22b/22d (proposed) would drop the gold-FSM target and learn $Z$ jointly via trajectory self-consistency. Not yet run.
- **Catastrophe-labels atom** ([catastrophe-labels.md](../arch/singularity/catastrophe-labels.md)) is still a finite-difference fold/cusp/none detector (Phase 6/7 Wave B), not a prescribed potential in the loss.

## What

Is the energy function $E(x)$ in the partition-function formulation prescribed from catastrophe theory (e.g., a fold, cusp, or swallowtail potential), learned end-to-end via gradient descent, or inferred from empirical margin and score differences?

The partition function $Z = \sum_q \exp(-\beta E(q | x))$ governs the distribution over next states. The choice of $E$ determines whether the system is a disciplined singular-geometry model or a learned softmax with extra structure.

## Why

This choice cascades through:
- **Interpretability**: A prescribed catastrophe potential is transparent and debuggable; a learned $E$ is black-box.
- **Generalization**: Catastrophe-theory structure may transfer; generic learning may not.
- **Stability**: Learned $E$ may not respect domain physics or constraints.
- **Computational cost**: Prescribed forms compute instantly; learning adds optimization overhead.

The answer determines feasibility of [Catastrophe Labels](../arch/singularity/catastrophe-labels.md) and success criteria for [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md).

## Interface

**Affected zettels:** [Catastrophe Labels](../arch/singularity/catastrophe-labels.md), [Stratified Partition Function](../arch/energy/stratified-partition-function.md), [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md), [E5 — IDF Rare-Stratum Weighting Ablation](../exp/e5-idf-ablation.md)

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
- **Affects:** [Catastrophe Labels](../arch/singularity/catastrophe-labels.md), [Stratified Partition Function](../arch/energy/stratified-partition-function.md), [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md)
- **Math:** [Architecture.md §Catastrophe-Theoretic Enrichment](../Architecture.md#catastrophe-theoretic-enrichment)
