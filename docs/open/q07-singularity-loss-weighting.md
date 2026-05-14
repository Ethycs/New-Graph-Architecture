# Does σ(x) get weighted into loss, detection-only, or scheduled?

**Cluster:** open
**Status:** resolved-2026-05-12 — detection-only
**Tags:** #singularity #loss #weighting #training

## Status update — 2026-05-12 (Phase 23e)

De-facto answered: **detection-only (variant B)**. σ is consumed at runtime by `SingularityDetector` and `ControlPolicy` to emit NORMAL / RECOVERY / ABSTAIN verdicts on the typed FSM and on the PCG-X regime graph, and is logged per step to `decision_trace.jsonl`. It is not multiplied into the training loss in any current TPN or PCG-X runner. See [research_log2.md §Phase 23e](../../research_log2.md) — "σ + control on the regime graph (the TPN ↔ PCG-X bridge)" — and [control-policy.md](../arch/substrate/control-policy.md).

The "weighted into loss" and "scheduled" variants (A, C) were not ablated; the project converged on detection-only because (i) σ is bimodal in practice (Phase 23e: RECOVERY band 0.3 ≤ σ < 0.7 is empty), so loss-weighting would mostly be a no-op except on the rare ABSTAIN tail, and (ii) the partition-function probe (Phase 22a) and the predictive projection (Phase 23) provide the gradient-side structure that "σ in loss" was meant to give.

## What

The singularity score $\sigma(x)$ combines margin, contradiction, illegal pressure, and loop risk. Should it be:
- **(A)** multiplied into the training loss: $\mathcal{L} = \sigma(x) \cdot \ell(f(x), y)$ (emphasize singular samples)?
- **(B)** used only at runtime for failure detection (no training impact)?
- **(C)** weighted into loss with a schedule: start with A, transition to B after warm-up?

The choice affects learning dynamics and whether the model actively learns to avoid singularities or just flags them post-hoc.

## Why

- **Weighted into loss**: Model prioritizes singular samples; risk of training instability if $\sigma$ is noisy.
- **Detection-only**: Clean training; singularity becomes post-hoc diagnosis, may miss early signals.
- **Scheduled**: Warm-up learns robust features, then focuses on singular cases; balances stability and specificity.

This is load-bearing for [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md). The choice determines whether singularity theory shapes learning or merely interprets it.

## Interface

**Affected zettels:** [Energy-Weighted Loss](../arch/energy/energy-weighted-loss.md), [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md), [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md), [E5 — IDF Rare-Stratum Weighting Ablation](../exp/e5-idf-ablation.md)

**Decision criteria:**
- Compare test loss, rare-event recall, and singular-region accuracy across A, B, C.
- Check training stability: do weighted variants oscillate?
- Measure generalization: does weighted training overfit to singular samples?
- Target: rare-recall > 70%, overall accuracy not degraded.

## Build steps

- Implement three loss variants: (A) $\sigma(x) \cdot \ell$, (B) $\ell$ alone, (C) piecewise (epochs 0–N: variant A, epochs N–max: variant B).
- Run on [E2 — Real BabyAI / MiniGrid](../exp/e2-real-babyai.md) and [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md).
- Measure: test accuracy, rare-event recall, false-alarm rate, training loss trajectory.
- If A is unstable, adopt C with tuned warmup. If B detects singularities well, stay with B.
- Set schedule (e.g., warmup for 10% of epochs) if choosing C.

## Links

- **See also:** [How is the energy function E(x) specified or learned?](./q02-energy-function-spec.md), [Does singularity score σ(x) beat margin alone?](./q10-failure-prediction-baseline.md)
- **Affects:** [Energy-Weighted Loss](../arch/energy/energy-weighted-loss.md), [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md), [E5 — IDF Rare-Stratum Weighting Ablation](../exp/e5-idf-ablation.md)
- **Math:** [Experiments.md §E4-Singularity-Detector](../Experiments.md#e4--singularity-detector)
