# Does singularity score σ(x) beat margin alone for failure prediction?

**Cluster:** open
**Status:** conditionally-answered — grammar-class-conditional (margin-saturation regime)
**Tags:** #singularity #failure-prediction #margins #e4-load-bearing

## Status update — 2026-05-05 → 2026-05-12 (Phases 10, 14, 15, 18, 23e)

**The answer is conditional on margin saturation:** σ beats margin **iff margin is not already saturated** on the task. This is now well-replicated across grammars and is the load-bearing summary for q10.

| phase | grammar | margin_auroc | sigma_uplift (mean, n=5 unless noted) | q10 strict bar (≥ +0.03) |
|---|---|---:|---:|---|
| Phase 10 (ListOps shallow) | listops | ~0.96 (saturated) | ≈ −0.035 | XFAIL |
| Phase 11/12 (ListOps deepened) | listops | ≈ 0.85 (saturated) | mean ≈ −0.098 | XFAIL |
| Phase 14 (Python expressions) | python_expr | ≈ 0.72 (headroom) | **+0.022** (4/5 seeds positive) | **PARTIAL** — 2/5 seeds pass; mean below bar |
| Phase 15 (Python big) | python_big | varies | sign-flipped back | XFAIL on mean |
| Phase 16 (JSON) | json | — | cross-grammar synthesis, see log | mixed |
| Phase 18 (control-flow Python) | python_control | — | σ_structural_uplift positive multi-seed | new positive signal |

See [research_log.md §Phase 14 — "first external benchmark: Python expressions, σ wins q10 strict bar"](../../research_log.md), [research_log.md §Phase 10 — "σ has a narrow prediction regime"](../../research_log.md), [research_log.md §Phase 18](../../research_log.md), and [research_log2.md §Phase 23e](../../research_log2.md).

**The refined claim — three roles, validated across grammars** (Phase 15):

1. σ as **failure-AUROC predictor** — beats margin only when margin is not saturated.
2. σ as **structural-ambiguity AUROC predictor** — Phase 13 ablation: disabling σ (A3) collapses `sigma_structural_auroc` from 0.904 to 0.500 (chance). σ is the only flag in the E14 ablation matrix that actually moves the needle. See [research_log.md §Phase 13](../../research_log.md).
3. σ as **regime-level OOD detector** — Phase 23e: σ discriminates train vs. held-out on every grammar, driven by the illegal signal on the regime graph. First time ABSTAIN fires in project history. See [research_log2.md §Phase 23e](../../research_log2.md).

**Original decision criterion (≥ +5 AUROC points uniformly) is too strict** for what the architecture actually does. Reframe: σ is grammar-class-conditional on failure-AUROC and unconditionally load-bearing on structural-AUROC and regime-OOD.

What's still open:

- Per-grammar AUROC of σ separating train vs. eval rows (Phase 23e "what's next").
- RECOVERY band (0.3 ≤ σ < 0.7) is empty in current sweep — is the band rare in principle, or under-calibrated at N=40?

## What

The composite singularity score $\sigma(x) = w_1 \cdot m^{-1} + w_2 \cdot \mathrm{contradiction} + w_3 \cdot \mathrm{loop\_pressure} + w_4 \cdot \mathrm{illegal\_pressure}$ combines multiple signals. Is $\sigma$ strictly better than margin alone for predicting agent failures, or does margin capture most of the signal with less complexity?

This is the **E4 load-bearing question**: does singularity theory actually improve failure prediction, or is it theoretical ornamentation?

## Why

If margin alone achieves > 85% AUROC for failure detection, the composite $\sigma$ adds theoretical complexity without empirical gain. If $\sigma$ consistently outperforms margin by > 5 AUROC points, singularity theory is justified as a core design principle.

The answer determines whether [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md) is kept, refined, or replaced with margin-only uncertainty.

## Interface

**Affected zettels:** [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md), [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md), [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md)

**Decision criteria:**
- Measure AUROC (failure detection) for: margin alone, entropy alone, full $\sigma(x)$.
- Compute: precision@top-k, recall@fixed-FPR, cost saved by early intervention.
- Ablate $\sigma$ components; attribute improvement to each term.
- Target: $\sigma$ beats margin by ≥ 5 AUROC points, or margin is sufficient and $\sigma$ is retired.

## Build steps

- Run [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md) on [E2 — Real BabyAI / MiniGrid](../exp/e2-real-babyai.md), [E8 — Transfer Experiment](../exp/e8-transfer-experiment.md), and [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md).
- Define failure: illegal transition, timeout, or low-quality action selected.
- Compute: AUROC(margin), AUROC($\sigma$), component ablations.
- Fit weights $w_i$ on validation set via logistic regression.
- Report confusion matrix: high-margin-high-$\sigma$, high-margin-low-$\sigma$, etc.
- If $\sigma$ ≥ margin + 5 AUROC, keep; else simplify to margin.

## Links

- **See also:** [Does σ(x) get weighted into loss, detection-only, or scheduled?](./q07-singularity-loss-weighting.md), [How is the energy function E(x) specified or learned?](./q02-energy-function-spec.md)
- **Affects:** [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md), [E4 — Singularity Detector Validation](../exp/e4-singularity-auroc.md), [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md)
- **Math:** [Experiments.md §E4-Singularity-Detector](../Experiments.md#e4--singularity-detector)
