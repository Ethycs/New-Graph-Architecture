# E4 — Singularity Detector Validation

**Cluster:** exp
**Status:** spec
**Tags:** #singularity #failure-prediction #load-bearing

## What

Test the singularity detector σ(x) against actual failures: illegal transitions, low-margin predictions, contradictions, and loop-risk indicators. Measure AUROC of σ(x) vs binary failure label across E0, E1, E2 results. Baseline comparators: max softmax confidence, entropy, margin alone. The detector combines margin, contradiction collisions, illegal-pressure, and loop-pressure into a weighted score. Success is AUROC > baseline by ≥ 3%.

## Why

The singularity detector is load-bearing only if it predicts failures better than simpler uncertainty measures. If margin alone achieves the same AUROC, the full singularity machinery adds no value. This experiment proves whether the four singularity modes (margin, contradiction, illegal-pressure, loop-pressure) collectively capture failure modes that simpler classifiers miss. This is essential for the claim that singularities are a useful abstraction.

## Interface

**Reads:**
- predictions and margins from E0, E1, E2
- failure labels: per-sample binary {0=success, 1=failure}
- confidence, entropy, and margin vectors for all samples
- low-margin region annotations

**Writes:**
- [`metrics.jsonl`](../drivers/metrics-jsonl.md): AUROC for each baseline and σ(x); precision at top-k
- [`results.jsonl`](../drivers/results-jsonl.md): singularity score per sample, failure ground truth, ROC curve points
- diagnostic plots: ROC curves for all methods, confusion matrix, failure-type breakdown

## Build steps

1. Gather predictions from E0, E1, E2; label failures: low margin, illegal transition, no-progress loop, tied scores.
2. Compute baseline scores: max softmax, entropy, margin = score[0] - score[1].
3. Build singularity score σ(x) = w₁ · margin⁻¹ + w₂ · contradiction + w₃ · loop_pressure + w₄ · illegal_pressure.
4. Tune weights via grid search on validation set to maximize validation AUROC.
5. Evaluate AUROC on test set for σ(x) and all baselines.
6. Compute precision-at-k: top-k risky samples, what fraction are actual failures.
7. Emit ROC curves, confusion matrices, and metrics.

## Links

- **See also:** [E0 — MNIST Typed-State Sanity Run](./e0-mnist.md), [E1 — Synthetic BabyAI Grid Run](./e1-synthetic-babyai.md), [E2 — Real BabyAI / MiniGrid](./e2-real-babyai.md)
- **Drives:** validates singularity concept as useful failure detector
- **Driven by:** (failure-label-generator — external)
- **Math:** AUROC = trapz(tpr, fpr); σ(x) = Σᵢ wᵢ φᵢ(x)
- **Open:** (singularity-weight-tuning, failure-definition — open questions)
