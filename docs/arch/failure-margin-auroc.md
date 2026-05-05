# Failure-vs-Margin AUROC

**Cluster:** arch
**Status:** spec
**Tags:** #metric #validation #performance

## What
Computed validation metric: Area Under the ROC Curve of the [Singularity Detector σ(x)](./singularity-detector.md) score $\sigma(x)$ against ground-truth error labels $y \in \{0, 1\}$. High AUROC (> 0.85) indicates that the detector discriminates failing and passing observations well; low AUROC signals that the geometry and feature design are misaligned.

## Why
AUROC is threshold-invariant and interpretable. It measures whether, on average, a random error observation has higher $\sigma$ than a random correct observation. This metric drives the inner loop of architecture tuning: if AUROC is low, prototype separation, [Hyperbolic Distance Loss](./hyperbolic-distance-loss.md) weights, or [Singularity Types Catalog](./singularity-types.md) definitions need adjustment. If high, the detector is reliable and can be deployed.

## Interface
**Inputs:**
- Predictions: $\sigma(x_i)$ for each observation $i$.
- Labels: $y_i \in \{0, 1\}$ (error/correct, typically derived from loss or downstream validation).
- Optional class weights for imbalanced data.

**Outputs:**
- AUROC value in $[0, 1]$.
- Confidence interval (from bootstrap or cross-validation).
- Precision-recall curve and optimal threshold $\theta^*$.
- Per-type AUROC breakdown: AUROC for each [Singularity Types Catalog](./singularity-types.md) subpopulation.

## Build steps
- Collect held-out test batch with observations and labels.
- Compute $\sigma(x_i)$ for each observation using trained [Singularity Detector σ(x)](./singularity-detector.md).
- Compute ROC curve: plot TPR vs. FPR across all thresholds.
- Integrate to get AUROC; compute confidence via 1000-iteration bootstrap.
- Compute precision-recall curve; find $\theta^*$ that maximizes $F_1$ or a task-specific metric.
- Optionally stratify by [Singularity Types Catalog](./singularity-types.md): recompute AUROC for each type to identify weak types.
- Log results to experiment tracker with hyperparameters.

## Links
- **See also:** [Singularity Detector σ(x)](./singularity-detector.md), [Singularity Types Catalog](./singularity-types.md)
- **Drives:** (detector-calibration-loop — planned experiment)
- **Driven by:** (observation-log, validation-labels — external inputs)
- **Math:** [Mathematics.md §ROC & Performance Metrics](../Mathematics.md#roc)
- **Open:** (threshold-selection-criteria — open question)
