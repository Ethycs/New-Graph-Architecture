# Metric Collectors

**Cluster:** exp
**Status:** spec
**Tags:** #metrics #logging #collectors #evaluation

## What

Standardized collectors for accuracy, illegal-action rate, AUROC (failure prediction), ECE (calibration), transfer gap, attention FLOPs, trainable-parameter count, and loop rate. Each collector consumes predictions, ground truth, and optional metadata; emits JSON rows to [`metrics.jsonl`](../drivers/metrics-jsonl.md) with schema `{experiment, model, metric_name, value, metadata}`. Collectors are composable: e.g., per-sample margin flows into both accuracy and AUROC computation. Used by all experiments.

## Why

Consistent metric computation across E0–E9 is essential for valid comparison. Manual metric computation introduces bugs and inconsistency. Standardized collectors ensure reproducibility and make ablation results trustworthy. They also support streaming: metrics are computed and emitted incrementally as experiments run, allowing early stopping or reweighting decisions. This is load-bearing infrastructure for the entire experimental harness.

## Interface

**Reads:**
- per-sample predictions: {label, confidence, margin, singularity_score, ...}
- ground truth labels or outcomes (success/failure)
- model weights (for param count)
- execution logs (for loop detection, FLOPs)

**Writes:**
- metrics.jsonl: {experiment, model, metric_name, value, optional_metadata}
- per-metric logs: accuracy.jsonl, flops.jsonl, auroc.jsonl, etc.

## Build steps

1. Define metric interfaces: each collector is a callable consuming (predictions, ground_truth, metadata) → row.
2. Accuracy collector: mean(predictions == ground_truth).
3. Illegal-action collector: mean(action_is_illegal) per step.
4. AUROC collector: binary classification (low-margin=1, high-margin=0) vs ground-truth failure.
5. ECE collector: binned calibration error, bins=10.
6. Transfer-gap collector: (test_accuracy - train_accuracy).
7. FLOPs collector: via torch.profiler or manual operation counting.
8. Param-count collector: sum of trainable model parameters.
9. Loop-rate collector: fraction of steps repeating action or state in window of size 3.
10. Wire collectors into experiment runners; emit metrics.jsonl after each run.

## Links

- **See also:** [E0 — MNIST Typed-State Sanity Run](./e0-mnist.md), [E1 — Synthetic BabyAI Grid Run](./e1-synthetic-babyai.md), [Ablation Matrix A0–A9](./ablation-matrix.md), [Evidence-Level Tracker](./evidence-tracker.md)
- **Drives:** all experiments (metric output)
- **Driven by:** (standard PyTorch, sklearn, numpy)
- **Math:** AUROC = trapz(tpr, fpr); ECE = Σ_bin |accuracy_bin - confidence_bin| / n_bins; loop_rate = (repeat_count / total_steps)
- **Open:** (metric-precision-targets, streaming-aggregation-strategy — open questions)
