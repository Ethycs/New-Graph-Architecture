# E0 — MNIST Typed-State Sanity Run

**Cluster:** exp
**Status:** spec
**Tags:** #sanity-check #classifier-to-graph #baseline

## What

A minimal end-to-end experiment validating the classifier-to-graph pipeline on sklearn digits. Given a trained classifier over {0,1,2,...,9}, extract typed labels, compute margin distributions, flag low-margin singular regions, and construct a confusion graph from classification covariance. Success is demonstrating that uncertain (low-margin) predictions cluster in interpretable confusion zones.

## Why

This is the foundational sanity check: does ordinary classifier output naturally lift into typed structure? The experiment isolates the classifier→typed-label→singularity bridge without involving state machines, grounding, or hyperbolic geometry. It tests whether margin-based singularity detection captures actual ambiguity. Load-bearing claim: classifier uncertainty *induces* useful graph and singularity structure.

## Interface

**Reads:**
- sklearn digits dataset (`load_digits()`)
- trained logistic regression or small MLP classifier
- confusion matrix from predictions

**Writes:**
- [`metrics.jsonl`](../drivers/metrics-jsonl.md): rows with `{"experiment": "e0", "metric": "accuracy", "value": 0.978, ...}`, `{"metric": "low_margin_accuracy", ...}`, `{"metric": "confusion_graph_density", ...}`
- [`results.jsonl`](../drivers/results-jsonl.md): graph adjacency as JSON, singularity flags per sample
- diagnostic plots: margin distribution, confusion graph heatmap

## Build steps

1. Train logistic regression on sklearn digits; hold test set.
2. Compute softmax scores and label predictions on test set.
3. Extract top-2 class scores and compute margin = score[0] - score[1].
4. Flag bottom 10% as "singular/low-margin"; top 90% as "non-singular."
5. Measure accuracy split: non-singular vs singular regions.
6. Build confusion graph: edge between classes i,j if co-confusion rate > threshold.
7. Emit [`metrics.jsonl`](../drivers/metrics-jsonl.md) and [`results.jsonl`](../drivers/results-jsonl.md) rows; visualize confusion graph.

## Links

- **See also:** [Dataset Adapter — MNIST Typed](./dataset-mnist-typed.md), [Metric Collectors](./metric-collectors.md)
- **Drives:** (baseline for singularity detector validation)
- **Driven by:** (sklearn-digits — external dataset)
- **Math:** margin is score[0] - score[1]; singularity iff margin < quantile(margins, 0.1)
- **Open:** (confusion-graph-threshold — open question)
