# Dataset Adapter — MNIST Typed

**Cluster:** exp
**Status:** spec
**Tags:** #dataset #mnist #sklearn #confusion-graph

## What

Adapter producing typed labels and confusion-graph adjacency from sklearn `load_digits()` (8×8 handwritten digits). Produces triples: (image, digit_class, margin). Confusion graph: edge between classes i, j iff co-confusion rate (both misclassified as each other) > 0.05. Typed labels are simply class indices {0–9}. Used by E0.

## Why

E0 needs a simple, deterministic dataset that tests the classifier→typed-label→confusion-graph pipeline without external dependencies. sklearn digits is lightweight and reproducible. The confusion graph captures which digit pairs are hard to distinguish (e.g., 1↔8, 5↔9), validating that classifier uncertainty induces structured relationships. This is a minimal sanity check dataset.

## Interface

**Reads:**
- sklearn.datasets.load_digits()

**Writes:**
- tuple stream: (image, typed_label, margin, is_singular, ground_truth_label)
- confusion_graph.json: {edges: [[i,j,coconf_rate], ...]}
- metadata: {n_classes: 10, n_features: 64, n_train: 1200, n_test: 397}

## Build steps

1. Load `load_digits()` and split into train (1200) and test (397).
2. Train logistic regression on train set; predict on test set.
3. Extract softmax scores and margins for all test samples.
4. Compute confusion matrix; identify co-confusion pairs (both misclassified as each other).
5. Build confusion graph: vertices {0–9}, edge (i,j) iff co_conf(i,j) > 0.05.
6. Flag low-margin samples (bottom 10%) as singular.
7. Emit tuple stream, confusion graph JSON, and metadata.

## Links

- **See also:** [E0 — MNIST Typed-State Sanity Run](./e0-mnist.md), [Metric Collectors](./metric-collectors.md)
- **Drives:** E0 data input
- **Driven by:** sklearn.datasets
- **Math:** co_conf(i,j) = (pred==i and true==j and pred!=true).sum() / (true==j).sum()
- **Open:** (none; this is a simple fixed adapter)
