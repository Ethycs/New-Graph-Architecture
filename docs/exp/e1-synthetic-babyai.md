# E1 — Synthetic BabyAI Grid Run

**Cluster:** exp
**Status:** spec
**Tags:** #graph-mask #classifier-structure #load-bearing

## What

A controlled in-process grid-world benchmark with 7 typed task states and 5 task types, testing whether graph-legality masking of classifier predictions reduces illegal transitions and improves accuracy when the state ID is hidden from the classifier. The key finding: graph mask gives +10 accuracy points and 0% illegal rate when classifier lacks state context.

## Why

This experiment isolates the graph-masking claim without requiring external dependencies (BabyAI/MiniGrid). It measures the efficiency gain of externalizing state-machine structure: if a classifier trained without state identity can be corrected by graph legality constraints, then the graph is load-bearing. This is a toy test but directly supports the core hypothesis that graph structure removes search burden from the classifier.

## Interface

**Reads:**
- synthetic task traces: (features, current_state, next_state, legal_next_states) tuples
- classifier weights (trained on features → next_state prediction)

**Writes:**
- [`metrics.jsonl`](../drivers/metrics-jsonl.md): accuracy with/without graph mask, illegal-transition rate, margin distribution
- [`results.jsonl`](../drivers/results-jsonl.md): per-sample mask application, singularity flags
- ablation table: accuracy vs feature richness (with/without current-state ID)

## Build steps

1. Generate synthetic traces over 7 states {Parse, Navigate, ResolveDoor, Pickup, Deliver, Interact, Done} and 5 task types.
2. Train classifier on (task_type, grid_features) → next_state without current-state ID.
3. Extract legal transitions graph from trace data; build adjacency matrix.
4. Evaluate classifier alone: measure accuracy and illegal-transition rate.
5. Apply graph mask: set p(illegal_next) = 0, renormalize legal distribution.
6. Measure accuracy and illegal rate post-mask; compute singularity regions (lowest 10% margin).
7. Emit metrics and results; tabulate accuracy gain.

## Links

- **See also:** [E0 — MNIST Typed-State Sanity Run](./e0-mnist.md), [E2 — Real BabyAI / MiniGrid](./e2-real-babyai.md), [Dataset Adapter — Synthetic BabyAI Grid](./dataset-synthetic-babyai-grid.md)
- **Drives:** baseline for graph-mask efficiency
- **Driven by:** (synthetic-task-generator — in-process)
- **Math:** illegal_rate = (p_pred[illegal] > 0.5).mean(); accuracy_gain = acc_masked - acc_free
- **Open:** (mask-renormalization — open question)
