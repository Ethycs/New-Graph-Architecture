# Dataset Adapter — Synthetic BabyAI Grid

**Cluster:** exp
**Status:** spec
**Tags:** #dataset #synthetic #grid-world #in-process

## What

In-process generator for synthetic grid-task traces: 7 typed task states × 5 task types, each producing (features, current_state, next_state, legal_next_states) tuples. Task states are {Parse, Navigate, ResolveDoor, Pickup, Deliver, Interact, Done}. Task types are {goto_object, pickup_object, open_door, put_near, toggle}. Transitions follow a hand-coded state machine. Features are grid position, grid occupancy map, and task-specific constraints. Used by E1 and ablation tests.

## Why

BabyAI/MiniGrid are not always available in sandboxed environments. A faithful synthetic generator allows controlled testing of the graph-mask hypothesis without external dependencies. The synthetic setup still exercises typed states, legal transitions, classifier prediction, and singularity detection. It is deterministic and reproducible, making it ideal for rapid ablation testing.

## Interface

**Reads:**
- task type (one of 5)
- episode length (default 20)
- RNG seed

**Writes:**
- trace: list of {features, current_state, next_state, legal_next_states, ground_truth_label}
- legal_graph.json: adjacency matrix for all 7 states
- metadata: {n_states: 7, n_tasks: 5, n_features: 16, state_names: [...]}

## Build steps

1. Define state machine: transitions (Parse→Navigate, Navigate→{ResolveDoor,Pickup}, etc.) for each task type.
2. Implement feature encoder: (task_type, grid_position, grid_occupancy, goal_pos) → 16-d feature vector.
3. For each task type and episode, generate n_steps transitions by sampling from legal next-states.
4. Record features, current state, ground-truth next state, and legal-next-states set.
5. Build adjacency matrix from state machine; emit as legal_graph.json.
6. Emit trace, legal graph, and metadata.

## Links

- **See also:** [E1 — Synthetic BabyAI Grid Run](./e1-synthetic-babyai.md), [Ablation Matrix A0–A9](./ablation-matrix.md), [Metric Collectors](./metric-collectors.md)
- **Drives:** E1 and ablation-matrix data input
- **Driven by:** (in-process generator, no external dependency)
- **Math:** legal_graph[s][s'] = 1 iff (s,s') is a valid transition for any task type
- **Open:** (task-type-coverage, feature-dimensionality — open questions)
