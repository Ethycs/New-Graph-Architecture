# Dataset Adapter — ALFWorld Loader

**Cluster:** exp
**Status:** spec
**Tags:** #dataset #alfworld #textworld #embodied-simulation #e9

## What

Loader bridging TextWorld abstract policies and embodied ALFRED-style household-task environments. Produces tuples: (abstract_state_text, embodied_observation, typed_fields, legal_actions, procedural_constraints). Abstract state is TextWorld graph of object/room/location relationships; embodied observation is egocentric RGB-D and proprioceptive data. Typed fields capture task-relevant objects and constraints (e.g., "find cup in kitchen"). Procedural constraints: actions {find, take, open, put, clean, heat}. Used by E9.

## Why

ALFWorld is closer to real-world agent tasks: household planning with both high-level symbolic reasoning and low-level embodied control. Testing the architecture on ALFWorld validates that typed-graph structure generalizes beyond pure-action or pure-reasoning domains. The dual representation (abstract + embodied) also tests the architecture's ability to bridge abstraction levels. This is important for E9 because it shows the system handles heterogeneous information sources.

## Interface

**Reads:**
- ALFWorld environment (TextWorld + ALFRED simulator)
- task specification (JSON: goal, object, location, procedural constraints)

**Writes:**
- tuple: (abstract_state_text, embodied_obs_dict, typed_fields, legal_actions, constraints)
- trace.jsonl: per-step abstract state, embodied obs, actions, rewards
- metadata: {task_id, goal, n_steps, success, reason_for_failure}

## Build steps

1. Initialize ALFWorld environment for a given task.
2. Extract abstract state: TextWorld graph object/location/relationship snapshot.
3. Extract embodied observation: simulator's egocentric image, depth, proprioception.
4. Build typed fields: parse task goal into {target_object, target_location, preconditions}.
5. Determine legal actions: filter full action set by embodied state and constraints.
6. For each step: update abstract state, embodied obs, legal actions; emit tuple.
7. On episode end, compute success; emit metadata.

## Links

- **See also:** [E9 — Full Agent Trace Benchmark](./e9-full-trace-benchmark.md), [Dataset Adapter — Real MiniGrid/BabyAI Wrapper](./dataset-minigrid-wrapper.md), [Dataset Adapter — ScienceWorld Loader](./dataset-scienceworld-loader.md)
- **Drives:** E9 data input (ALFWorld domain)
- **Driven by:** alfworld, textworld, alfred-simulator
- **Math:** (minimal; tuple emission is structured extraction)
- **Open:** (abstraction-level-alignment, procedural-constraint-coverage — open questions)
