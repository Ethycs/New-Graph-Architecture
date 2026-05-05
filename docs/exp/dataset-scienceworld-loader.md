# Dataset Adapter — ScienceWorld Loader

**Cluster:** exp
**Status:** spec
**Tags:** #dataset #scienceworld #reasoning #experimentation #e9

## What

Loader for ScienceWorld interactive text environment (elementary-school science reasoning tasks). Produces tuples: (task_text, observation_text, typed_fields, legal_actions, reasoning_context). Task specifies a science experiment or question. Typed fields extract: {hypothesis, objects_available, action_type, measurement_type}. Legal actions: {observe, measure, mix, heat, apply_force, predict, report_answer}. Reasoning context tracks prior observations and deductions. Used by E9.

## Why

ScienceWorld is substantially different from BabyAI and ALFWorld: it emphasizes experimentation, observation, reasoning, and explanation over navigation or household procedural control. Testing the architecture on ScienceWorld validates generalization to reasoning-heavy agent behavior and scientific hypothesis formation. It is the furthest from the original design space, so success here demonstrates broad applicability. This is critical for E9's claim that the architecture works across diverse agent modalities.

## Interface

**Reads:**
- ScienceWorld environment (text-based, interactive)
- task specification (goal, initial setup, available materials)

**Writes:**
- tuple: (task_text, observation_text, typed_fields, legal_actions, reasoning_context)
- trace.jsonl: per-step observation, action, reward, reasoning state
- metadata: {task_id, task_type, n_steps, success, final_answer}

## Build steps

1. Initialize ScienceWorld environment for a task.
2. Extract task text: goal and initial setup description.
3. Extract observation text: current state of experiment and environment.
4. Build typed fields: parse task into {hypothesis, objects, action_category, measurement_target}.
5. Determine legal actions: {observe, measure, mix, heat, apply_force, predict, report}.
6. Track reasoning context: prior observations, deductions, rule applications.
7. For each step: emit tuple with updated observation and reasoning state.
8. On episode end, compute success (correct answer or hypothesis); emit metadata.

## Links

- **See also:** [E9 — Full Agent Trace Benchmark](./e9-full-trace-benchmark.md), [Dataset Adapter — ALFWorld Loader](./dataset-alfworld-loader.md), [Dataset Adapter — Real MiniGrid/BabyAI Wrapper](./dataset-minigrid-wrapper.md)
- **Drives:** E9 data input (ScienceWorld domain)
- **Driven by:** scienceworld
- **Math:** (minimal; emphasis on text extraction and reasoning state tracking)
- **Open:** (reasoning-state-representation, hypothesis-evaluation — open questions)
