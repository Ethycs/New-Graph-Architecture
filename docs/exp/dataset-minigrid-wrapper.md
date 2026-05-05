# Dataset Adapter — Real MiniGrid/BabyAI Wrapper

**Cluster:** exp
**Status:** spec
**Tags:** #dataset #minigrid #observation-converter #real-benchmark

## What

Wrapper around MiniGrid/BabyAI environment providing: observation-to-typed-feature converter (image → {position, grid-occupancy, goal-vector}), instruction-to-typed-grammar converter (natural language → {action-primitive, object-type, location-constraint}), and level selector for curated benchmark suite. Emits tuples: (encoded_observation, typed_fields, legal_next_actions, instruction_embedding). Handles gym.Env interface. Used by E2 and E9.

## Why

Real BabyAI provides grounded language, compositional instructions, and a clean task graph. The wrapper bridges the gap between raw RL/vision environment and the typed-graph architecture by producing structured feature representations from raw observations and language. This is essential for E2 reproducibility and E9 integration testing. The adapter also standardizes level selection and metric collection across a benchmark suite.

## Interface

**Reads:**
- minigrid environment (gym.Env)
- MiniGrid level specification (level_id: str)
- instruction text (LLM-generated or human)

**Writes:**
- tuple: (encoded_obs, typed_fields, legal_actions, instruction_emb)
- trace.jsonl: per-step observations, actions, rewards, instruction
- metadata: {level_id, instruction, n_steps, success}

## Build steps

1. Install minigrid; load specified level.
2. Build observation converter: raw_image → CNN encoder → {pos_vector, grid_occupancy, goal_emb}.
3. Build instruction converter: text → [BPE/BERT] → {action_prim_logits, object_type_logits, location_logits}.
4. For each step in episode: step environment, encode observation and instruction, record legal action mask from env.
5. Emit tuple and append to trace.jsonl.
6. On episode end, compute success and emit metadata.

## Links

- **See also:** [E2 — Real BabyAI / MiniGrid](./e2-real-babyai.md), [E9 — Full Agent Trace Benchmark](./e9-full-trace-benchmark.md), [Metric Collectors](./metric-collectors.md)
- **Drives:** E2 and E9 data input
- **Driven by:** minigrid, gym, language model
- **Math:** (minimal; encoding is standard CNN + language model)
- **Open:** (feature-encoder-depth, instruction-coverage — open questions)
