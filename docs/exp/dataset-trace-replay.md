# Dataset Adapter — Research-Agent Trace Replay

**Cluster:** exp
**Status:** spec
**Tags:** #dataset #research-agent #trace-replay #deployment-nearest

## What

Loader for offline traces from research agents (e.g., Claude Code, ReAct-style agents, or other LLM-based planners). Replays episodes from saved JSON traces: each trace contains observations, actions, LLM prompts, tool calls, and outcomes. Typed fields extract: {agent_state, available_tools, active_goal, context_window}. Produces tuples suitable for training a policy that predicts next action given observation and goal. Closest to target deployment scenario. Used by E9 (optional advanced variant).

## Why

Research agents that invoke tools, maintain context, and plan multi-step workflows are the deployment target. Using real traces from such agents makes E9 maximally realistic. The architecture should be able to predict the next action (tool call or decision) given the agent's state and history. This tests whether the typed-graph machinery helps with real-world agent control, not just toy benchmarks. Success here is the strongest validation of the architecture's practical value.

## Interface

**Reads:**
- trace.jsonl files from agent runs: each line is {observation, action, tool_call, result, next_observation, metadata}

**Writes:**
- tuple: (observation_dict, action_label, legal_actions, typed_fields, context_dict)
- training_data.jsonl: standardized tuples for model training
- metadata: {n_traces, n_steps, n_unique_agents, action_type_distribution}

## Build steps

1. Load and parse agent trace files (JSON).
2. For each step in a trace:
   - Extract observation (agent state, environment, context window).
   - Extract action (tool name, parameters, or decision).
   - Identify legal actions: tools available, constraints, retry limits.
   - Build typed fields: active goal, tool category, input type, output type.
   - Record context: prior steps, goals, error history.
3. Standardize all tuples to uniform format.
4. Emit training_data.jsonl and metadata.

## Links

- **See also:** [E9 — Full Agent Trace Benchmark](./e9-full-trace-benchmark.md), [Dataset Adapter — ALFWorld Loader](./dataset-alfworld-loader.md), [Dataset Adapter — ScienceWorld Loader](./dataset-scienceworld-loader.md)
- **Drives:** E9 data input (research-agent domain, optional)
- **Driven by:** agent trace logging system
- **Math:** (minimal; standardization is structural)
- **Open:** (context-window-length, agent-trace-format-standardization — open questions)
