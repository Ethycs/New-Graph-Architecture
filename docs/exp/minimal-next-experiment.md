# Minimal Next Experiment

**Cluster:** exp
**Status:** spec
**Tags:** #next-step #prescribed #5-model-comparison #do-this-first

## What

The single most actionable next experiment: BabyAI/MiniGrid 5-model comparison. Run on a curated subset of 5 BabyAI levels (Goto, Pickup, OpenDoor, PutNear, Toggle mixed with and without key-opening sequences). Compare 5 model variants: flat classifier, typed classifier, graph FSM, graph+singularity, hyperbolic graph. Primary metrics: success rate, samples to 90% success, illegal-action rate, steps per episode, and whether the singularity detector flags failures. This is the minimal publishable result. Do this first.

## Why

All prior experiments (E0, E1, synthetic) are either toys or missing real dependencies. E2 is ambitious. The minimal-next experiment is a wedge: it is big enough to be credible, small enough to finish in days, and directly comparable to prior work (BabyAI baselines). If this 5-model comparison shows the graph FSM and hyperbolic variants reduce sample complexity or illegal actions, the architecture has earned continuation. If they don't, we have learned something and can pivot. This experiment answers: is the core idea promising on a real domain?

## Interface

**Reads:**
- BabyAI environment (or fallback: synthetic-babyai-grid)
- 5 pre-trained or initialized model variants
- level specifications: 5 BabyAI tasks × 2 modalities (with/without key logic) = 10 level+model pairs

**Writes:**
- results.jsonl: per-level per-model {level_id, model_id, success_rate, samples_to_90pct, illegal_rate, avg_steps, notes}
- comparison_table.md: markdown table of all results
- learning_curves.png: sample efficiency curves for each model

## Build steps

1. Install BabyAI/MiniGrid or fall back to synthetic-babyai-grid.
2. Select 5 representative BabyAI levels: Goto, Pickup, OpenDoor, PutNear, Toggle (or synthetic equivalents).
3. Initialize or train 5 model variants: flat, typed, graph, graph+singularity, hyperbolic.
4. For each of 5 levels × 5 models, run n_episodes=100 with progressive training, tracking samples to 90% success.
5. Measure illegal-action rate, average steps, and success margin distribution.
6. Emit results.jsonl and comparison table.
7. Plot learning curves; highlight which model variant is most sample-efficient.

## Links

- **See also:** [E2 — Real BabyAI / MiniGrid](./e2-real-babyai.md), [E1 — Synthetic BabyAI Grid Run](./e1-synthetic-babyai.md), [Dataset Adapter — Synthetic BabyAI Grid](./dataset-synthetic-babyai-grid.md), [Dataset Adapter — Real MiniGrid/BabyAI Wrapper](./dataset-minigrid-wrapper.md)
- **Drives:** decision to continue or pivot architecture
- **Driven by:** (minigrid-instruction-loader, synthetic-task-generator — external)
- **Math:** samples_to_90pct = min(n) s.t. success_rate(first_n_samples) ≥ 0.9
- **Open:** [Which BabyAI level set should be the canonical E2 benchmark?](../open/q04-babyai-dataset-choice.md), (sample-budget-per-model — open question)
