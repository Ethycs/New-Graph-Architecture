# Which BabyAI level set should be the canonical E2 benchmark?

**Cluster:** open
**Status:** orphaned-2026-05-13 — project pivoted to grammar substrates
**Tags:** #babyai #minigrid #benchmark #curriculum #orphaned

## Status update — 2026-05-13

The project pivoted away from BabyAI/MiniGrid onto **grammar substrates** as the canonical benchmark surface. Across Phases 11–25 the de-facto canonical suite is:

- **listops** (V=11) — synthetic depth-tracker
- **python_expr** (V=14) — `ast.parse` + `tokenize` over real Python expressions
- **python_big** (V=24) — Python expressions + calls + defs + returns
- **json** (V=26) — published JSON grammar
- **python_control** (V=37) — Python with control flow

See [research_log.md §Phase 14](../../research_log.md), [research_log.md §Phase 18](../../research_log.md), [research_log2.md §Phase 23](../../research_log2.md), and [research_log2.md §Phase 24](../../research_log2.md). All Phase 22a / 23 / 23b / 23d / 23e / 24 / 25 sweeps run on this 5-grammar suite.

E2 (Real BabyAI / MiniGrid) was never run past synthetic E1. The question as posed — which BabyAI subset — is moot in the current direction.

If a reframed version of this question is wanted, it would be: *which set of published grammars is canonical, and how is "comparable across grammars" defined?* The 5-grammar list above is the de-facto answer, locked by the cross-grammar synthesis tables in Phases 16/18.

## What

BabyAI/MiniGrid has an official curriculum of difficulty levels (e.g., GoToLocal, GoToObj, PutNextLocal, ...) and custom synthetic environments. Should we lock a specific canonical subset as the canonical benchmark for [E2 — Real BabyAI / MiniGrid](../exp/e2-real-babyai.md), or design a synthetic grid task that mimics BabyAI structure without the library dependency?

The choice affects reproducibility, comparability with other work, and what "success on E2" means.

## Why

- **Official curriculum**: Direct comparison to BabyAI papers, known difficulty scaling, broad acceptance.
- **Synthetic custom**: Full control over grammar complexity, state-space size, singularity patterns; reproducible without library; risk of being too toy-like.
- **Hybrid**: Official curriculum as primary, synthetic as fallback/ablation.

This decision gates whether E2 results are cited as "on BabyAI" or "on BabyAI-like synthetic," affecting credibility and adoption of the full E0–E9 pipeline.

## Interface

**Affected zettels:** [E2 — Real BabyAI / MiniGrid](../exp/e2-real-babyai.md), [E8 — Transfer Experiment](../exp/e8-transfer-experiment.md), [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md)

**Decision criteria:**
- Is BabyAI/MiniGrid available and stable in target runtime?
- Does official curriculum have a clean subset (e.g., first 5 levels) suitable for ablation?
- Can synthetic substitute match difficulty distribution?
- Target: one canonical split with reproducible results.

## Build steps

- Check BabyAI/MiniGrid availability; if missing, design synthetic BabyAI-like grammar and states (e.g., 6–10 states, 4–8 actions, task decomposition).
- Commit to either: (A) official levels 1–5, (B) official curriculum with custom split, or (C) synthetic with documented correspondence.
- Write fixed task-generation code; version it.
- Run baseline (flat policy) and graph policy on canonical split; report splits and hyperparameters.
- Document why this choice; make it reproducible and citable.

## Links

- **See also:** [What defines "successful transfer" in E8?](./q05-transfer-success-criterion.md), [What is the encoder freeze schedule in reservoir readout?](./q09-reservoir-freeze-schedule.md)
- **Affects:** [E2 — Real BabyAI / MiniGrid](../exp/e2-real-babyai.md), [E8 — Transfer Experiment](../exp/e8-transfer-experiment.md), [E9 — Full Agent Trace Benchmark](../exp/e9-full-trace-benchmark.md)
- **Math:** [Experiments.md §BabyAI-Task-Selection-and-Setup](../Experiments.md#babyai-task-selection-and-setup)
