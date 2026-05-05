# E6 — Group-Quotient Attention

**Cluster:** exp
**Status:** spec
**Tags:** #group-action #quotient-attention #efficiency #flops

## What

Implement orbit-based attention quotienting: group equivalent objects (same type, role, IDF stratum) into orbits V/H, then compute attention over orbit pairs at O(|V/H|²) cost. For high-IDF or singular orbits, re-expand and recompute local attention. Measure FLOPs, latency, task success rate, and re-expansion accuracy. Baseline: dense O(n²) attention. Hypothesis: quotient attention achieves parity or better success with 50–80% fewer FLOPs.

## Why

Full object-to-object attention scales as O(n²), which is expensive for large task or trace sequences. Group-action quotienting reduces this by exploiting symmetry: many objects are interchangeable from the task's perspective (all rooms, all small objects, etc.). By grouping and attending over equivalence classes, we reduce compute without losing information. This tests whether semantic structure (via group actions) translates to real efficiency gains. Load-bearing claim: quotienting is computational without sacrificing task quality.

## Interface

**Reads:**
- task traces with object inventories (type, role, IDF)
- dense attention baseline (e.g., transformer head)
- group structure on objects (equivalence relation or quotient map)

**Writes:**
- [`metrics.jsonl`](../drivers/metrics-jsonl.md): FLOPs (dense vs quotient), latency (wall-time), success_rate, re_expansion_accuracy
- [`results.jsonl`](../drivers/results-jsonl.md): orbit assignments per sample, attention reweight matrices, re-expansion flags
- diagnostic: FLOPs breakdown (orbit attention + re-expansion), speedup curve

## Build steps

1. Parse object inventories from task traces; assign type and IDF stratum to each.
2. Define quotient map: objects → orbits via (type, IDF_tier) equivalence.
3. Implement dense-attention baseline (PyTorch): compute full QK^T, mask, softmax.
4. Implement quotient attention: compute orbit-pair attention (|V/H|² FLOPs); reweight per-object.
5. For high-IDF or singular orbits, re-expand: recompute local pairwise attention, merge.
6. On test traces, measure: wall-clock time, FLOPs (via torch.profiler), task success, re-expansion cost.
7. Emit metrics and FLOPs breakdown.

## Links

- **See also:** [Orbit-Pair Attention](../arch/orbit-pair-attention.md), [E2 — Real BabyAI / MiniGrid](./e2-real-babyai.md), [E9 — Full Agent Trace Benchmark](./e9-full-trace-benchmark.md)
- **Drives:** efficiency gain from group-action structure
- **Driven by:** (task-object-inventory — external), [Orbit-Pair Attention](../arch/orbit-pair-attention.md)
- **Math:** quotient_attention = O(|V/H|²) vs dense O(n²); reexpansion = high-IDF orbits only
- **Open:** [When does an orbit re-expand from quotient form?](../open/q08-quotient-reexpansion-threshold.md), (group-structure-definition — open question)
