# Open Questions

The open cluster contains 12 atomic notes recording unresolved design decisions and pending empirical questions. Each note describes what is unknown, why it matters, what experiments would answer it, and which architecture/experiment atoms it affects. Read these to understand what is still being decided.

**Status legend** (introduced 2026-05-14 after a docs/open ↔ research-log reconciliation):

- **open** — unresolved, still gated on a specific experiment
- **resolved** — answered by an implemented design or commit
- **conditionally-answered** — answer is task-/regime-conditional, captured in the note
- **orphaned** — the framing has been superseded by an architectural reframe; the question as posed no longer has a current consumer

## Notes

- [q01 — What hyperbolic dimension is needed for the task graph?](./q01-hyperbolic-dim.md) — **open**; null on synthetic data, gated on hierarchical data (Phase 2/3).
- [q02 — How is the energy function E(x) specified or learned?](./q02-energy-function-spec.md) — **conditionally-answered**: hybrid via predictive projection + partition probe (Phases 22a + 23).
- [q03 — How are IDF weights updated at runtime?](./q03-idf-runtime-schedule.md) — **open** (stale; IDF flags dead on E14 path, Phase 13).
- [q04 — Which BabyAI level set should be the canonical E2 benchmark?](./q04-babyai-dataset-choice.md) — **orphaned**: project pivoted to a 5-grammar suite (listops / python_expr / python_big / json / python_control).
- [q05 — What defines "successful transfer" in E8?](./q05-transfer-success-criterion.md) — **open** and untouched; cross-grammar σ-weight transfer is degenerate at single seed (Phase 18).
- [q06 — Is the group action H specified or discovered?](./q06-group-action-discovery.md) — **open** but dormant (Phase 4 group atoms shipped degenerate).
- [q07 — Does σ(x) get weighted into loss, detection-only, or scheduled?](./q07-singularity-loss-weighting.md) — **resolved**: detection-only (Phase 23e).
- [q08 — When does an orbit re-expand from quotient form?](./q08-quotient-reexpansion-threshold.md) — **orphaned**: quotient replaced by upstream partition (Phases 21 + 23).
- [q09 — What is the encoder freeze schedule in reservoir readout?](./q09-reservoir-freeze-schedule.md) — **resolved**: no frozen-encoder commitment in TPN (commit 8d67c17).
- [q10 — Does singularity score σ(x) beat margin alone for failure prediction?](./q10-failure-prediction-baseline.md) — **conditionally-answered**: grammar-class-conditional on margin saturation (Phases 10/14/15/18/23e).
- [q11 — Does hyperbolic dimension reduction actually materialize at scale?](./q11-hyperbolic-vs-euclidean-tradeoff.md) — **open**; same status as q01.
- [q12 — How is "loop risk" scored in σ(x)?](./q12-loop-risk-detection.md) — **resolved**: N-gram on regime graph (Phase 23e).

## See also

- [Architecture index](../arch/_index.md)
- [Experiment index](../exp/_index.md)
- [Drivers index](../drivers/_index.md)
- [Root README](../README.md)
