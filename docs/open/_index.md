# Open Questions

The open cluster contains 12 atomic notes recording unresolved design decisions and pending empirical questions. Each note describes what is unknown, why it matters, what experiments would answer it, and which architecture/experiment atoms it affects. Read these to understand what is still being decided.

## Notes

- [q01 — What hyperbolic dimension is needed for the task graph?](./q01-hyperbolic-dim.md) — Does $d = 4$–$8$ suffice, or does required dimension scale with graph size?
- [q02 — How is the energy function E(x) specified or learned?](./q02-energy-function-spec.md) — Prescribed catastrophe potential vs learned MLP vs hybrid.
- [q03 — How are IDF weights updated at runtime?](./q03-idf-runtime-schedule.md) — Fixed from training, exponential moving average, or per-epoch recompute?
- [q04 — Which BabyAI level set should be the canonical E2 benchmark?](./q04-babyai-dataset-choice.md) — Official curriculum vs custom synthetic vs hybrid.
- [q05 — What defines "successful transfer" in E8?](./q05-transfer-success-criterion.md) — Absolute accuracy, relative-to-flat, or relative-to-oracle criterion?
- [q06 — Is the group action H specified or discovered?](./q06-group-action-discovery.md) — Grammar-specified orbits vs learned symmetry detection.
- [q07 — Does σ(x) get weighted into loss, detection-only, or scheduled?](./q07-singularity-loss-weighting.md) — Whether singularity score shapes training or only interprets inference.
- [q08 — When does an orbit re-expand from quotient form?](./q08-quotient-reexpansion-threshold.md) — IDF threshold vs learned policy vs singularity-triggered re-expansion.
- [q09 — What is the encoder freeze schedule in reservoir readout?](./q09-reservoir-freeze-schedule.md) — Frozen from epoch 0, warm-up then freeze, or selective unfreeze?
- [q10 — Does singularity score σ(x) beat margin alone for failure prediction?](./q10-failure-prediction-baseline.md) — The E4 load-bearing question: does composite σ add value over margin?
- [q11 — Does hyperbolic dimension reduction actually materialize at scale?](./q11-hyperbolic-vs-euclidean-tradeoff.md) — Does per-op overhead eat the dimensional savings at practical graph sizes?
- [q12 — How is "loop risk" scored in σ(x)?](./q12-loop-risk-detection.md) — N-gram history, autocorrelation, learned head, or ensemble?

## See also

- [Architecture index](../arch/_index.md)
- [Experiment index](../exp/_index.md)
- [Drivers index](../drivers/_index.md)
- [Root README](../README.md)
