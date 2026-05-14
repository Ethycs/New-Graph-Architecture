# Drivers Atoms

The drivers cluster contains 8 atomic notes defining the shared contracts between the architecture half and the experiment half of the system. Every file format, flag schema, and CLI interface that both halves must agree on lives here. Read this cluster first — it is the spine of the entire build.

## Notes

- [Config (YAML)](./config.md) — Single YAML source-of-truth: grammar path, FSM path, embedding dim, temperature, IDF alpha, ablation selection.
- [Typed Score Record Contract](./typed-score-record.md) — JSON-Lines wire format for classifier outputs Y; used by singularity detector and all consumers.
- [Graph FSM Spec](./graph-fsm-spec.md) — YAML/JSON file format for $(V, E, w_v, g_v, m_v)$; loaded by legality mask and transition validator.
- [Ablation Flag Set](./ablation-flags.md) — Boolean knobs A0–A9 controlling which components are enabled; keystone of the ablation sweep.
- [metrics.jsonl](./metrics-jsonl.md) — Append-only stream of scalar metrics; single source of truth for all performance measurements.
- [results.jsonl](./results-jsonl.md) — Append-only per-sample predictions and ground-truth; enables post-hoc failure analysis.
- [decision_trace.jsonl](./decision-trace-jsonl.md) — Per-step interpretability stream; full chain typed-scores → mask → σ → energy → partition → control verdict → output node-tuple. The "audit by construction" wire format (substrate-agnostic; v1.1).
- [CLI Runner](./cli-runner.md) — `python run.py --experiment Ek --ablation Aj` unified entry point for all training and evaluation.

## See also

- [Architecture index](../arch/_index.md)
- [Experiment index](../exp/_index.md)
- [Open questions](../open/_index.md)
- [Root README](../README.md)
