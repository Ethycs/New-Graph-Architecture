# CLI Runner

**Cluster:** driver
**Status:** spec
**Tags:** #driver #contract #entry-point #orchestration

## What

A unified command-line entry point `python run.py` that orchestrates the entire pipeline: dataset loading, model construction, training, evaluation, and metrics collection. It accepts experiment ID (E0–E9), ablation variant (A0–A9), config path, and seed; reads [Config (YAML)](./config.md), [Graph FSM Spec](./graph-fsm-spec.md), and [Ablation Flag Set](./ablation-flags.md); then executes the architecture half (training) and experiment half (evaluation) in sequence and writes all output streams under `runs/<run_id>/`. Both the arch trainer and the exp runners are invoked through this CLI, ensuring identical semantics and context propagation.

## Why

Without a single entry point, architecture and experiment might use different dataset loading logic, different seeds, different config parsing. This causes subtle divergence: one half trains on perturbed data, the other evaluates on unperturbed data; one uses seed 42, the other 123. The CLI runner is the central dispatch: it is the single source of truth for environment setup, dataset loading, and ablation selection. Any code that needs to run an experiment does `python run.py --experiment E2 --ablation A1 --config configs/babyai.yaml --seed 42`. This contract ensures reproducibility and makes it obvious what a run is testing.

## Interface

**File path (runtime):**
- Entry script: `run.py` at the repository root.
- Output directory: `runs/<run_id>/` where `run_id = <experiment>_<ablation>_seed<seed>`. Override with `--output`.
- Per-run files:
  - `runs/<run_id>/config_snapshot.yaml` — snapshot of resolved [Config (YAML)](./config.md).
  - `runs/<run_id>/graph_fsm.yaml` — snapshot of [Graph FSM Spec](./graph-fsm-spec.md).
  - `runs/<run_id>/ablation_snapshot.yaml` — resolved [Ablation Flag Set](./ablation-flags.md) tuple.
  - `runs/<run_id>/scores.jsonl` — [Typed Score Record Contract](./typed-score-record.md) stream.
  - `runs/<run_id>/results.jsonl` — [results.jsonl](./results-jsonl.md) stream.
  - `runs/<run_id>/metrics.jsonl` — [metrics.jsonl](./metrics-jsonl.md) stream.
  - `runs/<run_id>/model_final.pt` — final checkpoint (optional).

**Command syntax:**
```bash
python run.py \
  --experiment E2 \
  --ablation A0 \
  --config configs/babyai.yaml \
  --seed 42 \
  --output runs/E2_A0_seed42
```

**Argument schema:**

| Flag | Type | Required | Default | Example | Semantic note |
|---|---|---|---|---|---|
| `--experiment` | str (enum) | required | — | `E2` | One of `E0`…`E9`. Selects the dataset, task structure, and metric set. |
| `--ablation` | str (enum) | required | — | `A0` | One of `A0`…`A9` per [Ablation Flag Set](./ablation-flags.md). |
| `--config` | str (path) | required | — | `configs/babyai.yaml` | Path to [Config (YAML)](./config.md). |
| `--seed` | int | optional | `42` | `42` | Seeds numpy/torch/python; written to every metrics/results record. |
| `--output` | str (path) | optional | `runs/<run_id>` | `runs/E2_A0_seed42` | Output directory; created if missing. |
| `--device` | str | optional | `"cpu"` | `"cuda:0"` | Torch device. Overrides `Config.device`. |
| `--batch-size` | int | optional | from config | `64` | Overrides `Config.batch_size`. |
| `--num-epochs` | int | optional | from config | `20` | Overrides `Config.num_epochs`. |
| `--checkpoint-dir` | str (path) | optional | `null` | `runs/E2_A0_seed42` | Directory to load/save checkpoints. |
| `--skip-train` | bool flag | optional | `false` | `--skip-train` | Skip training; load checkpoint from `--checkpoint-dir`. |
| `--skip-eval` | bool flag | optional | `false` | `--skip-eval` | Skip evaluation phase. |
| `--verbose` | bool flag | optional | `false` | `--verbose` | Verbose logging. |

**Exit codes:** `0` on success; `1` on fatal error (missing config, unsupported `--experiment`, schema-version mismatch, etc.).

**Environment setup (executed in order):**
1. Parse args; load [Config (YAML)](./config.md); reject unsupported `schema_version`.
2. Compute `run_id = f"{experiment}_{ablation}_seed{seed}"`; create `runs/<run_id>/` if missing; refuse to overwrite an existing non-empty run directory unless `--output` is explicitly different.
3. Apply seed to numpy, torch, python `random` (and `torch.cuda` when applicable).
4. Merge CLI overrides into the loaded config; write `runs/<run_id>/config_snapshot.yaml`.
5. Resolve ablation flags from [Ablation Flag Set](./ablation-flags.md); write `runs/<run_id>/ablation_snapshot.yaml`.
6. Load [Graph FSM Spec](./graph-fsm-spec.md) from `Config.graph_fsm_path`; copy to `runs/<run_id>/graph_fsm.yaml`.
7. Dispatch dataset loader by `Config.dataset` (e.g., `mnist → exp/dataset_mnist_typed.py`).
8. Construct model via arch builder with the resolved config and ablation flags.
9. Train (unless `--skip-train`); arch trainer writes per-epoch records to `metrics.jsonl`.
10. Evaluate (unless `--skip-eval`); evaluation writes per-sample `scores.jsonl` and `results.jsonl` plus aggregate `metrics.jsonl` lines.
11. Print one-line summary to stdout (`run_id`, final accuracy, illegal-rate, AUROC) and exit.

**Versioning:**
- CLI surface version: `"1.0"`. Adding optional flags is non-breaking; renaming or removing flags bumps to `"2.0"` and is documented in `--help`.
- The CLI itself reads `schema_version` from [Config (YAML)](./config.md), [Graph FSM Spec](./graph-fsm-spec.md), and [Ablation Flag Set](./ablation-flags.md); on any unsupported version it aborts with exit code `1`.

**Producers:** developer-authored CLI script at repo root (`run.py`).
**Consumers:** every arch builder under `arch/` (entered via `arch/typed_field_pipeline.py`); every E0–E9 runner under `exp/` (entered via `exp/metric_collectors.py`); CI/CD pipelines; human experimenters; [Evidence-Level Tracker](../exp/evidence-tracker.md) reads the resulting `runs/*/`.

**Invocation patterns:**
```bash
# Single run
python run.py --experiment E2 --ablation A0 --config configs/babyai.yaml --seed 42

# Sweep all ablations for one experiment
for ablation in A0 A1 A2 A3 A4 A5 A6 A7 A8 A9; do
  python run.py --experiment E2 --ablation $ablation --config configs/babyai.yaml --seed 42
done

# Reproduce from archived snapshot
python run.py --experiment E2 --ablation A0 --config runs/E2_A0_seed42/config_snapshot.yaml --seed 42

# Skip training, reuse checkpoint
python run.py --experiment E2 --ablation A0 --config configs/babyai.yaml --seed 42 \
  --skip-train --checkpoint-dir runs/E2_A0_seed42
```

## Build steps

1. Create `run.py` at repo root with `argparse` covering the table above; `--help` text is the canonical place to document each flag.
2. Implement config loading and CLI-merge (CLI overrides config); write `config_snapshot.yaml` immediately after merge.
3. Implement `run_id` derivation and idempotent `runs/<run_id>/` directory creation; guard against accidental overwrite.
4. Implement RNG seeding (numpy, torch, python `random`, `torch.cuda` when present).
5. Implement experiment-to-dataset dispatch table: `E0→mnist`, `E1→babyai-synthetic`, `E2→babyai`, `E3→babyai`, `E4→reuses E0/E1/E2 outputs`, `E5→babyai`, `E6→babyai`, `E7→babyai`, `E8→babyai`, `E9→{babyai,alfworld,scienceworld}`.
6. Implement training orchestration: build arch model, run training loop, hook [Metric Collectors](../exp/metric-collectors.md).
7. Implement evaluation orchestration: load test split, write [Typed Score Record Contract](./typed-score-record.md), [results.jsonl](./results-jsonl.md), aggregate [metrics.jsonl](./metrics-jsonl.md).
8. Implement error handling: on schema-version mismatch, missing files, or invalid `--ablation`, exit with code 1 and a clear message.
9. Add an end-to-end smoke test: noop ablation that loads a config, validates a graph FSM spec, and writes empty `metrics.jsonl` / `results.jsonl` files for `runs/<run_id>/`.

## Links

- **See also:** [Config (YAML)](./config.md) (what CLI loads), [Ablation Flag Set](./ablation-flags.md) (how CLI applies variants), [Graph FSM Spec](./graph-fsm-spec.md) (snapshotted into the run dir), [metrics.jsonl](./metrics-jsonl.md) and [results.jsonl](./results-jsonl.md) (what CLI writes)
- **Drives:** [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md), every dataset adapter (`exp/dataset-*.md`), [Metric Collectors](../exp/metric-collectors.md), [Evidence-Level Tracker](../exp/evidence-tracker.md)
- **Driven by:** human or CI/CD invocation
- **Open:** (q07-checkpoint-loading — config or CLI arg?), (q08-distributed-training — parallelize over seeds and ablations?)
