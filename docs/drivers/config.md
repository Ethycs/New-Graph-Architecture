# Config (YAML)

**Cluster:** driver
**Status:** spec
**Tags:** #driver #contract #configuration #source-of-truth

## What

A single YAML configuration file serves as the source of truth for both architecture and experiment harnesses. It defines the grammar specification path, graph FSM path, embedding dimensions, temperature, IDF weighting parameter (α), group specification, ablation flags, dataset name, and random seed. Both (model-builder — arch component) and (experiment-runner — exp component) read this file to bootstrap their entire execution context.

## Why

Configuration drift between architecture and experiment—different grammar paths, mismatched embedding dimensions, divergent temperature settings—causes systematic bias and makes results unreproducible. A shared YAML contract ensures both halves start from identical semantics. Without it, claims about ablations collapse because each ablation variant might load slightly different grammars or hyperparameters. The config is the keystone: it ties together [Graph FSM Spec](./graph-fsm-spec.md) loading, [Ablation Flag Set](./ablation-flags.md) selection, and [Typed Score Record Contract](./typed-score-record.md) schema versioning.

## Interface

**File path (runtime):**
- Source configs live at `configs/<dataset>.yaml` (e.g. `configs/babyai.yaml`, `configs/mnist.yaml`).
- Each run is identified by a `run_id` of the form `<experiment>_<ablation>_seed<seed>` (e.g. `E2_A0_seed42`). [CLI Runner](./cli-runner.md) builds this run_id and snapshots the merged config to `runs/<run_id>/config_snapshot.yaml`.

**Schema (YAML):**
```yaml
schema_version: "1.0"
grammar_spec_path: ./grammars/babyai.grammar.yaml
graph_fsm_path: ./graphs/babyai.fsm.yaml
graph_fsm_source: hand                        # "hand" (load graph_fsm_path) or "compiled" (run grammar-compiler)
group_spec_path: ./groups/babyai.group.yaml   # optional; required iff group_quotient_enabled
embedding_dim: 64
temperature: 0.8
idf_alpha: 1.0
ablation: A0                                  # one of A0–A9
dataset: babyai                               # one of mnist | babyai-synthetic | babyai | alfworld | scienceworld
seed: 42
batch_size: 64                                # optional; CLI may override
num_epochs: 20                                # optional; CLI may override
device: cpu                                   # optional; "cpu" | "cuda:0" | ...
```

**Field schema:**

| Field | Type | Units | Required | Example | Semantic note |
|---|---|---|---|---|---|
| `schema_version` | str | semver | required | `"1.0"` | Bumped when fields change; readers compare on load. |
| `grammar_spec_path` | str (path) | — | required | `./grammars/babyai.grammar.yaml` | Source of grammar; produces label set used by [Typed Score Record Contract](./typed-score-record.md). |
| `graph_fsm_path` | str (path) | — | required | `./graphs/babyai.fsm.yaml` | Path to [Graph FSM Spec](./graph-fsm-spec.md); both halves load this exact file. |
| `graph_fsm_source` | str (enum) | — | required | `"hand"` | `"hand"` loads from `graph_fsm_path`; `"compiled"` invokes [Grammar Compiler](../arch/substrate/grammar-compiler.md) on `grammar_spec_path` and uses its output. Both producers must emit equivalent artifacts; see V2 in [Ablation Flag Set §Variant axes](./ablation-flags.md#variant-axes-orthogonal-to-a0-a9). Defaults to `"hand"` until the compiler lands. |
| `group_spec_path` | str (path) \| null | — | required iff `group_quotient_enabled` | `./groups/babyai.group.yaml` | Group action spec; ignored when ablation disables quotient. |
| `embedding_dim` | int | dimensions | required | `64` | $d$ for $\mathbb{H}^d$; must equal `coordinates.dimension` in graph FSM spec. |
| `temperature` | float | — | required | `0.8` | Softmax temperature $T$; range typically $[0.5, 2.0]$. |
| `idf_alpha` | float | — | required | `1.0` | Laplace smoothing $\alpha$ in $\log((N+\alpha)/(n_\lambda+\alpha))$. |
| `ablation` | str (enum) | — | required | `"A0"` | One of `A0`…`A9` defined in [Ablation Flag Set](./ablation-flags.md). |
| `dataset` | str (enum) | — | required | `"babyai"` | Selects loader. One of `mnist`, `babyai-synthetic`, `babyai`, `alfworld`, `scienceworld`. |
| `seed` | int | — | required | `42` | Seeds numpy/torch/python; written to every metrics/results record. |
| `batch_size` | int | samples | optional | `64` | CLI flag `--batch-size` overrides. |
| `num_epochs` | int | epochs | optional | `20` | CLI flag `--num-epochs` overrides. |
| `device` | str | — | optional | `"cpu"` | Torch device; CLI flag `--device` overrides. |

**Versioning:**
- `schema_version` is required. Current: `"1.0"`.
- Adding optional fields with defaults is non-breaking (still `"1.0"`).
- Renaming, type changes, or required-field additions bump to `"1.1"` (minor; reader supports backwards-compat) or `"2.0"` (major; reader rejects without explicit migration).
- Readers MUST compare `schema_version` on load; on unsupported major version, abort with a clear error naming both observed and supported versions.

**Producers:** human-authored YAML at `configs/*.yaml`; loaded by `config_loader.py` (driver code).
**Consumers:** [CLI Runner](./cli-runner.md) (entry); arch components [Grammar Compiler](../arch/substrate/grammar-compiler.md), [Graph FSM](../arch/graph/graph-fsm.md), [Hyperbolic Embedding](../arch/hyperbolic/hyperbolic-embedding.md), [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md); exp components [Metric Collectors](../exp/metric-collectors.md), [Evidence-Level Tracker](../exp/evidence-tracker.md), every dataset adapter under `exp/dataset-*.md`.

## Build steps

1. Define a `ConfigSchema` Pydantic model (or dataclass with validators) covering every field in the table above; enforce types, enum membership for `ablation` and `dataset`, and the `group_spec_path` ↔ `group_quotient_enabled` invariant.
2. Implement `config_loader.py` with `load(path) → ConfigSchema` and `dump(config, path)`; `load` reads `schema_version`, applies migration shims, and raises on unsupported versions.
3. Author template configs `configs/mnist.yaml`, `configs/babyai-synthetic.yaml`, `configs/babyai.yaml`, `configs/alfworld.yaml`, `configs/scienceworld.yaml` with commented examples.
4. Wire [Grammar Compiler](../arch/substrate/grammar-compiler.md), [Graph FSM](../arch/graph/graph-fsm.md), [Hyperbolic Embedding](../arch/hyperbolic/hyperbolic-embedding.md), and [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md) to accept a `ConfigSchema` instance instead of ad-hoc kwargs.
5. Wire [CLI Runner](./cli-runner.md) to call `config_loader.load`, merge CLI overrides (`--seed`, `--batch-size`, `--num-epochs`, `--device`) onto the loaded config, and write the merged result to `runs/<run_id>/config_snapshot.yaml` before any training/evaluation work begins.
6. Add a regression test that loads each template, round-trips it, and validates field values match.
7. Document migration steps inline next to `schema_version`; each version bump appends a note.

## Links

- **See also:** [Ablation Flag Set](./ablation-flags.md) (which flags are set), [Graph FSM Spec](./graph-fsm-spec.md) (what graph to load), [Typed Score Record Contract](./typed-score-record.md) (label set comes from grammar spec)
- **Drives:** [CLI Runner](./cli-runner.md), [Grammar Compiler](../arch/substrate/grammar-compiler.md), [Graph FSM](../arch/graph/graph-fsm.md), [Hyperbolic Embedding](../arch/hyperbolic/hyperbolic-embedding.md), [Typed Field Pipeline](../arch/typed/typed-field-pipeline.md), [Evidence-Level Tracker](../exp/evidence-tracker.md)
- **Driven by:** (none; this is root)
- **Open:** (q01-config-variants — dataset-specific files override defaults, or profile selection?)
