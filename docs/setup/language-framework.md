# Language and Framework

**Status:** decided
**Cluster:** setup

## Summary table

| Category | Choice | Version |
|---|---|---|
| Python | CPython | 3.11.9 |
| Package/env manager | Pixi via `pyproject.toml` `[tool.pixi]` | pixi >= 0.34 |
| Deep-learning framework | PyTorch | 2.4.1 (CUDA 12.1 wheels) |
| Hyperbolic geometry | geoopt | 0.5.0 |
| Pre-trained encoder source | HuggingFace `transformers` + `sentence-transformers` | 4.44.2 / 3.1.1 |
| Sklearn (E0 only) | scikit-learn | 1.5.2 |
| Vision baseline encoder | torchvision | 0.19.1 |
| Grid environment | minigrid (Farama) | 2.3.1 |
| ALFWorld stack | alfworld + textworld | 0.3.5 / 1.6.1 |
| ScienceWorld stack | scienceworld | 1.2.0 |
| Test framework | pytest + pytest-cov + hypothesis | 8.3 / 5.0 / 6.112 |
| Linter / formatter | ruff (lint and format) | 0.6.9 |
| Type checker | mypy strict | 1.11 |
| Logging | stdlib `logging` + custom JSONL writer | stdlib |
| CLI parsing | argparse | stdlib |
| Config schema validation | pydantic v2 | 2.9.2 |
| JSONL record validation | pydantic v2 (shared with config) | 2.9.2 |
| Seeding | custom `drivers/seeding.py` over `random`/numpy/torch | n/a |
| YAML loader | PyYAML | 6.0.2 |

## Decisions

### Python 3.11.9
The project targets `requires-python = ">= 3.11"`, so 3.11 is the floor. We pin to 3.11 (not 3.12) because Phase 5 dependencies (`alfworld`, `textworld`, the Java bridge in `scienceworld`) still ship 3.11 wheels first and most reliably, and PyTorch 2.4 CUDA 12.1 wheels for 3.11 are battle-tested. 3.11 also gives us stdlib `tomllib`, exception groups, and the faster interpreter, while locking out 3.12-only typing syntax that would creep into `arch/` and break older sims.

### Pixi via `pyproject.toml` with `[tool.pixi]`
The repo already commits to a single `pyproject.toml` with `[tool.pixi.workspace]` and `[tool.pixi.tasks]`. We keep one manifest because the project also needs an installable Python package (`hatchling` build backend, editable install) and splitting into a separate `pixi.toml` would force two sources of truth for dependencies. Channels: `conda-forge`. PyPI deps go under `[tool.pixi.pypi-dependencies]` for libraries (transformers, geoopt, alfworld) that are PyPI-only.

### PyTorch 2.4.1 (no JAX)
Phase 3 hyperbolic ops (geoopt), Phase 5 frozen encoder (`frozen-encoder-backbone.md`), Phase 5 trainable readouts (`typed-readout-layer.md`), and Phase 4 orbit-pair attention (`orbit-pair-attention.md`) all sit cleanly in torch. HuggingFace `transformers`, `sentence-transformers`, `torchvision`, `minigrid` examples, and geoopt all assume torch. JAX would force re-binding every pre-trained checkpoint and re-implementing geoopt's Riemannian optimizers, with no Phase 6 experiment that would benefit. One framework keeps `requires_grad=False` semantics consistent across reservoir training and end-to-end ablation A8.

### geoopt 0.5.0 for hyperbolic geometry
[hyperbolic-embedding.md](../arch/hyperbolic/hyperbolic-embedding.md) specifies Riemannian gradient descent on Poincaré and Lorentz models. geoopt provides `PoincareBall`, `Lorentz`, `ManifoldParameter`, and Riemannian Adam/SGD as torch optimizers. geomstats is heavier and pulls in non-torch backends; hypll is younger and lacks Lorentz parity; a custom implementation would block Phase 3 on numerical-stability work that is not the research contribution.

### Pre-trained encoder via `transformers` 4.44.2 plus `sentence-transformers` 3.1.1
E2 ([e2-real-babyai.md](../exp/e2-real-babyai.md)) needs an instruction encoder (text to typed grammar fields). E9 ([e9-full-trace-benchmark.md](../exp/e9-full-trace-benchmark.md)) extends that to ALFWorld and ScienceWorld text observations. `sentence-transformers` gives a frozen-embedding API that matches [frozen-encoder-backbone.md](../arch/substrate/frozen-encoder-backbone.md) build step 1 (verify all parameters have `requires_grad=False`). `transformers` covers the underlying tokenizer and model classes; `torchvision` 0.19.1 stays in the dep list for the MiniGrid raw-image branch in [dataset-minigrid-wrapper.md](../exp/dataset-minigrid-wrapper.md).

### scikit-learn 1.5.2
Required only by E0 ([e0-mnist.md](../exp/e0-mnist.md)): `load_digits`, `LogisticRegression`, and AUROC for [e4-singularity-auroc.md](../exp/e4-singularity-auroc.md) baseline numbers. Pinning here avoids letting sklearn drift into a Phase 1 dep. Not used in any other E experiment.

### MiniGrid 2.3.1, ALFWorld 0.3.5, ScienceWorld 1.2.0
- MiniGrid (Farama) provides E1 (synthetic), E2, and the BabyAI subset for E9.
- ALFWorld pulls `textworld` 1.6.1 (Java + old gym) and is needed only for E9.
- ScienceWorld pulls a Java JDK and `py4j` and is needed only for E9.

These three conflict on `gym` vs `gymnasium` versions and on numpy minor versions. Isolate them in pixi feature environments: a `default` env (no sim deps), `e2` (adds `minigrid`), `e9-alfworld` (adds `alfworld` + `textworld`), `e9-scienceworld` (adds `scienceworld` + JDK). The CLI runner picks the right env via `pixi run -e <env>`. This avoids second-best workarounds (vendored forks).

### pytest 8.3 with pytest-cov 5.0 and hypothesis 6.112
pytest is the only realistic choice. pytest-cov gates phase acceptance criteria (build-order's "Acceptance" lines reference jsonl schema correctness). hypothesis is load-bearing for [metrics-jsonl.md](../drivers/metrics-jsonl.md) and [typed-score-record.md](../drivers/typed-score-record.md) schema invariants (round-trip a generated record through pydantic, assert equality). pytest-benchmark is deferred behind a `[feature.bench]` env; only Phase 4 ([e6-group-quotient-attention.md](../exp/e6-group-quotient-attention.md) FLOPs claim) needs it.

### ruff 0.6.9 (lint + format)
Replaces black, isort, flake8, pyupgrade, pep8-naming. Rule set: `E`, `F`, `I`, `B`, `UP`, `SIM`, `RUF`. Line length 100. `target-version = "py311"`. Format on save; lint on commit. One tool, one config block, no second-best fallback.

### mypy 1.11 strict on `drivers/` and `arch/`
Pyright is faster but mypy is the better fit because pydantic v2 ships first-class mypy plugin support, and the driver atoms ([config.md](../drivers/config.md), [metrics-jsonl.md](../drivers/metrics-jsonl.md), [results-jsonl.md](../drivers/results-jsonl.md), [typed-score-record.md](../drivers/typed-score-record.md)) all require strict cross-field invariants that pydantic-v2-with-mypy enforces at type-check time. Strict for `drivers/` and `arch/` (the contracts and the model). Soft (`--ignore-missing-imports`) for `exp/` runners that import sim libs without stubs.

### stdlib `logging` plus custom JSONL writers
[metrics-jsonl.md](../drivers/metrics-jsonl.md), [results-jsonl.md](../drivers/results-jsonl.md), and [typed-score-record.md](../drivers/typed-score-record.md) all specify exact field schemas, append-only semantics, and `schema_version` checks. structlog adds an indirection without solving the schema problem. Build a `drivers/jsonl_writer.py` that takes a pydantic record, validates, serializes one JSON object per line with `\n` terminator, and `fsync`s. stdlib `logging` covers human-readable diagnostics; the JSONL writer covers machine-readable streams. They are separate sinks.

### argparse for the CLI
[cli-runner.md](../drivers/cli-runner.md) already specifies a flat 11-flag surface with explicit types, defaults, and choices. argparse is stdlib, supports `choices=[...]` for the `--experiment`/`--ablation` enums, generates `--help` text directly from the spec table, and adds zero deps. Typer/click would impose decorator-based introspection that buys nothing on a flat command.

### pydantic v2 (2.9.2) for config
[config.md](../drivers/config.md) build step 1 names "Pydantic model (or dataclass with validators)". v2 gives discriminator-based schema_version handling, native enum validation for `ablation` and `dataset`, the `group_spec_path` versus `group_quotient_enabled` cross-field check via `model_validator`, and YAML round-trip via PyYAML + `model_dump`. dataclasses-plus-tomllib was second-best; the missing capability is cross-field validators, which we would have re-implemented anyway.

### pydantic v2 for JSONL record validation
Same library as config so that `MetricsRecord`, `ResultsRecord`, and `TypedScoreRecord` (the wire format of Y) share one validator stack, one error format, and one `schema_version` check. jsonschema would force a second toolchain. Validation runs on every write (so corrupt records never reach disk) and on every read (so [evidence-tracker.md](../exp/evidence-tracker.md) aborts loudly on a schema-version mismatch).

### Reproducibility and seeding
A single `drivers/seeding.py` exposes `set_seed(seed: int) -> None` that:
1. Sets `PYTHONHASHSEED` (pre-import, via env var in pixi tasks).
2. Calls `random.seed(seed)`, `numpy.random.seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`.
3. Sets `torch.use_deterministic_algorithms(True)` and `torch.backends.cudnn.deterministic = True`, `cudnn.benchmark = False`.

CLI flow: `--seed` flows into `Config.seed` (pydantic), into `set_seed(...)`, and into every `MetricsRecord.seed` and `ResultsRecord.seed` field, so every JSONL line is self-identifying.

### GPU vs CPU baseline
- **CPU sufficient:** E0 (sklearn digits), E1 (in-process 7-state grid), E4 (post-hoc AUROC over E0/E1 outputs), E5 (IDF ablation on E1).
- **GPU required (CUDA 12.1, one device):** E2 (real BabyAI with frozen text encoder), E3 (hyperbolic dim sweep with Riemannian optimization at scale), E6 (FLOPs benchmark needs torch.profiler on CUDA), E7 (reservoir vs end-to-end fine-tuning), E8 (transfer), E9 (multi-domain integration).

The pixi `cuda` feature env adds `pytorch-cuda=12.1`; the default env is CPU-only so contributors without GPUs can run Phase 0 and the critical-path E0 to E5 ramp without pulling 4 GB of CUDA wheels.

## Initial pyproject.toml skeleton

```toml
[project]
name = "new-graph-architecture"
version = "0.1.0"
requires-python = ">= 3.11, < 3.12"
dependencies = []

[build-system]
build-backend = "hatchling.build"
requires = ["hatchling"]

[tool.pixi.workspace]
channels = ["conda-forge"]
platforms = ["linux-64"]

[tool.pixi.pypi-dependencies]
new-graph-architecture = { path = ".", editable = true }

# Core deps shared by every environment
[tool.pixi.dependencies]
python = "3.11.*"
numpy = ">=1.26,<2.1"
scipy = ">=1.13"
pyyaml = "6.0.*"
networkx = ">=3.3"

[tool.pixi.pypi-dependencies.core]
pydantic = "==2.9.2"
torch = "==2.4.1"
torchvision = "==0.19.1"
geoopt = "==0.5.0"
transformers = "==4.44.2"
sentence-transformers = "==3.1.1"
scikit-learn = "==1.5.2"

# Linter, formatter, type-checker, tests live in dev features
[tool.pixi.feature.dev.dependencies]
ruff = "==0.6.9"
mypy = "==1.11.*"
pytest = "==8.3.*"
pytest-cov = "==5.0.*"
hypothesis = "==6.112.*"

# CUDA wheels are opt-in
[tool.pixi.feature.cuda.dependencies]
pytorch-cuda = "12.1.*"

# E2 environment: MiniGrid only
[tool.pixi.feature.e2.pypi-dependencies]
minigrid = "==2.3.1"

# E9 environments: simulators that conflict on shared deps
[tool.pixi.feature.e9-alfworld.pypi-dependencies]
alfworld = "==0.3.5"
textworld = "==1.6.1"

[tool.pixi.feature.e9-scienceworld.pypi-dependencies]
scienceworld = "==1.2.0"

[tool.pixi.environments]
default = { features = [], solve-group = "default" }
dev = { features = ["dev"], solve-group = "default" }
cuda = { features = ["dev", "cuda"], solve-group = "cuda" }
e2 = { features = ["dev", "e2"], solve-group = "default" }
e2-cuda = { features = ["dev", "e2", "cuda"], solve-group = "cuda" }
e9-alfworld = { features = ["dev", "e9-alfworld"], solve-group = "e9a" }
e9-scienceworld = { features = ["dev", "e9-scienceworld"], solve-group = "e9s" }

[tool.pixi.tasks]
install = "pixi install"
test = { cmd = "pytest -q --cov=arch --cov=drivers --cov=exp", env = { PYTHONHASHSEED = "0" } }
lint = "ruff check . && ruff format --check ."
format = "ruff format ."
typecheck = "mypy drivers arch"
run-e0 = { cmd = "python run.py --experiment E0 --ablation A0 --config configs/mnist.yaml --seed 42", env = { PYTHONHASHSEED = "0" } }
run-e1 = { cmd = "python run.py --experiment E1 --ablation A0 --config configs/babyai-synthetic.yaml --seed 42", env = { PYTHONHASHSEED = "0" } }

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]

[tool.mypy]
python_version = "3.11"
strict = true
files = ["drivers", "arch"]
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["minigrid.*", "alfworld.*", "textworld.*", "scienceworld.*", "geoopt.*"]
ignore_missing_imports = true
```

## Version-pinning policy

Every dependency is pinned to an exact version (`==X.Y.Z`) in the manifest, and `pixi.lock` is committed to VCS so contributors get bit-identical environments. Bumps are deliberate: open a branch named `bump/<package>-<version>`, run `pixi update <package>`, run `pixi run test` and `pixi run typecheck`, and commit both the manifest change and the regenerated lock together. No automated bumps until Phase 6 lands; the cost of a silent torch or transformers regression mid-experiment outweighs the benefit of monthly updates.

## What's NOT decided here (and where it lives)

- Test fixtures and pytest layout: see [test-fixtures.md](test-fixtures.md).
- Repo directory layout (where `arch/`, `drivers/`, `exp/` live and how `run.py` imports them): deferred (waiting on the grammar-compiler ordering decision flagged in [build-order.md](../build-order.md)).
- CI workflow and matrix: deferred (after the first green `pixi run test` on `default` and `e2` envs).
- Checkpoint serialization format and `--checkpoint-dir` semantics: tracked in [cli-runner.md](../drivers/cli-runner.md) open question q07.
- Distributed training across seeds and ablations: tracked in [cli-runner.md](../drivers/cli-runner.md) open question q08.

## Links

- [README](../README.md)
- [build-order](../build-order.md)
- [drivers/_index.md](../drivers/_index.md)
- [pixi_guide](../../pixi_guide.md)
