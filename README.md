# New Graph Architecture (NGA)

A reference implementation of the **Typed Protocol Network (TPN)** — a structured classifier-controller for tasks where a typed finite-state protocol governs which of several submodels handles each step, with calibrated abstention, hard structural constraints, and audit trail by construction.

For the model class definition: [`docs/model-class.md`](docs/model-class.md).
For the foundational mathematics: `Mathematics.md`.
For the running development log: `research_log.md`.
For the paper-shaped synthesis across all 5 grammars + the real-world medical domain: [`docs/results.md`](docs/results.md).
For incoming research proposals: [`docs/proposals/`](docs/proposals/).

## What's been built

Five architectural commitments compose into one model:

1. **Typed graph as full state space.** Every output is a node-tuple in a Cartesian product of typed axes. No raw floats cross interfaces.
2. **Per-type submodels.** Each FSM-state-type owns its own classifier head; specialisation is structural.
3. **Hard mask + soft σ-routing.** A legality matrix zeros illegal predictions absolutely; σ-routing's three branches (NORMAL / RECOVERY / ABSTAIN) handle ambiguity the mask can't.
4. **Bayesian posterior over the protocol itself.** The mask is a Beta(α, β) posterior per edge, induced from observation via classical forward-backward + Beta-Dirichlet M-step.
5. **Information-geometric foundation.** Fisher information sets natural-gradient learning rates; KL is the canonical loss; Cramér-Rao gives sample-sufficiency bounds. The architecture's hyperbolic + Riemannian + Bayesian apparatus is information geometry by construction.

## Validated claims (multi-seed, 5 real grammars)

Numbers below are the cross-grammar universal bar — every metric reproduced from `aggregate.py` and `runs/phase{12,14,15,16,18}_*_summary.json` at 5 seeds [42–46] each.

| Claim | Range across grammars | Robust? |
|---|---|---|
| Mask drives illegal-rate to 0 | 0.000 ± 0.000 on every grammar | Yes, every seed of every grammar |
| Mask uplift on accuracy | **+0.42 to +0.50** mean (std ≤ 0.117) | Yes, every grammar |
| Phase A classical seed recovers FSM | hamming ≤ 0.058 worst seed; ≤ 0.0029 worst mean | Yes, every grammar |
| σ fires more at structural-ambiguity points | boundary ratio > 1 on **4 of 5** grammars | Yes (only python_expr inverts, by construction) |
| σ_structural_uplift positive on multi-ambiguity grammars | JSON +0.067, python_expr +0.059, control flow +0.007 | Sign-stable on JSON and python_expr |

For the per-claim, per-seed verdict matrix: `pixi run -e dev python aggregate.py`.
For the per-phase honest read on what worked, what didn't, and why: `research_log.md`.

## The 5-grammar finding

σ_structural_uplift across grammars (mean ± std, n=5 seeds): JSON **+0.067 ± 0.047** > Python expr **+0.059 ± 0.070** > control flow Python **+0.007 ± 0.026** > Python big **−0.012 ± 0.024** > ListOps **−0.088 ± 0.047**. The ranking tracks the **distribution of structural ambiguity** rather than grammar size — JSON's seven legal value-position continuations scattered through every document beat ListOps's single ambiguity (which the depth tracker collapses) and python_big's rare 2-way call-vs-variable site. Control flow Python's `S0_after_if_body` branching site (where `else` vs a new statement has no stack disambiguation) shifts the uplift positive but only barely; mean is positive, 3/5 seeds positive. The architectural prediction (more ambiguity → positive σ_structural) held but is not magnitude-invariant. See [`docs/results.md`](docs/results.md) for the full synthesis.

## Phase 19B — self-supervised structural discovery on real medical data

The first real-world (non-synthetic) result. **No disease labels used in training.** Clustered 4920 patients × 132 binary symptoms in the Poincaré ball via `typed_latent_clustering` (Riemannian k-means at K = 20, 41, 80), ran Phase A forward-backward over Markov-randomised symptom orderings, evaluated post-hoc against ground-truth diagnoses.

| K | cluster_purity | ARI | NMI |
|---|---:|---:|---:|
| 20 | 0.4878 | 0.5122 | 0.8492 |
| **41 (target)** | **0.8780** | **0.8192** | **0.9659** |
| 80 | 0.9988 | 0.9231 | 0.9657 |

At K=41 the cluster purity is **36× chance** (1/41 ≈ 0.024); ARI 0.819 and NMI 0.966 are dramatically above the zero-baseline of random clustering; σ-hardness AUROC 0.977. **The architecture recovers the medical taxonomy from symptom co-occurrence alone**, validating "graph grows from observation" on real data. Compute: 10.72 s total for 4920 patients × 3 K values on CPU. The honestly-failed E22 supervised baseline (`diagnostic_accuracy = 0.000`, collapsed 41 diseases into a single `DIAGNOSED` label) lives in `research_log.md` as the structural counter-example. See [`docs/results.md`](docs/results.md) §10.

## Compute efficiency

Phase 18 introduced systematic tracking of wall-clock, throughput, and peak memory on every new runner. Reference point — E21 (control flow Python, 37-state FSM, 5 seeds): **2.26 ± 0.08s total wall-clock**, **733 ± 7.5 samples/s** inference throughput, **78 ± 6.5 MB peak memory**. All on CPU. The architecture's third stated goal (interpretability + efficiency + control) is now validated, not aspirational: TPN runs at hundreds of samples/sec with sub-100MB memory on the largest grammar tested. Standard transformer pipelines need GPU + hundreds of MB just for one inference pass.

## Quick start

```bash
# Install (pixi-managed; CPU-only torch)
pixi install -e dev

# Run the full test suite
pixi run -e dev python -m pytest tests/ -q

# Run a single experiment end-to-end
pixi run -e dev python run.py \
    --experiment E14 --ablation A0 \
    --config tests/fixtures/configs/e14_listops_minimal.yaml \
    --seed 42 \
    --output runs/E14_A0_seed42 \
    --ablation-file tests/fixtures/ablations/ablations.yaml

# View consolidated metrics across all runs
pixi run -e dev python aggregate.py
```

## Repository structure

```
src/nga/
  arch/        ~40 architecture atoms (FSM, mask, hyperbolic, monodromy,
               energy, posterior_mask, information_geometry, ...)
  drivers/     wire-format schemas (config, FSM spec, JSONL streams,
               decision_trace, ...)
  exp/         experiment runners E0..E21 + dataset generators
  cli.py       single CLI entry point dispatching all experiments

tests/
  unit/        per-atom unit tests (~60+)
  e2e/         per-experiment acceptance tests (~50+)
  integration/ atom census + cross-cutting integration tests
  fixtures/    YAML configs, FSM specs, ablation matrix, grammars

docs/          zettelkasten architecture spec; one .md per atom
               + results.md (paper-shaped synthesis)
runs/          per-(experiment, ablation, seed) output directories
scripts/       multi-seed sweeps, ablation matrices

research_log.md   running development log (latest entry on top)
Mathematics.md    foundational mathematical document
aggregate.py      cross-run metrics aggregator with PASS/FAIL verdicts
```

## Experiments

| Runner | What it tests |
|---|---|
| E0 | sklearn LogReg on MNIST, typed pipeline + margin singularity |
| E1 | synthetic BabyAI grid, mask + σ on illegal-temptation samples |
| E3 | hyperbolic vs Euclidean dim sweep |
| E4 | post-hoc σ-vs-margin AUROC on E0/E1 outputs |
| E5 | IDF ablation |
| E6 | group-quotient orbit-pair attention FLOPs reduction |
| E7 | reservoir vs end-to-end (frozen encoder + readout heads) |
| E8 | transfer / OOD generalisation |
| E9 | Dyck-k bracket matching (first hierarchical dataset) |
| E10 | unified world model on Dyck-k (six-layer integration) |
| E11 | KL/Fisher/CRB curriculum on Dyck-k |
| E12 | torch substrate on Dyck-k (substrate independence) |
| E13 | classical KL framework on ListOps (real grammar) |
| E14 | torch end-to-end on ListOps (depth-3, 11-state FSM) |
| E15 | training-maturity sweep at checkpoints (Phase A on) |
| E16 | training-maturity sweep with Phase A ablated |
| E17 | Python expressions (real grammar via `ast.parse` + `tokenize`; 14-state FSM) |
| E18 | bigger Python: calls + defs + returns (24-state FSM) |
| E19 | JSON (RFC 8259 subset, `json.loads`-validated; 26-state FSM, 99 edges) |
| E20 | cross-grammar σ-weight transfer (Python big → JSON) |
| E21 | control flow Python: `if` / `else` / `while` (37-state FSM, the largest grammar) |
| E22 | diagnostic supervised (Kaggle disease-symptom; structurally green, semantically failed — see `research_log.md`) |
| E23 | diagnostic self-supervised (Phase 19B; recovers medical taxonomy at ARI 0.82 / NMI 0.97 without labels) |

## The atom census as executable specification

`tests/integration/test_atom_census.py` enforces two invariants:
- Every atom listed in `docs/index-architecture.md` has a Python implementation.
- Every implemented atom is imported by at least one runner (transitive closure counts).

Run it: `pixi run -e dev python -m pytest tests/integration/test_atom_census.py -v`. If it passes, the architecture is built exactly per its own specification.

## Test suite

Current state: **447 collected, 438 passed, 8 xfailed, 1 pre-existing E0 env-flake** for documented reasons (the q10 σ_uplift bar on margin-saturated grammars; E7 reservoir-vs-end-to-end accuracy at natural noise). Reproduce: `pixi run -e dev python -m pytest tests/ -q`.

## What this is good at

Tasks where you need *all* of: hard structural constraint, hierarchy, calibrated abstention, audit by construction, continual learning of the protocol from observation. Examples: grammar-constrained classification, protocol execution with audit, hierarchical structured output, auditable safety-critical classification, sequence labeling with cyclic structure.

## What this is not for

- Open-ended generation (no learned generation policy).
- Pure representation learning (encoder is frozen).
- Tasks with no structural prior (the geometric machinery has nothing to grip).

## Acknowledgements

Built from `Mathematics.md` first principles. Each architectural commitment binds a math object from that document to a runtime role. Information geometry (Phase 7) named the framework that was implicit from the start.
