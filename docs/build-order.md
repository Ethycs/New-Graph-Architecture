# Build Order

The order in which to actually build the system. Drivers first because they define the contracts that lock the rest of the work in place. Then arch atoms in dependency order. Then experiments, starting with the cheapest validation and ramping toward the full benchmark.

Read alongside [README](README.md) and the cluster indexes ([drivers](drivers/_index.md), [arch](arch/_index.md), [exp](exp/_index.md), [open](open/_index.md)).

## Phase 0 — Drivers (the contracts)

Build these first. Without them, every arch and exp note is unimplementable. This phase is dependency-ordered: each driver below consumes the ones above it.

1. [config.md](drivers/config.md) — the YAML; everything reads it. Defines `schema_version`, `embedding_dim`, `seed`, dataset enum, ablation enum.
2. [graph-fsm-spec.md](drivers/graph-fsm-spec.md) — vertices/edges/coordinates that arch and exp share. Constrains `embedding_dim` and the label set used downstream.
3. [typed-score-record.md](drivers/typed-score-record.md) — the wire format of Y. Length of `dist` is fixed by graph-fsm-spec; `z_H` length is fixed by config. Defines canonical `confidence`/`margin` semantics.
4. [ablation-flags.md](drivers/ablation-flags.md) — the boolean knobs; A0–A9 are tuples. Selected by config; consumed by every flag-aware arch atom.
5. [metrics-jsonl.md](drivers/metrics-jsonl.md), [results-jsonl.md](drivers/results-jsonl.md) — output streams. Both share `run_id`/`experiment`/`ablation`/`seed`/`step` with each other and with typed-score-record.
6. [cli-runner.md](drivers/cli-runner.md) — the entry point that ties Phase 0 together. Loads config + graph FSM + ablation tuple, derives `run_id`, snapshots inputs to `runs/<run_id>/`, and routes the arch and exp halves through one shared environment.

**Acceptance:** a `python run.py --experiment E0 --ablation A0 --config configs/mnist.yaml --seed 42` invocation can load a config, validate a graph FSM spec, resolve the A0 tuple, and write empty `runs/E0_A0_seed42/{metrics,results,scores}.jsonl` files for a noop pipeline. No model code yet.

## Phase 1 — Architecture: minimal viable typed pipeline

Goal: classifier → typed Y → graph mask → predictions. Validates Phase 0 with real data; everything below is in dependency order.

1. [typed-field-pipeline](arch/typed/typed-field-pipeline.md) — observation $x \to$ grammar objects $\to$ typed feature vector $\mathbf{t}$. Front of the pipeline.
2. [typed-score-record (arch atom)](arch/typed/typed-score-record.md) — packs $(q_t, \lambda_t, p_t, z_t, \mathbf{e}_t)$ and serializes per [typed-score-record contract](drivers/typed-score-record.md).
3. [graph-fsm](arch/graph/graph-fsm.md) — loads [graph-fsm-spec](drivers/graph-fsm-spec.md) into the in-memory $(V, E, w_v, g_v, m_v)$ tuple.
4. [graph-legality-mask](arch/graph/graph-legality-mask.md) — builds the boolean adjacency mask, applies it pre-softmax. Consumed by typed-field-pipeline at inference.
5. [margin-uncertainty](arch/typed/margin-uncertainty.md) — top-1 minus top-2 score; populates `margin` in [typed-score-record contract](drivers/typed-score-record.md) and [results.jsonl](drivers/results-jsonl.md).
6. [confusion-graph](arch/typed/confusion-graph.md) — accumulates $(y_{\text{true}}, y_{\text{hat}}, m)$ tuples from results.jsonl into a directed multigraph for diagnostic feedback.

**Acceptance:** [E0 MNIST](exp/e0-mnist.md) runs end-to-end and writes `accuracy`, `low_margin_accuracy`, `confusion_graph_density` into metrics.jsonl plus per-sample `singular_flag` (margin-only) into results.jsonl.

## Phase 2 — Architecture: singularity + stratification

Goal: aggregate margin/contradiction/loop/illegal signals into σ(x) and validate it against ground-truth errors. Margin-based singularity detection works without hyperbolic embedding; deeper geometric stratification arrives in Phase 3.

1. [singularity-types](arch/singularity/singularity-types.md) — taxonomy (low-margin, decision-tie, contradiction, illegal, loop, stabilizer-jump). Defines what σ(x) is summing over.
2. [catastrophe-labels](arch/singularity/catastrophe-labels.md) — annotates FSM edges with fold/cusp/swallowtail tags; supplies priors to σ(x).
3. [singularity-detector](arch/singularity/singularity-detector.md) — combines the above signals into σ(x) ∈ [0,1]. Reads `margin` from Phase 1; writes `singular_flag` and σ score into [results.jsonl](drivers/results-jsonl.md).
4. [failure-margin-auroc](arch/singularity/failure-margin-auroc.md) — computes AUROC of σ(x) against $y \in \{0,1\}$; writes `auroc_failure` into [metrics.jsonl](drivers/metrics-jsonl.md).
5. [behavioral-stratum-tagger](arch/singularity/behavioral-stratum-tagger.md) — discrete categorical tagging over the [singularity-types](arch/singularity/singularity-types.md) catalog. No geometry; consumed by [singularity-detector](arch/singularity/singularity-detector.md). The geometric Whitney content lands in Phase 5 inside [stratified-partition-function](arch/energy/stratified-partition-function.md).

**Acceptance:** [E1 synthetic BabyAI](exp/e1-synthetic-babyai.md) validates graph mask + σ(x) against an in-process oracle; [E4 singularity AUROC](exp/e4-singularity-auroc.md) shows σ(x) AUROC ≥ margin-only AUROC by ≥ 3% on E0/E1 outputs (resolves [q10](open/q10-failure-prediction-baseline.md)).

## Phase 3 — Architecture: hyperbolic geometry

Goal: replace the Euclidean fallback with $\mathbb{H}^d$ embedding so prototypes, distances, and stratification become hierarchy-aware. Each atom below is gated by `hyperbolic_geometry_enabled` from [ablation-flags](drivers/ablation-flags.md); if the flag is off (A6), Phase 1's Euclidean code path runs instead.

1. [hyperbolic-embedding](arch/hyperbolic/hyperbolic-embedding.md) — embeds the typed graph into Poincaré or Lorentz $\mathbb{H}^d$; produces `coordinates.node_embeddings` consumable by [graph-fsm-spec](drivers/graph-fsm-spec.md).
2. [graph-prototype-vectors](arch/hyperbolic/graph-prototype-vectors.md) — per-type anchors $\mathbf{p}_v \in \mathbb{H}^d$. Depends on hyperbolic-embedding.
3. [graph-extrusion](arch/hyperbolic/graph-extrusion.md) — extrudes the discrete graph into a stratified manifold (cells + tubes). Consumed by [stratified-partition-function](arch/energy/stratified-partition-function.md) in Phase 5, which is where Whitney conditions are actually checked.
4. [hyperbolic-distance-loss](arch/hyperbolic/hyperbolic-distance-loss.md) — training loss that pulls observations toward prototype, repels off-type prototypes. Depends on graph-prototype-vectors.
5. [gromov-hyperbolicity-diagnostic](arch/hyperbolic/gromov-hyperbolicity-diagnostic.md) — validation metric: estimate δ on a sample of 4-tuples; flag if the embedding is not sufficiently tree-like. Runs after each training cycle.

**Phase 2 / Phase 5 split:** Phase 2 ships the lightweight [behavioral-stratum-tagger](arch/singularity/behavioral-stratum-tagger.md) (discrete categorical, no geometry). The geometric Whitney content lands later inside [stratified-partition-function](arch/energy/stratified-partition-function.md) (Phase 5), which is the only consumer that actually needs it. There is no upgrade path between the two; they are different concerns sharing only a name.

**Acceptance:** [E3 hyperbolic vs Euclidean](exp/e3-hyperbolic-vs-euclidean.md) shows accuracy at $d=8$ hyperbolic ≥ accuracy at $d=16$ Euclidean (resolves [q01](open/q01-hyperbolic-dim.md), informs [q11](open/q11-hyperbolic-vs-euclidean-tradeoff.md)).

## Phase 4 — Architecture: group quotient

Goal: compress the graph by group action so attention runs at orbit granularity. Each atom is gated by `group_quotient_enabled`.

1. [group-action-on-graph](arch/group/group-action-on-graph.md) — builds permutation generators and the action $\sigma : H \times V \to V$ from [graph-fsm-spec](drivers/graph-fsm-spec.md).$m_v$.
2. [orbit-quotient-space](arch/group/orbit-quotient-space.md) — collapses $V$ into $V/H$ with multiplicities. Depends on group-action-on-graph.
3. [stabilizer-signature](arch/group/stabilizer-signature.md) — per-state stabilizer record; flags singular states (non-trivial stabilizer) and feeds them into [singularity-detector](arch/singularity/singularity-detector.md). Depends on group-action-on-graph.
4. [orbit-pair-attention](arch/group/orbit-pair-attention.md) — attention over $V/H \times V/H$ instead of $V \times V$; broadcasts back to per-state weights. Depends on orbit-quotient-space.

**Acceptance:** [E6 group-quotient attention](exp/e6-group-quotient-attention.md) shows ≥ 50% FLOPs reduction at parity success rate on E2 traces.

## Phase 5 — Architecture: energy & reservoir

Goal: turn the typed pipeline into an energy-shaped Boltzmann sampler with a frozen reservoir backbone. The grammar compiler also lands here because it's the offline tool that produces both the FSM spec and the prototype seeds; later phases can be retrofitted to use compiler output once it exists.

1. [frozen-encoder-backbone](arch/substrate/frozen-encoder-backbone.md) — pre-trained encoder with `requires_grad=False`. Gated by `reservoir_frozen` (default true; A8 unfreezes).
2. [typed-readout-layer](arch/typed/typed-readout-layer.md) — per-type MLP heads on top of the frozen encoder. Depends on frozen-encoder-backbone.
3. [energy-function-E](arch/energy/energy-function-E.md) — scalar pressure combining cost, uncertainty, contradiction, loop, progress. Reads `margin` (Phase 1) and σ signals (Phase 2). Open spec at [q02](open/q02-energy-function-spec.md) and [q07](open/q07-singularity-loss-weighting.md).
4. [stratified-partition-function](arch/energy/stratified-partition-function.md) — merges Whitney stratification of the extruded manifold (formerly the geometric content of `whitney-stratification-layer`) with the Boltzmann normalization $Z = \sum_\lambda Z_\lambda$. Depends on [graph-extrusion](arch/hyperbolic/graph-extrusion.md), [hyperbolic-embedding](arch/hyperbolic/hyperbolic-embedding.md), and [energy-function-E](arch/energy/energy-function-E.md). Provides per-stratum probability $P(\lambda)$ and per-state $P(x)$.
5. [energy-weighted-loss](arch/energy/energy-weighted-loss.md) — re-weights base loss by $w(E(x_*))$ to upweight singular/high-energy states. Depends on energy-function-E and the typed-readout outputs.
6. [grammar-compiler](arch/substrate/grammar-compiler.md) — DSL that compiles a declarative grammar spec into the FSM, prototype seeds, and legality mask. Drops in as the canonical producer for [graph-fsm-spec](drivers/graph-fsm-spec.md) and [graph-prototype-vectors](arch/hyperbolic/graph-prototype-vectors.md) once present; Phases 1–4 can run with hand-authored YAML until then.

**Optional / off-critical-path arch atoms** (none of them block any E experiment, but they extend the singularity-theory side of the system):

- [weighted-plumbing-graph](arch/graph/weighted-plumbing-graph.md) — enriches the FSM with intersection-form weights for surface-singularity realization.
- [dynkin-ade-classification](arch/graph/dynkin-ade-classification.md) — recognizes when the graph is an ADE quiver.
- [graph-realization-functor](arch/graph/graph-realization-functor.md) — lifts an enriched graph to a singularity object.
- [singularity-extraction-functor](arch/graph/singularity-extraction-functor.md) — the inverse direction.

Build these only after Phase 5 lands and only if you need the algebraic-geometry route for a specific claim.

**Acceptance:** [E7 reservoir vs end-to-end](exp/e7-reservoir-vs-end2end.md) shows reservoir readout achieves ≥ 90% of end-to-end accuracy with < 10% trainable parameters (resolves [q09](open/q09-reservoir-freeze-schedule.md)).

## Phase 6 — Experiments: ramp

Starting with the cheapest, validating each architectural phase as it lands. Each experiment writes [metrics.jsonl](drivers/metrics-jsonl.md) and [results.jsonl](drivers/results-jsonl.md) under `runs/<run_id>/`; [Evidence-Level Tracker](exp/evidence-tracker.md) reads them all.

1. [E0 — MNIST](exp/e0-mnist.md) — validates Phase 1 on sklearn digits; smoke-tests the typed-score → confusion-graph → margin pipeline. No state machine.
2. [E1 — Synthetic BabyAI](exp/e1-synthetic-babyai.md) — validates Phase 1 + Phase 2; in-process 7-state grid; isolates the graph-mask claim and the margin-based singularity flag.
3. [E4 — Singularity AUROC](exp/e4-singularity-auroc.md) — Phase 2; consumes E0/E1/E2 outputs to test σ(x) AUROC vs margin/entropy/confidence baselines. Load-bearing for the singularity claim ([q10](open/q10-failure-prediction-baseline.md)).
4. [E5 — IDF ablation](exp/e5-idf-ablation.md) — Phase 1+2; toggles `idf_weighting_enabled` (A5) and measures rare-state recall. Resolves [q03](open/q03-idf-runtime-schedule.md).
5. [E3 — Hyperbolic vs Euclidean](exp/e3-hyperbolic-vs-euclidean.md) — Phase 3; sweep over $d \in \{2,4,8,16,32\}$ to test the hyperbolic-compression claim. Resolves [q01](open/q01-hyperbolic-dim.md), [q11](open/q11-hyperbolic-vs-euclidean-tradeoff.md).
6. [E6 — Group-quotient attention](exp/e6-group-quotient-attention.md) — Phase 4; FLOPs vs success-rate on E2 traces; gates on [q06](open/q06-group-action-discovery.md), [q08](open/q08-quotient-reexpansion-threshold.md).
7. [E7 — Reservoir vs end-to-end](exp/e7-reservoir-vs-end2end.md) — Phase 5; toggles `reservoir_frozen` (A8) and compares trainable params + sample efficiency. Resolves [q09](open/q09-reservoir-freeze-schedule.md).
8. [E2 — Real BabyAI](exp/e2-real-babyai.md) — full integration on real MiniGrid/BabyAI; primary reproducibility target. Requires Phases 1–4 (and Phase 5 reservoir if A0 baseline includes the frozen encoder). Resolves [q04](open/q04-babyai-dataset-choice.md).
9. [E8 — Transfer](exp/e8-transfer-experiment.md) — Phases 1–4; final structural claim — does the typed/hyperbolic abstraction transfer zero-shot across BabyAI families? Resolves [q05](open/q05-transfer-success-criterion.md).
10. [E9 — Full Trace Benchmark](exp/e9-full-trace-benchmark.md) — full architecture across BabyAI + ALFWorld + ScienceWorld; the integration test. Requires every prior phase plus the dataset adapters under [exp/](exp/_index.md).

## Cross-cutting

- [Ablation matrix](exp/ablation-matrix.md) — runs `A0`–`A9` for every E experiment that has a meaningful ablation hook; the canonical proof that each architectural component is load-bearing.
- [Metric Collectors](exp/metric-collectors.md) — the shared writer for [metrics.jsonl](drivers/metrics-jsonl.md); update its canonical metric table whenever a new metric joins the standard set.
- [Evidence-Level Tracker](exp/evidence-tracker.md) — re-aggregates `runs/*/metrics.jsonl` after every E experiment closes; promotes claims from `synthetic_benchmark` to `real_benchmark` to `load_bearing` as evidence accumulates.
- [Open questions](open/_index.md) — each question is referenced by the experiment that resolves it; close each one as evidence lands.

## Critical-path summary

Shortest path to a publishable result:

> **Phase 0 → Phase 1 → Phase 2 → E0 → E1 → E4 → E5**

That sequence stands up the contracts, the typed-score + graph-mask + margin trio, and the singularity detector, then validates the load-bearing claims on cheap benchmarks. It produces the table that closes [q10](open/q10-failure-prediction-baseline.md) and [q03](open/q03-idf-runtime-schedule.md), and it is the minimum needed before committing to the more expensive Phase 3 / Phase 4 / E2 / E9 work.

## Build-order considerations surfaced during driver polish

Notes captured here (not edits to arch/exp content) for the next pass:

- **`reservoir_frozen` default semantics — RESOLVED as variant axis V1.** A0 stays frozen, A8 stays unfrozen. E7 reports the matched pair `(A0_frozen, A8_unfrozen)` rather than treating A8 as a deviation from a single baseline. See [Ablation Flag Set §Variant axes](drivers/ablation-flags.md#variant-axes-orthogonal-to-a0-a9) V1. No code or runtime changes; reporting templates list both rows.
- **`A4` semantics.** A4 keeps `singularity_detector_enabled=true` but disables routing on σ(x). The arch component implementing this gating is currently implicit; the next arch pass should name where in [singularity-detector](arch/singularity/singularity-detector.md) the "compute but don't act" branch lives.
- **`whitney-stratification-layer` split — RESOLVED by atom rename and merge.** The original atom is gone. Phase 2 ships [behavioral-stratum-tagger](arch/singularity/behavioral-stratum-tagger.md), a discrete categorical tagger over the [singularity-types](arch/singularity/singularity-types.md) catalog (no geometry, no Whitney conditions). Phase 5 ships [stratified-partition-function](arch/energy/stratified-partition-function.md), which merges the geometric Whitney stratification of the extruded manifold with the Boltzmann normalization $Z$ that consumes it. The two atoms are different concerns; there is no upgrade path between them.
- **Grammar compiler ordering — RESOLVED as variant axis V2.** [grammar-compiler](arch/substrate/grammar-compiler.md) stays in Phase 5; while absent, only `Config.graph_fsm_source = "hand"` is selectable. When it lands, `"compiled"` becomes available as an additional producer that must emit artifacts equivalent to the hand-authored path. An acceptance test (`test_compiler_output_matches_hand_authored`) gates the variant; cheap experiments (E0–E5) run the full `(A × V2)` grid, expensive ones (E2, E9) run only the resolved variant. See [Ablation Flag Set §Variant axes](drivers/ablation-flags.md#variant-axes-orthogonal-to-a0-a9) V2 and the new `graph_fsm_source` field on [Config](drivers/config.md).
- **Dataset adapters.** The Phase 6 ramp depends on the adapters under [exp/](exp/_index.md) (`dataset-mnist-typed`, `dataset-synthetic-babyai-grid`, `dataset-minigrid-wrapper`, `dataset-alfworld-loader`, `dataset-scienceworld-loader`, `dataset-trace-replay`). Each adapter should land just before the first experiment that needs it; the build order above implicitly assumes that.
