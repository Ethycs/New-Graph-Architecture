# E24 — Universal Graph Extraction (Phase 20 Wave A)

**Cluster:** exp
**Status:** implemented (Wave A sanity tier)
**Tags:** #phase-20 #graph-extraction #sanity #synthetic-substrate

## What

The integration runner for the graph-extraction proposal. Pipes the six-step pipeline end-to-end: harvest → discretise + K-select → count transitions → Phase A (Beta-Dirichlet M-step) → type-discover → compare. Wave A ships the **synthetic substrate** sanity tier: generate a known typed Markov chain over $K_{\text{true}}$ states with a noisy-centroid embedding, run the extraction, and confirm that the extracted FSM matches the ground-truth FSM by normalised Hamming distance and by held-out predictive likelihood. Wave B will replace the synthetic substrate with `TorchEnergyTrainer` activations on Python big and JSON.

## Why

Phase 20's central hypothesis (`docs/proposals/graph-extraction.md`) is that a typed graph is latent in the activations of any sufficiently trained network on a structurally regular corpus. E24 is the **decisive integration runner** that makes that hypothesis testable: it composes the three new atoms (`hidden_state_harvester`, `bayesian_nonparametric_k`, and the runner's own extraction logic) with the existing posterior / forward-backward stack, then evaluates the extracted graph against ground truth. The synthetic tier (Wave A) validates the machinery without needing a trained-model checkpoint; the cross-substrate tiers (Wave B → D) plug in real encoders without changing the pipeline.

## Interface

- **Runner entry point:** `run_e24_synthetic(config, ablation, run_id, output_dir, seed) -> E24Result`.
- **CLI:** `python -m nga.cli --experiment E24 --ablation A0 --config tests/fixtures/configs/e24_graph_extraction_minimal.yaml --seed 42 --output runs/E24_A0_seed42 --ablation-file tests/fixtures/ablations/ablations.yaml`.
- **Config knobs:** `K_true`, `K_range_min`, `K_range_max`, `n_samples`, `sequence_length`, `embed_dim`, `noise_scale`, `holdout_n_samples`, `k_selection_criterion` ∈ {`holdout_nll`, `bic`, `elbow`}.

## Output artefacts (under `runs/E24_A0_seed{seed}/`)

- `metrics.jsonl` — per-metric records: `K_star`, `K_ground_truth`, `extracted_hamming_normalised`, `holdout_nll_per_token_extracted`, `holdout_nll_per_token_chain`, `holdout_nll_improvement_per_token`, `phase_a_hamming_normalised`, `cluster_purity`, plus all five compute-efficiency metrics.
- `results.jsonl` — per-row results: `step`, `sample_id`, `y_true` (ground-truth state ID), `y_hat` (discovered cluster ID), `singular_flag`, `sigma_score` (currently zero; not load-bearing in the synthetic runner).
- `extracted_graph.json` — the discovered graph itself: `K_star`, `types`, `extracted_mask`, `ground_truth_legality`, `best_permutation`, `criterion`, `K_range`, `K_scores`.
- `config_snapshot.yaml`, `ablation_snapshot.yaml` — provenance (emitted by the CLI shell).

`decision_trace.jsonl` is **intentionally not populated** by the Wave-A synthetic runner — there is no FSM-step argmax for which a per-row trace is well-defined. Wave B's real-substrate variants will emit it.

## Acceptance bars (Wave A sanity)

The e2e suite (`tests/e2e/test_e24_graph_extraction.py`) enforces:

1. `K_star == K_ground_truth` — K-selection recovers the true cardinality.
2. `extracted_hamming_normalised <= 0.10` — the Tier 1 sanity bar from the proposal.
3. `holdout_nll_improvement_per_token > 0` — extracted FSM beats a uniform-transition chain on held-out trajectories.
4. `phase_a_hamming_normalised == 0` — posterior-derived legality matrix is self-consistent.
5. `cluster_purity > 5 / K` — material recovery above chance (chance is 1/K).
6. All five compute-efficiency metrics present and non-negative.
7. `extracted_graph.json` is well-formed.

## Links

- **See also:** [Hidden State Harvester](../arch/substrate/hidden-state-harvester.md), [Bayesian Nonparametric K](../arch/energy/bayesian-nonparametric-k.md), [Forward Backward](../arch/energy/forward-backward.md), [Posterior Mask](../arch/energy/posterior-mask.md).
- **Driven by:** the graph-extraction proposal in [docs/proposals/graph-extraction.md](../proposals/graph-extraction.md).
- **Drives:** the Wave B / C / D real-substrate runs (each will reuse `extract_graph` with a different harvester).
- **Math:** the extraction is a map (trained network, corpus) → (Beta posterior on typed graph). Phase A's correctness on hand-authored vertex sets carries to extracted vertex sets conditional on the discretisation (the math is the same).
- **Open:** [[open.qNN-extraction-depth]] — which layer of a multi-layer network to harvest from; [[open.qNN-equivariance-emergence]] — whether the extracted base graph respects an emergent equivariance.
