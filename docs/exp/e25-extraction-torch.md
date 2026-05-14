# E25 — Graph Extraction on the Torch Substrate (Phase 20 Wave B Tier 1)

**Cluster:** exp
**Status:** implemented (Tier 1 sanity on python_big)
**Tags:** #phase-20 #graph-extraction #wave-b #tier-1 #python-big

## What

Plug the universal-graph-extraction pipeline (E24) into our own `FrozenEncoderTorch` substrate on the python_big grammar. The runner harvests the encoder's hidden states over a python_big dataset, runs the six-step extraction (cluster → K-select → count → Phase A → type-discover → compare), and reports best-permutation normalised Hamming against the hand-authored `python_big.fsm.yaml` legality matrix.

## Why

E24 validated the extraction *machinery* on a synthetic typed Markov chain where the substrate was constructed to make the FSM trivially recoverable. E25 is the first **real-substrate** test of the universality hypothesis from `docs/proposals/graph-extraction.md`: does the extraction recover the FSM from a real grammar's encoder hidden states, without ever seeing FSM labels? The python_big grammar (24 vertices, 89 edges) is the natural single-grammar entry point — it is the grammar where most of Phase 14–18's σ structural-uplift results landed.

## Interface

- **Runner entry point:** `run_e25(config, ablation, fsm, run_id, output_dir, seed, n_programs=80, encoder_hidden_dim=32, k_selection_criterion='bic', K_range_pad=5, holdout_fraction=0.2) -> E25Result`.
- **CLI:** `python -m nga.cli --experiment E25 --ablation A0 --config tests/fixtures/configs/e25_extraction_torch_minimal.yaml --seed 42 --output runs/E25_A0_seed42 --ablation-file tests/fixtures/ablations/ablations.yaml`.
- **Output artefacts:** `metrics.jsonl`, `results.jsonl`, `config_snapshot.yaml`, `ablation_snapshot.yaml`, `extracted_graph.json` (the extracted legality matrix, the gold legality matrix, the FSM vertex IDs in canonical order, the chosen permutation when alignable).

## Substrate

`FrozenEncoderTorch.fit` is a no-op flag flip — the encoder is randomly initialised under the seed and its parameters all carry `requires_grad=False`. The harvest's `hidden_states` is therefore a deterministic, seeded, non-linear projection of the per-token raw features into the trainer's embedding space, **without any training that targets the FSM**. This is the honest sanity test: if the extraction recovers the FSM, it is because token-level features alone (the encoder's input) already cluster informatively; if it does not, a trained encoder is required, and Wave C+ delivers one.

## Wave B Tier 1 result on python_big (seed 42, n_programs=80)

| Metric | Value | Interpretation |
|---|---:|---|
| `V_ground_truth` | 24 | python_big's hand-authored FSM vertex count |
| `K_star` | 22 | K-selection (BIC) picked 22 — off by 2 |
| `extracted_hamming_normalised` | 1.000 (sentinel) | K_star ≠ V, so permutation alignment is undefined; sentinel emitted |
| `cluster_purity` | 0.3553 | **8.5× chance** (chance = 1/24 ≈ 0.042) |
| `holdout_nll_per_token_extracted` | ~2.07 | extracted transition matrix's NLL |
| `holdout_nll_per_token_chain` | ~3.09 | uniform-chain baseline NLL |
| `holdout_nll_improvement_per_token` | **+1.015 nat/token** | extracted FSM decisively beats uniform chain |
| `n_total_steps` | 2066 | tokens across 80 programs |
| `total_wall_clock_seconds` | 0.27 | end-to-end on CPU |
| `extraction_throughput_steps_per_sec` | 7587 | |
| `peak_memory_kb` | ~358 MB | mostly torch import overhead |

## Reading the result honestly

The Tier 1 sanity bar in the proposal is normalised Hamming ≤ 0.05 on ≥ 3/5 grammars. On python_big with a **frozen-random-projection encoder**, that bar is **not met** — K-selection picks 22 instead of 24, and the Hamming sentinel triggers. However, the other extraction-quality signals are positive:

- **Cluster purity 0.3553 = 8.5× chance** says the clusters carry real FSM-state information.
- **Held-out NLL improvement +1.015 nat/token** says the extracted transition matrix predicts unseen python_big trajectories materially better than a uniform-chain baseline.

The architectural finding: token-level raw features carry **non-trivial but incomplete** FSM-state information through a frozen random projection. A *trained* encoder is required to pin down the exact vertex count and meet the strict Hamming bar; Wave C (an external small transformer trained on the same corpus) is the right next test of the universality hypothesis.

## Acceptance bars (Wave B sanity tests)

The e2e suite (`tests/e2e/test_e25_extraction_torch.py`) enforces:

1. All standard artefacts emitted (`metrics.jsonl`, `results.jsonl`, `config_snapshot.yaml`, `ablation_snapshot.yaml`, `extracted_graph.json`).
2. `K_star` within `K_range_pad` of `V` (selection is plausible).
3. `extracted_hamming_normalised` and `cluster_purity` recorded as floats in `[0, 1]`. **No strict numerical bar on Hamming at this tier** — the architectural reading lives in `research_log.md`, not in a gating assertion.
4. `cluster_purity > 2 / V` (materially above chance).
5. All five compute-efficiency metrics present and non-negative.
6. `extracted_graph.json` well-formed and carries the extracted + gold legality matrices.

## Links

- **See also:** [E24 Graph Extraction](./e24-graph-extraction.md) (synthetic sanity tier), [Hidden State Harvester](../arch/substrate/hidden-state-harvester.md), [Bayesian Nonparametric K](../arch/energy/bayesian-nonparametric-k.md), [Forward Backward](../arch/energy/forward-backward.md).
- **Driven by:** [docs/proposals/graph-extraction.md](../proposals/graph-extraction.md) Tier 1 sanity.
- **Drives:** Wave C (external transformer); the multi-grammar bundle that will run E25 on python_expr / python_big / JSON / control flow / ListOps with multi-seed.
- **Open:** whether a trained encoder (Phase B in E18 etc.) on the same substrate brings K_star to V and Hamming under 0.05 — the natural Wave-B follow-up.
