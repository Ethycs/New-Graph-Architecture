# E28 — Predictive Control Graph Extractor (Phase 23 MVP + Phase 23e σ/control bridge)

**Cluster:** exp
**Status:** implemented (5-grammar smoke + σ + control + held-out-eval mode)
**Tags:** #phase-23 #pcg-x #regime-graph #sigma #control-policy #decision-trace

## What

E28 is the reference runner for the **Predictive Control Graph Extractor** (PCG-X) — the Phase 23 architectural reframing of "universal graph extraction" into "practical regime / control-graph extractor from an arbitrary network's activations." It implements the two operational legs of the mantra **partition by prediction, merge by behavior, control by intervention**: argmax-of-next-state-head as the partition, bisimulation quotient as the behavioral merge. The control-by-intervention leg is Phase 23c (open).

Phase 23e bolted σ + the 3-branch control policy onto the extracted regime graph. Every E28 run now also emits `decision_trace.jsonl` with one record per harvested step, recording σ on the regime-graph context (margin and decision-tie from the model's FSM softmax; illegal from the regime edge set; loop-risk from same-program revisits) and the control verdict (NORMAL / RECOVERY / ABSTAIN) from `ControlPolicy.decide()` with the regime legality adjacency.

## Why

E25–E27 showed that strict-Hamming universal graph extraction is encoder-bounded, not pipeline-bounded. The reframing is: stop trying to recover the gold FSM exactly; extract the *regime graph the substrate actually visits* and audit predictions against it. E28 ships the descriptive half (extraction + per-regime stats + edges with Beta confidence). The σ + control wiring closes the bridge from the symbolic stack (Phases 0–19's σ/mask/control) to the regime stack: the same audit-trail machinery now operates on either the typed FSM or the PCG-X regime graph without two implementations.

The held-out-eval mode is the operational diagnostic the σ + control bridge unlocks: on training data σ is degenerate (every observed transition is by construction a regime edge, so the illegal signal is identically zero); on held-out data σ has something to grip and discriminates train vs. eval on every grammar.

## Interface

**Runner entry point:**
```python
run_e28(
    *, fsm, run_id, output_dir, seed,
    grammar="python_big", n_programs=None,
    encoder_hidden_dim=32, projection_z_dim=32, projection_hidden_dim=64,
    n_encoder_train_epochs=50, n_projection_train_epochs=30,
    target_n_regimes=None, adversarial_token_weight=0.0,
    eval_n_programs=None, eval_seed=None,        # Phase 23e additions
) -> E28Result
```

**Pipeline (training set):**
1. Sample a synthetic dataset for `grammar` via `GRAMMAR_DISPATCH[grammar]["loader"](fsm, n, seed)`.
2. Train a state-conditioned MLP encoder on next-state prediction (Wave-C substrate).
3. Train a `PredictiveProjection` head (h → z) with three losses: next-state CE (partition signal), entropy regression (Morse-lite uncertainty), failure BCE (risk). Optional adversarial token head with gradient reversal.
4. Argmax-of-next-state-head partitions the corpus into cells; per-cell stats accumulated (support, failure_rate, entropy_mean, margin_to_tie).
5. `bisimulation_quotient` merges cells into regimes by behavioral equivalence (criterion=`"full"`).
6. Emit `control_graph.json` (regimes, edges with Beta(α, β) confidence, diagnostics).
7. **(Phase 23e)** Build the regime legality adjacency from the edge set; identify goal regimes (failure_rate < 0.05); iterate the steps and emit one `DecisionTraceRecord` per step with σ, control verdict, mask context. Trace label is `"A0_train"`.

**Held-out-eval mode (when `eval_n_programs` is supplied):**

After the training pipeline above, sample a fresh dataset with `eval_seed` (defaults to `seed + 1`), run the trained encoder + projection over it, map argmax FSM states through the training cluster map (FSM states absent from training cells map to a `regime_unknown` sentinel that auto-fires the illegal signal), and emit the trace on the held-out slice instead of the training one. Trace label is `"A0_eval"`. `control_graph.json` is unchanged either way (it always reflects training-set extraction).

**Output artefacts:**
- `control_graph.json` — regimes (id, support, failure_rate, entropy_mean, mean_margin_to_tie, dominant_current_state, purity), edges (src, dst, probability, Beta α/β, count), diagnostics.
- `decision_trace.jsonl` — schema-1.1 records, one per step. `mask.enabled=false` (PCG-X does not apply a hard mask to the prediction), names use the `regime_K` / `regime_unknown` convention.

## Acceptance bars

The e2e suite `tests/e2e/test_e28_pcg_extractor.py` enforces (on `python_expr`, V=14, n_programs=8, 4 epochs each):

1. `decision_trace.jsonl` exists and is non-empty.
2. Every row carries σ in [0, 1], the documented `sigma_signals` keys, a valid control verdict, and self-consistent thresholds. `control.sigma_observed == sigma_total` per row.
3. Trace rows use the `regime_K` (or `regime_unknown`) naming convention for `top1_state`, `mask.pre_mask_argmax`, `mask.post_mask_argmax`, and optional `top2_state` / `mask.current_state`.
4. At least one row has a non-None `mask.current_state` (intra-program continuity exercised).
5. `control_graph.json` still emitted with the documented fields.
6. Held-out-eval rows carry `ablation="A0_eval"`. Rows whose `top1_state == "regime_unknown"` have `sigma_signals["illegal"] == 1.0`.

## Phase 23e operational stats (5-grammar sweep, N_PROGRAMS=40, train_seed=42, eval_seed=43)

| grammar | slice | rows | NORM% | ABS% | mean σ | mean illegal_signal |
|---|---|---:|---:|---:|---:|---:|
| listops | train / eval | 49 / 94 | 100 / 100 | 0 / 0 | 0.052 / **0.101** | 0.000 / 0.000 |
| python_expr | train / eval | 866 / 648 | 100 / 100 | 0 / 0 | 0.030 / **0.034** | 0.000 / 0.023 |
| python_big | train / eval | 985 / 1149 | 100 / 100 | 0 / 0 | 0.027 / **0.048** | 0.000 / 0.080 |
| json | train / eval | 353 / 338 | 100 / 99.7 | 0 / 0.3 | 0.081 / **0.101** | 0.000 / 0.086 |
| python_control | train / eval | 766 / 981 | 100 / 99.6 | 0 / 0.4 | 0.033 / **0.065** | 0.000 / 0.102 |

Reproduce via `pixi run -e dev python scripts/phase23e_sigma_control_stats.py`. σ discriminates train vs. eval on every grammar; RECOVERY band stays empty at this scale (σ is bimodal; intermediate band would populate at lower thresholds or larger N).

## Links

- **See also:** [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md), [Control Policy](../arch/substrate/control-policy.md), [Predictive Projection](../arch/graph/predictive-projection.md), [Bisimulation Quotient](../arch/graph/bisimulation-quotient.md), [Decision Trace JSONL](../drivers/decision-trace-jsonl.md).
- **Driven by:** [E26 Trained-Encoder Extraction](./e26-extraction-trained-encoder.md) (the Wave-C substrate that E28 reuses), [E27 Partition Probe](./e27-extraction-partition-probe.md) (the probe-head pattern), and `docs/proposals/graph-extraction.md` (the reframed deliverable).
- **Drives:** Phase 23c interventions (still open), Phase 23 pretrained-substrate test (GPT-2 small, open), σ-threshold recalibration on the regime graph (open).
