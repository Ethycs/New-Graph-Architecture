# E30 — PCG-X on a Frozen Pretrained Substrate (Phase 24)

**Cluster:** exp
**Status:** implemented (5-grammar sweep, seed 42)
**Tags:** #phase-24 #pcg-x #pretrained-substrate #gpt2 #tier-3

## What

E30 is the proposal's deferred Tier-3 test: PCG-X over a **frozen pretrained** causal LM, with no grammar-specific training of the substrate. The default substrate is GPT-2 small (124M, 12 transformer blocks, 768-d hidden), loaded from `~/models/hf/hub/` (DVC-tracked, `TRANSFORMERS_OFFLINE=1` so the loader never reaches out to the hub). The model is not trained on the grammar; PCG-X asks whether useful grammar-shaped regimes emerge from activations of a substrate that has never seen the grammar.

The implementation reuses every PCG-X stage from E28 except the substrate. Phase 23e's σ + control + decision-trace bridge drops in unchanged — the regime graph emitted on a pretrained substrate is audited by the same machinery as the typed FSM.

## Why

E26–E29 showed that strict-Hamming extraction is encoder-bounded, and that substrate quality is the binding constraint on PCG-X regime purity. E30 closes the loop: if the substrate is the binding constraint, then a *good off-the-shelf* substrate that never saw this grammar should still produce useful regimes — and a model with web-scale pretraining priors should beat a tiny model trained from scratch on this corpus. Phase 24 confirms both directions at the seed-42 reading.

## Interface

**Runner entry point:**

```python
run_e30(
    *, fsm, run_id, output_dir, seed,
    grammar="python_expr",
    n_programs=None,
    model_id="gpt2",
    harvest_layer=6,
    device=None,                                # default cuda if available
    projection_z_dim=32, projection_hidden_dim=64,
    n_projection_train_epochs=30,
    target_n_regimes=None,
    adversarial_token_weight=0.0,
    eval_n_programs=None, eval_seed=None,       # held-out trace mode, from E28
) -> E30Result
```

**Pipeline:**

1. **Load frozen pretrained substrate.** `AutoTokenizer.from_pretrained(model_id, use_fast=True)` and `AutoModelForCausalLM.from_pretrained(model_id)`. Every parameter is set `requires_grad_(False)` and the model is moved to GPU when available.
2. **Sample dataset** via `GRAMMAR_DISPATCH[grammar]["loader"](fsm, n, seed)`. Group samples by `program_id`.
3. **Per-program harvest.** Build per-program text by joining `observed_token`s with single-space separators; tokenize once with `return_offsets_mapping=True`; for each grammar step find the *last* BPE position whose start-char is strictly inside the step's char range; one forward pass with `output_hidden_states=True`; take the mid-layer hidden state at each per-step BPE position. Truncation is defensive: if a tokenized program exceeds the model's `n_positions` (1024 for GPT-2 small), the steps past the cap reuse the last in-window vector and the run records `n_programs_truncated_to_context`.
4. **PCG-X tail (identical to E28).** `PredictiveProjection` trained on next-state CE + entropy regression + failure BCE; argmax-of-next-state partition; bisimulation merge to `target_K = V`; `control_graph.json` emission.
5. **σ + control trace** via `_emit_regime_decision_trace` from E28 (substrate-agnostic helper). Held-out-eval mode re-uses the same frozen substrate against a fresh dataset sampled with `eval_seed`.

**Output artefacts:** `control_graph.json` (regimes + edges + diagnostics; `substrate` field records the model id and harvest layer) and `decision_trace.jsonl` (one record per step in the configured trace slice).

## Phase 24 results (seed 42, n_programs=80, gpt2, layer 6/12, 22 s total on a 6 GB GPU)

| grammar | V | merged regimes | mean purity | proj next-state acc |
|---|---:|---:|---:|---:|
| listops | 11 | 10 | **0.918** | 0.993 |
| python_expr | 14 | 11 | 0.710 | 0.895 |
| python_big | 24 | 23 | 0.758 | 0.930 |
| json | 26 | 25 | 0.673 | 0.896 |
| python_control | 37 | 36 | 0.761 | 0.924 |
| **mean** | — | — | **0.764** | **0.928** |

Cross-substrate comparison at the same n_programs:

| substrate | grammar-specific training? | mean purity |
|---|:---:|---:|
| Wave-C MLP, state-conditioned (E28) | yes | 0.790 |
| **Frozen GPT-2 (E30)** | **no** | **0.764** |
| Small transformer from scratch (E29) | yes | 0.663 |

The frozen pretrained substrate matches the trained-on-grammar MLP within multi-seed noise and decisively beats the from-scratch transformer trained on the grammar (+10.1 pp). The cleanest evidence so far that the binding constraint on PCG-X regime quality is substrate quality, not pipeline-fit.

Reproduce via `pixi run -e dev python scripts/phase24_gpt2_substrate_sweep.py`.

## Phase 25 — layer-ablation (where the state structure lives in GPT-2)

`scripts/phase25_layer_ablation_sweep.py` runs E30 at five harvest layers on all 5 grammars. Per-layer mean purity (5 grammars, seed 42, 101 s on GPU):

| layer | mean purity | mean projection next-state acc |
|---|---:|---:|
| L0 (embedding output) | 0.556 | 0.570 |
| L2 (early) | 0.659 | 0.774 |
| L6 (mid — current default) | 0.750 | 0.916 |
| L10 (late) | **0.785** | **0.960** |
| L12 (final block) | 0.673 | 0.825 |

The peak-then-drop shape holds on every grammar individually. Three findings:

1. **Mid-to-late layers carry the structure; the final block does not.** L12's −11.2 pp drop from L10 is the cleanest evidence that the final transformer block is the wrong place to harvest for state-extraction — it specialises for the LM head's next-token job, not for grammar-state representation.
2. **L0 (embedding alone) gives 0.556 mean purity**, well above the 1/V chance baseline. Pure tokenisation already half-resolves the FSM partition for Python-flavoured grammars; the L0→L10 lift of +22.9 pp is what deep contextualisation buys.
3. **L6 vs L10 is within seed variance.** A cross-process re-run of `phase24_gpt2_substrate_sweep.py` after flipping the default to L10 gave mean purity 0.750, not 0.785 — same seed, different CUDA-nondeterministic RNG state. The qualitative shape is robust; the L6/L10 distinction is not. `_DEFAULT_HARVEST_LAYER` therefore stays at 6, with a code comment pointing at Phase 25's finding. Multi-seed bootstrap (Phase 26) is the right next step before changing the default.

Per-layer regime-graph visualisations for python_control (V=37) at `runs/phase25_layer_evolution/python_control_layer_evolution.png` show the structural evolution: yellow/orange (low purity) at L0–L2, mostly green at L6–L10, drifting back to yellow at L12.

## Environment notes

- `transformers >=4.40,<5.0` is a required dependency for E30.
- `pyproject.toml` declares `pytorch-gpu` plus `system-requirements.cuda = "12"` so conda-forge resolves a CUDA-enabled torch build.
- E30 sets `HF_HOME = ~/models/hf` and `TRANSFORMERS_OFFLINE=1` defensively at import time. The DVC-tracked area is the trust boundary; nothing reaches the HuggingFace hub at runtime.
- Tested with `model_id="gpt2"`. TinyLlama-1.1B-Chat-v1.0 and Qwen2.5-1.5B-Instruct are also DVC-tracked under `~/models/hf/hub/` and should drop in by changing `model_id`; size-vs-purity ablation is open.

## Acceptance bars

None gated by the e2e suite yet — Phase 24 ships as an operational result, not a regression bar. An E30 e2e test mirroring `test_e28_pcg_extractor.py` is a defensive follow-up.

## Links

- **See also:** [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md), [Control Policy](../arch/substrate/control-policy.md), [Predictive Projection](../arch/graph/predictive-projection.md), [Bisimulation Quotient](../arch/graph/bisimulation-quotient.md), [Decision Trace JSONL](../drivers/decision-trace-jsonl.md).
- **Driven by:** [E28 PCG Extractor](./e28-pcg-extractor.md) (the PCG-X reference runner; E30 reuses its helpers), [Graph extraction proposal](../proposals/graph-extraction.md) (Tier 3 — pretrained substrate, finally executed).
- **Drives:** multi-seed bootstrap on E30; layer ablation (harvest_layer sweep); larger DVC-tracked substrates (TinyLlama-1.1B, Qwen2.5-1.5B); real-corpus follow-up where GPT-2's 1024 token context will require chunking.
