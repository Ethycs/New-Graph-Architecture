# E7 — Reservoir Readout vs End-to-End

**Cluster:** exp
**Status:** spec
**Tags:** #reservoir-computing #parameter-efficiency #sample-efficiency

## What

Compare two training regimes on E2 (BabyAI) or E1 (synthetic): end-to-end training of the full classifier/graph/distributor pipeline vs freezing the encoder and training only a linear readout atop fixed hidden representations. Measure trainable parameters, samples to convergence, accuracy, calibration (ECE), and transfer to a held-out level. Hypothesis: reservoir readout (frozen encoder + learned readout) achieves 90%+ of full-training accuracy with <10% the parameters and faster convergence.

## Why

If the typed-graph machinery works as a structured "reservoir," then a frozen pre-trained encoder (from language models or RL policies) should provide sufficient features, and a lightweight readout should suffice. This tests the reservoir-computing interpretation: is the architecture fundamentally about better structured decoding rather than learned features? If yes, this dramatically reduces training cost and enables quick transfer. Load-bearing claim: typed-graph readout is an efficient decoder over fixed representations.

## Interface

**Reads:**
- task traces from E1 or E2
- pre-trained encoder (frozen)
- test set with ground-truth labels

**Writes:**
- [`metrics.jsonl`](../drivers/metrics-jsonl.md): trainable_params, samples_to_90pct, final_accuracy, ece, transfer_accuracy (for both regimes)
- [`results.jsonl`](../drivers/results-jsonl.md): learning curves, per-sample predictions, calibration data
- comparison table: parameter ratio, sample efficiency, transfer gap

## Build steps

1. Initialize or load a pre-trained encoder (e.g., vision CNN, BERT).
2. Extract hidden representations h_t for all training traces.
3. Train readout (end-to-end): full backprop through encoder + readout; measure params and convergence.
4. Train readout (reservoir): freeze encoder, train only readout layer; measure params and convergence.
5. Evaluate both on held-out test level; measure accuracy, ECE, speed.
6. On a separate transfer-target level, freeze learned readout from source; measure transfer accuracy.
7. Emit metrics and learning curves.

## Links

- **See also:** [E2 — Real BabyAI / MiniGrid](./e2-real-babyai.md), [E1 — Synthetic BabyAI Grid Run](./e1-synthetic-babyai.md), [Typed Field Pipeline](../arch/typed-field-pipeline.md)
- **Drives:** parameter-efficiency claim for typed readout
- **Driven by:** (encoder-loader — external), [Typed Readout Layer](../arch/typed-readout-layer.md)
- **Math:** ECE = mean|confidence - accuracy| over bins; param_ratio = (reservoir_params) / (full_params)
- **Open:** [What is the encoder freeze schedule in reservoir readout?](../open/q09-reservoir-freeze-schedule.md), (readout-architecture — open question)
