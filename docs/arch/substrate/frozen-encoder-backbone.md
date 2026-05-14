# Frozen Encoder Backbone

**Cluster:** arch
**Status:** deprecated (Phase 0–18 default; superseded by paradigm-neutral substrate, commit 8d67c17)
**Tags:** #reservoir #training #efficiency #deprecated

> **Deprecated.** Commit 8d67c17 dropped the frozen-encoder commitment from the TPN architectural contract. The encoder is no longer required to be frozen — training, fine-tuning, and end-to-end backprop substrates are all in-scope. This note is kept for historical reference of the Phase 0–18 default; for the current substrate framing see [model-class.md](../../model-class.md). The Phase 24 result that established frozen pretrained substrates as one *valid* choice (not a contractual one) is in [results.md](../../results.md) and [insights.md](../../insights.md).

## What

The encoder backbone $h_\theta(x)$ (a pre-trained LLM or universal transformer) is held constant during retrieval training. Only the lightweight readout layer is updated. This follows the reservoir-computing paradigm: the large, expressive encoder acts as a fixed dynamical system that maps inputs to a rich hidden representation, while learning is restricted to the small output stage. The frozen backbone reduces training overhead, stabilizes gradient flow, and allows fine-tuning on downstream tasks without destabilizing the learned representation.

## Why

Training a full LLM is expensive and risks catastrophic forgetting of pre-trained knowledge. By freezing the encoder, we leverage its existing semantic understanding while learning only the graph-retrieval-specific readout. This enables sample-efficient training (gradient flow is simpler, no large-scale backprop through the encoder), reduced memory footprint, and rapid experimentation. It also ensures that the encoder does not overfit to the retrieval task and remains generalizable. Reservoir computing is a proven paradigm for efficient few-shot learning.

## Interface

**Inputs:**
- Input state: $x$ (text, structured data, or feature vector).
- Frozen encoder: $h_\theta$ with parameters $\theta$ (immutable during training).

**Outputs:**
- Latent representation: $\mathbf{h} = h_\theta(x) \in \mathbb{R}^d$ (fixed).
- State embedding: passed to [Typed Readout Layer](../typed/typed-readout-layer.md) for further processing.

## Build steps

1. **Load pre-trained encoder:** Instantiate a frozen LLM or encoder backbone (e.g., a fine-tuned BERT, T5, or GPT variant). Verify that all parameters have `requires_grad=False` (in PyTorch) or equivalent.
2. **Verify freezing:** Run a quick check to ensure no gradients flow through the encoder. Attempt a forward-backward pass on a small batch and confirm encoder gradients are None.
3. **Prepare inputs:** Tokenize or embed input states and pass them through the encoder. Cache the latent representations if the same states are encountered multiple times (e.g., during curriculum learning).
4. **Connect to readout:** Feed the frozen embeddings $\mathbf{h}$ directly to the [Typed Readout Layer](../typed/typed-readout-layer.md). Do not insert additional trainable layers between encoder and readout.
5. **Monitor representation quality:** Track the entropy and singular-value spectrum of $\mathbf{h}$ during training to ensure the encoder is not collapsing or becoming inactive.
6. **Document encoder version:** Record the exact pre-trained checkpoint, architecture, and hyperparameters for reproducibility.

## Links

- **See also:** [Typed Readout Layer](../typed/typed-readout-layer.md)
- **Drives:** [Typed Readout Layer](../typed/typed-readout-layer.md), (readout-ablation — planned experiment)
- **Driven by:** (pretrained-encoder-spec — external input)
- **Math:** [Mathematics.md §Reservoir Computing](../../Mathematics.md#reservoir)
- **Open:** (encoder-selection-criteria — open question)
