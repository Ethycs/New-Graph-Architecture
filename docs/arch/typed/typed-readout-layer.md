# Typed Readout Layer

**Cluster:** arch
**Status:** spec
**Tags:** #readout #typed-output #training

## What

The readout layer $h_t: \mathbb{R}^d \to Y_t$ is a small, trainable neural network that maps frozen encoder embeddings to a typed output space. For each state type $t$ (e.g., question, retrieval, decision, synthesis), there is a dedicated readout head $h_t$ that predicts type-specific quantities: next-state embeddings, energy coefficients, or action logits. The readout layer is the sole trainable component during reservoir training, keeping model capacity small and learning efficient.

## Why

A single monolithic readout risks learning spurious correlations and entangling different task types. Typed readouts modularize the output: each head is specialized for one semantic category, improving interpretability and allowing type-specific regularization. The small size means gradient flow is clean, few hyperparameters to tune, and fast iteration. This design aligns with the typed graph architecture, where every node has an explicit type; the readout respects this structure.

## Interface

**Inputs:**
- Frozen embedding: $\mathbf{h} \in \mathbb{R}^d$ from [Frozen Encoder Backbone](../substrate/frozen-encoder-backbone.md).
- State type: $t \in T = \{\text{question}, \text{retrieval}, \text{decision}, \ldots\}$.
- Optional embedding: $\mathbf{z}_H \in \mathbb{R}^{d'}$ (learned orbit embedding, if applicable).

**Outputs:**
- Typed output $\mathbf{y}_t \in Y_t$:
  - For continuation: logits over next states, shape $(|V|,)$ or $(|V/H|,)$ if orbit-compressed.
  - For energy: scalar energy $E(x)$ (or per-feature energies).
  - For latent: embedding $\mathbf{y}_{t, \text{emb}} \in \mathbb{R}^{d}$ (for pooling or aggregation).

## Build steps

1. **Define output types:** Enumerate all state types $T$ that appear in the typed graph. For each type, specify the output shape and semantics (e.g., logits, energy, embedding).
2. **Construct readout heads:** For each type $t \in T$, implement a small MLP: $h_t(\mathbf{h}) = \text{MLP}_t(\mathbf{h})$ with 1-2 hidden layers. Typical architecture: $\mathbb{R}^d \to \mathbb{R}^{128} \to \mathbb{R}^{|Y_t|}$.
3. **Add orbit embedding (optional):** If using orbit-quotient compression ([Orbit Quotient Space](../group/orbit-quotient-space.md)), include a learned orbit embedding table $\mathbf{Z}_H \in \mathbb{R}^{|V/H| \times d'}$ and apply attention or gating between frozen $\mathbf{h}$ and orbit embeddings.
4. **Ensure parameter isolation:** All weights in readout heads are independent; no weight sharing across types unless semantically justified.
5. **Initialize readout weights:** Use standard initialization (e.g., Xavier uniform). For energy heads, initialize biases to zero (energy is zero-centered).
6. **Hook to loss:** Connect typed readout outputs to the loss function (cross-entropy for logits, energy-weighted loss for energy, etc.).

## Links

- **See also:** [Frozen Encoder Backbone](../substrate/frozen-encoder-backbone.md), [Orbit Quotient Space](../group/orbit-quotient-space.md)
- **Drives:** [Energy Function E(x)](../energy/energy-function-E.md), (readout-architecture-search — planned experiment)
- **Driven by:** [Frozen Encoder Backbone](../substrate/frozen-encoder-backbone.md)
- **Math:** [Mathematics.md §Neural Readouts](../../Mathematics.md#readouts)
- **Open:** (type-specific-regularization — open question)
