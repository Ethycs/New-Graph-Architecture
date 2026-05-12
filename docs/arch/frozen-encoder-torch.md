# Frozen Encoder Torch

**Cluster:** arch
**Status:** implemented
**Tags:** #torch #frozen #encoder #reservoir #substrate-independence

## What

The torch-backed sibling of [Frozen Encoder Backbone](./frozen-encoder-backbone.md). A small `torch.nn.Sequential` MLP that is randomly initialised under a deterministic `seed` and whose **all parameters have `requires_grad=False`**. The public contract mirrors the sklearn sibling: `transform(x)` is a one-shot projection, `is_frozen` is `True`, and any attempt to call a fit-style update raises. The encoder is the "reservoir" floor of the TPN's substrate independence claim: switching from sklearn to torch should not change the architecture's structural metrics, only its representational ceiling.

## Why

The TPN commitment "frozen encoder; all gradient flows through symbolic structure" is the load-bearing identifiability story (gradients route through prototypes / posterior / heads, not through representation learning). The torch sibling is necessary for Phase 8's substrate-independence ablation — without it, the Phase 8 result that torch capacity adds +15pp absolute accuracy at the same Phase A recovery cannot be measured. It also keeps the torch trainer self-contained: a torch runner doesn't have to detour through sklearn for its encoder.

## Interface

- **Constructor:** `FrozenEncoderTorch(input_dim, output_dim, hidden_dim, seed)` — deterministic random init under `seed`; all `nn.Parameter`s set to `requires_grad=False`.
- **`transform(x: torch.Tensor) -> torch.Tensor`:** one-shot projection from $\mathbb{R}^{\text{input_dim}}$ to $\mathbb{R}^{\text{output_dim}}$.
- **`is_frozen -> bool`:** `True` by construction.
- **Optional `embed_to_poincare(x)`:** apply `tanh` to the final layer's output and rescale into the open Poincaré ball if downstream training expects ball coordinates.

## Build steps

- Build `nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim))` under a `torch.Generator(device='cpu').manual_seed(seed)` context.
- Iterate over `self.parameters()` and set `requires_grad=False`.
- `transform(x)`: `with torch.no_grad(): return self.net(x)`.
- For Poincaré coordinates: append a final `tanh` and a norm rescaling that keeps $\|x\| \leq 1 - \epsilon$.

## Links

- **See also:** [Frozen Encoder Backbone](./frozen-encoder-backbone.md), [Torch Energy Trainer](./torch-energy-trainer.md), [Typed Readout Torch](./typed-readout-torch.md).
- **Drives:** every torch runner from E12 onward; the substrate-independence ablation.
- **Driven by:** the encoder seed (deterministic init) and the runner's input pipeline.
- **Math:** a fixed random MLP is a sample from a function-space prior; for `tanh` activations the conditional Gaussian process limit at infinite width is well-defined.
- **Open:** when (Phase 20+) we explore replacing the random-projection floor with a learned encoder, this atom is the natural place to plug in a JEPA-style self-supervised encoder.
