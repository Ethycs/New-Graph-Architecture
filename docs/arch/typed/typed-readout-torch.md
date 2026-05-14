# Typed Readout Torch

**Cluster:** arch
**Status:** implemented
**Tags:** #torch #typed #readout #per-type-mlp #substrate-independence

## What

The torch-backed sibling of [Typed Readout Layer](typed-readout-layer.md). One small `nn.Sequential(nn.Linear, nn.ReLU, nn.Linear)` MLP per `type_id`, with logits output. The public contract mirrors the sklearn sibling — `fit`, `predict`, `predict_proba`, `decision_function`, `n_trainable_params`, `n_trainable_params_per_type`, `is_fitted`. Heads are fully independent; fitting head A never touches head B; predict-style methods dispatch on `type_id` and raise `KeyError` for unknown IDs or `RuntimeError` for unfitted heads.

## Why

The typed-readout commitment is that specialisation is structural — each FSM state-type owns its own classifier head, and the per-type isolation must hold regardless of substrate. The torch sibling supplies the substrate that lets Phase B gradient training go through the readout heads (sklearn's LogisticRegression is closed-form fit and has no gradient interface). Without this atom, Phase B's single-loss energy trainer cannot reach the heads, and the "all gradient flows through structure" commitment is broken at the readout layer.

## Interface

- **Constructor:** `TypedReadoutTorch(input_dim, hidden_dim, n_classes, type_ids, seed)` — pre-allocates one `nn.Sequential(Linear, ReLU, Linear)` per `type_id`.
- **`fit(x, y, type_id)`:** fits the head for `type_id` only; raises `KeyError` if `type_id` is unknown.
- **`predict(x, type_id) -> argmax`**, **`predict_proba(x, type_id) -> softmax`**, **`decision_function(x, type_id) -> raw logits`**.
- **`n_trainable_params() -> int`**, **`n_trainable_params_per_type() -> dict[TypeId, int]`**.
- **`is_fitted(type_id) -> bool`** — `False` until the first `fit` call for that type.

## Build steps

- Lazy-import torch at module load time; raise `ImportError` with install instructions if unavailable.
- Pre-allocate one `nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes))` per `type_id`.
- `fit`: optimise the head's parameters via `torch.optim.Adam` on cross-entropy (or the energy-trainer's gradient signal when called inside Phase B).
- Predict methods: dispatch on `type_id`, return the head's output without softmax (logits) or with softmax (`predict_proba`).
- Maintain a `_fitted: dict[TypeId, bool]` flag set on first successful `fit`.

## Links

- **See also:** [Typed Readout Layer](typed-readout-layer.md), [Torch Energy Trainer](../energy/torch-energy-trainer.md), [Frozen Encoder Torch](../substrate/frozen-encoder-torch.md).
- **Drives:** every torch runner that exercises Phase B; the multi-seed sweeps in Phases 14–18.
- **Driven by:** the typed FSM (which fixes the `type_ids` at construction) and the energy trainer (which supplies the gradient signal during Phase B).
- **Math:** independent per-type MLPs; the typed isolation is structural, not learned (no shared weights across types unless explicitly tied).
- **Open:** weight tying across types — under what structural conditions (e.g., a type-orbit equivalence) is shared weights warranted? Currently every head is independent by construction.
