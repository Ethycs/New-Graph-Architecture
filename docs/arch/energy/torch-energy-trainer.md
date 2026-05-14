# Torch Energy Trainer

**Cluster:** arch
**Status:** implemented
**Tags:** #torch #energy-based #poincare #gradient #phase-9

## What

The gradient-trained sibling of [Energy Minimisation Trainer](energy-minimization-trainer.md). Where the classical trainer interleaves Riemannian SGD on numpy prototype arrays with closed-form Beta-posterior updates, this atom lifts every learnable quantity into a single `nn.Module` whose `forward` IS the energy at the observed (input → true_next_state) transition. Backprop through that scalar walks the gradient back through (i) prototypes (an `nn.Parameter` of shape $(V, d)$ living in the open Poincaré ball), (ii) the posterior legality bias in the form of two `nn.Parameter` tensors `alpha_log` and `beta_log` of shape $(V, V)$ (log-space for stability), and (iii) the typed-readout heads' weights via [Typed Readout Torch](../typed/typed-readout-torch.md). The encoder is preserved as frozen-by-construction via [Frozen Encoder Torch](../substrate/frozen-encoder-torch.md).

## Why

Phase 8's substrate ablation (E11 sklearn vs E12 torch) showed that torch capacity adds +15pp absolute accuracy at the same Phase A recovery — confirming substrate independence with capacity reservation. The torch trainer is the substrate that buys representational capacity in Phase B (gradient training over prototypes and readout heads) without giving up Phase A (the classical posterior update). The hybrid story — Phase A's deterministic FSM recovery seeded into Phase B's gradient training — is structurally cleaner than either alone, and the torch trainer is the implementation that makes Phase B possible.

## Interface

- **Constructor:** `TorchEnergyTrainer(TorchTrainerConfig)` — config carries energy coefficients $\alpha, \beta, \gamma, \eta, \kappa$, learning rate, frozen encoder, readout heads, and warm-start `(posterior_alpha_init, posterior_beta_init)` from a numpy PosteriorMask.
- **`forward(x, true_next_state) -> energy`:** returns a scalar energy; differentiable in all three learnable parameter blocks.
- **`step(x, true_next_state)`:** computes the energy, calls `loss.backward()`, applies the optimiser step, and re-projects prototypes into the Poincaré ball with boundary clipping.
- **`snapshot_posterior_mask() -> PosteriorMask`:** emits a numpy `PosteriorMask` for downstream analysis tools. One-way: the snapshot does not feed back into the gradient graph.

## Build steps

- Define `class TorchEnergyTrainer(nn.Module)` with the three parameter blocks.
- Compute the Poincaré distance via the arccosh form $d(x, y) = \mathrm{arccosh}(1 + 2 \|x - y\|^2 / ((1 - \|x\|^2)(1 - \|y\|^2)))$ with three numerical guards: clip squared-norms to $\leq 1 - \epsilon$, clip the arccosh argument to $\geq 1 + \epsilon$, and soft-project coordinates via norm-rescaling with gradients.
- Compute the energy at the true transition (and at every legal transition for the softmax / KL term).
- Training loss = mean energy at the true next-state prototype plus an optional KL term.
- Use a standard `torch.optim.Adam` (or SGD); apply gradient clipping to keep updates inside the safe region near the Poincaré boundary.

## Links

- **See also:** [Energy Minimisation Trainer](energy-minimization-trainer.md), [Frozen Encoder Torch](../substrate/frozen-encoder-torch.md), [Typed Readout Torch](../typed/typed-readout-torch.md), [Posterior Mask](posterior-mask.md).
- **Drives:** E12 (substrate-independence runner) and E14 (torch-native end-to-end training); every multi-seed sweep on Python expr / big / JSON / control flow Python (Phases 14–18) uses this trainer.
- **Driven by:** the warm-start PosteriorMask snapshot from a Phase A run (when present).
- **Math:** Riemannian SGD on the Poincaré ball; closed-form sigmoid posterior mean from the log-odds parametrisation; the energy is a differentiable mirror of [Energy Function](energy-function-E.md).
- **Open:** [[open.qNN-torch-phase-a-coupling]] — should the posterior `alpha_log / beta_log` parameters receive natural-gradient rescaling, or is plain Adam enough?

## Known plumbing gap: readout heads receive zero gradient

The constructor registers `_readout_heads` as `nn.ModuleDict` and feeds them into Adam via `self.parameters()`, but `forward` computes `logits = -d_poincare(observation, prototypes) + legality_bias[current_state]` and never invokes the heads. The readout-head parameters therefore receive zero gradient on every step — Adam updates them with zeros, which is a no-op. The bug surfaced in Phase 20 Wave B and is pinned by `tests/unit/test_torch_energy_trainer.py::test_readout_heads_receive_gradient` (`pytest.xfail(strict=True)`).

Fixing it is an architectural decision rather than a typo: the readout heads should either (a) **additively compose** with the distance-based logits (`logits = -d + bias[cur] + head[type(cur)](../observation)`), or (b) **replace** them entirely, with the distance term recast as a regulariser. Most existing callers construct `TypedReadoutTorch(type_ids=[0])` (a single global head), so the minimum viable fix under (a) is one line. The XFAIL flips XPASS the moment that line lands and the marker should be removed.
