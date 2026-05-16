# Predictive Projection

**Cluster:** arch
**Status:** implemented (Phase 23)
**Tags:** #phase-23 #pcg-x #regime-graph #projection #probe-heads

## What

A small MLP projection `h → z` from arbitrary base-network hidden states to a **regime-space representation** $z$, paired with three (optionally four) small probe heads sitting on top of $z$:

- **Next-state head** — predicts where the trajectory goes next. The argmax over its logits is the PCG-X partition signal; each argmax cell is a candidate regime.
- **Entropy-regression head** — predicts the entropy of the next-state distribution at this step. Morse-lite uncertainty signal.
- **Failure head** — predicts whether the next step is wrong / illegal / out-of-distribution. Risk signal feeding `ControlPolicy`.
- **Adversarial token head** *(optional, off by default)* — gradient-reversed token-prediction head used to strip content variance from $z$ on frozen base networks that lack state-conditioning.

The atom is intentionally cheap (single-hidden-layer MLP plus linear heads) so it can be run on every layer of any pretrained network being analysed.

## Why

Phase 21 found that an overcluster-then-bisimulate pipeline on the trained-from-scratch encoder is inert: the substrate's natural equivalence is finer than the FSM, and the quotient cannot recover the FSM by post-hoc merging similar transition distributions. Phase 23's reframe — **partition by prediction, merge by behaviour, control by intervention** — moves the partition step upstream: instead of merging similar clusters after the fact, partition by `argmax(next_state_logits)`. The Myhill–Nerode coarsening is then done by the predictive head itself; the resulting cells correspond to FSM states approximately (purity 0.76–0.85) without an explicit quotient step.

This atom owns the first leg of that pipeline: turn arbitrary `h` into a regime-space `z` whose argmax cells are the regimes the network actually visits.

## Interface

- **`PredictiveProjectionConfig`** — frozen dataclass: `z_dim`, `hidden_dim`, `n_states`, `entropy_weight`, `failure_weight`, `adversarial_token_weight`, `n_tokens`.
- **`PredictiveProjection`** — `nn.Module` exposing `.project(h) -> z`, `.next_state_logits(z)`, `.entropy_regression(z)`, `.failure_logits(z)`, and (when enabled) `.adversarial_token_logits(z)` via a gradient-reversal layer.
- **`train_predictive_projection(...)`** — trainer combining next-state CE + entropy MSE + failure BCE (+ optional adversarial token CE through the gradient-reversal layer) on harvested `(h, next_state, entropy_target, failure_target)` tuples.

## Build steps

- Project: `z = MLP(h)` with a single hidden layer and ReLU.
- Forward each probe head from $z$ via its own linear layer.
- Combine losses: $\mathcal{L} = \text{CE}(\hat y_{\text{next}}, y_{\text{next}}) + w_{\text{ent}} \cdot \text{MSE}(\hat H, H) + w_{\text{fail}} \cdot \text{BCE}(\hat f, f) [+ w_{\text{adv}} \cdot \text{CE}(\hat t, t)\text{ through GRL}]$.
- After training, the partition over harvested steps is `argmax(next_state_logits)` per step; the gap between top-1 and top-2 logits is the **margin-to-tie-wall** (per-step regime-confidence certificate).

## Links

- **See also:** [Bisimulation Quotient](bisimulation-quotient.md) (post-hoc consolidator; mostly inert at argmax cells), [Hidden State Harvester](../substrate/hidden-state-harvester.md), [Labelled Hypergraph](labelled-hypergraph.md) (Phase 26 lift of the regime graph produced here), [Singularity Detector σ(x)](../singularity/singularity-detector.md), [E28 PCG Extractor](../../exp/e28-pcg-extractor.md), [E30 PCG Extractor (Pretrained)](../../exp/e30-pcg-extractor-pretrained.md).
- **Drives:** Phase 23 / E28 (Wave-C substrate), Phase 23d / E29 (small transformer substrate), Phase 24 / E30 (frozen pretrained GPT-2 substrate), Phase 25 (layer-ablation sweep).
- **Driven by:** [Hidden State Harvester](../substrate/hidden-state-harvester.md) (input `h`), the gold FSM (for next-state supervision in current runners; self-supervised replacement is proposed under Phase 22d).
- **Math:** Single-hidden-layer MLP projection; per-head softmax / sigmoid as appropriate; gradient reversal $\partial \mathcal{L}_{\text{adv}} / \partial z = -\lambda \cdot \partial / \partial z$ on the adversarial-token branch only.
