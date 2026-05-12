# Hidden State Harvester

**Cluster:** arch
**Status:** implemented
**Tags:** #graph-extraction #phase-20 #harvester #substrate-agnostic

## What

Substrate-agnostic adapter that yields a contiguous stream of hidden states from any trained system with an accessible activation surface. Given a corpus and a callable / torch module / pre-computed iterable, the harvester returns a `HarvestResult` carrying parallel `hidden_states` (shape `(N_total_steps, hidden_dim)`) and `sample_index` (shape `(N_total_steps,)`) arrays — the input format the graph-extraction pipeline (cluster → count → Phase A) consumes.

## Why

The first step of `docs/proposals/graph-extraction.md` is "collect hidden states $\{h_t\}$ at every step / token / layer of interest." The proposal's load-bearing claim — that typed structure is latent in the activations of any sufficiently trained network — only becomes testable once we have a *uniform* way to harvest activations from substrates as different as our TorchEnergyTrainer, a small transformer, and a pretrained GPT-2-small. The harvester is that uniform interface. Without it, every substrate would need its own bespoke probing wrapper and the proposal's universality claim could not be operationalised.

## Interface

- **`HiddenStateHarvester.from_callable(fn, per_sample_emits_sequence=False)`** — wrap any `x -> h` callable. When `per_sample_emits_sequence=True`, the callable returns a `(T, d)` stack for sequence models.
- **`HiddenStateHarvester.from_torch_module(module, layer_name=None, per_sample_emits_sequence=False)`** — register a forward hook on `module.<layer_name>` (or the module's top-level output if `layer_name is None`), then capture per-forward-pass outputs. Sets `module.eval()` + `torch.no_grad()`; never trains.
- **`HiddenStateHarvester.from_sequences(iterable)`** — wrap a pre-computed iterable of `(T_i, d)` arrays for cases where activations were extracted by other means.
- **`harvest(corpus) -> HarvestResult`** — iterate the corpus, call the encode function, stack the results.
- **`close()`** / context-manager protocol — releases any registered torch hook.

`HarvestResult` exposes `hidden_states`, `sample_index`, `n_steps`, `n_samples`, `hidden_dim`.

## Build steps

- Define `HarvestResult` as a slotted container that validates shape consistency between `hidden_states` and `sample_index`.
- Provide three classmethod constructors mapping each substrate flavour to a common `encode_fn` callable.
- `harvest`: iterate the corpus; for per-sample-vector mode, append `(1, d)`; for per-sample-sequence mode, append `(T_i, d)`. Concatenate at the end.
- Torch adapter: register `register_forward_hook` on the named submodule; the hook stashes the output in a captured dict; `_encode` retrieves it after each forward pass; `close` unregisters via the handle.
- Defensive shape checks raise `ValueError` early so the caller doesn't get a silently corrupt harvest.

## Links

- **See also:** [Typed Latent Clustering](./typed-latent-clustering.md), [Bayesian Nonparametric K](./bayesian-nonparametric-k.md), [Forward Backward](./forward-backward.md).
- **Drives:** E24 (the Phase 20 graph-extraction runner, in development).
- **Driven by:** the substrate the user wants to probe (any callable, any torch module, or any pre-extracted activation stream).
- **Math:** the harvest is a function $X \to (\mathbb{R}^d)^{N_{\text{total}}}$ that factors through the network's chosen layer; activations are the natural latent coordinates on the model's encoding manifold.
- **Open:** which layer to harvest from (depth-sensitivity is an experimental risk in the proposal); whether to harvest from multiple depths simultaneously to build the recursive-TPN structure.
