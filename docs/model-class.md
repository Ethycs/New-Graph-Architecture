# Typed Protocol Network (TPN)

A single-name handle for the model class this architecture defines. Use **Typed Protocol Network** in writing; **TPN** as the abbreviation.

## One-line definition

A TPN is a finite-state typed protocol (the FSM) carrying a network of per-type submodels, with Bayesian posterior over its own transition structure, hard structural masking, soft σ-routing for abstention, and information-geometric credit assignment.

## What's load-bearing

Six commitments together define the class. None alone is novel; the *composition* is.

1. **Typed graph as the system's full state space.** Every observation, prediction, signal is a vertex in some typed axis; the system's true state is a node-tuple in their Cartesian product. No raw floats cross the system's interfaces.
2. **Per-type submodels.** Each type (or group of FSM states sharing a type) gets its own classifier head. Submodels are independent; specialisation is structural.
3. **Hard structural mask + soft σ-routing.** The legality matrix zeros illegal predictions absolutely; σ-routing's three branches (NORMAL / RECOVERY / ABSTAIN) handle ambiguity that the mask doesn't address.
4. **Bayesian posterior over the protocol itself.** The mask isn't a hand-coded constant; it's a Beta(α, β) posterior per edge that grows from observation. The classical Baum-Welch + Bayesian M-step performs cold-start induction.
5. **Information-geometric foundation.** Fisher information sets natural-gradient learning rates; KL divergence is the canonical loss; Cramér-Rao bounds set sample-sufficiency stopping rules. The architecture's hyperbolic + Riemannian + Bayesian apparatus is information geometry by construction.
6. **Labelled hypergraph with residual.** The regime graph is not a discrete graph of opaque labels — it is a hypergraph where each regime carries its canonical statistical identity (a KL-distance signature), its human-named coordinates (FSM state, dominant SAE features), and its residual feature support (mathematically canonical but semantically unnamed) as separate fields. Each transition is a hyperedge carrying the feature-delta across the boundary. The data structure owns the semantic gap rather than hiding it; the interpretability claim is calibrated per regime by `len(named) / (len(named) + len(residual))`. The current discrete regime graph $(V, E)$ is the strict projection of this structure (forget enrichment ⇒ recover $(V, E)$ unchanged).

## What a TPN is good at

Tasks where you need *all* of: hard structural constraint, hierarchical / cyclic / symmetric structure, calibrated abstention, audit trail by construction, and continual learning of the protocol from observation. Concrete examples:

- Grammar-constrained classification (parsing, tagging, structured prediction).
- Protocol execution with audit (medical triage, regulatory workflows, customer routing).
- Multi-stage decision pipelines with calibrated abstention.
- Hierarchical structured output (form-filling, code completion within syntax).
- Auditable safety-critical classification (aviation, finance, healthcare).
- Sequence labeling with cyclic structure (matched brackets, dialogue acts, conversational protocols).

## What a TPN is not for

- Open-ended generation. No learned generation policy; this is structured prediction.
- Tasks with no structural prior. The geometric machinery has nothing to grip.
- Sparse-reward RL. No value function; supervised + Bayesian only.

## Substrate choice (the encoder is not part of the architectural contract)

Phases 0–18 used a *frozen* random-projection encoder by convention; Phases 20–23 showed that this was an *implementation choice*, not a TPN commitment. The formal definition above does not constrain the encoder at all — the typed graph, posterior, mask, σ, and per-cell heads sit on top of *whatever encoded representation* the substrate provides. Concretely:

- **Frozen substrate (`FrozenEncoderTorch`):** the original convention. Useful for the "all gradient flows through symbolic structure" identifiability argument; bounded by the encoder's seed-init geometry (Phase 20 Wave B).
- **Trained-from-scratch substrate (Wave C / E26):** an MLP trained on next-state prediction with state-conditioned input. Substantially better substrate quality (Phase 23: purity 0.42 → 0.80).
- **Frozen pretrained substrate (E30 / PCG-X on GPT-2 small):** mid-layer activations from an off-the-shelf model that **was not trained on the grammar**. The most general case. Phase 24 result: mean purity 0.764 across 5 grammars at L6 (block 6 of 12) — matches the state-conditioned MLP trained on the grammar (E28: 0.790) and beats the from-scratch small transformer trained on the grammar (E29: 0.663). PCG-X provides the interpretation layer; the substrate's pretraining priors carry the state structure. Phase 25's layer-ablation sweep (scripts/phase25_layer_ablation_sweep.py) found a robust peak-then-drop curve through depth: L0 ≈ 0.56, L6 ≈ 0.75, L10 ≈ 0.78, L12 ≈ 0.67. The final transformer block is the *wrong* place to harvest for state extraction — it specialises for next-token prediction. Mid-to-late (L6–L10) is the sweet spot; the L6/L10 distinction is within CUDA-nondeterministic seed variance and pending multi-seed bootstrap.

The architectural commitment is the symbolic stack on top, not the substrate underneath.

The same point applies to the *audited graph*: σ + the 3-branch `ControlPolicy` are substrate-agnostic over the graph they reason about. `ControlPolicy` accepts either a `GraphFSM` (the typed FSM, used by E0/E1/E9) or a raw `legality_matrix: np.ndarray` (the extracted PCG-X regime graph, used by E28 since Phase 23e). The σ ensemble takes scalar signals (margin, decision-tie, illegal, loop-risk, stabiliser, KL-surprise) and never references the graph directly, so the same detector runs over either. `decision_trace.jsonl` is therefore one schema with two producer families: the typed-FSM runners use vertex IDs like `"q_open"`; the PCG-X runner uses `"regime_K"` (or `"regime_unknown"` for argmaxes outside the trained cell set). The interpretability surface is the same in both cases.

## Comparison to nearest neighbours

| Existing class | What they have | What TPN adds |
|---|---|---|
| Cascade classifier | Sequential classifiers with rejection | State-dependent routing, learned protocol, calibrated confidence, audit trail. |
| Mixture-of-Experts | Learned soft routing | Hard FSM routing + soft σ override; auditable; deterministic. |
| HMM with classifier emissions | Latent state + emission | Bayesian posterior over transitions; hyperbolic geometry; group-equivariant attention; audit. |
| Grammar-constrained decoding | Mask at decode time | Mask + cycle detection + audit + posterior over the grammar itself. |
| Conditional Random Field | Sequence-level structured prediction | Tree / cyclic structure; per-type submodels; calibrated abstention. |

## Formal definition

A TPN is a tuple

$$\mathcal{T} = (V, T, \tau, M, \{f_t\}_{t\in T}, \pi, \sigma, g, \mathcal{H})$$

where:

- $V$ is a finite vertex set (the typed FSM's states);
- $T$ is a finite type alphabet and $\tau \colon V \to T$ assigns a type to each vertex;
- $M \in \{0, 1\}^{V \times V}$ is a hard legality mask induced by a Beta posterior $\pi_{ij} = \mathrm{Beta}(\alpha_{ij}, \beta_{ij})$ via $M_{ij} = \mathbb{1}[\mathbb{E}\pi_{ij} > 1/2]$ (strict; the skeptical-prior default);
- $\{f_t\}_{t \in T}$ is a family of per-type classifier heads $f_{\tau(i)} \colon \mathcal{X} \to \Delta^{|V|-1}$, whose outputs are masked by row $M_{i, \cdot}$ before softmax;
- $\sigma \colon \mathcal{X} \times V \to \mathbb{R}_{\geq 0}$ is the singularity score $\sigma = \sum_k w_k \cdot s_k$ over the additive ensemble $s_k \in$ {margin, decision-tie, illegal, loop-risk, KL-surprise, stabiliser}, governing the routing $\rho \in$ {NORMAL, RECOVERY, ABSTAIN};
- $g$ is a Riemannian metric: hyperbolic on prototypes ($\mathbb{D}^d$, the open Poincaré ball), Fisher–Rao on $\pi$, and the natural gradient $g^{-1} \nabla$ governs all gradient updates;
- $\mathcal{H} = (R, E_H)$ is the **labelled hypergraph** with $R$ a set of regimes each carrying a canonical KL signature, a `named` coordinate dict, and a `residual` feature list, and $E_H$ a set of hyperedges each carrying a feature-delta and an optional boundary geometry. The discrete graph $(V, E)$ is the strict projection $\pi_{\mathrm{disc}}(\mathcal{H})$ — backward compatibility is automatic.

A TPN is **trained** in two phases:

- **Phase A** (classical, deterministic on a fixed corpus): Baum–Welch forward–backward in log space + Beta-Dirichlet M-step on $(\alpha, \beta)$, recovering $M$ from observation alone.
- **Phase B** (gradient, optional): backprop through energy $E(x, j) = \sum_k \lambda_k \phi_k(x, j)$ at the observed transition, jointly updating $\{f_t\}$, prototype positions in $\mathbb{D}^d$, and the log-odds bias $\log \alpha - \log \beta$.

A TPN is **deployed** by emitting a node-tuple decision trace at every step: (`output_node_tuple`, `edge_traversed`, `mask_version_id`, `posterior_summary`, `axis_node_ids`). No raw floats cross the system's interfaces; the trace is the audit artefact by construction.

## Reference implementation

This repository (`nga`) is the reference implementation. The atom census (`tests/integration/test_atom_census.py`) is the executable specification: every architectural commitment is an atom; every atom has a caller; every claim has a test.

For the foundational mathematics, see `Mathematics.md`. For the paper-shaped synthesis across five grammars and one real-world domain, see [`docs/results.md`](results.md). For incoming research proposals (Phase 20+ work), see [`docs/proposals/`](proposals/).
