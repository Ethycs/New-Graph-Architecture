# Typed Protocol Network (TPN)

A single-name handle for the model class this architecture defines. Use **Typed Protocol Network** in writing; **TPN** as the abbreviation.

## One-line definition

A TPN is a finite-state typed protocol (the FSM) carrying a network of per-type submodels, with Bayesian posterior over its own transition structure, hard structural masking, soft σ-routing for abstention, and information-geometric credit assignment.

## What's load-bearing

Five commitments together define the class. None alone is novel; the *composition* is.

1. **Typed graph as the system's full state space.** Every observation, prediction, signal is a vertex in some typed axis; the system's true state is a node-tuple in their Cartesian product. No raw floats cross the system's interfaces.
2. **Per-type submodels.** Each type (or group of FSM states sharing a type) gets its own classifier head. Submodels are independent; specialisation is structural.
3. **Hard structural mask + soft σ-routing.** The legality matrix zeros illegal predictions absolutely; σ-routing's three branches (NORMAL / RECOVERY / ABSTAIN) handle ambiguity that the mask doesn't address.
4. **Bayesian posterior over the protocol itself.** The mask isn't a hand-coded constant; it's a Beta(α, β) posterior per edge that grows from observation. The classical Baum-Welch + Bayesian M-step performs cold-start induction.
5. **Information-geometric foundation.** Fisher information sets natural-gradient learning rates; KL divergence is the canonical loss; Cramér-Rao bounds set sample-sufficiency stopping rules. The architecture's hyperbolic + Riemannian + Bayesian apparatus is information geometry by construction.

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
- Pure representation learning. Encoder is frozen; this isn't contrastive/diffusion.
- Tasks with no structural prior. The geometric machinery has nothing to grip.
- Sparse-reward RL. No value function; supervised + Bayesian only.

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

$$\mathcal{T} = (V, T, \tau, M, \{f_t\}_{t\in T}, \pi, \sigma, g)$$

where:

- $V$ is a finite vertex set (the typed FSM's states);
- $T$ is a finite type alphabet and $\tau \colon V \to T$ assigns a type to each vertex;
- $M \in \{0, 1\}^{V \times V}$ is a hard legality mask induced by a Beta posterior $\pi_{ij} = \mathrm{Beta}(\alpha_{ij}, \beta_{ij})$ via $M_{ij} = \mathbb{1}[\mathbb{E}\pi_{ij} > 1/2]$ (strict; the skeptical-prior default);
- $\{f_t\}_{t \in T}$ is a family of per-type classifier heads $f_{\tau(i)} \colon \mathcal{X} \to \Delta^{|V|-1}$, whose outputs are masked by row $M_{i, \cdot}$ before softmax;
- $\sigma \colon \mathcal{X} \times V \to \mathbb{R}_{\geq 0}$ is the singularity score $\sigma = \sum_k w_k \cdot s_k$ over the additive ensemble $s_k \in$ {margin, decision-tie, illegal, loop-risk, KL-surprise, stabiliser}, governing the routing $\rho \in$ {NORMAL, RECOVERY, ABSTAIN};
- $g$ is a Riemannian metric: hyperbolic on prototypes ($\mathbb{D}^d$, the open Poincaré ball), Fisher–Rao on $\pi$, and the natural gradient $g^{-1} \nabla$ governs all gradient updates.

A TPN is **trained** in two phases:

- **Phase A** (classical, deterministic on a fixed corpus): Baum–Welch forward–backward in log space + Beta-Dirichlet M-step on $(\alpha, \beta)$, recovering $M$ from observation alone.
- **Phase B** (gradient, optional): backprop through energy $E(x, j) = \sum_k \lambda_k \phi_k(x, j)$ at the observed transition, jointly updating $\{f_t\}$, prototype positions in $\mathbb{D}^d$, and the log-odds bias $\log \alpha - \log \beta$.

A TPN is **deployed** by emitting a node-tuple decision trace at every step: (`output_node_tuple`, `edge_traversed`, `mask_version_id`, `posterior_summary`, `axis_node_ids`). No raw floats cross the system's interfaces; the trace is the audit artefact by construction.

## Reference implementation

This repository (`nga`) is the reference implementation. The atom census (`tests/integration/test_atom_census.py`) is the executable specification: every architectural commitment is an atom; every atom has a caller; every claim has a test.

For the foundational mathematics, see `Mathematics.md`. For the paper-shaped synthesis across five grammars and one real-world domain, see [`docs/results.md`](results.md). For incoming research proposals (Phase 20+ work), see [`docs/proposals/`](proposals/).
