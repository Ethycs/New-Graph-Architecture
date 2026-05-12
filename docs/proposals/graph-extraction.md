# Universal Graph Extraction: Recovering Typed Protocol Networks from Homogeneously Trained Substrates

**Status:** proposed
**Phase target:** Phase 20
**Date posted:** 2026-05-08
**Author:** TPN research team

## Abstract

We propose and evaluate a procedure for extracting a Typed Protocol Network (typed graph + Beta(α, β) posterior over edges + σ-routed soft mask) from any homogeneously trained neural network with an accessible hidden-state stream. The extraction composes established primitives — Riemannian / Euclidean clustering, Bayesian-nonparametric K-selection, classical forward-backward with a Beta-Dirichlet M-step, and transition-profile type discovery — into a single end-to-end pipeline. The central hypothesis is that the typed graph is *latent* in the activations of any sufficiently trained network on a structurally regular corpus, and that the extracted graph matches hand-authored ground-truth FSMs by Hamming distance and held-out predictive likelihood within a stated confidence band. If validated, TPN is recategorised from a designed architecture to a universal post-hoc analysis with consequences for interpretability, structural distillation, and calibrated abstention on arbitrary pretrained models.

## Background

The TPN framework (this repository, Phases 0–19B) defines a class of structured-prediction models where a typed FSM is the system's full state space, per-type submodels score within-edge content, a Beta(α, β) posterior governs edge legality, and a singularity score σ routes to NORMAL / RECOVERY / ABSTAIN. To date the FSM has been hand-authored in every grammar tested (ListOps, Python expr, Python big, JSON, control flow Python, diagnostic). Phase 19B established that the same Bayesian / Riemannian machinery can recover semantic taxonomy without supervised labels — the medical taxonomy of 41 diseases at NMI 0.97 — by clustering an embedding manifold and running Phase A on cluster transitions. The proposal generalises that result: instead of clustering raw inputs, cluster the **hidden-state stream of a trained network**, and treat the resulting cluster-transition structure as the network's latent typed graph. The reformulation is licensed by the fiber-bundle picture (the typed graph is the discrete base of an equivariant bundle whose fibers are the trained network's prototype manifold) and is adjacent to work on causal abstraction (Geiger et al.), fixed-point analysis of RNNs (Sussillo & Barak 2013), and circuit-extraction in mechanistic interpretability.

## Central hypothesis

**H₀.** For any homogeneously trained network N on a corpus C of sufficient structural regularity, the typed graph G(N, C) extracted by the procedure below is statistically indistinguishable, by normalised Hamming distance and held-out predictive likelihood, from the typed-graph ground truth (where available) within a stated confidence band:

E[Hamming(G(N, C), G_truth)] ≤ 0.10  AND  NLL_holdout(G) − NLL_chain ≤ −1 nat/token.

**H₁.** There exists a corpus / network family where extraction systematically fails — bounding the architecture's universality claim and identifying the structural-regularity precondition empirically.

## What works immediately (mathematical, no experiment needed)

These claims are licensed by composition of established results and do not require new experimentation:

1. **The fiber-bundle reformulation.** Standard differential geometry: bundles compose, equivariant maps compose, connections are well-defined. The reframing is notation, not a claim.
2. **TPN as equivariant GNN with Bayesian edge admissibility.** Definitionally true once the bundle picture is accepted; Cohen & Welling 2016 give the message-passing mathematics, the Beta posterior is a layered addition.
3. **Graph extraction is a well-defined algorithm.** Each step (cluster, count, Phase A, type-discover) has a closed-form or proven algorithm; the composition is a valid map from "trained network" to "typed graph."
4. **Phase A on extracted vertex sets is mathematically identical to Phase A on hand-authored vertex sets.** Conditional on the discretisation, Phase A's deterministic convergence and substrate-independence carry over unchanged.
5. **Holonomy = σ_loop_risk.** Equivalent under the bundle identification; non-trivial holonomy ↔ loop curvature ↔ existing σ component.
6. **Phase A as connection induction.** Forward-backward computes parallel transport on the discrete connection over the base graph. Identification is immediate.
7. **Information-geometric decomposition.** Riemannian fibration ⇒ horizontal / vertical split of Fisher information ⇒ block-diagonal natural gradient. Standard result.
8. **Hierarchical TPN at depth 2 under level-factorisation.** Beta-product factorisation is well-defined; Phase A composes hierarchically given factorisation.

## What requires experimental validation

1. Whether discretised hidden states are Markov on the chosen K. Non-Markov residuals would make Phase A fit a confident-but-wrong skeleton.
2. Whether K-selection stabilises across priors / restarts.
3. **Whether extracted graphs match hand-authored ones** (the decisive experiment) — Hamming distance and held-out predictive likelihood vs known FSM.
4. Whether σ ensemble survives on extracted graphs (σ_uplift / σ_structural_uplift are grammar-conditional).
5. Whether equivariance is enforced or merely emergent in extracted base graphs.
6. Hierarchical Phase A convergence in practice (Lyapunov argument available; empirical monotonicity not).
7. Cross-substrate generality (transformer vs MLP vs CNN vs our Poincaré-prototype substrate).
8. Held-out predictive likelihood ≥ chain baseline.
9. Cluster purity vs domain ground truth at scale beyond Phase 19B.
10. Stability under corpus growth.

## Methods

Pipeline (the integration runner):

1. **Harvest.** Hook into N at chosen depth(s), collect hidden states {h_t} over corpus C.
2. **Discretise.** Cluster {h_t} via Riemannian k-means (Poincaré substrates) or Euclidean k-means (otherwise) into K vertices.
3. **K-selection.** Stick-breaking DP-mixture or held-out predictive likelihood to choose K.
4. **Count.** Transition counts c[cluster(h_t), cluster(h_{t+1})] over consecutive pairs.
5. **Phase A.** Forward-backward + Beta-Dirichlet M-step on counts → induced legality mask + Beta posteriors per edge.
6. **Type discovery.** Secondary clustering of vertices by transition-profile similarity → type assignments.

Substrates evaluated:

- **Sanity.** Our own TorchEnergyTrainer on Python big and JSON. Ground-truth FSM available; Hamming target ≤ 0.05.
- **External.** Small transformer (~10M params) trained on the same corpora. Ground truth same; Hamming target ≤ 0.10.
- **Stress.** Pretrained GPT-2-small on a Python source corpus. No ground truth; held-out predictive likelihood is the only metric.

New atoms required (3):

- `hidden_state_harvester.py` — model-hook abstraction returning (h_t)_{t ∈ C}.
- `bayesian_nonparametric_k.py` — DP-mixture stick-breaking over the embedding manifold; returns K and labels.
- `extracted_graph_runner.py` — composes harvest → cluster → K-select → count → Phase A → type-discover → emit `GraphFSMSpec`.

Existing atoms reused (no modification):

- `forward_backward.py`, `posterior_mask.py`, `typed_latent_clustering.py`, `singularity_detector.py`, `decision_trace_jsonl.py`, `axis_quantizer.py`.

## Validation protocol

Three tiers, each with a pre-registered acceptance criterion:

| Tier | Substrate | Metric | Acceptance |
|---|---|---|---|
| Sanity | TorchEnergyTrainer | Hamming(G_extracted, G_authored) | ≤ 0.05 on ≥ 3/5 grammars |
| External | Small transformer | Hamming + held-out NLL improvement over chain | ≤ 0.10 Hamming AND ≥ 1 nat/token NLL improvement on ≥ 3/5 grammars |
| Stress | Pretrained GPT-2-small | Held-out NLL improvement over chain on Python corpus | ≥ 0.5 nat/token (looser, no ground-truth FSM available) |

Multi-seed (5 seeds, 42–46) on each tier; report mean ± std. Multi-K stability check: fit at K, K ± 10%, K ± 25%; expect graph to be subgraph-stable across perturbations.

## Sudden large implications (conditional on H₀)

1. **Universality.** TPN stops being a clean-room architecture and becomes a *universal post-hoc analysis* applicable to any homogeneously trained network. The model class is a lens, not a design.
2. **Substrate independence at full strength.** Already shown sklearn ↔ torch for Phase A; extraction extends it across unrelated families (transformer, MLP, CNN). Substrate-independence becomes near-tautological.
3. **Audit trail for any pretrained model.** Decision traces, mask version IDs, σ routing — attached to a small GPT-2 or any pretrained network at inference time. This is a genuinely new interpretability primitive, adjacent to but distinct from sparse autoencoders.
4. **Compute-efficient inference distillation.** Train the big model expensively; extract its graph cheaply; run TPN inference at sub-100 MB CPU on the extracted skeleton.
5. **Calibrated abstention layer for arbitrary pretrained models.** σ + posterior mask attached to extracted graphs gives any base model a free abstention / audit interface — the LLM-hallucination-detection primitive that the field currently lacks.
6. **Symbolic-vs-neural debate empirically settled within scope.** If typed structure is latent in activations, the dichotomy dissolves; if not, you have a clean experimental boundary on when typed structure exists in a network.
7. **Recursive TPN falls out of multi-layer extraction.** Run extraction at multiple network depths; the hierarchy of extracted graphs is the recursive TPN.

## Risks and limitations

- **Markov assumption.** If hidden-state dynamics are non-Markov at the chosen K, Phase A fits a wrong skeleton. Detected by a per-K residual Markov test before extraction is trusted.
- **Depth sensitivity.** Which layer to harvest from is unspecified; planned ablation across layers in Tier 2.
- **Equivariance not certified end-to-end.** Current atoms are equivariant by construction at orbit-pair level; whether the extracted graph respects an emergent equivariance is unverified.
- **Scope.** Validation is on grammars of size ≤ 37 states and corpora ≤ 1000 samples. Million-token regimes and learned encoders at scale are out of scope; positive results would lower the prior cost of testing those regimes but not establish them.

## Timeline

- **Wave A (3 days).** Atoms — `hidden_state_harvester.py`, `bayesian_nonparametric_k.py`, `extracted_graph_runner.py`. Unit tests per atom.
- **Wave B (1 week).** Tier 1 (sanity, our substrate, 5 grammars × 5 seeds).
- **Wave C (1 week).** Tier 2 (external, small transformer trained from scratch on the same grammars).
- **Wave D (1 week).** Tier 3 (pretrained GPT-2-small on real Python source).
- **Wave E (3 days).** Synthesis — research log entry, `docs/results.md` update, `aggregate.py` verdicts, decision on H₀.

Total: ~4 weeks at single-developer cadence, CPU-only for Tiers 1–2, modest GPU access required for Tier 3.

## Decision criterion

The proposal is judged a success iff Tier 1 AND Tier 2 acceptance criteria are met on ≥ 3/5 grammars. Below that bar, the universality claim is retracted; the architecture remains valid as a designed model class but loses its post-hoc interpretation. Tier 3 is reported regardless and treated as exploratory.

## Related work

- Cohen & Welling 2016 — group-equivariant convolutional networks.
- Weiler & Cesa 2019 — steerable CNNs.
- Sussillo & Barak 2013 — fixed-point analysis of trained RNNs.
- Geiger et al. — causal abstraction.
- Anthropic circuits / sparse-autoencoder feature dictionaries — mechanistic interpretability of trained transformers.

## See also

- [results.md §10](../results.md) — Phase 19B self-supervised structural discovery (the closest precedent in this repository).
- [model-class.md](../model-class.md) — TPN model class definition.
- `research_log.md` — running phase log; the Phase 20 entry that consumes this proposal will reference this file.
