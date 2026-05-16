# Architecture Atoms

The arch cluster contains 50 atomic notes covering the typed-graph inference architecture: state machines, scoring, hyperbolic embedding, singularity detection, group-quotient compression, energy-based training, substrate adapters, the PCG-X control bridge, and the Phase 26 labelled-hypergraph layer. Atoms are grouped into 7 themed sub-folders that correspond to the section headings below. Read these if you want to understand *how* the system works at a design level.

Note: the original `whitney-stratification-layer` and `partition-function-Z` atoms have been replaced by [Behavioral Stratum Tagger](singularity/behavioral-stratum-tagger.md) (P2, discrete categorical tagging, no geometry) and [Stratified Partition Function](energy/stratified-partition-function.md) (P5, merges Whitney stratification of the extruded manifold with the Boltzmann normalization $Z$). See [build-order.md](../build-order.md) for the rationale.

## Notes

### Graph & State Machine — `graph/`

- [Graph FSM](graph/graph-fsm.md) — The $(V, E, w_v, g_v, m_v)$ tuple that makes a bare graph into a legal-move machine.
- [Graph Legality Mask](graph/graph-legality-mask.md) — Layer that zeros out illegal next-state logits before softmax.
- [Weighted Plumbing Graph](graph/weighted-plumbing-graph.md) — FSM enrichment carrying intersection-form weights for singularity realization.
- [Dynkin / ADE Classification](graph/dynkin-ade-classification.md) — Recognizes when a task graph is an ADE quiver, unlocking classical singularity tools.
- [Graph Realization Functor](graph/graph-realization-functor.md) — Maps an enriched graph to a singularity object (plumbing, ADE, or graph hypersurface).
- [Singularity Extraction Functor](graph/singularity-extraction-functor.md) — Inverse direction: maps a singularity back to its combinatorial graph skeleton.
- [Product Graph](graph/product-graph.md) — Typed product of per-axis FSMs; the node-tuple identity space for outputs.
- [Bisimulation Quotient](graph/bisimulation-quotient.md) — Behavioural merge of cells into regimes; the discrete shadow of the stratified partition (PCG-X foundation).
- [Labelled Hypergraph](graph/labelled-hypergraph.md) — Phase 26 lift of the PCG-X regime graph: regimes carry KL signature + named coordinates + residual features; transitions are hyperedges carrying feature-deltas.
- [KL Regime Signature](graph/kl-regime-signature.md) — Coordinate-free KL fingerprint of a regime's conditional output distribution; powers principled K-choice and regime identity (Phase 26).
- [Predictive Projection](graph/predictive-projection.md) — Phase 23 projection `h → z` with next-state / entropy / failure / optional adversarial-token heads; the partition signal feeding PCG-X.
- [Labelled Hypergraph](graph/labelled-hypergraph.md) — Lifts the regime graph to a hypergraph carrying named labels + residual features per regime; owns the semantic gap explicitly.
- [KL Regime Signature](graph/kl-regime-signature.md) — Coordinate-free KL signatures and gap-detection thresholding for canonical regime identity.

### Typed Scoring — `typed/`

- [Typed Score Record](typed/typed-score-record.md) — Structured tuple $(q_t, \lambda_t, p_t, z_t, \mathbf{e}_t)$ produced per inference step.
- [Margin Uncertainty](typed/margin-uncertainty.md) — Score gap between top-1 and top-2 predictions; primary singularity signal.
- [Confusion Graph](typed/confusion-graph.md) — Directed graph of classification errors and near-misses built from evaluation runs.
- [Typed Field Pipeline](typed/typed-field-pipeline.md) — Front-half pipeline: raw observation → grammar objects → Typed Score Record.
- [Typed Readout Layer](typed/typed-readout-layer.md) — Small trainable heads mapping frozen embeddings to typed output spaces.
- [Typed Readout (Torch)](typed/typed-readout-torch.md) — Wave-C PyTorch implementation of the typed readout head.
- [Typed Latent Clustering](typed/typed-latent-clustering.md) — Per-type clustering in latent space used by the readout head.

### Hyperbolic Geometry — `hyperbolic/`

- [Hyperbolic Embedding](hyperbolic/hyperbolic-embedding.md) — Embeds the task graph into $\mathbb{H}^d$ preserving hierarchy as hyperbolic distance.
- [Graph Prototype Vectors](hyperbolic/graph-prototype-vectors.md) — Learned anchor points in $\mathbb{H}^d$ per node type.
- [Graph Extrusion](hyperbolic/graph-extrusion.md) — Converts the discrete graph into a continuous stratified geometric object.
- [Hyperbolic Distance Loss](hyperbolic/hyperbolic-distance-loss.md) — Training objective pulling observations toward typed-state prototypes in $\mathbb{H}^d$.
- [Gromov δ Diagnostic](hyperbolic/gromov-hyperbolicity-diagnostic.md) — Computes the $\delta$ constant to validate tree-likeness of the embedding.
- [Information Geometry](hyperbolic/information-geometry.md) — Fisher-information geometry on the typed-classifier output simplex.
- [Axis Quantizer](hyperbolic/axis-quantizer.md) — Per-typed-axis vertex-id quantizer used for `output_node_tuple` in decision traces.

### Singularity — `singularity/`

- [Singularity Detector σ(x)](singularity/singularity-detector.md) — Aggregate anomaly score combining margin, contradiction, loop, and illegal-pressure signals.
- [Behavioral Stratum Tagger](singularity/behavioral-stratum-tagger.md) — Discrete categorical tagging from the singularity-types catalog (P2; no geometry).
- [Singularity Types Catalog](singularity/singularity-types.md) — Enumerated taxonomy of singularity kinds with canonical feature signatures.
- [Failure-vs-Margin AUROC](singularity/failure-margin-auroc.md) — Validation metric: AUROC of σ(x) against ground-truth error labels.
- [Catastrophe Labels](singularity/catastrophe-labels.md) — Annotates edges with fold/cusp/swallowtail tags marking expected bifurcations.
- [Monodromy Consistency](singularity/monodromy-consistency.md) — Checks loop-conjugacy consistency of stratum tags along closed paths.
- [Sigma Weight Optimizer](singularity/sigma-weight-optimizer.md) — Optimizes the σ-component weights against held-out failure labels.

### Group Quotient — `group/`

- [Group Action on Graph](group/group-action-on-graph.md) — A group $H$ acts on vertices/edges, identifying structurally equivalent states.
- [Orbit Quotient Space](group/orbit-quotient-space.md) — Compressed graph $|\Gamma|/H$ whose vertices are orbits rather than individual states.
- [Orbit-Pair Attention](group/orbit-pair-attention.md) — Attention computed at orbit granularity, reducing cost from $O(n^2)$ to $O(|V/H|^2)$.
- [Stabilizer Signature](group/stabilizer-signature.md) — Per-state record of stabilizer subgroup; non-trivial stabilizer = singular state.

### Energy & Training — `energy/`

- [Energy Function E(x)](energy/energy-function-E.md) — Scalar pressure per state combining cost, uncertainty, contradiction, loop, and progress.
- [Stratified Partition Function](energy/stratified-partition-function.md) — Whitney stratification of the extruded manifold + Boltzmann normalization $Z = \sum_\lambda Z_\lambda$ (P5; merges the old `whitney-stratification-layer` and `partition-function-Z`).
- [Energy-Weighted Loss](energy/energy-weighted-loss.md) — Training loss reweighted by state energy to prioritize hard/singular examples.
- [Energy Minimization Trainer](energy/energy-minimization-trainer.md) — Wave-B trainer driving E(x) downhill against the typed targets.
- [Torch Energy Trainer](energy/torch-energy-trainer.md) — Wave-C PyTorch trainer using energy + phase-A posterior; current default substrate.
- [Forward-Backward Pass](energy/forward-backward.md) — Posterior smoothing pass over typed sequences; powers loop and contradiction signals.
- [Posterior Mask](energy/posterior-mask.md) — Beta(α, β) per-edge legality posterior; the Bayesian core of the mask layer.
- [Bayesian Nonparametric K](energy/bayesian-nonparametric-k.md) — Dirichlet-process / CRP K-selection for grammar-discovery experiments.

### Substrate, Grammar & Control — `substrate/`

This bucket carries the encoder, the readout-side glue, the grammar compiler that bootstraps the FSM, the activation-harvester, and the PCG-X control policy that bridges σ/regime signals to NORMAL/RECOVERY/ABSTAIN verdicts. The bucket name replaces the older "Reservoir + Grammar" framing because the frozen-encoder commitment was dropped from the TPN contract (commit 8d67c17); the substrate is now paradigm-neutral.

- [Frozen Encoder Backbone](substrate/frozen-encoder-backbone.md) — **Deprecated**: kept for Phase 0–18 historical reference; see [model-class.md](../model-class.md) for current substrate framing.
- [Frozen Encoder (Torch)](substrate/frozen-encoder-torch.md) — PyTorch implementation of the encoder side; used by Wave-C runs.
- [Hidden State Harvester](substrate/hidden-state-harvester.md) — Extraction of activations from the substrate (Phase 20+); the input side of graph extraction.
- [Grammar Compiler](substrate/grammar-compiler.md) — DSL that compiles declarative state-type specs into typed graphs, prototypes, and legality masks.
- [Control Policy](substrate/control-policy.md) — σ-thresholded NORMAL / RECOVERY / ABSTAIN router; substrate-agnostic decision surface consumed by both the typed FSM and the PCG-X regime graph (Phase 23e).
- [SAE Adapter](substrate/sae-adapter.md) — Protocol over a sparse-feature decomposer with identity + mock implementations; named-vs-residual feature split used by the labelled hypergraph (Phase 26 interface; Phase 28 will land a real pretrained SAE).
- [SAE Adapter](substrate/sae-adapter.md) — Sparse feature decomposition with named/residual split; produces the per-regime feature labels consumed by the labelled hypergraph.

## See also

- [Drivers index](../drivers/_index.md)
- [Experiment index](../exp/_index.md)
- [Open questions](../open/_index.md)
- [Root README](../README.md)
