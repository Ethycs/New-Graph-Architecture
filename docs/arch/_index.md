# Architecture Atoms

The arch cluster contains 30 atomic notes covering the typed-graph inference architecture: state machines, scoring, hyperbolic embedding, singularity detection, group-quotient compression, energy-based training, and the reservoir+grammar stack. Read these if you want to understand *how* the system works at a design level.

Note: the original `whitney-stratification-layer` and `partition-function-Z` atoms have been replaced by [Behavioral Stratum Tagger](./behavioral-stratum-tagger.md) (P2, discrete categorical tagging, no geometry) and [Stratified Partition Function](./stratified-partition-function.md) (P5, merges Whitney stratification of the extruded manifold with the Boltzmann normalization $Z$). See [build-order.md](../build-order.md) for the rationale.

## Notes

### Graph & State Machine

- [Graph FSM](./graph-fsm.md) — The $(V, E, w_v, g_v, m_v)$ tuple that makes a bare graph into a legal-move machine.
- [Graph Legality Mask](./graph-legality-mask.md) — Layer that zeros out illegal next-state logits before softmax.
- [Weighted Plumbing Graph](./weighted-plumbing-graph.md) — FSM enrichment carrying intersection-form weights for singularity realization.
- [Dynkin / ADE Classification](./dynkin-ade-classification.md) — Recognizes when a task graph is an ADE quiver, unlocking classical singularity tools.
- [Graph Realization Functor](./graph-realization-functor.md) — Maps an enriched graph to a singularity object (plumbing, ADE, or graph hypersurface).
- [Singularity Extraction Functor](./singularity-extraction-functor.md) — Inverse direction: maps a singularity back to its combinatorial graph skeleton.

### Typed Scoring

- [Typed Score Record](./typed-score-record.md) — Structured tuple $(q_t, \lambda_t, p_t, z_t, \mathbf{e}_t)$ produced per inference step.
- [Margin Uncertainty](./margin-uncertainty.md) — Score gap between top-1 and top-2 predictions; primary singularity signal.
- [Confusion Graph](./confusion-graph.md) — Directed graph of classification errors and near-misses built from evaluation runs.
- [Typed Field Pipeline](./typed-field-pipeline.md) — Front-half pipeline: raw observation → grammar objects → Typed Score Record.

### Hyperbolic Geometry

- [Hyperbolic Embedding](./hyperbolic-embedding.md) — Embeds the task graph into $\mathbb{H}^d$ preserving hierarchy as hyperbolic distance.
- [Graph Prototype Vectors](./graph-prototype-vectors.md) — Learned anchor points in $\mathbb{H}^d$ per node type.
- [Graph Extrusion](./graph-extrusion.md) — Converts the discrete graph into a continuous stratified geometric object.
- [Hyperbolic Distance Loss](./hyperbolic-distance-loss.md) — Training objective pulling observations toward typed-state prototypes in $\mathbb{H}^d$.
- [Gromov δ Diagnostic](./gromov-hyperbolicity-diagnostic.md) — Computes the $\delta$ constant to validate tree-likeness of the embedding.

### Singularity

- [Singularity Detector σ(x)](./singularity-detector.md) — Aggregate anomaly score combining margin, contradiction, loop, and illegal-pressure signals.
- [Behavioral Stratum Tagger](./behavioral-stratum-tagger.md) — Discrete categorical tagging from the singularity-types catalog (P2; no geometry).
- [Singularity Types Catalog](./singularity-types.md) — Enumerated taxonomy of singularity kinds with canonical feature signatures.
- [Failure-vs-Margin AUROC](./failure-margin-auroc.md) — Validation metric: AUROC of σ(x) against ground-truth error labels.
- [Catastrophe Labels](./catastrophe-labels.md) — Annotates edges with fold/cusp/swallowtail tags marking expected bifurcations.

### Group Quotient

- [Group Action on Graph](./group-action-on-graph.md) — A group $H$ acts on vertices/edges, identifying structurally equivalent states.
- [Orbit Quotient Space](./orbit-quotient-space.md) — Compressed graph $|\Gamma|/H$ whose vertices are orbits rather than individual states.
- [Orbit-Pair Attention](./orbit-pair-attention.md) — Attention computed at orbit granularity, reducing cost from $O(n^2)$ to $O(|V/H|^2)$.
- [Stabilizer Signature](./stabilizer-signature.md) — Per-state record of stabilizer subgroup; non-trivial stabilizer = singular state.

### Energy

- [Energy Function E(x)](./energy-function-E.md) — Scalar pressure per state combining cost, uncertainty, contradiction, loop, and progress.
- [Stratified Partition Function](./stratified-partition-function.md) — Whitney stratification of the extruded manifold + Boltzmann normalization $Z = \sum_\lambda Z_\lambda$ (P5; merges the old `whitney-stratification-layer` and `partition-function-Z`).
- [Energy-Weighted Loss](./energy-weighted-loss.md) — Training loss reweighted by state energy to prioritize hard/singular examples.

### Reservoir + Grammar

- [Frozen Encoder Backbone](./frozen-encoder-backbone.md) — Pre-trained encoder held fixed; only the readout layer trains (reservoir computing).
- [Typed Readout Layer](./typed-readout-layer.md) — Small trainable heads mapping frozen embeddings to typed output spaces.
- [Grammar Compiler](./grammar-compiler.md) — DSL that compiles declarative state-type specs into typed graphs, prototypes, and legality masks.

## See also

- [Drivers index](../drivers/_index.md)
- [Experiment index](../exp/_index.md)
- [Open questions](../open/_index.md)
- [Root README](../README.md)
