# Graph Prototype Vectors

**Cluster:** arch
**Status:** spec
**Tags:** #embedding #state-representation #anchor

## What
Assign each node in the hyperbolic embedding a learned prototype vector $\mathbf{p}_v \in \mathbb{H}^d$ that anchors the typed semantic type of that node. The prototype summarizes the grammatical, role-based, and task-structural identity of the node, independent of its observed transitions.

## Why
Prototypes decouple node identity from observed behavior. A node with the same type (e.g., "loop guard," "decision branch") should have similar prototypes even if it appears in different graph contexts. This enables transfer learning and regularization: the training loss can push observations of a typed node toward its prototype, while the prototype itself is learned from all instances of that type. Without prototypes, each node is treated as unique, losing semantic generalization.

## Interface
**Inputs:**
- Hyperbolic embedding $\phi: V(G) \to \mathbb{H}^d$.
- Node type partition $\mathcal{T} = \{T_1, T_2, \ldots, T_k\}$ (e.g., control-flow types).
- Observations $\{(v_i, \mathbf{o}_i)\}$ where $\mathbf{o}_i \in \mathbb{H}^d$ is an observed state.

**Outputs:**
- Prototype map $\mathbf{p}: V(G) \to \mathbb{H}^d$.
- Type embedding $\boldsymbol{\tau}: \mathcal{T} \to \mathbb{R}^k$ (learned type features).

## Build steps
- Initialize prototype $\mathbf{p}_v$ as the mean of all observations with type $\tau(v)$, projected into $\mathbb{H}^d$.
- Learn type embeddings $\boldsymbol{\tau}_t$ for each type $t \in \mathcal{T}$ via shared encoder; prototype $\mathbf{p}_v = \text{decode}(\boldsymbol{\tau}_{\tau(v)})$.
- During training, add term to [Hyperbolic Distance Loss](hyperbolic-distance-loss.md): $\lambda_p \sum_i D_{\text{hyp}}(\mathbf{o}_i, \mathbf{p}_{\tau(v_i)})^2$.
- Freeze prototypes after convergence or update them as moving averages of observed centroids.
- Evaluate prototype quality by coverage: fraction of observations within margin of their prototype.

## Links
- **See also:** [Hyperbolic Embedding](hyperbolic-embedding.md), [Hyperbolic Distance Loss](hyperbolic-distance-loss.md), [Singularity Types Catalog](../singularity/singularity-types.md)
- **Drives:** (prototype-stability, cross-graph-transfer — planned experiments)
- **Driven by:** (typed-graph-contract, observation-log — external inputs)
- **Math:** [Mathematics.md §Riemannian Centers](../../Mathematics.md#riemannian)
- **Open:** (type-hierarchy-fusion — open question)
