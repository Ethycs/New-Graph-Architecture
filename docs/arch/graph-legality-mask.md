# Graph Legality Mask

**Cluster:** arch
**Status:** spec
**Tags:** #graph-legality #mask-layer #classifier-constraint

## What
A legality mask is a learned or rule-defined layer that zeros out logits (log-probabilities) over illegal next-state transitions before softmax. It enforces that the classifier cannot propose a move that violates the FSM's edge set $(V, E)$. Specifically, for each current state $q_t$, it sets $\log p(q' \mid q_t) \to -\infty$ if $(q_t, q') \notin E$.

## Why
A classifier trained on observation data may learn spurious correlations and propose high-probability jumps to disconnected graph nodes. This causes routing failures, loop violations, and task sequencing errors. The mask is a hard structural constraint that makes the classifier graph-aware: no probability mass can flow to illegal moves. This is the bridge between learned scores and graph topology.

## Interface
- **Input:** (from [Typed Score Record](./typed-score-record.md)) logits/scores over all possible next states $q'$.
- **Output:** (to classifier softmax) masked logits where illegal transitions are zeroed; soft distribution $p(q_{t+1} \mid q_t, x)$ respects edge set $E$.

## Build steps
- Construct a boolean adjacency matrix $A[q, q'] = \mathbb{1}_{(q,q') \in E}$ from the FSM's edge set.
- For each current state $q_t$, extract the corresponding row $A[q_t, :]$.
- Invert the row: $M[q_t, :] = 1 - A[q_t, :]$ (mask is 1 for illegal transitions).
- Multiply logits element-wise by a large negative constant where mask is 1: $\ell'[q'] = \ell[q'] - M[q_t, q'] \cdot \lambda_{\infty}$, where $\lambda_{\infty}$ is large (e.g., 1e6).
- Apply softmax to masked logits; verify that illegal transitions have probability $\lesssim 10^{-6}$.
- Cache the adjacency matrix and mask computation for efficiency; recompute only if FSM changes.

## Links
- **See also:** [Graph FSM](./graph-fsm.md), [Typed Field Pipeline](./typed-field-pipeline.md)
- **Drives:** (illegal-jump-rate-test — planned experiment)
- **Driven by:** [Graph FSM](./graph-fsm.md)
- **Math:** [Architecture.md §Catastrophe graph moves](../Architecture.md#catastrophe-graph-moves) — the legality condition is a tame constraint in the classifier's decision space.
- **Open:** (mask-differentiability — backprop through mask for end-to-end FSM learning, or keep fixed?)
