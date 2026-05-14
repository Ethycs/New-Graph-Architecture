# Confusion Graph

**Cluster:** arch
**Status:** spec
**Tags:** #confusion-graph #diagnostic #error-graph

## What
A confusion graph is a directed graph of classification errors and near-misses: vertices are task states (true and predicted labels), and weighted edges record the frequency of confusions $(q_{\text{true}}, q_{\text{pred}})$ and low-margin near-confusions. It is built incrementally from classifier runs and serves as both a diagnostic tool (where is the model struggling?) and a stratification aid (which state pairs are behaviorally close?).

## Why
Accuracy alone hides structure. A confusion graph reveals which state pairs are systematically confused, which transitions are rare, and where low-margin regions cluster. This feeds back into graph refinement (should some states be merged or split?) and singularity understanding (do confused pairs correspond to predicted singularities?). For a deployed agent, the confusion graph is a window into actual failure modes.

## Interface
- **Input:** (from [Typed Score Record](typed-score-record.md) and [Margin Uncertainty](margin-uncertainty.md)) per-step records of $(q_{\text{true}}, q_{\text{pred}}, m)$ from evaluation runs.
- **Output:** (to analysis and [Graph FSM](../graph/graph-fsm.md) refinement) directed multigraph $\mathcal{C} = (V, E_{\text{err}}, w_{\text{err}})$ where $w_{\text{err}}(q, q')$ is the confusion count or rate, plus optional low-margin edge set $E_{\text{near}}$.

## Build steps
- Initialize an empty directed weighted graph with the task state set $V$ as vertices.
- For each evaluation example, record the tuple $(q_{\text{true}}, q_{\text{pred}}, m)$.
- If $q_{\text{true}} \neq q_{\text{pred}}$, increment edge weight $w_{\text{err}}(q_{\text{true}}, q_{\text{pred}})$ by 1.
- If $q_{\text{true}} = q_{\text{pred}}$ but $m < \tau$ (low margin), increment a separate near-miss counter or color edge with lower weight.
- Compute confusion rates: $\text{rate}(q, q') = w_{\text{err}}(q, q') / \text{count}(q_{\text{true}} = q)$.
- Identify high-confusion pairs (rate $> 10\%$ or top-$k$ by count).
- Output as a sparse adjacency matrix or edge list for visualization and further analysis.

## Links
- **See also:** [Margin Uncertainty](margin-uncertainty.md), [Typed Score Record](typed-score-record.md), [Graph FSM](../graph/graph-fsm.md)
- **Drives:** (error-analysis, state-merging-test — planned experiments)
- **Driven by:** [Margin Uncertainty](margin-uncertainty.md)
- **Math:** [Architecture.md ��Hyperbolic singularity idea](../../Architecture.md#hyperbolic-singularity-idea) — confusion edges correspond to stratum-crossing regions.
- **Open:** (confusion-stratification — open question: stratify by confusion density or FSM structure?)
