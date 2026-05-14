# Typed Score Record

**Cluster:** arch
**Status:** spec
**Tags:** #typed-score #score-record #classifier-output

## What
A Typed Score Record $(Y)$ is the per-step output of the observation encoder and classifier. It is a structured tuple $(q_t, \lambda_t, p_t, z_t, \mathbf{e}_t)$ where $q_t$ is the predicted next state, $\lambda_t$ is the behavioral stratum label (Search, Verify, Write, etc.), $p_t$ is the soft probability distribution, $z_t$ is the hyperbolic embedding, and $\mathbf{e}_t$ is an optional dense embedding. This is the interface between the neural model and the graph.

## Why
Raw classifier logits are untyped and geometrically incoherent. A structured record ties the model output to the agent's semantic and geometric concepts. It enables downstream components to reason about confidence, uncertainty, stratum crossing, and singularity proximity. Without it, scores are just numbers; with it, scores become navigable agent state descriptors.

## Interface
- **Input:** (from [Typed Field Pipeline](typed-field-pipeline.md)) raw context $x$, classifier logits $\ell_q$ and $\ell_\lambda$ over states and strata.
- **Output:** (to [Graph Legality Mask](../graph/graph-legality-mask.md), [Margin Uncertainty](margin-uncertainty.md), [Confusion Graph](confusion-graph.md)) structured record $Y = (q_t, \lambda_t, p_t, z_t, \mathbf{e}_t)$ with type hints and optional metadata.

## Build steps
- Apply softmax to state logits $\ell_q$ to get soft distribution $p_q = \mathrm{softmax}(\ell_q / T)$.
- Take argmax to get predicted state $q_t = \arg\max_q \ell_q$.
- Apply softmax to stratum logits $\ell_\lambda$ to get stratum distribution $p_\lambda = \mathrm{softmax}(\ell_\lambda / T)$ and label $\lambda_t = \arg\max_\lambda \ell_\lambda$.
- Extract or compute hyperbolic embedding $z_t$ (either from model output or by mapping state to hyperbolic prototype).
- Optionally extract dense embedding $\mathbf{e}_t$ from the model's penultimate layer.
- Pack into a dataclass or named tuple with field names, types, and docstrings.
- Validate: check that $\sum p_q = 1$, check for NaN, check that $q_t \in V$ (is a valid state).

## Links
- **See also:** [Typed Field Pipeline](typed-field-pipeline.md), [Margin Uncertainty](margin-uncertainty.md), [Confusion Graph](confusion-graph.md)
- **Drives:** (score-record-validation-test — planned experiment)
- **Driven by:** [Typed Field Pipeline](typed-field-pipeline.md)
- **Math:** [Architecture.md §Classifier version](../../Architecture.md#classifier-version) — the score record is the output of the multi-component classifier.
- **Open:** (score-record-schema — include confidence intervals or only point estimates?)
