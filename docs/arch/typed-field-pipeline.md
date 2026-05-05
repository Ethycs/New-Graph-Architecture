# Typed Field Pipeline

**Cluster:** arch
**Status:** spec
**Tags:** #typed-field #pipeline #observation-to-score

## What
The Typed Field Pipeline is the front half of the architecture: observation $x$ (agent context, query, evidence) flows through grammar objects and type classifiers to produce a Typed Score Record $Y$. It comprises: (1) raw observation encoder, (2) grammar rule application, (3) typed-feature bottleneck, (4) stratum and state classifiers, and (5) score record packaging. This is the observation-to-decision bridge.

## Why
Raw observations (text, embeddings, tool outputs) are unstructured. The pipeline imposes semantic and grammatical structure via intermediate typed representations. This enables the downstream graph and singularity detectors to reason about agent state semantically, not just statistically. Without structure, the classifier is a black box; with it, every intermediate layer is inspectable and auditable.

## Interface
- **Input:** observation $x$ (context, evidence, user intent, current state).
- **Output:** Typed Score Record $Y = (q_t, \lambda_t, p_t, z_t, \mathbf{e}_t)$ and intermediate grammar objects for inspection/debugging.

## Build steps
- **Stage 1 (Encoding):** apply an encoder (LLM, neural network, or tfidf) to observation $x$ to produce initial feature vector or embedding.
- **Stage 2 (Grammar):** apply grammar rules (templates, constraint checks, type validators) to extract semantic fields (query_clarity, evidence_strength, contradiction_pressure, loop_risk, answer_readiness). Each field is a scalar or structured object.
- **Stage 3 (Bottleneck):** pass grammar outputs through a learned or rule-based bottleneck to produce a compact typed-feature vector $\mathbf{t}$ with named components matching the agent's type vocabulary.
- **Stage 4 (Classifiers):** train two classifiers in parallel: (a) stratum classifier on $\mathbf{t}$ → $\lambda_t$ and (b) state classifier on $\mathbf{t}$ → logits over $V$. Also compute hyperbolic embedding $z_t$.
- **Stage 5 (Packaging):** collect results into a Typed Score Record, validate types, and pass to downstream.
- **Testing:** validate that grammar rules are reproducible, that bottleneck preserves interpretability, and that classifiers agree with human labels on a gold set.

## Links
- **See also:** [Typed Score Record](./typed-score-record.md), [Graph FSM](./graph-fsm.md), [Margin Uncertainty](./margin-uncertainty.md)
- **Drives:** (interpretability-audit, pipeline-stage-ablation — planned experiments)
- **Driven by:** (observation-format — external driver)
- **Math:** [Architecture.md §Classifier version](../Architecture.md#classifier-version) — the pipeline realizes the multi-component classifier scoring rule.
- **Open:** (grammar-learning — hand-craft grammar rules, or learn end-to-end?)
