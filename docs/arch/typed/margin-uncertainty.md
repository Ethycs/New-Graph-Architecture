# Margin Uncertainty

**Cluster:** arch
**Status:** spec
**Tags:** #margin-uncertainty #uncertainty-signal #singularity-detector

## What
Margin uncertainty is the score gap between the top-1 and top-2 predicted states: $m(x) = s_{q_1}(x) - s_{q_2}(x)$ where $q_1, q_2$ are the highest and second-highest-scoring next states. This is the primary uncertainty signal used to flag decision boundaries and singularities. Low margin (e.g., $m < 0.15$) indicates stratum crossing, tie or near-tie decisions, and regions where the classifier is unstable.

## Why
Confidence alone is not enough; a model can be 90% sure of a wrong answer. Margin detects when the top choice is barely ahead of alternatives—a signature of decision boundaries and singular points. In agent terms, low margin means "we are near a mode switch" or "multiple plans look equally good." This is exactly where singularity diagnostics apply and where the system needs extra caution or intervention.

## Interface
- **Input:** (from [Typed Score Record](typed-score-record.md)) top-2 state scores $s_{q_1}, s_{q_2}$ (logits or calibrated probabilities).
- **Output:** (to [Confusion Graph](confusion-graph.md), singularity detector, and diagnostics) margin value $m \in [0, \infty)$ and boolean flag is_singular $:= \mathbb{1}_{m < \tau}$ where $\tau$ is a threshold (typically 0.1–0.2).

## Build steps
- Extract top-1 and top-2 scores from the classifier output.
- Compute margin: $m = s_1 - s_2$. If there is only one legal state, set $m = \infty$ (no uncertainty).
- Calibrate scores if needed (e.g., via Platt scaling or temperature tuning) so that margin is comparable across runs.
- Set threshold $\tau$ empirically or via a held-out validation set (e.g., threshold that maximizes F1 for detecting actual errors).
- Flag singularity if $m < \tau$.
- Optionally compute higher-order margins (3-way tie, etc.) for richer diagnostics.
- Log margin distribution for each state and stratum as a diagnostic.

## Links
- **See also:** [Typed Score Record](typed-score-record.md), [Confusion Graph](confusion-graph.md), [Graph Legality Mask](../graph/graph-legality-mask.md)
- **Drives:** (margin-calibration-test, singularity-detection-accuracy — planned experiments)
- **Driven by:** [Typed Score Record](typed-score-record.md)
- **Math:** [Architecture.md §Hyperbolic monodromy](../../Architecture.md#hyperbolic-quotient-feature) — low margin corresponds to stratum-boundary crossing.
- **Open:** (margin-threshold — learn adaptively per task, or use fixed global value?)
