# Catastrophe Labels

**Cluster:** arch
**Status:** spec
**Tags:** #labels #bifurcation #enrichment

## What
Annotate graph edges with catastrophe-theoretic tags—fold, cusp, swallowtail, butterfly—that mark where bifurcations and phase transitions are expected. These labels come from catastrophe-theory analysis of the control parameter space and encode intrinsic instability. They inform [Singularity Detector σ(x)](singularity-detector.md) priors and ground truth for training.

## Why
Catastrophe theory predicts that certain transitions are inherently unstable and must bifurcate. Tagging such edges elevates their prior singularity probability in the detector, reducing false negatives at known-risky locations. Moreover, catastrophe labels provide weak supervision for [Failure-vs-Margin AUROC](failure-margin-auroc.md): an observation at a cusp edge is *expected* to exhibit high $\sigma$, so it is a positive example even if the downstream error label is initially unknown.

## Interface
**Inputs:**
- Typed graph $G_{\text{typed}}$.
- Control parameter space dimension and structure (from domain analysis).
- Bifurcation diagram or singularity set (e.g., from catastrophe-theory literature or symbolic computation).

**Outputs:**
- Edge label map: each $e \in E(G) \mapsto \{\text{fold}, \text{cusp}, \text{swallowtail}, \text{butterfly}, \text{none}\}$.
- Confidence per label.
- Implied prior probabilities: $P(\text{singularity} | \text{edge type})$.

## Build steps
- Review domain specification of control parameters (e.g., loop counts, constraint thresholds, flag states).
- Identify bifurcation set in parameter space using catastrophe theory or numerical bifurcation analysis (e.g., AUTO, MatCont, or custom solver).
- For each edge, compute which control parameter trajectory(ies) it traverses; check against bifurcation set.
- Tag edge with catastrophe type(s) if its trajectory is near a bifurcation point.
- Estimate confidence: proximity to actual bifurcation surface determines label confidence.
- Embed confidence into prior: $P(\sigma = 1 | \text{cusp}) > P(\sigma = 1 | \text{regular})$.
- Use priors to initialize [Singularity Detector σ(x)](singularity-detector.md) weights and to weight training examples.

## Links
- **See also:** [Singularity Detector σ(x)](singularity-detector.md), [Singularity Types Catalog](singularity-types.md)
- **Drives:** (bifurcation-location-recovery, catastrophe-label-alignment — planned experiments)
- **Driven by:** (domain-specification — external input)
- **Math:** [Mathematics.md §Catastrophe Theory](../../Mathematics.md#catastrophe)
- **Open:** (bifurcation-set-validation — open question)
