# Singularity Detector σ(x)

**Cluster:** arch
**Status:** spec
**Tags:** #detection #anomaly #geometry

## What
Combined anomaly score $\sigma(x) \in [0, 1]$ that aggregates five independent signals—low margin, contradiction, loop pressure, illegal-pressure, stabilizer-jump—into a single confidence that observation $x$ lies at a singularity. Each signal is normalized and weighted; high $\sigma$ predicts error or bifurcation.

## Why
No single feature reliably detects all singularities. Low margin catches cases where the state is equidistant from multiple prototypes. Contradiction flags loops or cycles in the logic. Loop pressure and illegal pressure detect constraint violations. Stabilizer-jump captures phase transitions. By ensemble voting, the detector becomes robust to individual feature failures and noise, improving AUROC on unseen tasks.

## Interface
**Inputs:**
- Observation $\mathbf{o} \in \mathbb{H}^d$.
- Node type $\tau(v)$ and prototype $\mathbf{p}_v$.
- Loop/contradiction labels (from graph analysis).
- Illegal-pressure and stabilizer-jump signals (from domain rules).

**Outputs:**
- Score $\sigma(x) \in [0, 1]$.
- Per-signal decomposition $[\sigma_{\text{margin}}, \sigma_{\text{contra}}, \sigma_{\text{loop}}, \sigma_{\text{illegal}}, \sigma_{\text{stabilizer}}]$.
- Predicted label (singularity yes/no at threshold $\theta$).

## Build steps
- Compute margin $m = D_{\text{hyp}}(\mathbf{o}, \mathbf{p}_v) - D_{\text{hyp}}(\mathbf{o}, \mathbf{p}_{v'})$ to nearest other type. $\sigma_{\text{margin}} = \sigmoid(-m / \tau_{\text{margin}})$.
- Look up node's contradiction flags from [Singularity Types Catalog](./singularity-types.md) (e.g., contradicts parent constraint). $\sigma_{\text{contra}} = \text{normalize}(\text{count contradictions})$.
- Query loop-pressure signal from graph embedding: $\sigma_{\text{loop}} = \text{loop-distance}/\text{max-distance}$.
- Query illegal-pressure from domain rules (e.g., type-incompatible transition). $\sigma_{\text{illegal}} = \text{rule-violation-count} / \text{total-rules}$.
- Detect stabilizer-jump: compare gradient norm or embedding change rate to baseline. $\sigma_{\text{stabilizer}} = \sigmoid(\text{rate} / \tau_{\text{stab}})$.
- Aggregate: $\sigma(x) = w_m \sigma_{\text{margin}} + w_c \sigma_{\text{contra}} + w_l \sigma_{\text{loop}} + w_i \sigma_{\text{illegal}} + w_s \sigma_{\text{stabilizer}}$, normalized by $\sum w = 1$.

## Links
- **See also:** [Singularity Types Catalog](./singularity-types.md), [Behavioral Stratum Tagger](./behavioral-stratum-tagger.md), [Catastrophe Labels](./catastrophe-labels.md)
- **Drives:** [Failure-vs-Margin AUROC](./failure-margin-auroc.md), (singularity-detection-recall — planned experiment)
- **Driven by:** (observation-log, domain-rules — external inputs)
- **Math:** [Mathematics.md §Bifurcation & Catastrophe](../Mathematics.md#catastrophe)
- **Open:** (signal-weighting-tuning, threshold-calibration — open questions)
