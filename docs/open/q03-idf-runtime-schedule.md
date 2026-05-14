# How are IDF weights updated at runtime?

**Cluster:** open
**Status:** open
**Tags:** #idf #rare-events #online-learning #weight-schedule

## What

For a given task $\lambda$ or transition, the IDF weight is $\mathrm{idf}(\lambda) = \log\frac{N + \alpha}{n_\lambda + \alpha}$, where $n_\lambda$ is the frequency of $\lambda$ in training data.

At inference or online training, is $n_\lambda$ fixed (from training), updated via exponential moving average, or recomputed per episode/epoch? When do rare events become less rare?

## Why

IDF weighting is designed to highlight rare-failure modes and rare-grammar states during training ([E5 — IDF Rare-Stratum Weighting Ablation](../exp/e5-idf-ablation.md)). But at runtime:
- Fixed IDF may over-weight stale rare events.
- EMA tracks recent drift (useful for covariate shift).
- Per-epoch recompute is expensive but adaptive.

The schedule affects learning stability, rare-event recall, and whether [Singularity Detector σ(x)](../arch/singularity/singularity-detector.md) stays aligned with task semantics as the agent encounters new regimes.

## Interface

**Affected zettels:** (idf-weighting — arch component), [Energy-Weighted Loss](../arch/energy/energy-weighted-loss.md), [E5 — IDF Rare-Stratum Weighting Ablation](../exp/e5-idf-ablation.md), [E7 — Reservoir Readout vs End-to-End](../exp/e7-reservoir-vs-end2end.md)

**Decision criteria:**
- Measure rare-state recall and false-alarm rate under distribution shift.
- Compare: fixed-idf vs. EMA(τ) vs. per-epoch recompute.
- Check stability: does frequent recompute cause training oscillation?
- Target: rare-recall loss < 5%, overall accuracy maintained.

## Build steps

- Implement fixed, EMA, and per-epoch variants.
- Create distribution-shift benchmark: train on uniform grammar, test on biased grammar (e.g., rare states 3× more frequent).
- Measure: recall on rare states, precision on common states, total accuracy, training variance.
- Fit EMA timescale $\tau$ (e.g., 0.95, 0.99) to minimize rare-recall loss.
- If fixed idf degrades after shift, adopt EMA or adaptive per-epoch schedule.

## Links

- **See also:** [What defines "successful transfer" in E8?](./q05-transfer-success-criterion.md), [When does an orbit re-expand from quotient form?](./q08-quotient-reexpansion-threshold.md)
- **Affects:** (idf-weighting — arch component), [Energy-Weighted Loss](../arch/energy/energy-weighted-loss.md), [E5 — IDF Rare-Stratum Weighting Ablation](../exp/e5-idf-ablation.md)
- **Math:** [Experiments.md §E5-IDF-Rare-Stratum-Weighting](../Experiments.md#e5--idf-rare-stratum-weighting)
