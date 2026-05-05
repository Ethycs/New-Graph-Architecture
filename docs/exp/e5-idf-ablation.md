# E5 — IDF Rare-Stratum Weighting Ablation

**Cluster:** exp
**Status:** spec
**Tags:** #idf-weighting #rare-events #sample-efficiency

## What

Train the typed-graph classifier with and without IDF (inverse document frequency) weighting on rare task states, rare transitions, rare singularities, and rare failure modes. Measure: overall accuracy, rare-state recall, rare-failure recall, and false-alarm rate. IDF weight is log(N + α) / (nλ + α) applied per-sample in the training loss. Hypothesis: IDF improves recall on rare types without collapsing overall accuracy.

## Why

Rare states and transitions (e.g., uncommon task orderings, edge-case grammar rules) are both harder to learn and more valuable to get right. IDF weighting increases gradients for rare samples, forcing the classifier to allocate capacity to them. Without IDF, the model may fit well on common cases but fail on rare corners. This tests whether stratified re-weighting improves robustness and recall on edge cases, supporting the claim that IDF is a load-bearing component.

## Interface

**Reads:**
- task traces from E1 or E2
- state frequency distribution (for IDF computation)
- task-type and transition-type labels
- pre-trained baseline classifier

**Writes:**
- [`metrics.jsonl`](../drivers/metrics-jsonl.md): accuracy, rare_state_recall, rare_failure_recall, false_alarm_rate (with and without IDF)
- [`results.jsonl`](../drivers/results-jsonl.md): per-sample IDF weight, loss contribution, classification outcome
- comparison table: performance delta (with_idf - without_idf) by metric

## Build steps

1. Compute state frequency from training traces; IDF weight for each state idf(λ) = log(N + 1) / (nλ + 1).
2. Train baseline classifier on all traces with uniform weighting.
3. Train IDF-weighted classifier: loss_per_sample = idf(state_λ) · cross_entropy(pred, label).
4. Evaluate both on held-out test set; compute accuracy, precision, recall.
5. For rare states (bottom 10% frequency), measure recall separately; compare.
6. For rare-failure cases (singularity labels), measure failure-recall and false-alarm rate.
7. Emit metrics and comparison table.

## Links

- **See also:** [E1 — Synthetic BabyAI Grid Run](./e1-synthetic-babyai.md), [E2 — Real BabyAI / MiniGrid](./e2-real-babyai.md), [E4 — Singularity Detector Validation](./e4-singularity-auroc.md)
- **Drives:** validates IDF as load-bearing for rare-event handling
- **Driven by:** (task-trace-loader — external)
- **Math:** idf(λ) = log((N + α) / (nλ + α)); L_weighted = Σᵢ idf(λᵢ) · ℓ(f(xᵢ), λᵢ)
- **Open:** [How are IDF weights updated at runtime?](../open/q03-idf-runtime-schedule.md), (rare-stratum-definition — open question)
