# When does an orbit re-expand from quotient form?

**Cluster:** open
**Status:** open
**Tags:** #quotient #reexpansion #idf #compression

## What

In [Orbit Quotient Space](../arch/orbit-quotient-space.md), object orbits are compressed into single representatives during attention, then optionally re-expanded for downstream processing. When should an orbit re-expand from compressed quotient form back to full: at a fixed IDF threshold, via learned decision, or triggered by detected singularities?

## Why

Re-expansion is expensive (breaks compression), but necessary when:
- A rare object in the orbit becomes task-critical.
- The orbit contains a singularity or boundary case.
- Accumulated representational error exceeds tolerance.

The schedule affects compression efficiency and whether the system gracefully scales representation complexity. Misaligned re-expansion can waste compute or miss rare cases.

## Interface

**Affected zettels:** [Orbit Quotient Space](../arch/orbit-quotient-space.md), (idf-weighting — arch component), [Singularity Detector σ(x)](../arch/singularity-detector.md), [E6 — Group-Quotient Attention](../exp/e6-group-quotient-attention.md)

**Decision criteria:**
- Measure: re-expansion frequency vs. task performance gain.
- Compare: fixed IDF threshold vs. learned policy vs. singularity-triggered.
- Check: does re-expansion recover compressed-error cost?
- Target: re-expand < 10% of orbits, accuracy loss < 2 points.

## Build steps

- Implement three re-expansion strategies: (A) IDF($\lambda$) > threshold, (B) learned binary classifier (expand or not), (C) expand if $\sigma(x) > \sigma_{\mathrm{crit}}$.
- Run [E6 — Group-Quotient Attention](../exp/e6-group-quotient-attention.md) with each.
- Measure: re-expansion rate, accuracy recovered, total attention FLOPs, latency.
- Fit threshold / learned weights on validation set.
- If A (IDF-threshold) achieves target efficiently, lock it; else explore B or C.

## Links

- **See also:** [How are IDF weights updated at runtime?](./q03-idf-runtime-schedule.md), [Is the group action H specified or discovered?](./q06-group-action-discovery.md)
- **Affects:** [Orbit Quotient Space](../arch/orbit-quotient-space.md), (idf-weighting — arch component), [E6 — Group-Quotient Attention](../exp/e6-group-quotient-attention.md)
- **Math:** [Architecture.md §Quotients, Monodromy, Boundary Memory](../Architecture.md#quotients-monodromy-boundary-memory)
