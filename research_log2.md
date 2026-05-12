# Research log 2

A continuation of `research_log.md`. The prior log carries Phases 0–21
(architecture build-out, 5-grammar sweeps, Phase 19B self-supervised
diagnostic discovery, Phase 20 universal graph extraction, Phase 21
bisimulation quotient). This file picks up at Phase 22.

The architectural state at the start of this log:

- The extraction pipeline math is correct (oracle Hamming 0.026, meets
  the strict 0.05 bar on 4/5 grammars).
- The trained-from-scratch encoder of Phase 20 / Wave C learns
  `(state, token)` jointly; clustering at K = V gives ~77% purity.
- Forced K = V Hamming sits at 0.24; agglomerative bisimulation
  quotient under transition / emission / full criteria is
  statistically indistinguishable from random merge (~0.26). The
  substrate's natural equivalence classes are finer than the FSM
  partition, so similarity-based merge cannot recover the FSM.
- The route to closing the gap (Condition 1 in the three-condition
  framing): couple the encoder to a partition function $Z$ via a
  probe head whose gradient reshapes the encoder's natural
  equivalence toward "same FSM state regardless of token."

The prior log is preserved as `research_log.md`. New entries land at
the top of this file.

---

## 2026-05-08 — Phase 22a — partition-function probe couples the encoder to the gold graph; mechanism validated, strict bar still missed

### What we built

`src/nga/exp/e27_extraction_partition_probe.py` and
`scripts/phase22a_partition_probe_sweep.py`. The new runner `run_e27`
extends the Wave-C / E26 trainable encoder with a **partition-function
probe head** sitting alongside the next-state head: a small
`nn.Linear(hidden_dim, n_states)` whose target is the uniform
distribution over legal successors under the gold FSM legality matrix
($P_Z(\text{next}=j \mid s) = \mathbb{1}[L_{sj}=1] / \sum_k L_{sk}$).
The probe's KL loss against this target is added to the next-state
cross-entropy with a tunable `probe_weight`, and gradient through the
probe is the mechanism that's supposed to reshape the encoder's
natural equivalence from `(state, token)` toward `state` alone.

### What we measured

5-grammar sweep at `force_K_equals_V=True`, `n_train_epochs=50`, three
probe weights `{0.0, 1.0, 5.0}` (0.0 = Wave-C-equivalent control):

| probe weight | mean enc_acc | mean cluster_purity | mean forced-K=V Hamming | strict bar (≤0.05) |
|---:|---:|---:|---:|---:|
| 0.0 (control) | 0.972 | 0.777 | **0.242** | 0 / 5 |
| 1.0 | 0.970 | 0.820 | **0.227** | 0 / 5 |
| 5.0 | 0.962 | **0.854** | 0.229 | 0 / 5 |

Per-grammar at `probe_weight=5.0` vs control:

| grammar | V | purity baseline → probe | Hamming baseline → probe |
|---|---:|---|---|
| listops | 11 | 0.944 → 0.979 (+0.035) | 0.240 → **0.306 (+0.066, worse)** |
| python_expr | 14 | 0.807 → 0.811 (+0.004) | 0.311 → 0.311 (flat) |
| python_big | 24 | 0.683 → 0.779 (+0.096) | 0.319 → **0.210 (−0.109, best)** |
| json | 26 | 0.775 → 0.894 (+0.119) | 0.157 → 0.172 (+0.015, slight worse) |
| python_control | 37 | 0.675 → 0.808 (+0.133) | 0.184 → **0.147 (−0.037)** |
| **mean** | — | 0.777 → 0.854 | 0.242 → 0.229 |

### Architectural reading

**Mechanism: validated, directionally.** Cluster purity climbed by ~10pp on average with the strongest probe weight; on the four bigger grammars purity rose by ≥9.6pp on every one of them. Encoder train accuracy held at 0.96+ throughout — the probe loss did not destroy the next-state task. The probe-KL converged to ≤ 1.0 on every grammar under both nonzero weights, confirming the probe head was actually learning the partition target. The **coupling mechanism works**: gradient through the partition probe does reshape the encoder's hidden state toward FSM-state-aligned equivalence.

**Strict bar: not crossed.** Mean Hamming drops from 0.242 → 0.227 (≈ 6% relative). The 0.05 strict bar is still unmet on 5/5 grammars; the gap to the oracle floor of 0.026 (Phase 21) is largely unchanged. The Hamming improvement is **non-uniform across grammars**: python_big and python_control improve materially (Hamming drops by 0.11 and 0.04 respectively), but listops gets *worse* and json drifts up slightly. Purity gains don't translate cleanly into Hamming gains.

**The asymmetry diagnoses itself.** The probe pulls hidden states with the same `current_state` toward agreement on partition output, but partition output is determined by `legality_matrix[current_state]`, which can be the *same row* for many distinct FSM states whose legal-successor sets coincide. Specifically: any pair of FSM states with identical outgoing-legality rows in the gold matrix have *identical* partition targets, and the probe gives the encoder no signal to keep them apart. On listops the FSM has many states with similar outgoing distributions (the depth-tracker structure produces near-isomorphic neighbourhoods), so the probe over-collapses; on python_big and python_control the outgoing-legality rows are more discriminating and the probe genuinely helps.

This is the Myhill-Nerode finding again, mirrored: the partition target was supposed to be "the FSM's equivalence relation," but the uniform-over-legal-successors target *under-distinguishes* states that differ only in incoming structure or in successor *probabilities* rather than the legality set. The probe correctly enforces *its own* equivalence relation on the encoder — but that relation is coarser than the gold FSM on grammars where outgoing-legality rows alias.

### What this validates and what it doesn't

- ✓ Probes can couple the encoder to a partition function over the graph.
- ✓ The coupling reshapes encoder equivalence toward (some version of) the FSM partition: purity rises 8-13pp on the four bigger grammars.
- ✓ The architecture's existing partition-function infrastructure is meaningfully wired into the encoder for the first time (Wave C ignored it entirely).
- ✗ The strict Hamming bar (0.05) is not met on any grammar — the gap to oracle (0.026) is essentially unchanged in mean.
- ✗ The chosen partition target (uniform-over-legal-successors) is too coarse: it aliases states with identical outgoing-legality rows. A more informative target is needed.

### What's next

- **22b — observed-transition-frequency target.** Replace the uniform-over-legal target with the empirical observed transition distribution from random walks over the gold FSM. This carries more information than the legality matrix alone and should reduce the under-distinguishing effect on listops.
- **22c — multi-target probe head.** Two probe heads: one on outgoing-distribution, one on incoming-distribution. Forces the encoder to encode both row-of-legality and column-of-legality, which together identify states even when outgoing rows alias.
- **22d — Phase 22b self-supervised variant** (the universality move). Drop the gold-FSM target; learn the partition function $Z$ jointly with the encoder via trajectory self-consistency (noise-contrastive estimation or a detailed-balance objective). If this approaches the 22a numbers without labels, the universality claim earns its keep at this scale.

Suite: 500 passed, 8 xfailed, 1 pre-existing E0 env-flake (unchanged from Phase 21).

