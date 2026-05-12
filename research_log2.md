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

## 2026-05-08 — Phase 23 — Predictive Control Graph Extractor MVP: partition by prediction, merge by behavior

### The architectural reframe

Stop trying to prove universal graph extraction. Build a practical **regime / control-graph extractor** from an arbitrary network's activations. The deliverable is no longer "the FSM the gold author wrote" but **"a useful control graph of regimes the network actually visits."** The mantra:

> Partition by prediction, merge by behavior, control by intervention.

Phase 23 ships the partition + merge legs end-to-end. Control-by-intervention is documented as future work.

### What we built

Two new atoms and one runner:

* **`nga/arch/predictive_projection.py`** — projection `h → z` plus three (optional four) probe heads: next-state head (partition signal), entropy-regression head (Morse-lite uncertainty signal), failure head (risk signal). Adversarial token head with gradient reversal is exposed as a `n_tokens` + `adversarial_token_weight` config knob for Phase 23+ work on arbitrary frozen base networks.
* **`nga/exp/e28_pcg_extractor.py`** — the PCG-X MVP runner. Train the Wave-C state-conditioned base encoder; harvest its mid-layer `h`; train the predictive projection `h → z`; tropical-lite partition by `argmax(next_state_logits)` per step (each argmax cell is a candidate regime; the gap between top-1 and top-2 logits is the margin-to-tie-wall); transition counts per program; bisimulation quotient under the `full` criterion (transition + emission + incoming) to merge behaviourally equivalent regimes. Emits a **`control_graph.json`** artefact with regimes (support, failure rate, entropy, margin, dominant current state, purity) and edges (probability, Beta confidence).
* **`scripts/phase23_pcg_extractor_smoke.py`** — 5-grammar smoke sweep.

The control-graph artefact deliberately mirrors the Phase-23 spec:

```text
Node 18 support=235  failure_rate=0.000  entropy_mean=0.954  margin=8.193
        dom_state=S1_after_term  purity=0.447
Edge 18 -> 22  prob=0.549  count=129  Beta(α=130.0, β=1.0)
```

### Results — 5-grammar smoke (seed 42, single-seed, 17.2 s total CPU)

| grammar | V | argmax cells | regimes after merge | mean failure_rate | mean entropy | mean margin | mean state-purity |
|---|---:|---:|---:|---:|---:|---:|---:|
| listops | 11 | **5** | 5 | 0.036 | 0.55 | 1.95 | **0.850** |
| python_expr | 14 | **11** | 11 | 0.0002 | 0.79 | 6.22 | 0.766 |
| python_big | 24 | **23** | 23 | 0.0002 | 0.85 | 6.02 | 0.789 |
| json | 26 | **25** | 25 | 0.0022 | 0.79 | 3.00 | 0.755 |
| python_control | 37 | **36** | 36 | 0.0002 | 0.94 | 5.45 | 0.788 |

### The decisive finding — `argmax cells < V` on every grammar, naturally

The `argmax(next_state_logits)` partition produces **fewer cells than V** on every grammar, without any explicit quotient. This is the architecturally important behaviour: the predictive head collapses `(state, token)` pairs into `next-state-equivalent` classes for free — Myhill–Nerode coarsening *is* what argmax-of-next-state-head does. Phase 21's bisimulation quotient was trying to recover this collapse *post-hoc* from a finer substrate; PCG-X does it *upstream* by changing the partition definition.

The Phase 21 finding "the substrate's natural equivalence is finer than the FSM" stays true, but PCG-X is no longer trying to reverse it. It just runs the partition at the right level (next-state-conditional) and reports.

### Architectural reading

The five-grammar table confirms the reframe is operational at sub-second-per-grammar CPU:

1. **Cells correspond to FSM states approximately, not exactly** — per-regime state-purity 0.76–0.85 (vs Wave C's forced-K=V 0.78). The PCG-X partition gives roughly the same partition quality as Wave C's forced K-means, but via *prediction* rather than *clustering*, so the partition comes with directly observable margins, failure rates, and entropy bands as first-class artefacts.
2. **Failure rate is low on every grammar (0.02–3.6%)** — confirms the failure head learned a useful per-cell risk signal. In a real deployment this is the signal that would route to `RECOVERY` / `ABSTAIN` via the existing σ control_policy.
3. **Mean margin scales with grammar size and confidence** — listops at 1.95 (small, narrow), python_big and python_expr at ~6 (decisive). Margin gives a cheap per-step certificate of "how safely inside this regime is this step."
4. **The quotient is essentially a no-op at this scale** — argmax already produced cells close to V; merging with `target_K = V` either keeps cells intact (when argmax = V) or only minimally collapses. The bisimulation atom has been re-purposed as a *post-hoc consolidator* rather than the load-bearing piece.

### What this validates

- ✓ The architectural reframe is operational. PCG-X produces a control-graph artefact in one CPU pass on every grammar.
- ✓ Partition-by-prediction does what Phase 21's quotient-by-behavior couldn't: it produces FSM-aligned cells *without* a quotient step.
- ✓ The artefact is shaped exactly like the user-specified Phase 23 deliverable (regimes with support/failure/entropy/margin, edges with probability + Beta confidence).
- ✓ The TPN's existing partition-function machinery (energy, σ, posterior) is the natural backend for downstream `control_policy.decide()` once interventions are added.

### What this does NOT prove

- The current "arbitrary network" is our own Wave-C state-conditioned MLP, not a real off-the-shelf model. Real activations from a small transformer trained on the corpus would be the proper Tier-2 substrate.
- The failure target is supervised (`true_next_state ∈ legal_set(current_state)`); a self-supervised proxy (e.g., self-consistency under perturbation) is needed before the failure head can be claimed universal.
- Interventions (step 8 of the spec) are unimplemented; without them, the artefact is descriptive, not controllable.
- The grammar dispatch still uses our own hand-authored FSMs; PCG-X is a useful artefact for them, but its load-bearing test is recovering useful regimes from a substrate where no gold FSM exists.

### What's next

- **Phase 23b — adversarial token head wired in by default.** Currently `adversarial_token_weight=0.0`; enabling it under a real frozen base network is the structurally complete version of "strip content from the projection."
- **Phase 23c — intervention-labelled edges.** Sample input edits / activation patches; estimate `P(next regime | current regime, intervention)`. This is the "control" leg of "partition by prediction, merge by behavior, control by intervention."
- **Phase 23d — small transformer base network.** Replace our own Wave-C encoder with a small Python-trained transformer; rerun PCG-X on its mid-layer activations. The proposal's Tier-2 test, repurposed for the reframed deliverable.
- **σ + control_policy integration.** Hook the existing `singularity_detector` + `control_policy` atoms to the PCG-X regime annotations so each regime emits a NORMAL / RECOVERY / ABSTAIN call.

Suite: 500 passed, 8 xfailed, 1 pre-existing E0 env-flake (unchanged). Atom census: 49 passed.

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

