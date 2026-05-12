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

## 2026-05-08 — Phase 23d — transformer substrate: PCG-X is operational; the Phase 23b prediction is falsified

### What we built

* **`src/nga/arch/small_transformer.py`** — bare causal transformer LM (token embed + learned positional embed + `nn.TransformerEncoderLayer` ×n, gelu, norm-first, causal mask, linear LM head). `encode(x, harvest_layer=k)` returns `(final_hidden, harvested_layer_k_output)` for mid-layer harvesting. `train_small_transformer` trains via causal next-token cross-entropy, one program at a time.
* **`src/nga/exp/e29_pcg_extractor_transformer.py`** — the Phase 23d runner. Builds a per-grammar token vocab from `s.observed_token`, groups samples by `program_id`, trains a `d_model=64`, `n_heads=4`, `n_layers=2` transformer on the token sequences via causal LM. Harvests **layer-0 activations** aligned with the original sample ordering (one `(d_model,)` per (program, step)), then runs the standard PCG-X pipeline (predictive projection → argmax partition → bisimulation quotient).
* **`scripts/phase23d_transformer_pcg_sweep.py`** — 5-grammar × `adversarial_token_weight ∈ {0.0, 1.0}` comparison.

The substrate is now a real arbitrary-network: no `current_state` injected into the input, no per-step engineered features, no Wave-C state-conditioning. The transformer must learn FSM state implicitly from token context. This was the proposal's Tier-2 universality test, repurposed for the PCG-X deliverable.

### Results — 5-grammar sweep (seed 42, 79.7 s total CPU)

| grammar | V | adv_w | argmax cells | mean state-purity | mean failure | mean margin | xfm next-token acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| listops | 11 | 0.0 | 5 | **0.820** | 0.020 | 2.20 | 0.859 |
| listops | 11 | 1.0 | 4 | 0.745 | 0.024 | 1.93 | 0.859 |
| python_expr | 14 | 0.0 | 11 | **0.674** | 0.0003 | 2.63 | 0.612 |
| python_expr | 14 | 1.0 | 9 | 0.561 | 0.003 | 1.63 | 0.612 |
| python_big | 24 | 0.0 | 23 | **0.603** | 0.0005 | 1.62 | 0.600 |
| python_big | 24 | 1.0 | 22 | 0.456 | 0.0003 | 0.81 | 0.600 |
| json | 26 | 0.0 | 24 | **0.601** | 0.002 | 1.44 | 0.595 |
| json | 26 | 1.0 | 23 | 0.577 | 0.010 | 0.87 | 0.595 |
| python_control | 37 | 0.0 | 36 | **0.618** | 0.0008 | 1.49 | 0.718 |
| python_control | 37 | 1.0 | 34 | 0.512 | 0.0000 | 0.88 | 0.718 |

**Aggregate means:**

| adv_w | mean state-purity | mean argmax cells | mean margin | mean failure |
|---:|---:|---:|---:|---:|
| 0.0 (no adv) | **0.6632** | 19.8 | 1.877 | 0.0046 |
| 1.0 (adv on) | 0.5700 | 18.4 | 1.225 | 0.0076 |

### The two architectural findings

**(1) PCG-X is operational on a real arbitrary-network substrate.** The full pipeline runs end-to-end on 5 grammars in 80 seconds CPU. Layer-0 transformer activations carry enough FSM-relevant structure that argmax-of-next-state partitioning produces between 5 (listops) and 36 (python_control) cells with non-trivial state-purity (mean 0.66, range 0.60-0.82). The transformer's own next-token accuracy is only 0.60-0.86 at this scale (50K-param model, 12 training epochs, ≤2000 tokens per grammar), so the substrate is *weakly trained* compared to Wave-C's 0.97. **PCG-X still extracts a useful regime graph from this weakly-trained substrate.**

**(2) The Phase 23b prediction is falsified.** I predicted that the adversarial token head, which was net-negative on the state-conditioned MLP substrate, would be net-positive on the transformer substrate because the transformer has no state-conditioning and its activations carry heavy token-level surface variance. **The prediction was wrong.** Mean state-purity drops by 9pp with `adv_w=1.0` (0.663 → 0.570); mean margin shrinks 35%; *every grammar* loses purity. The adversarial head consistently hurts.

### Why the prediction failed (the post-mortem)

Three plausible reasons, each diagnostic:

1. **The transformer's activation space encodes state THROUGH token-context, not despite it.** Mid-layer activations of a causal LM are built to predict the next token from the full prefix; FSM state is the slow component of "what predicts the next token." Stripping the fast component (current-token content) doesn't just remove nuisance — it removes part of how state is encoded. The adversarial signal competes with the next-state-prediction signal *inside the same `z` projection*, so they pull against each other.
2. **The transformer is weakly trained at this scale.** Next-token accuracy 0.60-0.72 on the big grammars means the substrate's state signal isn't yet sharp; adversarial pressure has a smaller "real" target to preserve and can disrupt the projection more than at maturity.
3. **The adversarial-on-z design is too coarse.** The adversarial loss flows through the same `z` the predictive projection uses. A cleaner design routes the adversarial head through a *separate* branch that taps `z` but doesn't push back through it — preserving the predictive content while still measuring (and possibly using) the token-strippability signal as a diagnostic, not a training pressure.

Combined with Phase 23b's null on the state-conditioned MLP, the **honest verdict on the adversarial token head is: it does not help PCG-X on either substrate tested at this scale and configuration**. The mechanism is correct (token accuracy drops sharply when the head is on); the placement is wrong. A "diagnostic-only" version (measure but don't backprop) or a "weak adversarial weight + larger projection capacity" version are both worth trying, but the simplest take-away is to **leave `adversarial_token_weight=0.0` by default**.

### What this validates and what it refutes

- ✓ PCG-X works on a real causal-transformer substrate trained from scratch on the corpus. Sub-2-minute end-to-end CPU on 5 grammars.
- ✓ The control-graph artefact (regime nodes with support/failure/entropy/margin/dom_state/purity; edges with Beta confidence) is shaped identically across the MLP and transformer substrates.
- ✗ The Phase 23b prediction is falsified: stripping token content does not earn its keep on the transformer substrate at this scale.
- → The cleanest PCG-X configuration to ship is `adversarial_token_weight=0.0` (the Phase 23 MVP default). The adversarial head stays in the atom as a diagnostic switch and a future-research knob.
- ↘ Substrate quality is the binding constraint, not the post-substrate machinery. The Wave-C MLP at 0.97 next-token accuracy gives purity 0.79; the small transformer at 0.63 next-token accuracy gives purity 0.66. The PCG-X stack faithfully reflects substrate quality without inflating it.

### What's next

- **Phase 23c (control / interventions).** With the adversarial head fully evaluated, this is the next leg of the mantra. Sample input edits per regime, estimate `P(next regime | current regime, intervention)`.
- **σ + control_policy integration.** Hook the existing TPN routing atoms to the regime annotations PCG-X already emits.
- **Bigger transformer at the same scale of corpus.** If the substrate-quality bottleneck is real, a 4-layer 128-dim transformer should bring next-token accuracy up and purity correspondingly. Quick experiment, but it's a substrate-quality scan, not a load-bearing architectural question.

Suite: 500 passed, 8 xfailed, 1 pre-existing E0 env-flake (unchanged).

## 2026-05-08 — Phase 23b — adversarial token head: mechanism verified, null-to-negative on the state-conditioned substrate

### What we built

Wired the existing `PredictiveProjection` adversarial token head through `run_e28`: built a per-grammar token vocab from `s.observed_token`, plumbed `n_tokens` + `adversarial_token_weight` into the projection config, passed `token_ids` to `train_predictive_projection`. New sweep script `scripts/phase23b_adversarial_token_sweep.py` runs each grammar three times — `adversarial_token_weight ∈ {0.0, 0.5, 1.0}` — keeping every other knob fixed.

### Results — 5-grammar sweep (seed 42, 41 s total CPU)

| grammar | V | adv_w | argmax cells | mean state-purity | mean failure | mean margin | token_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| listops | 11 | 0.0 | 5 | **0.850** | 0.036 | 1.95 | — |
| listops | 11 | 0.5 | 5 | 0.748 | 0.014 | 1.77 | 0.104 |
| listops | 11 | 1.0 | 5 | 0.694 | 0.013 | 1.60 | **0.056** (below chance) |
| python_expr | 14 | 0.0 | 11 | 0.766 | 0.0002 | 6.22 | — |
| python_expr | 14 | 0.5 | 11 | 0.767 | 0.0000 | 4.32 | 0.275 |
| python_expr | 14 | 1.0 | 11 | 0.759 | 0.0000 | 4.67 | 0.364 |
| python_big | 24 | 0.0 | 23 | 0.789 | 0.0002 | 6.02 | — |
| python_big | 24 | 0.5 | 23 | 0.788 | 0.0000 | 5.44 | 0.486 |
| python_big | 24 | 1.0 | 23 | **0.790** | 0.0000 | 4.08 | 0.370 |
| json | 26 | 0.0 | 25 | 0.755 | 0.0022 | 3.00 | — |
| json | 26 | 0.5 | 25 | 0.773 | 0.0016 | 2.46 | 0.605 |
| json | 26 | 1.0 | 25 | **0.779** | 0.0000 | 1.70 | 0.296 |
| python_control | 37 | 0.0 | 36 | 0.788 | 0.0002 | 5.45 | — |
| python_control | 37 | 0.5 | 36 | 0.776 | 0.0002 | 2.63 | 0.608 |
| python_control | 37 | 1.0 | 36 | 0.768 | 0.0001 | 2.81 | 0.614 |

**Aggregate means:**

| adv_w | mean purity | mean argmax cells | mean failure | mean margin |
|---:|---:|---:|---:|---:|
| 0.0 (Phase 23 control) | **0.7897** | 20.0 | 0.0078 | **4.529** |
| 0.5 | 0.7703 | 20.0 | 0.0031 | 3.326 |
| 1.0 | 0.7581 | 20.0 | 0.0027 | 2.971 |

### Architectural reading

**Mechanism: verified.** Token accuracy under `adv_w=1.0` drops well below chance on listops (`tok_acc=0.056` vs `1/n_tokens ≈ 0.06`; encoder is actively scrambling token signal), and stays well below "without adversarial signal" baseline on every other grammar. The gradient-reversal layer is doing the work it was designed for.

**Effect on our substrate: null-to-negative.** Mean state-purity drops by 3pp (0.790 → 0.758) and mean margin shrinks 35% (4.5 → 3.0). The number of argmax cells is unchanged (the partition shape isn't altered by stripping token info from `z`); failure rate drops slightly because confident-wrong predictions are crowded out by less-confident predictions overall.

**Per-grammar split: token info matters differently per grammar.** Listops loses 16pp of purity — its FSM transitions are token-discriminative (different ops, brackets, ints have different state implications), and stripping token info costs the model real discriminative power. JSON gains a few pp — its state structure is largely token-independent (a value-position is a value-position whether the content is a number, string, or object), so stripping nuisance content actually helps. Python expr / big / control sit between, near-flat.

**The clean architectural finding: the adversarial token head is the *wrong tool* for our current substrate.** The Wave-C / E26 encoder concatenates `current_state` one-hot to its input by design, so the encoder already has access to the FSM state at every step. Token information is **complementary disambiguator**, not nuisance: given (`state=S1_after_term`, `token=NAME`) vs (`state=S1_after_term`, `token=NUMBER`), the next-state distributions genuinely differ. Stripping token info forces the projection to pick a single next-state distribution per state, losing the conditional structure the encoder learned.

**Where the adversarial head IS load-bearing: the arbitrary-frozen-network case (Phase 23d).** A frozen transformer has no `current_state` injected; the only thing identifying state in its activations is the contextual aggregation of past tokens. There, **stripping token-level surface information from `z` is exactly the right move** — content variance dominates, and the state signal is the lower-frequency component buried under it. Phase 23b's null-result on our state-conditioned substrate is the architecturally correct outcome for a substrate that already has the state signal. The same head should help meaningfully on Phase 23d.

### What this validates / refutes

- ✓ The adversarial head implementation is correct (gradient reversal verifiably scrambles token signal).
- ✓ The diagnostic stack distinguishes "head works" (token_acc drops) from "head helps" (purity rises) cleanly — both are reported and answer different questions.
- ✗ The adversarial head does NOT improve PCG-X on the state-conditioned MLP substrate. Net effect on purity is −3pp at adv_w=1.0; margin shrinks materially.
- → Predicted to help on Phase 23d (frozen transformer, no state-conditioning). Cheap to verify once the transformer substrate lands.

### What's next

- **Phase 23c — intervention-labelled edges.** Sample input edits per regime; estimate `P(next regime | current regime, intervention)`. This is the "control" leg.
- **Phase 23d — real frozen transformer base network.** Train a small transformer on Python source, no state-conditioning; run PCG-X on its mid-layer activations both with and without the adversarial token head; expect the head to be net-positive there.
- **σ + control_policy integration.** Hook the existing `singularity_detector` + `control_policy` atoms to the PCG-X regime annotations so each regime emits NORMAL / RECOVERY / ABSTAIN.

Suite: 500 passed, 8 xfailed, 1 pre-existing E0 env-flake (unchanged).

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

