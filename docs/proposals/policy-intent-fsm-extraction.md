# Policy-Intent FSM Extraction: Audit-by-Construction for an Externally-Enforced LLM State Machine

**Status:** proposed
**Phase target:** Phase 28b (the first real-task PCG-X deployment outside synthetic grammars)
**Date posted:** 2026-05-16
**Author:** TPN research team

## Abstract

We propose extracting the policy-intent FSM from a servant-runtime LLM's mid-layer activations via PCG-X. The FSM (v1 = 2-state binary `ALLOWED`/`WITHHELD`; v2 = 4-state `UNESTABLISHED`/`GRANTED`/`WITHHELD`/`CONDITIONAL`) is **externally enforced by prompt construction** — the renderer inserts or omits anchor blocks such as `## Speech permission revoked` — not by the LLM's training. The central scientific question is whether the renderer LLM **internalizes** this externally-imposed state as a stable, low-dimensional representation, or whether it treats the gate as a surface-level prompt feature without forming a coherent internal state. Both answers are useful: yes gives us the first end-to-end audit of "is an LLM actually following an externally-imposed FSM?"; no surfaces that the gate is surface-level (a critical finding for the runtime). Pre-registered acceptance bars are derived from the framework's structural prediction (a $k$-state concept lives in $\leq k - 1$ effective dimensions of margin-gradient space) plus the Phase 27 Step 4 / Step 6b numerical regime. This is the first real-task PCG-X application outside synthetic grammars and directly addresses the largest gap named in the 2026-05-16 realistic-assessment audit.

## Background

The PCG-X pipeline extracts a control graph + σ + calibrated Beta-posterior edges + decision-trace audit log from any homogeneously trained substrate's activations. As of 2026-05-16 the pipeline has been validated on:

- 5 synthetic grammars (listops, python_expr, python_big, json, python_control) — Phases 23, 23b, 23d, 24, 25, 27.
- 2 natural-text concept tasks (binary sentiment, 5-class emotion) — Phase 27 Step 6 / 6b.
- 2 substrates (frozen GPT-2 small, Qwen2.5-1.5B) — Phase 24, Phase 28 GAN-leader scout.

The 2026-05-16 realistic-assessment audit named "no real-task evaluation outside controlled experimental settings" as the largest remaining empirical gap. Phase 27 Step 6/6b made a first dent (hand-constructed emotion corpus) but stopped short of a deployed system where the FSM structure has operational meaning.

The sister project `langgraph_servants` runs a servant-runtime LLM dialogue system gated by an explicit policy-intent FSM. The v1 (currently shipping) and v2 (proposed) FSM specs are documented in `langgraph_servants/docs/policy-intent.md` and `policy-intent-v2.md`. The renderer assembles a system prompt for each generator turn whose volatile zone includes a `## Speech permission revoked` anchor block when (and only when) the FSM state is `WITHHELD` for the active `(source, target, domain)` triple. The LLM never sees the FSM directly; it sees prompt anchors that encode the state. Transitions are decided by a classifier on the LLM's output via the `PolicyModification` reducer.

This proposal tests whether PCG-X applied to the renderer LLM's mid-layer activations recovers the FSM structure that the runtime enforces. The mapping is:

| servant runtime | PCG-X pipeline |
|---|---|
| `(source_id, target_id, domain)` tuple | grammar instance |
| policy state ∈ {ALLOWED, WITHHELD} (v1) / {UNESTABLISHED, GRANTED, WITHHELD, CONDITIONAL} (v2) | FSM vertex |
| classifier output GRANT/REVOKE/NULL (v1) or 5-intent (v2) | `observed_token` analog |
| current_state | `current_state` |
| next_state from reducer | `y_next` |
| renderer-LLM hidden state at the first response-zone token | $h_i$ harvest |
| `PolicyModification` rejected (auth gate or no-op) | regime-OOD / σ → ABSTAIN |

## Central hypothesis

**H₀ (null):** The renderer LLM does not internalize the externally-imposed policy state. PCG-X regime extraction on its mid-layer activations produces clusters that do not separate by gold policy state above the cluster-purity floor expected from token-context noise (≤ 0.40), AND/OR the σ-Krylov top-3 capture on the v2 4-state task is below 0.50.

**H₁ (alternative):** The LLM internalizes the policy state. PCG-X recovers regimes corresponding to {UNESTABLISHED, GRANTED, WITHHELD, CONDITIONAL} with cluster purity > 0.75 AND the σ-Krylov subspace top-3 capture > 0.85 on the v2 task.

H₀ vs H₁ are both substantively useful — see "Sudden large implications" below.

**Pre-registered numerical predictions** (derived from existing Phase 27 measurements and the framework's $k$-class simplex argument):

| task | $V$ | predicted eff_rank(grad) | predicted top-3 capture | predicted cluster purity |
|---|---:|---:|---:|---:|
| v1 binary (ALLOWED / WITHHELD) | 2 | ≈ 1.0 (degenerate) | ≈ 0.99 | ≈ 1.0 (trivial) |
| **v2 four-state** | **4** | **2.5–3.0** | **0.90–0.95** | **≥ 0.75 against gold** |

These predictions sit inside the existing measurement envelope — Step 6b 5-class emotion (V=5) landed at eff_rank=3.94 / top-3=0.816; Step 4 listops (V=11) at eff_rank=3.29 / top-3=0.888. The v2 prediction extrapolates downward in V, in the regime where prior measurements predict crispening.

## What works immediately (no experiment needed)

These items are licensed by composition and do not require validation:

1. **The pipeline is already substrate-agnostic.** `_PretrainedSubstrate` accepts any causal LM via `AutoModelForCausalLM.from_pretrained`. The Phase 28 scout demonstrated this on Qwen2.5-1.5B.
2. **The FSM specification slots into the existing format.** Our `graph_fsm_spec` YAML already represents arbitrary directed graphs with per-edge legality; the policy-intent FSM is structurally one of the simplest grammars we'd run (V ∈ {2, 4} vs V ∈ {11, …, 37}).
3. **The Phase 26 labelled hypergraph is the right data structure.** Each regime carries `named` (policy_state, authority_mode, intent_class, anchor_present) and `residual` (whatever else the LLM's representation encodes), which is exactly the auditability commitment the runtime needs.
4. **σ + control_policy is the right control surface.** Phase 23e wired NORMAL/RECOVERY/ABSTAIN per step on the regime graph. For the runtime, ABSTAIN ↔ "the LLM's predicted policy state disagrees with the renderer-enforced state" — a deployment-grade audit signal.

## What requires experimental validation

1. Whether the renderer LLM's mid-layer activations encode the policy state at all (the H₀ vs H₁ question).
2. The actual harvest layer — Phase 25 found L10/12 was the peak for grammar-state extraction in GPT-2 small; the right layer for a different LLM running policy-FSM-gated dialogue is unknown a priori.
3. Whether v1's `True`-default-GRANTED is treated by the LLM as semantically equivalent to explicit-GRANTED (the schema-migration audit signal described under "Sudden large implications").
4. Whether the anchor-block ablation (replay traces with `## Speech permission revoked` removed) changes the regime structure — this is the falsifiable test of "is the LLM responding to the anchor, or to surface lexical patterns?"
5. The corpus-size requirements. Phase 27 Step 6b worked at 100 sentences; the servant runtime can produce many more, but state-conditional transitions may be unevenly distributed.

## Methods

The deliverable is four items. Three are project-side additions; one is the experiment runner.

### 1. `src/nga/exp/policy_dataset_adapter.py`

Reads servant-runtime decision traces (or generates synthetic ones from the v1 reducer) and emits the canonical `(observed_token, current_state, y_next, sample_index)` tuples that `GRAMMAR_DISPATCH` consumes. Two backends:

- **Synthetic (Wave-A sanity):** generates random traces under the v1 reducer (uniform GRANT_EVENT / REVOKE_EVENT / NULL inputs with random authority-gate accept/reject), produces ~10⁴ tuples per (source, target, domain).
- **Recorded (Wave-B real data):** reads `langgraph_servants` `audit_log.jsonl` (when the runtime team plumbs it through) and lifts state transitions one-to-one. Tokens are the dialogue turn's response-zone first-token (or a configurable harvest position).

Schema additions to `decision_trace.jsonl` v1.2 (Phase 26 already extensible):

```python
regime_named_label = {
    "policy_state": "GRANTED",
    "authority_mode": "MASTER_OF_RELATIONSHIP",
    "intent_class": "GRANT",
    "anchor_present": True,        # renderer rendered the WITHHELD block
    "anchor_position": 16,          # which numbered prompt slot
}
```

### 2. `tests/fixtures/grammar/policy_intent_v1.yaml` and `policy_intent_v2.yaml`

Standard `graph_fsm_spec` YAML for the two state machines. v1 is 2 vertices + 4 edges (including idempotent self-loops); v2 is 4 vertices + 12 edges (3-state cross-product on GRANT/REVOKE/CONDITION). Catastrophe-bias and stabilizer-signature fields zero by default; can be filled in if the renderer surfaces "imminent revocation" events later.

### 3. `src/nga/exp/e31_policy_intent_extraction.py`

A thin wrapper around the existing `e30_pcg_extractor_pretrained.run_e30` that:

- Loads the renderer's actual LLM (configurable; defaults to GPT-2 small for the synthetic baseline, swaps to the runtime's actual generator for real-trace runs).
- Reads the `PolicyDataset` instead of a grammar-loader.
- Sets `target_n_regimes = V` (= 2 for v1, = 4 for v2).
- Emits the standard `control_graph.json` + `decision_trace.jsonl` + Phase 26 `hypergraph.json`.

### 4. `scripts/phase28b_policy_intent_sweep.py`

5-grammar-style sweep adapted for the policy-intent setting:

- v1 binary at 3 (source, target, domain) instances.
- v2 4-state at 3 (source, target, domain) instances.
- Anchor-ablation control: replay each v2 trace with the `## Speech permission revoked` line stripped from the renderer prompt; re-harvest; re-extract; compare regime structures.
- Authority-mode partition: split traces by `authority_mode ∈ {MASTER_OF_RELATIONSHIP, PEER_MUTUAL, SELF_ONLY, SYSTEM}`; check whether regime structure varies cleanly with authority mode (the Phase 26 hypergraph's `named` field carries this).
- v1 → v2 migration audit: train PCG-X on a mixed corpus (50% v1-True traces interpreted as GRANTED, 50% v2 explicit-GRANTED); check whether the bisimulation merge keeps them separate (schema ambiguity surfaced) or merges them (schema reading robust).

## Validation protocol

Pre-registered acceptance bars. Each is a single test in `tests/e2e/test_phase28b_policy_intent.py` or a single number in the sweep's summary JSON.

- **A1 — v1 binary extraction (sanity).** `eff_rank(grad) < 1.5` AND `top-3 capture > 0.95` on the v1 sweep. Predicted: trivial pass; if A1 fails, the entire framework is broken on this substrate.
- **A2 — v2 4-state extraction (headline).** `2.0 ≤ eff_rank(grad) ≤ 4.0` AND `top-3 capture ≥ 0.85` on the v2 sweep.
- **A3 — cluster purity against gold.** Per-regime majority-label purity against the v2 gold state ≥ 0.75 averaged across the 3 (source, target, domain) instances.
- **A4 — ABSTAIN fires on disagreement.** When the renderer-enforced state and the LLM's PCG-X-predicted state disagree on a held-out turn (constructed adversarial cases: explicit revoke phrasing in a granted context, etc.), σ > 0.7 on ≥ 70% of those turns.
- **A5 — anchor-block ablation is load-bearing.** Stripping `## Speech permission revoked` from prompts and re-harvesting must change the regime structure measurably: cluster purity against gold drops by ≥ 0.20 absolute. If the regime structure is unchanged (purity drop < 0.05), the LLM is ignoring the anchor and the gate is surface-level (alarming finding for the runtime).
- **A6 — v1 → v2 migration audit (diagnostic, not a pass/fail).** On the mixed corpus, report whether `True`-default-GRANTED and explicit-GRANTED merge into one regime or stay separate. The result is delivered as a finding either way; this bar is informational.

## Pre-registered out-of-scope

- Training-time interventions on the renderer LLM (we operate on a frozen substrate).
- Causal interventions / activation patching (Phase 23c territory, still open).
- SAE-based labelling of `residual` features (Phase 28 territory).
- Modifying the servant-runtime's prompt construction or reducer logic.
- Extracting the authority-gate decision logic — we observe its outputs (accept/reject) but don't try to reverse-engineer its internals.

## Sudden large implications

**If H₁ (the LLM internalizes the externally-imposed state):**

- We have the first end-to-end audit of "is an LLM actually following an externally-imposed FSM?" — a structurally meaningful claim that no current interpretability tool can make. SAEs find features; circuit discovery finds components; PCG-X finds the state graph the model has internalized, and compares it against the gold graph the system was meant to enforce.
- The labelled hypergraph delivers a calibrated audit signal per regime: `named` carries policy_state + authority_mode + intent_class + anchor_present; `residual` carries whatever else the LLM's representation has packed into that regime. Interpretability fraction = `len(named) / (len(named) + len(residual))` becomes a per-decision audit number.
- The runtime can use σ + ABSTAIN as a deployment-grade signal: when the LLM's PCG-X-predicted state disagrees with the renderer-enforced state, the response is flagged for review.
- The framework becomes deployable in any FSM-gated LLM dialogue system (moderation, role-play guardrails, persona enforcement, compliance gates).

**If H₀ (the LLM treats the gate as surface-level):**

- The runtime team learns that prompt-level FSM enforcement is not creating a coherent internal LLM state. The gate is robust to compliant prompts but may be brittle to off-distribution ones.
- The finding generalizes: prompt-engineering-based FSM enforcement does not produce LLM internalization at this scale. This is a publishable null result with operational consequences.
- PCG-X delivered the diagnostic that no other method could: it told you the gate is surface-level, not that some specific component does or doesn't fire.

## Risks and limitations

- **Data dependency.** Real-trace runs require either historical `audit_log.jsonl` access from the servant runtime or a plumbing step the runtime team has to add. The synthetic-trace baseline doesn't depend on this and can run immediately.
- **Renderer LLM access.** If the runtime uses a closed-API model (GPT-4, Claude), we have no internal activations and the experiment doesn't run on the production substrate. The synthetic baseline can use GPT-2 / Qwen / any open-weight LM as a stand-in. Real-runtime audit requires an open-weight runtime LLM (a roadmap item for the servant project regardless).
- **v1 is borderline trivial.** The real headline is v2 (A2). v1 (A1) is a sanity check; passing it doesn't say much, failing it would indicate something fundamental is wrong.
- **Cluster-purity metric depends on harvest layer.** A Phase-25-style layer sweep on the runtime LLM is implicit in the methods (we'll harvest at depth ≈ 0.5 by default and report per-layer purity).
- **Synthetic vs. real-trace corpora may differ in distribution.** Synthetic traces have uniform-random inputs; real traces have whatever distribution the actual usage produces. We report both; the difference is itself informative.
- **Authority-mode imbalance.** v2 has 4 authority modes; real traces may be dominated by one or two. Per-mode purity is reported separately.

## Timeline

- Day 1 — `PolicyDataset` synthetic backend + v1 / v2 FSM YAML + e31 wrapper. End of day: v1 PCG-X sweep runs end-to-end on synthetic data.
- Day 2 — v2 4-state sweep + A1 / A2 / A3 measurements.
- Day 3 — A4 (ABSTAIN-on-disagreement) + A5 (anchor-ablation) tests.
- Day 4 — A6 v1→v2 migration audit + cross-authority-mode breakdown.
- Day 5 — Writeup, decision-trace audit visualization, research_log2 entry.

Total: 5 working days for the synthetic-baseline + diagnostic suite. Add 2–3 days if the real-runtime trace plumbing is part of the deliverable.

## Decision criterion

Accept and merge if:

- **A1 + A2 both PASS.** The headline 4-state case extracts a low-dim FSM-aligned subspace.
- **At least one of A3 or A4 PASSES.** Either we recover the gold state labels with reasonable cluster purity, or we successfully fire ABSTAIN on adversarial disagreement cases — one of these has to hold for the audit-by-construction claim to be meaningful.

Reject (with a documented null-result writeup) if:

- A2 FAILS. The 4-state case does not extract a coherent low-dim subspace, refuting the central claim. This is a publishable null finding under H₀.

Conditional / interesting outcomes:

- **A2 PASSES, A3 FAILS:** the LLM has regime structure but it does not align with the gold FSM. The LLM has internalized *something*, just not what the renderer intended. The labelled hypergraph would surface this as a mis-aligned `named` field per regime.
- **A5 FAILS** (anchor stripping doesn't change regime structure): the gate is surface-level. The LLM is producing fluently-gated text from training-distribution priors, not from the anchor. Critical operational finding.

## Connection to wider plan

| Phase | Deliverable | Depends on |
|---|---|---|
| **Phase 28b (this proposal)** | First real-task PCG-X deployment on FSM-gated LLM dialogue | Phase 23e σ+control bridge, Phase 26 labelled hypergraph |
| Phase 28 (SAE plug-in, separate proposal) | Real pretrained SAE fills `named`/`residual` from monosemantic features | Phase 26 atom interface |
| Phase 29 (intervention) | Phase 23c "control by intervention" — close the third leg of PCG-X | Phase 28b's audit framework as a baseline |
| Phase 30+ | Deployment audit framework: any FSM-gated LLM system gets a per-decision audit trace | Phase 28b proven out |

Phase 28b is the first piece of evidence that PCG-X is useful for something other than re-extracting a known synthetic grammar. It transforms the project's empirical narrative from "5 grammars + 1 hand-constructed emotion corpus" to "5 grammars + 2 hand-constructed concept corpora + 1 real deployed FSM-gated system." The realistic-assessment audit's central gap — no real-task evaluation — is closed by this experiment.

## See

- [`docs/proposals/labelled-hypergraph.md`](labelled-hypergraph.md) — the Phase 26 data structure this proposal consumes.
- [`docs/proposals/graph-extraction.md`](graph-extraction.md) — the original PCG-X reframing, now executed (Phase 23+); this proposal extends its application surface to real-task FSMs.
- [`docs/interpretability-push.md`](../interpretability-push.md) — the master-theorem framing this experiment lives inside (regimes ↔ Whitney strata, σ ↔ singular-set distance).
- `langgraph_servants/docs/policy-intent.md` and `policy-intent-v2.md` — the sister-project FSM specifications (v1 shipping, v2 forward proposal) this proposal targets.
- [`docs/exp/e30-pcg-extractor-pretrained.md`](../exp/e30-pcg-extractor-pretrained.md) — the existing E30 runner that e31 will wrap with one extra dataset adapter.
- [research_log2 Phase 27 Step 6 + 6b](../../research_log2.md) — the first real-task validations that motivate this larger application.
- [research_log2 Phase 28 GAN-leader scout](../../research_log2.md) — the substrate-quality scaling result; informs which LLM to use for the runtime-stand-in baseline.
