# Insights

Three conceptual reflections on what this architecture actually is once a frozen pretrained substrate sits underneath it: **what it's useful for**, **what it costs to run**, **how far the design stretches**. Companion to `docs/results.md` (the paper-shaped, number-heavy synthesis) and `docs/model-class.md` (the formal model-class definition). This document is for the question "what does the project mean as an engineering proposition" — not "what numbers did we measure."

Phase 24 established that a frozen pretrained LM at a mid-to-late layer matches a trained-on-grammar baseline (mean purity 0.764 vs 0.790) on the PCG-X regime-extraction task. Phase 25 established that the peak harvest layer is mid-to-late (L10 of 12 for GPT-2 small), with embedding-only at 0.556 mean and the final block dropping back to 0.673 because it specialises for next-token prediction rather than state representation. Everything below builds on those two findings.

---

## 1. Why a TPN is useful on a frozen pretrained substrate

The frozen substrate gives you good representations. The TPN adds a **control surface** over them. Five concrete things it buys, ranked by how distinct they are from existing tooling.

### 1.1 Calibrated abstention with a structural reason, not just a confidence number

A probe + threshold gives you "abstain when confidence < 0.7." The σ ensemble combines six signals — margin, decision-tie, *illegal-transition against the extracted regime graph*, loop-risk, stabiliser, KL-surprise — and emits an explanation. On Phase 23e's held-out eval the illegal signal fires on 2–10% of steps while the training-set illegal rate is exactly 0% by construction. That's a structural OOD signal pure confidence-thresholding doesn't capture: "this transition was never observed during regime extraction" is qualitatively different from "the softmax is flat here."

### 1.2 Graph-aware recovery, not just abstain

When σ lands in the middle band (uncertain but not catastrophic), `ControlPolicy.decide()` runs BFS over the regime adjacency from the current regime toward a goal regime (low failure-rate), and emits the first step on that path. `ROUTE_RECOVERY` is structurally different from `ABSTAIN`. Selective-prediction tooling has only abstain; constrained-decoding tools (Outlines, Guidance, LMQL) have graph-aware control but the graph is hand-authored; TPN's graph is **extracted from the substrate**, so the framework covers grammars too big or fluid to write by hand.

### 1.3 Per-step audit trail by construction, not as a logging afterthought

Every prediction produces a `DecisionTraceRecord` with the typed score, the mask state, the σ signal decomposition, the energy breakdown, the control verdict, the output node-tuple. For regulated domains (medical, financial, legal, safety-critical), this is the difference between "we abstained" and "we abstained because illegal_signal fired at σ_total = 0.74 of which 0.61 came from the illegal-transition contribution." Phase 23e showed the same schema works on both the typed FSM and the extracted PCG-X regime graph without modification.

### 1.4 The structural constraint emerges from observation and then enforces itself

Phase A's Bayesian Beta-Dirichlet posterior over regime edges converges deterministically on a fixed corpus. The resulting legality matrix is a hard mask applied *before* softmax — illegal transitions are zeroed absolutely. The grammar is discovered, then becomes constraint. Constrained-decoding tools take the grammar as input; the TPN learns it from the regime graph.

### 1.5 Trainable-parameter count an order of magnitude smaller than fine-tuning

Trainable footprint over a frozen 124M-parameter GPT-2: `PredictiveProjection` (~few thousand params) + Beta α/β log-counts per regime edge + per-type readout heads. Total trainable scale is in the low thousands. Compare to fine-tuning the full encoder. For domains with limited labelled data or strict deployment-cost budgets, this matters more than any algorithmic gain.

### 1.6 What the TPN does NOT add

Be explicit about the negative space:

- **Better raw next-token / next-state accuracy.** GPT-2's frozen mid-layer activations drive the projection to 0.928 mean next-state accuracy. A plain linear probe gets 90%+ of that. The TPN is a calibration / control / audit overlay, not an accuracy amplifier.
- **Better representations.** The substrate IS the representation; the TPN reads it, doesn't refine it. If the substrate is bad, the TPN faithfully reflects that (Phase 23d: from-scratch transformer at 0.63 next-token accuracy → 0.66 regime purity).
- **Open-ended generation.** No learned generation policy. The TPN routes the substrate's outputs; it doesn't synthesise new ones.

### 1.7 The engineering shape — when this wins, when it doesn't

**Wins when at least two of:**
- The label / output space has structure (typed, hierarchical, conditioned on state).
- Some inputs *should* trigger abstain rather than a forced choice.
- Per-decision audit is part of the deliverable.
- You cannot fine-tune the substrate (cost, data, licensing, or product reasons).
- The structural prior is too rich for hand-written rules but the substrate is too generic to enforce it alone.

**Concrete deployments that fit:**
- **Medical-triage classifier over a frozen domain LM.** Regime graph extracted from clinical notes; σ flags unfamiliar transition patterns; each prediction comes with the audit-field reason. Phase 19B's ARI 0.82 result is the half-built version of this.
- **Regulated document classification** (contracts, filings, financial-doc routing). Extract regimes from the corpus; σ + control gates predictions; trace satisfies compliance.
- **Structured-output extraction** (form-filling, schema-conformant code completion, JSON generation). Regime graph constrains output paths; ABSTAIN is a first-class outcome when the input doesn't match any extracted regime.
- **Conversational-protocol enforcement** over a frozen chat LM. Customer-service flow modelled as regimes; RECOVERY escalates to a human when σ flags drift into an unknown regime.

**Does not win when:** open-ended generation (chat, creative writing); flat unstructured labels (generic sentiment, intent classification with no state); maximising raw accuracy with full fine-tuning available and cheap.

### 1.8 One-liner

A TPN wrapping a frozen pretrained substrate is a **calibrated, structurally-aware, audit-trail-emitting control surface for tasks where the substrate's representations are good enough but the control / abstention / interpretability layer needs to be principled rather than ad-hoc**. The substrate does the perception; the TPN does the routing and the receipts. Distinct from selective prediction (no graph), constrained decoding (hand-authored graph), and probing (produces labels, not control verdicts).

---

## 2. Efficiency gains

Five distinct angles, ordered by how cleanly the existing phases support them. **Measured** = already done in the project; **defensible** = clean implication of existing findings, one-experiment-away; **untested** = research direction with plausible mechanism.

### 2.1 (measured) Trainable parameter count is already ~10⁴–10⁵× smaller than fine-tuning

When the substrate is frozen, the TPN's trainable footprint on top is the `PredictiveProjection` head (~1–2k params), the Beta α/β log-counts (~K² floats per regime graph), and optional per-type readout heads (~few-K params per type). Total trainable: **single digit thousands**. Compare to fine-tuning GPT-2 small (124M trainable), or training the small from-scratch transformer of Phase 23d (~few-hundred K params). The TPN is roughly **10⁴–10⁵× cheaper to train than fine-tuning the substrate**. Phase 18 measured 700–800 samples/sec on CPU at sub-100 MB memory for TPN inference proper — the substrate forward dominates; everything the TPN adds is in the noise.

### 2.2 (defensible) Layer truncation for state-extraction-only deployment

Phase 25's clean finding: L10 is the peak harvest layer; L12 drops 11.2 pp below it. For state-extraction tasks (audit, abstention, regime control — *not* generation), **the last two transformer blocks of GPT-2 small are not just unhelpful, they're actively worse than L10**. Concrete cut:

- Drop blocks L11 and L12: ~14M params off the 85M transformer-block budget.
- Drop the LM head (50,257 × 768 ≈ 38M params): gone, replaced by the TPN's projection head at ~few-thousand params.
- Keep embeddings + 10 transformer blocks + the small TPN head.
- **Total: 124M → ~72M params, ~42% reduction**, with no expected regime-purity loss because L10 was the peak anyway.

The principle generalises: a single layer-ablation sweep identifies the peak layer K for any pretrained substrate; everything past K can be discarded for state-extraction-only use. Not built; it's a one-day experiment.

### 2.3 (defensible) Inference-time compute via ABSTAIN-driven early exit

If σ can be computed from an intermediate layer's activations (not just the final layer's), then high-σ inputs can `ABSTAIN` *before* the full forward completes. Early-exit transformer literature does this for confidence; the TPN can do it for σ specifically. Concrete shape:

- Train one `PredictiveProjection` per harvest layer (say L4, L8, L10).
- Run the substrate forward layer-by-layer; at each layer, compute σ from that layer's projection.
- If σ ≥ θ_abstain at L4, abstain immediately — skip L5–L12. Save ~⅔ of the forward.
- If σ is borderline at L4, continue to L8; refine the decision.
- The "full forward + project at L10" path runs only for inputs with confident NORMAL routing.

For tasks where many inputs are easy (high mass on NORMAL routing) and the long tail needs only abstain, **average inference cost can drop substantially** without quality loss on the easy mass. Exact saving depends on the σ distribution on the deployment corpus. Untested but architecturally clean.

### 2.4 (untested) Distillation to a smaller substrate

Phase 24's finding that frozen GPT-2 matches a trained-on-grammar specialist suggests the regime graph is **substrate-agnostic in structure** — different substrates produce structurally similar regime graphs for the same task. If you train a *smaller* model from scratch with the regime graph (from a bigger frozen substrate) as the structured supervision target — not just label CE, but "match the regime trajectories the big model would produce" — you might compress the substrate itself.

Honest caveat: Phase 23d's small-transformer-from-scratch trained on the grammar's own tokens only hit 0.663 purity vs GPT-2's 0.764. So small-models-trained-on-task underperform big-frozen-pretrained. Whether distilling the regime structure from GPT-2 into a smaller model closes that gap is the experiment that would answer this. Standard knowledge-distillation literature suggests yes; not run.

### 2.5 (defensible) Regime-level caching and short-circuiting

Regimes coarsen state space dramatically: GPT-2's 768-d hidden vector compresses to one of ~K ≈ V regimes (10–37 on the project's grammars; thousands for a universal corpus). If many inputs land in the same regime with the same predicted-next regime, you can **cache the control verdict at the regime level** and short-circuit the rest of the pipeline on hits. The expensive thing (substrate forward) is unavoidable, but everything after it is amortisable.

For high-traffic deployments (regulated-text classifiers running 10⁶ docs/day, customer-service routing pipelines), this would reduce per-decision overhead beyond the substrate. Not built; mostly engineering, no new math required.

### 2.6 The honest summary

**Real measured efficiency gains today:**

- Trainable parameter count: ~10⁴–10⁵× smaller than fine-tuning.
- Inference latency: substrate forward dominates; TPN adds < 1 ms / sample of σ + control + trace overhead.

**Defensible compression gains, one experiment away:**

- ~42% substrate reduction by dropping post-peak layers + LM head for state-extraction-only deployment.
- Early-exit on ABSTAIN: average inference saving proportional to how much of the deployment corpus σ flags early.
- Regime-level caching: amortise post-substrate cost on repeat-regime inputs.

**Untested research direction:**

- Distill regime structure into a smaller substrate.

The deliverable framing: **"Sub-1k trainable parameters and ~40% substrate compression for tasks where state-extraction is the goal, not generation."** That's a real number with a clear domain (regulated classification, structured extraction, audit-required deployment) and a defensible mechanism (regime extraction + post-peak-layer truncation + tiny TPN head).

---

## 3. A complete control structure for GPT-2 across all tasks

Conceptually yes. The architectural pieces all extend; the question is which extensions are clean and which are open research.

### 3.1 What "complete control structure for all tasks" would mean

Concretely: GPT-2 frozen substrate + a *universal* regime graph extracted from a broad slice of its pretraining distribution + σ computed per token at generation/inference time + a 3-branch `ControlPolicy` that routes/abstains/recovers + per-token `decision_trace.jsonl` audit. The TPN becomes a **meta-layer over the LM** that lets the same frozen weights serve any task while emitting audit / abstention / structural-routing decisions over all of them.

### 3.2 What carries over directly

- **The substrate forward.** Already a frozen LM. No change.
- **Harvest-at-layer-K.** Layer 10 generalises (Phase 25); the equivalent peak generalises for other task families with the same sweep.
- **σ signals — 4 of 6.** Margin, decision-tie, illegal-transition (against whatever regime graph is current), loop-risk all carry over meaningfully to any sequential prediction. Stabiliser-risk needs a group action that doesn't generalise; KL-surprise needs an empirical conditional that needs construction for an arbitrary task.
- **The 3-branch `ControlPolicy`.** NORMAL / RECOVERY / ABSTAIN routing over a graph adjacency works for any directed graph. The σ thresholds are the only thing per-task that needs calibration.
- **The decision-trace schema.** Already substrate-agnostic per Phase 23e. Adding a new producer = adding a new caller, no schema change.

That's five of seven architectural pieces unchanged on a universal corpus.

### 3.3 What needs extension or reframing

**1. The universal regime graph.** Phase 24's graph was extracted from 80 programs of one synthetic grammar (a few thousand tokens). For "all tasks" you'd extract from a pretraining-scale activation sample — say 10⁸–10⁹ tokens — and the resulting graph would have thousands or tens of thousands of regimes. Computationally tractable: PCG-X scales linearly; harvesting 10⁹ tokens of GPT-2 activations is ~250 GPU-hours, ordinary infrastructure cost.

**2. Supervision signal for the projection head.** Currently the projection's next-state head supervises against a *known FSM*. For arbitrary tasks there is none. Three coherent extensions:

- **(a) Self-supervise on next-BPE-token.** Tautological — you recover GPT-2's own confidence partition. Useful for σ-as-uncertainty but adds no structural information GPT-2 doesn't already carry.
- **(b) Coarse external taxonomy.** Predict next-token's *category* (named entity / number / keyword / punctuation / syntactic-role). Cheap to label (tokenisers and POS taggers give it); produces a syntactic regime graph.
- **(c) Multi-headed projection.** Multiple supervision signals stacked — syntactic category + safety classification + topic / domain + task class. Each head produces its own partition; the regime is the *product* of categorical labels. The "typed graph" commitment from `docs/model-class.md` was always about a Cartesian product of typed axes; we just haven't built the multi-headed version yet.

(c) is the architecturally clean answer.

**3. Goal regimes per task.** Recovery currently steers toward "low failure rate" regimes — defined relative to gold FSM. For arbitrary tasks "failure" means different things: abstention wants high-confidence regimes; safety wants non-toxic regimes; task-completion wants goal-relevant regimes. The clean extension: a small *task-conditional* head that maps `(task_id, regime_id) → is_goal`. Per-task fine-tuning of this head is cheap (it's tiny); the substrate stays frozen. This is the engineering pattern that lets one regime graph serve many tasks.

**4. Generation, not just classification.** The biggest gap. For the TPN to act as a runtime guardrail during *generation*, it needs to intervene in the decoding loop. The obstacle: you need a `token_id → predicted_next_regime` map to bias GPT-2's logits toward legal regimes. Two paths:

- **Look-ahead** (expensive, exact): for each top-k candidate token, run GPT-2 one more step, project, check the regime. k forward passes per generation step. On GPT-2 small with top-k=20 this is ~5 ms / step on GPU.
- **Learned** (cheap, approximate): train a `token_id → next_regime` head once, use it at generation time. One additional small head; same trainable-footprint story.

This is the leap from "audit layer" to "control layer." All pieces exist; the harness hasn't been built.

### 3.4 The two binding constraints

1. **Building the universal regime graph** from a pretraining-scale activation sample. Engineering cost, not research uncertainty: ~250 GPU-hours of harvest + one PCG-X run. The bisimulation-merge stage scales; the predictive-projection training does too.
2. **The supervision signal for the projection head.** Coarse-taxonomy approach (b) ships fast; the architecturally satisfying multi-headed approach (c) is a real research project — defining the right axes, picking labels that are cheap and meaningful, validating that the product graph holds together.

If both are solved you have **GPT-2 + a universal control surface that is task-conditional via a tiny head, calibrated abstention via σ, structural routing via the `ControlPolicy`, and per-token audit via `decision_trace.jsonl` — all without retraining the substrate**.

### 3.5 What this delivers

For the same substrate weights:

- **Task-conditional abstention without retraining or RLHF.** Calibrated `<ABSTAIN>` based on regime structure, configurable per task by setting σ thresholds and goal regimes.
- **OOD detection at the regime-graph level.** Inputs whose regime trajectory falls outside the trained set fire `regime_unknown` and auto-abstain — a structural OOD signal that simple confidence-thresholding cannot match.
- **Audit-trail-by-construction for every generation.** Per-token regime ID, σ decomposition, control verdict. The compliance / regulated-deployment story is real.
- **Safety / toxicity guardrails as one task axis among many.** If "safety" is one of the multi-headed projection's axes, the regime graph encodes safety as a coordinate; the `ControlPolicy` can refuse to leave the "safe" sub-region. Without retraining the underlying model.
- **Composable structural constraints.** Multiple typed axes (syntactic + semantic + safety + task) compose as a Cartesian product; new constraints can be added by training new projection heads without touching the substrate.

The pitch: **the same pretrained LM serves many tasks under different control configurations, with the structural metadata extracted once and reused.** The cost model is "one big regime-graph extraction" + "small per-task head" + "small per-deployment threshold tuning." None of it requires gradient updates to the substrate.

### 3.6 Honest scope

This is a research project, not a small extension. A defensible roadmap from where the project sits today:

- **Phase 26 (foundation):** multi-seed bootstrap of Phase 24/25 numbers; one larger substrate (TinyLlama-1.1B) layer sweep. Establishes the science.
- **Phase 27 (universal regime graph MVP):** PCG-X on a small natural corpus (wikitext-2 or HumanEval subset), 10⁶–10⁷ tokens, single supervision head (next-syntactic-category). One week.
- **Phase 28 (multi-headed typed graph):** add a second projection head (e.g., named-entity-presence or toxicity). Build the product-graph regime structure. Two weeks.
- **Phase 29 (runtime guardrail at generation):** wire token→regime look-ahead into decoding. Demonstrate ABSTAIN and RECOVERY firing during actual GPT-2 generation. Two weeks.
- **Phase 30 (real corpus + real task):** test on a regulated-domain task. Multi-week, dataset-dependent.

The architecture supports it. Each step is independently a defensible mini-paper. The composition is the publishable shape: **"audit-by-construction, task-conditional control, no substrate retraining."**

---

## The engineering proposition, in one paragraph

A pretrained LM at a mid-to-late layer carries enough state structure that a tiny supervised head can extract a useful regime graph from its frozen activations — matching, on synthetic grammar benchmarks, a baseline trained specifically on the grammar. On top of that regime graph, a 3-branch σ-routed control policy plus a per-step decision trace gives calibrated abstention, graph-aware recovery, OOD detection via a `regime_unknown` sentinel, and audit-by-construction — without retraining the substrate, with sub-1k trainable parameters, and with potential ~40% substrate compression for state-extraction-only deployment. Extended with multi-headed typed projections and a token→regime look-ahead, the same machinery becomes a universal control structure over the pretrained model: one substrate, many tasks, configurable safety / abstention / routing axes, full decision audit. The piece-by-piece priors are well-precedented (DFA extraction, structural probing, bisimulation, selective prediction, constrained decoding); the composition — and the result that it works on a substrate the project did not train — is the contribution.
