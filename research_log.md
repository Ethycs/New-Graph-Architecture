# Research log

A running, candid record of what we measured and what it told us. Add new entries at the top.

## 2026-05-05 — Build wave A/B/C: missing atoms shipped, decision_trace lit up, E7 honest fail

Three waves landed today. They closed the implementation gap and added two new reportable claims plus one honest xfail; the architectural story (sigma still does not beat margin, hyperbolic still has nothing to compress) is unchanged.

### What was built

**Wave A — four missing atoms.** `grammar_compiler` (token-stream to typed FSM transitions), `frozen_encoder_backbone` (the reservoir-side encoder used in E7), `typed_readout_layer` (per-type linear head with mask gating), `control_policy` (the ledger-driven action selector). Each ships with unit tests; all pass.

**Wave B — integration.** E0 and E1 now emit `decision_trace.jsonl` with all six layers populated per step: mask action + sigma signals + energy contributions + control decision + monodromy class + raw scores. The E6 multi-orbit parity bug (orbit_pair_attention not weighting softmax by orbit size) was fixed; single-orbit parity remains 3.5e-15 and the multi-orbit case now also lands at machine epsilon (3.4e-15 reported by the runner).

**Wave C — three new runners.** E5 (IDF ablation: `accuracy_idf` vs `accuracy_no_idf`, reports `idf_uplift`); E7 (reservoir vs end-to-end: param counts and accuracies); E8 (transfer: train/test accuracy plus `transfer_gap`).

### What `decision_trace.jsonl` now records

For the first time we have a per-step record that ties together: which transitions the mask permitted, which sigma signals fired and at what magnitude, the energy contributions per term, the control_policy's chosen action, and the monodromy class assigned. This is the artefact the Architecture doc has promised since Phase 0; it is now actually written by E0 and E1.

### What the numbers say

| Bar | Status at natural settings |
|---|---|
| Phase 1 — E0 accuracy >= 0.90 | PASS |
| Phase 2 — mask uplift > 0, illegal_rate == 0 | PASS |
| Phase 2 — sigma > margin (strict q10) | FAIL (unchanged) |
| Phase 4 — single-orbit parity < 1e-6 | PASS (3.5e-15) |
| Phase 4 — multi-orbit parity < 1e-6 | PASS after fix (3.4e-15) |
| Phase 5 — reservoir params < 10% of end-to-end | PASS (21 vs 231, ratio 0.0909) |
| Phase 5 — reservoir accuracy >= 0.9x end-to-end | xfail at natural noise |
| Phase 5 — E5 idf_uplift | report-only (sign varies by seed/data) |
| Phase 5 — E8 transfer_gap | report-only (no bar) |

### Honest scope statement on E7

The E7 reservoir at `noise_scale=0.30` (the natural setting) loses to end-to-end on accuracy. Reservoir reports ~0.2 to 0.4; end-to-end ~0.7+. The cause is structural: a numpy random projection from 32-D features to 2-D output is too lossy on this data. Meeting the 0.9-of-end-to-end bar will require either (a) real frozen-encoder weights instead of a random projection, or (b) data hierarchical enough that 2-D suffices. The earlier `noise_scale=2.0` workaround was premature optimization and has been reverted; the bar is now `xfail` with the reason recorded, not silently tuned to pass. The param-efficiency claim (the architecturally interesting half of E7) still PASSES.

### What is now testable

`tests/integration/test_atom_census.py` enumerates every atom listed in the docs and asserts each has both an implementation file and at least one caller. The four Wave-A atoms are the ones that previously caused this test to fail; they no longer do.

### Live status updates

- q12 (multi-orbit parity at machine epsilon): PASS after softmax-weighting fix.
- q13 (reservoir matches end-to-end at <10% params): split — params PASS, accuracy xfail at natural noise.
- q14 (transfer): now measurable; report-only until we know what gap looks acceptable.
- q10 (sigma > margin) and q01 (hyperbolic compression) remain null on synthetic data, unchanged by this wave.

## 2026-05-05 — Phase 4 verdict: structural pass, claim degenerate

Phase 4 (group quotient) lands as scaffolding. The four atoms (group-action-on-graph, orbit-quotient-space, stabilizer-signature, orbit-pair-attention) compose correctly. E6 ran on a synthetic 12-vertex graph with Z/12 acting on all vertices.

Numbers: standard_flops 18,576, orbit_pair_flops 1,665, flops_reduction_ratio +0.9104, output_parity_l2 3.52e-15.

The 91% FLOPs reduction PASSES the >=50% bar nominally, but the test setup is degenerate. With Z/12 collapsing every vertex into a single orbit, the orbit-pair attention reduces to a 1x1 matrix and parity is trivially exact. Multi-orbit parity (the architecturally interesting case where some vertices are in non-trivial orbits while others are fixed) does not hit machine epsilon because orbit_pair_attention does not weight its softmax by orbit size. With Z/6 acting on the first 6 vertices and 6 fixed vertices, parity_l2 would be O(1) rather than O(1e-15).

**Phase 4 status:** atoms ship correctly; the FLOPs payoff exists in principle and is reportable; the parity claim is currently tested only on a degenerate single-orbit case. The fix is a softmax-weighting tweak in orbit_pair_attention (multiply by orbit_size[j] before softmax, equivalently add log(orbit_size[j]) to the score). Tracked as a Phase 4 follow-up; not blocking other phases.

**Sigma's stabilizer-signature signal still does not fire on E0/E1.** The synthetic dataset has no inherent symmetry, so the GroupAction passed to E0/E1 would have to be artificially imposed; stabilizer_risk would either be 0 everywhere or constant, neither of which would help q10 close. The genuine fix for q10 remains real-data E2.

## 2026-05-05 — Phase 0+1+2+3 architectural-signal assessment

### What's in the runs/ tree

| Run | What it tested | Headline numbers |
|---|---|---|
| E0_A0_seed42 | sklearn digits, full Phase 1+2 pipeline | accuracy 0.9733, low-margin acc 0.50, density 0.22 |
| E1_A0_seed42 | synthetic-BabyAI grid with adversarial illegal-temptation samples | accuracy 0.6422 (with mask) vs 0.0092 (without); illegal rate 0.0 vs 0.865 |
| E3_A0_seed42 | dim sweep euclidean vs hyperbolic on synthetic-BabyAI | hyperbolic d=8 0.7089, euclidean d=16 0.7722, hyperbolic uplift -0.063 |
| E4_A0_seed42 | post-hoc sigma vs margin AUROC on E0+E1 results | margin 0.957, sigma 0.923, sigma uplift -0.035 |

### Strong signal (1 of ~6 architectural claims)

**Graph mask is a hard structural prior.** E1 shows +63 pt mask uplift and a collapse from 86.5% to 0.0% illegal-transition rate. Without the mask the classifier is at chance because the synthetic dataset deliberately seeds illegal-temptation samples. This is unambiguous: external graph structure removes a search-space burden the classifier cannot reliably handle alone.

Caveat: the +63 pt is a designed effect. The synthetic dataset was constructed to make the mask claim land. On a less adversarial dataset the uplift would be ~3 to 5x smaller, but still positive.

### Null signals (2 of ~6 claims)

**Sigma does not beat margin.** Uplift -0.035 vs the +0.03 strict bar (q10). Margin AUROC = 0.957 is so high there is almost nothing left to predict. Sigma in Phase 2 has only margin + loop_risk + post-mask is_illegal as live signals; the rest are stubs. Structurally there is no information beyond margin for sigma to extract until Phase 4 stabilizer-jump and Phase 5 catastrophe priors land.

**Hyperbolic compression does not materialize.** H^8 underperforms R^16 by 6.3 pt. Matched-dim edge is +1.3 pt at d=8 and +2.5 pt at d=4, but well within likely seed variance (no bootstrap yet). Gromov delta on the trained prototypes is 0.08 to 0.15, indicating the embedded space is essentially Euclidean-flat — the synthetic data has no tree structure for negative curvature to exploit.

### Untested (3+ of 6 claims)

Group-quotient FLOPs (E6), reservoir-vs-end-to-end (E7), transfer (E8), energy-shaped loss, stratified partition function — none yet run. They are Phase 4 to Phase 5 atoms and downstream experiments.

### What the model is telling us

**The dataset is the bottleneck for the strict claims, not the architecture.** Hyperbolic geometry compresses trees and hierarchies; the synthetic features have neither, so H^d has nothing to compress. Sigma is meant to aggregate complementary signals that fire on stabilizer jumps, contradiction, and catastrophe-edge proximity; on a flat classification task with no real state dynamics those signals can't fire — margin alone captures everything.

**The plumbing is sound, the claims are dataset-bound.** The four bars that PASS (typed pipeline, graph mask, sigma no-degradation, matched-dim hyperbolic) all test properties intrinsic to the architecture. The two that FAIL (sigma beats margin, hyperbolic compresses) are bars that require richer data to express.

### Comparison to a typical architecture-paper bar

| Standard | Status |
|---|---|
| At least one real-world benchmark | NO — only synthetic + sklearn digits |
| Headline claim replicates on held-out data | NO — strict bars don't land |
| Each architectural choice is ablated | PARTIAL — ablation matrix exists but only A0 has been run |
| Multiple seeds | NO — single seed throughout |
| Comparison to a strong baseline | NO — sklearn LogReg is the only baseline |

If submitted as-is, reviewer comments would be: "interesting plumbing, but the load-bearing claims are not supported by the experiments shown."

### Verdict on the architecture's strength of signal

**One solid result, two informative nulls, three untested claims, working pipeline ready to test the rest.** Treat the current state as a substrate, not a result.

Three things, in order of importance:

1. The integration story (drivers + mask + typed scoring) is sound. That is the least surprising part of the architecture but the most reliable to ship. A typed-FSM-masked-classifier pipeline works as advertised.
2. The detection story (sigma over margin) is unsupported in this regime. The architecture's value-add over a plain confidence-thresholded classifier is currently zero. Either Phase 4-5 atoms light up genuinely new signals, or the singularity detector is decoration.
3. The geometry story (hyperbolic compression) requires hierarchical data to even be testable. On flat classification it cannot lose by much, but it cannot win either. It is a bet on data that has tree structure: agent task graphs, dependency graphs, knowledge hierarchies. Until that data is in the loop, the geometry is unjustified architectural overhead.

### Open questions promoted to live status

- q10 (sigma > margin): null on synthetic data; re-evaluate when Phase 4 stabilizer-jump signals land.
- q01 (hyperbolic dim < euclidean dim at parity accuracy): null on synthetic data; needs E2 real BabyAI traces.
- q11 (hyperbolic-vs-euclidean tradeoff): currently a wash; no compression payoff visible.

### What we are NOT doing now and why

- Multi-seed bootstrap of the matched-dim hyperbolic edge — would tighten the +1.3 pt claim into "real" or "noise". Cheap to do (~5 min). Deferred until E2 because the matched-dim claim is small enough that the effect has to land on real data to matter.
- E2 real BabyAI/MiniGrid — would test sigma > margin, hyperbolic compression, and transfer on data with actual hierarchical structure. Requires installing minigrid (Farama) and likely a training loop heavier than sklearn LogReg. The single highest-value next step from a "validate the architecture" lens.

### Decision: continue Phase 4 anyway

Phase 4 (group quotient: group-action-on-graph, orbit-quotient-space, stabilizer-signature, orbit-pair-attention) lands the stabilizer-signature signal that sigma needs to genuinely beat margin. This is the architectural bet that addresses the q10 null. E6 (group-quotient attention FLOPs) is a different claim with measurable units (FLOPs reduction) that can be tested without real-data training, via a synthetic graph with explicit symmetry. So Phase 4 is not just more scaffolding — it has a real chance of producing positive signal on q10 and producing a new positive claim under E6.

If Phase 4 still does not move sigma above margin, that is itself informative: it means the singularity detector is over-engineered, not just "data-bound". We will know after running.
