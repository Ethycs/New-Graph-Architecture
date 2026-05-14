# Control Policy

**Cluster:** arch
**Status:** implemented
**Tags:** #control #routing #sigma #decision-trace

## What

The σ-thresholded control surface. Maps a singularity score $\sigma \in [0, 1]$ to one of three categorical decisions matching `ControlAction.decision` in the decision-trace schema:

- $\sigma < \theta_\text{normal}$ → **ROUTE_NORMAL**: trust the model's prediction.
- $\theta_\text{normal} \le \sigma < \theta_\text{abstain}$ → **ROUTE_RECOVERY**: override with the next state on the BFS-shortest-path from `current_state` to the nearest goal state.
- $\sigma \ge \theta_\text{abstain}$ → **ABSTAIN**: emit the sentinel action `-1`.

The adjacency used by recovery BFS is **substrate-agnostic**: it comes from either a `GraphFSM` (the original typed FSM use case) or a raw `legality_matrix: np.ndarray` (e.g. the PCG-X regime graph, added in Phase 23e). Both paths produce identical decisions on the same adjacency.

## Why

The architecture promises three things σ alone cannot deliver: (a) a deterministic mapping from σ to a discrete action, (b) a recovery action that respects the graph's legality structure, and (c) an audit-trail-ready record of the decision and its reason. The control policy is the atom that lifts σ from "a number" to "a verdict with provenance." Without it, runners would each invent their own σ-threshold scheme and the `decision_trace.jsonl` reason field would diverge across experiments.

The substrate-agnostic adjacency path (Phase 23e) is what lets the same atom audit *either* the typed FSM (E0, E1, E9 — recovery on the hand-authored graph) *or* the PCG-X regime graph (E28 — recovery on the extracted control graph) without two implementations.

## Interface

**Constructor:**
```python
ControlPolicy(
    theta_normal: float = 0.3,
    theta_abstain: float = 0.7,
    fsm: GraphFSM | None = None,
    legality_matrix: np.ndarray | None = None,  # square bool, mutually exclusive with fsm
    goal_states: list[int] | None = None,
)
```

**Decision:**
```python
decide(
    sigma: float,
    model_prediction: int,
    current_state: int | None,
) -> ControlPolicyResult
```

`ControlPolicyResult` fields:
- `decision`: `"ROUTE_NORMAL" | "ROUTE_RECOVERY" | "ABSTAIN"`.
- `chosen_action`: `int`. Equals `model_prediction` for NORMAL, the BFS first-step for RECOVERY, `-1` for ABSTAIN.
- `reason`: short human-readable explanation echoed into `ControlAction.reason`.

**Threshold convention.** Both thresholds are inclusive on their upper bands. $\sigma = \theta_\text{normal}$ enters the recovery band (recovery is preferred when in doubt about model trust); $\sigma = \theta_\text{abstain}$ enters the abstain band (abstaining is preferred when in doubt about recovery). The boundaries are strictly monotone — increasing σ never re-grants trust.

**Recovery fallback.** Recovery is best-effort. When (a) neither `fsm` nor `legality_matrix` is supplied, (b) `goal_states` is missing or empty, (c) `current_state is None` (no graph location yet), or (d) no goal is reachable from `current_state` under the legality adjacency, the policy falls through to `ROUTE_NORMAL` with a `reason` string ending in `"recovery_unavailable"` so downstream traces can audit the cause.

## Build steps

- Validate thresholds satisfy `0 ≤ θ_normal ≤ θ_abstain ≤ 1`; raise `ValueError` otherwise.
- Validate `fsm` and `legality_matrix` are mutually exclusive; pick whichever was supplied as the adjacency.
- For each `decide` call:
  - σ ≥ θ_abstain → emit ABSTAIN with sentinel `-1`.
  - σ < θ_normal → emit ROUTE_NORMAL with `model_prediction`.
  - Else (recovery band): BFS over the legality adjacency from `current_state` to the closest member of `goal_states`. If found, return the first step on the path. Else fall through to ROUTE_NORMAL with `recovery_unavailable`.

## Links

- **See also:** [Singularity Detector σ(x)](../singularity/singularity-detector.md) (produces the σ input), [Decision Trace JSONL](../../drivers/decision-trace-jsonl.md) (the `ControlAction` sub-record this atom populates).
- **Drives:** [E1 Synthetic BabyAI](../../exp/e1-synthetic-babyai.md), [E9 Full Trace Benchmark](../../exp/e9-full-trace-benchmark.md), [E28 PCG Extractor](../../exp/e28-pcg-extractor.md).
- **Driven by:** the σ ensemble's output and either `GraphFSM.legality_matrix` or any extracted regime adjacency (PCG-X via `e28_pcg_extractor`).
