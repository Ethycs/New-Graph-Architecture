# Graph FSM

**Cluster:** arch
**Status:** spec
**Tags:** #graph-fsm #state-machine #typed-graph

## What
A graph FSM is the tuple $(V, E, w_v, g_v, m_v)$ that defines a finite-state automaton where vertices are task states, edges are legal transitions, and three label families encode weights (self-intersection or cost), grammatical types (task role), and metric annotations (monodromy or catastrophe class). This enriches a bare graph into a machine-drivable control structure.

## Why
A raw graph has no notion of legality, cost, or meaning. Without this tuple, a classifier cannot distinguish a valid next state from an illegal jump. Weights enable energy-based scoring; grammatical tags allow type-safe instruction; metric annotations attach singularity and catastrophe signals. Without it, the graph is inert combinatorics.

## Interface
- **Input:** (from [Graph FSM Spec](../../drivers/graph-fsm-spec.md)) task vertices, transition edges, optional costs and semantic labels.
- **Output:** (to [Graph Legality Mask](graph-legality-mask.md)) the enriched tuple $(V, E, w_v, g_v, m_v)$ defining the machine's legal move set and scoring bias.

## Build steps
- Parse task graph into vertex and edge lists; assign each vertex a unique state ID.
- Annotate each vertex $v$ with weight $w_v$ (cost, self-intersection, or noop weight).
- Assign each vertex a grammar tag $g_v$ from a vocabulary (Search, Verify, Write, ResolveConflict, etc.).
- Assign each transition $(u, v)$ an edge weight encoding transition cost or probability bias.
- Annotate critical vertices with metric flags $m_v$ (catastrophe class, monodromy order, singularity type).
- Store and version as a structured record (JSON or Python dataclass) for reproducibility.

## Links
- **See also:** [Graph Legality Mask](graph-legality-mask.md), [Typed Score Record](../typed/typed-score-record.md), [Typed Field Pipeline](../typed/typed-field-pipeline.md)
- **Drives:** (classifier-routing-test — planned experiment)
- **Driven by:** [Graph FSM Spec](../../drivers/graph-fsm-spec.md)
- **Math:** [Architecture.md §Typed Zig-Zag Pipeline](../../Architecture.md#typed-zig-zag-pipeline) — the ($G_{\mathrm{raw}}$ → $G_{\mathrm{typed}}$) lift.
- **Open:** (fsm-learning — learn $w_v, g_v, m_v$ from agent traces or hand-engineer?)
