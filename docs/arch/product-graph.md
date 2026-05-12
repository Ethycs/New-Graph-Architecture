# Product Graph (lazy)

**Cluster:** arch
**Status:** implemented
**Tags:** #product-graph #node-tuple #lazy #typed-axes #cartesian

## What

The system's full state is a node-tuple in the Cartesian product of typed axes. Direct materialisation of the product is exponential in the number of axes (product of vertex counts). `ProductGraph` is the **lazy, sparse** materialiser: cells are only allocated when visited, so storage is $O(\text{visited})$ rather than $O(\prod_a |V_a|)$. Each axis is a typed graph with a totally ordered vertex set; a `NodeTuple` is the projection of the system's state onto each axis.

## Why

A TPN's commitment is "every observation, prediction, and signal is a vertex in some typed axis; the system's true state is a node-tuple in their Cartesian product." Without a lazy realisation, this commitment is computationally infeasible for $\ge 3$ axes — three axes of 50 vertices each is $125\,000$ cells, four is $6.25 \times 10^6$, and most never get visited. Lazy product-graph materialisation keeps the commitment honest at runtime while preserving the audit guarantee (a `NodeTuple` always exists for any state the system has been in).

## Interface

- **`NodeTuple`** — an ordered tuple of vertex IDs, one per axis; hashable, comparable, serialisable into the decision trace's `output_node_tuple` field.
- **`ProductGraph`** — constructed from a list of typed axis graphs; `visit(tuple)` materialises the cell if absent; `is_visited(tuple)` is the O(1) check; `n_cells_materialised()` is the audit count.
- **Drives:** [Decision Trace JSONL](../drivers/decision-trace-jsonl.md)'s `output_node_tuple` field, and the recursive / cross-attention composition in `docs/proposals/graph-extraction.md` (the proposal identifies cross-attention between two TPNs as intra-attention on the product graph).
- **Driven by:** the typed FSM (each axis is one) and the runner's per-step state.

## Build steps

- `NodeTuple = collections.namedtuple` over (`axis_0_vertex`, ..., `axis_K-1_vertex`); deterministic ordering; `__hash__` over the tuple bytes.
- `ProductGraph.__init__(axes)`: store axis references; `_cells: dict[NodeTuple, CellPayload]` for the lazy allocator.
- `visit(tuple)`: if absent, allocate a fresh `CellPayload` (currently a counter; can be extended with per-cell statistics).
- `n_cells_materialised`: `len(self._cells)`.
- Consider weak references for long-running streams so unvisited cells don't accumulate.

## Links

- **See also:** [Graph FSM](./graph-fsm.md), [Decision Trace JSONL](../drivers/decision-trace-jsonl.md), [Typed Latent Clustering](./typed-latent-clustering.md).
- **Drives:** Phase 19B's self-supervised diagnostic runner (E23) used `product_graph` for sparse materialisation over 4920 patients × discovered clusters.
- **Driven by:** the typed-axis FSMs ([Graph FSM](./graph-fsm.md)).
- **Math:** Cartesian product of finite typed graphs; the lazy materialisation is an inverse-limit / sheaf-of-cells construction.
- **Open:** under what conditions can the system collapse equivalent node-tuples (orbit-quotient on the product) to keep materialisation polynomial in the orbit count rather than the raw cell count?
