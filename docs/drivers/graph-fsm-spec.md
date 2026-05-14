# Graph FSM Spec

**Cluster:** driver
**Status:** spec
**Tags:** #driver #contract #state-machine #legality-mask

## What

The Graph FSM Spec is a YAML/JSON file format describing the task state machine: vertices $V$ (task states), edges $E$ (legal transitions), vertex weights $w_v$ (importance/specificity), geometry $g_v$ (hyperbolic radius or curvature), and monodromy $m_v$ (symmetry group action at each node). Architecture loads this to build the legality mask that constrains the classifier's next-state prediction. Experiments load the same spec to validate whether predicted transitions are legal and compute transition-legality metrics.

## Why

The graph FSM is the combinatorial skeleton. Without it, the classifier can predict any next state; legal constraints are lost. With it, two systems must agree on the legal transition set: the architecture enforces the mask during training/inference, and the experiment checks whether the architecture's outputs satisfy the mask. If the spec drifts between them—if architecture sees edge `(Parse, Navigate)` as legal but experiment does not—then ablation studies become meaningless. The FSM spec is also the core input to [Graph Legality Mask](../arch/graph/graph-legality-mask.md) and to (transition-validator — exp).

## Interface

**File path (runtime):** path is read from [Config (YAML)](./config.md).graph_fsm_path. Source files live at `graphs/<dataset>.fsm.yaml` (e.g. `graphs/babyai.fsm.yaml`). [CLI Runner](./cli-runner.md) snapshots the resolved file alongside the config at `runs/<run_id>/graph_fsm.yaml` for auditability. Format: YAML (canonical) or JSON (interchange).

**Schema (YAML):**
```yaml
schema_version: "1.0"
name: babyai_task_graph
vertex_count: 7
edge_count: 12
vertices:
  - id: Parse
    label: Parse
    layer: 0
    w_v: 1.0
    g_v: 0.1
    m_v: null
  - id: Navigate
    label: Navigate
    layer: 1
    w_v: 2.0
    g_v: 0.5
    m_v: S3
  - id: ResolveDoor
    label: ResolveDoor
    layer: 1
    w_v: 1.5
    g_v: 0.4
    m_v: null
  # ... remaining vertices
edges:
  - source: Parse
    target: Navigate
    label: parse_to_navigate
    weight: 1.0
  - source: Navigate
    target: ResolveDoor
    label: navigate_to_door
    weight: 1.0
  # ... remaining edges
coordinates:
  space: hyperbolic           # "hyperbolic" | "euclidean"
  dimension: 64               # MUST equal Config.embedding_dim
  node_embeddings:
    Parse: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Navigate: [0.5, -0.3, 0.21, 0.04, -0.11, 0.18, 0.07, 0.42]
    # ... one per vertex id
validation:
  acyclic: false
  start_nodes: [Parse]
  end_nodes: [Done]
  invariant_edges: ["Parse → *"]
```

**Top-level field schema:**

| Field | Type | Units | Required | Example | Semantic note |
|---|---|---|---|---|---|
| `schema_version` | str | semver | required | `"1.0"` | Same versioning rules as [Config (YAML)](./config.md). |
| `name` | str | — | required | `"babyai_task_graph"` | Human-readable identifier; logged with every run. |
| `vertex_count` | int | nodes | required | `7` | MUST equal `len(vertices)`. Validated on load. |
| `edge_count` | int | edges | required | `12` | MUST equal `len(edges)`. Validated on load. |
| `vertices` | array<Vertex> | — | required | see below | Each `id` MUST appear as a label in the grammar spec (so it can show up in [Typed Score Record Contract](./typed-score-record.md).label). |
| `edges` | array<Edge> | — | required | see below | Defines the legal transition set $E$. |
| `coordinates` | Coordinates | — | required | see below | Geometric embedding shared by arch and exp. |
| `validation` | Validation | — | optional | see below | Sanity-check metadata; loaders enforce when present. |

**Vertex schema:**

| Field | Type | Units | Required | Example | Semantic note |
|---|---|---|---|---|---|
| `id` | str | — | required | `"Navigate"` | Unique vertex id; used as `label` in [Typed Score Record Contract](./typed-score-record.md) and `y_true`/`y_hat` in [results.jsonl](./results-jsonl.md). |
| `label` | str | — | required | `"Navigate"` | Human-readable label; usually equals `id`. |
| `layer` | int | depth | optional | `1` | Topological depth for visualization. |
| `w_v` | float ≥ 0 | weight | required | `2.0` | Vertex weight; consumed by IDF when `idf_weighting_enabled` (see [Ablation Flag Set](./ablation-flags.md)). |
| `g_v` | float | curvature/radius | required | `0.5` | Geometric coefficient; with `coordinates.space == "hyperbolic"` interpreted as radius/specificity. |
| `m_v` | str \| null | group label | required | `"S3"` | Stabilizer/monodromy group label; consumed by [Group Action on Graph](../arch/group/group-action-on-graph.md) and [Stabilizer Signature](../arch/group/stabilizer-signature.md). `null` means trivial. |

**Edge schema:**

| Field | Type | Units | Required | Example | Semantic note |
|---|---|---|---|---|---|
| `source` | str | — | required | `"Parse"` | Must reference a vertex `id`. |
| `target` | str | — | required | `"Navigate"` | Must reference a vertex `id`. |
| `label` | str | — | required | `"parse_to_navigate"` | Human-readable transition name. |
| `weight` | float ≥ 0 | — | required | `1.0` | Edge weight; consumed by graph-aware loss/attention. |

**Coordinates schema:**

| Field | Type | Required | Example | Semantic note |
|---|---|---|---|---|
| `space` | str (enum: `"euclidean"` \| `"hyperbolic"`) | required | `"hyperbolic"` | When `hyperbolic_geometry_enabled=false` (ablation A6), arch may swap to `"euclidean"`. |
| `dimension` | int | required | `64` | MUST equal [Config (YAML)](./config.md).embedding_dim and the length of `z_H` in [Typed Score Record Contract](./typed-score-record.md). |
| `node_embeddings` | object<str, array<float>> | required | `{Parse: [...], …}` | One entry per vertex id; each array has length `dimension`. |

**Validation schema (optional):**

| Field | Type | Required | Example | Semantic note |
|---|---|---|---|---|
| `acyclic` | bool | optional | `false` | Loaders flag a warning if the graph contradicts this claim. |
| `start_nodes` | array<str> | optional | `[Parse]` | Designated entry points. |
| `end_nodes` | array<str> | optional | `[Done]` | Designated terminals. |
| `invariant_edges` | array<str> | optional | `["Parse → *"]` | Pattern strings; edges that must exist in all variants. |

**Legality matrix construction:**
- [Graph Legality Mask](../arch/graph/graph-legality-mask.md) reads `vertices` and `edges`, builds adjacency $A \in \{0,1\}^{|V| \times |V|}$ where $A[i,j] = 1$ iff edge `(vertices[i].id, vertices[j].id)` exists.
- At inference: given current state $q_t$ and classifier scores $s_\lambda(x)$, set masked logits $\ell'_j = \ell_j - (1 - A[q_t, j]) \cdot \lambda_\infty$, then softmax.

**Versioning:**
- `schema_version` required. Current: `"1.0"`.
- Adding optional fields is non-breaking.
- Renaming or changing types bumps the version; readers MUST refuse unsupported majors.

**Producers:** human-curated YAML or output of (grammar→graph compiler — code TBD); written under `graphs/`.
**Consumers:** [Graph FSM (arch)](../arch/graph/graph-fsm.md) and [Graph Legality Mask](../arch/graph/graph-legality-mask.md) (build mask), [Hyperbolic Embedding](../arch/hyperbolic/hyperbolic-embedding.md) and [Graph Prototype Vectors](../arch/hyperbolic/graph-prototype-vectors.md) (consume `coordinates.node_embeddings`), [Group Action on Graph](../arch/group/group-action-on-graph.md) and [Stabilizer Signature](../arch/group/stabilizer-signature.md) (consume `m_v`), [Metric Collectors](../exp/metric-collectors.md) and (transition-validator — exp) for legality metrics.

## Build steps

1. Define `GraphFSMSpec`, `Vertex`, `Edge`, `Coordinates`, `Validation` dataclasses with YAML/JSON serialization and the validators above (`vertex_count`, `edge_count`, embedding length, edge endpoints reference real vertex ids, `dimension` equals `Config.embedding_dim`).
2. Implement `fsm_loader.py` with `load(path) → GraphFSMSpec`; rejects unsupported `schema_version`.
3. Implement `build_legality_matrix(spec) → np.ndarray` and cache by `(spec.name, spec.schema_version)`.
4. Wire [Graph FSM (arch)](../arch/graph/graph-fsm.md) and [Graph Legality Mask](../arch/graph/graph-legality-mask.md) to consume `GraphFSMSpec` and call `build_legality_matrix` once at construction time.
5. Wire (transition-validator — exp, lives inside [Metric Collectors](../exp/metric-collectors.md)) to load the same spec and compute `transition_legal` for each [results.jsonl](./results-jsonl.md) record.
6. Author template FSM files under `graphs/`: `mnist.fsm.yaml`, `babyai-synthetic.fsm.yaml`, `babyai.fsm.yaml`, `alfworld.fsm.yaml`, `scienceworld.fsm.yaml`.
7. Add a regression test that loads each template, builds the legality matrix, and confirms `start_nodes`/`end_nodes` are reachable.

## Links

- **See also:** [Config (YAML)](./config.md) (path to spec file; `embedding_dim` constraint), [Ablation Flag Set](./ablation-flags.md) (`graph_mask_enabled`, `hyperbolic_geometry_enabled`), [Typed Score Record Contract](./typed-score-record.md) (`label` ↔ vertex `id`)
- **Drives:** [Graph FSM (arch)](../arch/graph/graph-fsm.md), [Graph Legality Mask](../arch/graph/graph-legality-mask.md), [Hyperbolic Embedding](../arch/hyperbolic/hyperbolic-embedding.md), [Graph Prototype Vectors](../arch/hyperbolic/graph-prototype-vectors.md), [Group Action on Graph](../arch/group/group-action-on-graph.md), [Stabilizer Signature](../arch/group/stabilizer-signature.md), [Metric Collectors](../exp/metric-collectors.md)
- **Driven by:** grammar DSL compiler or manual curation under `graphs/`
- **Open:** (q03-edge-weights — used for masking or analysis only?), (q04-monodromy-representation — encode stabilizer groups compactly?)
