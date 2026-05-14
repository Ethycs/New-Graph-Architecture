# Grammar Compiler

**Cluster:** arch
**Status:** spec
**Tags:** #grammar #specification #dsl

## What

The grammar compiler is a domain-specific language (DSL) and execution engine that translates a declarative specification of state types, transitions, and legality rules into a typed graph object, prototype vectors, and a legality mask. Instead of learning the graph structure from data, the compiler constructs it from explicit rules: "Question states have these fields; Retrieval states reference Document indices; a transition from Q to R is legal iff the Q contains a query term." The output is a fully typed, labeled, validated graph ready for attention, energy, and training.

## Why

Learning graph structure from weak supervision (e.g., implicit in sequences) is error-prone, slow, and hard to debug. A specification-first approach ensures the graph structure is interpretable, correct, and auditable. The grammar also enables fast, type-aware validation of state transitions, preventing illegal states from polluting training data. This is essential for safety-critical reasoning tasks (medical, legal) where structure must be verifiable. The compiler also generates prototype vectors and masks that guide the model, reducing the hypothesis space and improving sample efficiency.

## Interface

**Inputs:**
- Grammar DSL (YAML or textual format) specifying:
  - State types: $T = \{t_1, \ldots, t_m\}$ with fields and constraints.
  - Transition rules: $(t_i, t_j, \text{legality condition})$ tuples.
  - Prototype vectors or constraints (e.g., "Question ≈ [query, context, uncertainty]").
  - Embedding space structure (dimensional constraints, metric properties).

**Outputs:**
- Typed graph: $\Gamma = (V, E, \ell_V, \ell_E)$ where $\ell_V(v) \in T$ and $\ell_E(e)$ is a transition type.
- Prototype vectors: $\{\mathbf{p}_t \in \mathbb{R}^d : t \in T\}$ (anchors for each type).
- Legality mask: boolean matrix $M \in \{0,1\}^{|V| \times |V|}$ where $M[i,j] = 1$ iff the edge $i \to j$ is legal.
- Validation report: type coverage, transition saturation, detected inconsistencies.

## Build steps

1. **Write grammar specification:** Author a YAML or text file defining state types, their fields (text, references, counts), and transition rules. Example: "Transition from Question to Retrieval requires a non-empty query field."
2. **Implement type parser:** Build a parser that reads the grammar and extracts type definitions, field constraints, and transition legality rules.
3. **Generate prototype vectors:** For each state type, create a prototype embedding (either hand-crafted based on semantics, or initialized from representative examples). Store in a prototype table.
4. **Construct legality mask:** For each pair of states $(v_i, v_j)$, evaluate all transition rules between their types. Populate the legality mask $M[i,j]$.
5. **Validate graph:** Check that all rules are consistent, no type-transitivity cycles unless explicitly allowed, and no orphaned states (unreachable from initial states).
6. **Export and cache:** Serialize the typed graph, prototype vectors, and legality mask to disk for rapid loading in training/inference.

## Links

- **See also:** (typed-graph-spec — external driver), [Typed Readout Layer](../typed/typed-readout-layer.md)
- **Drives:** [Typed Readout Layer](../typed/typed-readout-layer.md), (grammar-correctness-audit — planned experiment)
- **Driven by:** (typed-graph-spec — external driver)
- **Math:** [Mathematics.md §Type Theory and Grammars](../../Mathematics.md#grammars)
- **Open:** (grammar-expressiveness-bounds — open question)
