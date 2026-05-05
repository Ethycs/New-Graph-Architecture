# Group Action on Graph

**Cluster:** arch
**Status:** spec
**Tags:** #symmetry #quotient #compression

## What

A group $H$ acts on the vertices and edges of a retrieval graph $\Gamma$ by permutation, identifying structurally equivalent states. For instance, if two retrieval tools are functionally identical, $H$ may permute them; if two subtasks have the same logic, $H$ folds them into a single orbit. The action respects edge structure and labels, so orbits of states are maximal equivalence classes under this symmetry.

## Why

Retrieval graphs often contain redundancy: duplicate tools, symmetric subtasks, equivalent reasoning paths. Without identifying these symmetries, attention and compression schemes treat them as distinct, wasting compute and learning capacity. Grouping equivalent states under an orbit structure enables downstream compression (quotient graphs) and principled attention reweighting. Without group action, every state looks unique, and O(n²) attention is irreducible.

## Interface

**Inputs:**
- Graph $\Gamma = (V, E)$ with vertex and edge labels.
- Group $H$ (permutation group on $V$).

**Outputs:**
- Action map $\sigma: H \times V \to V$ satisfying $\sigma(gh, v) = \sigma(g, \sigma(h, v))$ and edge preservation.
- Orbit partition $\text{Orb}(H, \Gamma)$: set of equivalence classes under the action.

## Build steps

1. **Identify symmetries:** Scan the typed graph for duplicate or equivalent subtask/tool vertices (same semantics, signature, label). Record which pairs belong to the same symmetry class.
2. **Construct permutation generators:** For each symmetry class, define generator permutations (e.g., swap tool A ↔ tool B). Compose them to form $H$.
3. **Verify action:** Check that each generator preserves edges and labels. Confirm $H$ forms a group under composition.
4. **Compute orbits:** Partition $V$ under the action. Each orbit is a maximal set of vertices that $H$ can shuffle among themselves.
5. **Assign orbit IDs:** Label each orbit with a unique index. Record the stabilizer subgroup of each orbit.
6. **Store orbit membership:** Create lookup tables: vertex → orbit ID, and orbit ID → canonical representative(s).

## Links

- **See also:** [Stabilizer Signature](./stabilizer-signature.md), [Orbit Quotient Space](./orbit-quotient-space.md)
- **Drives:** [Orbit-Pair Attention](./orbit-pair-attention.md)
- **Driven by:** (typed-graph-spec — external driver)
- **Math:** [Mathematics.md §Symmetry and Orbits](../Mathematics.md#symmetry)
- **Open:** (orbit-detection-automatic — open question)
