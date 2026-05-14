# Orbit Quotient Space |Γ|/H

**Cluster:** arch
**Status:** spec
**Tags:** #quotient #compression #symmetry

## What

Given a group $H$ acting on a retrieval graph $\Gamma$, the orbit quotient $|\Gamma|/H$ is a collapsed graph where each vertex is an orbit (equivalence class under $H$), and edges connect orbits if there exist representatives with an edge between them. This quotient is a compressed shadow of $\Gamma$, retaining the essential retrieval structure while eliminating redundant states.

## Why

Quotient graphs reduce the state space from $|V|$ to $|V|/|H|$, sometimes dramatically. This enables compressed attention—computing affinities between orbits instead of individual states, dropping cost from O(n²) to O((n/k)²). It also clarifies the gross topology of the retrieval landscape: bottlenecks, convergence funnels, and singular regions emerge when redundancy is stripped away. Without quotient structure, you cannot systematically compress attention.

## Interface

**Inputs:**
- Typed graph $\Gamma = (V, E, \ell)$ with labels.
- Group action $H$ with orbit partition $\text{Orb}(H, \Gamma)$.

**Outputs:**
- Quotient graph $\Gamma/H = (V/H, E/H, \ell/H)$ where $V/H$ is the set of orbits.
- Lift map: orbit → set of representative vertices in $\Gamma$.
- Multiplicity: for each quotient edge, how many edges in $\Gamma$ connect its lifts.

## Build steps

1. **Identify orbits:** Use the group action to partition $V$ into orbits. Assign each orbit a canonical representative.
2. **Define quotient vertices:** Each orbit becomes a single vertex in $\Gamma/H$. Store the orbit ID and its multiplicity (number of vertices it contains).
3. **Define quotient edges:** For each orbit pair $(O_1, O_2)$, check whether any representative of $O_1$ has an edge to any representative of $O_2$. If yes, add an edge to $E/H$.
4. **Label quotient edges:** Collapse or aggregate edge labels from $\Gamma$. Record multiplicity (number of distinct edges in $\Gamma$ projecting to this quotient edge).
5. **Preserve metadata:** Attach to each quotient vertex the set of original vertices it represents, and any aggregated statistics (e.g., average cost).
6. **Create lift tables:** Build reverse lookup: for each quotient edge/vertex, list all its preimages in $\Gamma$.

## Links

- **See also:** [Group Action on Graph](group-action-on-graph.md), [Orbit-Pair Attention](orbit-pair-attention.md)
- **Drives:** [Orbit-Pair Attention](orbit-pair-attention.md), (quotient-compression-eval — planned experiment)
- **Driven by:** [Group Action on Graph](group-action-on-graph.md)
- **Math:** [Mathematics.md §Quotient Spaces](../../Mathematics.md#quotient)
- **Open:** (aggregate-edge-labels — open question)
