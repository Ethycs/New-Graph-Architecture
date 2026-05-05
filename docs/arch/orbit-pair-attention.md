# Orbit-Pair Attention

**Cluster:** arch
**Status:** spec
**Tags:** #attention #quotient #compression

## What

Attention is computed at the granularity of orbits, not individual states. Each query is an orbit (or representative thereof), each key/value is an orbit, and the attention matrix is $(|V/H|, |V/H|)$ instead of $(|V|, |V|)$. When an orbit is queried, its attention weights are broadcast to all member states; when retrieving, a single orbit's value is replicated to all its members. This reduces computation from O(n²) to O((n/k)²) where k is the average orbit size.

## Why

Treating individual states as independent in attention leads to redundant computation when many states are symmetric. Orbit-pair attention respects the symmetry structure, avoiding wasted compute on equivalent state pairs. It also refines attention: symmetric states now share a single attention weight, forcing the model to attend to them uniformly—preventing spurious differentiation of interchangeable states. This is essential for principled compression and for preventing overfitting to redundant structure.

## Interface

**Inputs:**
- Quotient graph $\Gamma/H = (V/H, E/H)$ with orbit IDs.
- State embeddings $\mathbf{s} \in \mathbb{R}^{|V/H| \times d}$ (one per orbit, or one per representative).
- Orbit membership lookup: state ID → orbit ID.

**Outputs:**
- Attention matrix $\mathbf{A} \in \mathbb{R}^{|V/H| \times |V/H|}$ with softmax normalization.
- State weights $\mathbf{w} \in \mathbb{R}^{|V|} = $ broadcast of $\mathbf{A}$ to all member states.
- Aggregated output $\mathbf{o} \in \mathbb{R}^{|V| \times d}$ per state (replicated within orbit).

## Build steps

1. **Embed orbits:** For each orbit, compute a representative embedding (e.g., mean of member state embeddings, or embedding of canonical representative).
2. **Compute orbit queries/keys/values:** Apply standard Q/K/V projections to the $(|V/H|, d)$ orbit embedding matrix.
3. **Compute orbit attention:** $\text{softmax}(Q K^\top / \sqrt{d})$ over the quotient graph vertices. Shape: $(|V/H|, |V/H|)$.
4. **Broadcast weights:** For each orbit pair $(O_i, O_j)$ with attention weight $\mathbf{A}[i,j]$, assign this weight to every (state in $O_i$, state in $O_j$) pair.
5. **Aggregate values:** For each state, sum the weighted values from all orbits (including its own). Replicate across orbit members to maintain symmetry.
6. **Output:** Return per-state weighted output $\mathbf{o}$, maintaining the symmetry structure.

## Links

- **See also:** [Orbit Quotient Space](./orbit-quotient-space.md), [Group Action on Graph](./group-action-on-graph.md)
- **Drives:** (orbit-attention-speedup — planned experiment)
- **Driven by:** [Orbit Quotient Space](./orbit-quotient-space.md)
- **Math:** [Mathematics.md §Attention Mechanisms](../Mathematics.md#attention)
- **Open:** (orbit-embedding-aggregation — open question)
