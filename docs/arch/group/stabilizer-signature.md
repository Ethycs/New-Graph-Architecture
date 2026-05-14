# Stabilizer Signature

**Cluster:** arch
**Status:** spec
**Tags:** #symmetry #singularity #stability

## What

For each state (or orbit) in a retrieval graph, the stabilizer subgroup $\text{Stab}_H(v)$ is the set of group elements $h \in H$ that fix $v$. A state with trivial stabilizer ($\{e\}$) is generic; a state with non-trivial stabilizer is singular. The stabilizer signature is a record per state: stabilizer ID, generators, size, and fixed-point structure. This signature flags which states are brittle, which are resilient to symmetry perturbations.

## Why

Singular states (those with non-trivial stabilizers) mark bottlenecks, choice points, or degenerate regions in the retrieval landscape. They are sensitive to perturbations that break symmetry, and attention/training on them may be unstable. By identifying singular states upfront, we can apply specialized handling: higher learning rates, explicit regularization, or dedicated energy-based weighting. Without stabilizer tracking, singular regions hide in the data, leading to unexplained training instability and poor generalization.

## Interface

**Inputs:**
- Orbit partition $\text{Orb}(H, \Gamma)$ with group $H$.
- State/vertex IDs in the graph.

**Outputs:**
- Stabilizer signature for each state: $\{\text{sid}, \text{generators}, |\text{Stab}|, \text{fixed\_edges}, \text{is\_singular}\}$.
- Singularity map: state ID → boolean (True if stabilizer is non-trivial).
- Singular state list: all states with non-trivial stabilizers.

## Build steps

1. **Compute stabilizers:** For each vertex $v \in V$, compute $\text{Stab}_H(v) = \{h \in H : h(v) = v\}$ by testing each generator and composing.
2. **Assign stabilizer IDs:** Group states by their stabilizer. Assign each unique stabilizer a canonical ID.
3. **Record generators:** For each stabilizer, store its minimal generating set (typically small for finite groups).
4. **Detect singularity:** A state is singular if $|\text{Stab}(v)| > 1$. Mark in a boolean flag.
5. **Compute fixed-point structure:** For each singular state, identify which edges and neighbor states are preserved by the stabilizer. This reveals the local rigidity.
6. **Build lookup tables:** Store stabilizer ID → all states with that stabilizer, and state ID → stabilizer signature for O(1) lookup during training.

## Links

- **See also:** [Group Action on Graph](group-action-on-graph.md), [Orbit Quotient Space](orbit-quotient-space.md)
- **Drives:** [Energy-Weighted Loss](../energy/energy-weighted-loss.md), (singular-state-analysis — planned experiment)
- **Driven by:** [Group Action on Graph](group-action-on-graph.md)
- **Math:** [Mathematics.md §Stabilizer Subgroups](../../Mathematics.md#stabilizers)
- **Open:** (singular-state-regularization — open question)
