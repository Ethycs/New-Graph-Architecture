# Dynkin / ADE Classification

**Cluster:** arch
**Status:** spec
**Tags:** #dynkin #ade #singularity-classification

## What
Dynkin/ADE classification recognizes when a task graph $G$ belongs to the ADE family (types $A_n, D_n, E_6, E_7, E_8$). ADE graphs are rare, highly structured quivers with Cartan/Killing forms that biject with simple (du Val) singularities. Detection of ADE structure unlocks classical singularity-theoretic machinery and enables canonical resolution geometry.

## Why
Most task graphs are wild quivers with no singularity correspondence. But if a graph happens to be ADE, it lies in the most-understood corner of singularity theory: vanishing cycles, root systems, Picard–Lefschetz theory, and monodromy are all explicit. Recognizing ADE structure automatically provides canonical intersection forms, vanishing-cycle diagrams, and singularity type. Without this check, you miss a chance to inherit classical results.

## Interface
- **Input:** (from [Graph FSM](./graph-fsm.md) or [Weighted Plumbing Graph](./weighted-plumbing-graph.md)) graph $G = (V, E, w_v)$ with optional intersection form.
- **Output:** (to [Singularity Extraction Functor](./singularity-extraction-functor.md)) boolean flag is_ADE, Dynkin type (A/D/E), index $n$, canonical Cartan form, list of simple roots.

## Build steps
- Check if the graph is a tree (no cycles). If cyclic, return is_ADE = False.
- Enumerate possible tree structures and compare to known Dynkin diagrams (A_n is a path; D_n is a path with one 3-valent node; E_6/E_7/E_8 are fixed shapes).
- For each candidate match, compute the Cartan matrix $C$ and its signature. Verify that $C$ has all positive diagonal entries and non-positive off-diagonal (for $a_{ij} < 0$ iff $i,j$ adjacent).
- Compute determinant and rank: for ADE, $\det C > 0$ and rank equals the number of vertices.
- If all checks pass, extract simple roots and validate against database of Dynkin roots.
- Return type label (A_n, D_n, E_6, E_7, or E_8) and canonical Cartan form.

## Links
- **See also:** [Graph FSM](./graph-fsm.md), [Weighted Plumbing Graph](./weighted-plumbing-graph.md), [Graph Realization Functor](./graph-realization-functor.md)
- **Drives:** (ade-classification-test — planned experiment)
- **Driven by:** [Weighted Plumbing Graph](./weighted-plumbing-graph.md)
- **Math:** [Architecture.md §Dynkin graph to ADE singularity](../Architecture.md#three-routes-to-singularities) — ADE classification and realization.
- **Open:** (ade-parametrization — learn ADE graph parameters, or fix them?)
