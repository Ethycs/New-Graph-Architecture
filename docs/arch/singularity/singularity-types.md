# Singularity Types Catalog

**Cluster:** arch
**Status:** spec
**Tags:** #typology #classification #reference

## What
Enumerated taxonomy of singularity kinds realized in the typed task graph, each with a canonical feature signature. Types include: low-margin (confused states), decision-tie (equidistant prototypes), contradiction (violated logical constraints), illegal (domain-rule violation), loop (unexpected cycle or reentry), and stabilizer-jump (abrupt phase shift). Each type maps to codimension and bifurcation class.

## Why
Singularities are not monolithic. Different types have different remedies: low-margin states need sharper prototypes; contradictions need re-typing; illegal violations need domain-rule repair; loops need topology fixes. Cataloging types enables targeted diagnosis and repair. Without a catalog, singularities are opaque labels; with one, they guide intervention.

## Interface
**Inputs:**
- Observation batch and error labels.
- Graph structure, types, constraints, and rules.

**Outputs:**
- Per-observation singularity type(s) (may be multi-label).
- Type frequency distribution.
- Type-to-feature mapping: which components of [Singularity Detector σ(x)](singularity-detector.md) flag each type.

## Build steps
- **Low-margin:** order prototypes by distance to observation; $\text{type\_low\_margin} \iff (d_1 - d_2) < m_{\text{threshold}}$.
- **Decision-tie:** detect if two or more competing prototypes are within small distance $\delta_{\text{tie}}$; common in branching nodes.
- **Contradiction:** check if observation satisfies all parent and sibling constraints; flag if any violated.
- **Illegal:** query domain rule engine for type incompatibilities or forbidden transitions.
- **Loop:** test if shortest hyperbolic path from observation to same node is short (< depth threshold).
- **Stabilizer-jump:** record embedding velocity; $\text{type\_stabilizer\_jump} \iff v > v_{\text{threshold}}$.
- Aggregate counts; report distribution and per-type feature profiles.

## Links
- **See also:** [Singularity Detector σ(x)](singularity-detector.md), [Catastrophe Labels](catastrophe-labels.md)
- **Drives:** (singularity-type-distribution, type-specific-repair — planned experiments)
- **Driven by:** (observation-log, domain-rules — external inputs)
- **Math:** [Mathematics.md §Bifurcation Theory](../../Mathematics.md#bifurcation)
- **Open:** (type-coexistence-analysis — open question)
