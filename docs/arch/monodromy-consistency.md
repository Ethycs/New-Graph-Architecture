# Monodromy Consistency

**Cluster:** arch
**Status:** implemented
**Tags:** #monodromy #closed-walk #group-equivariance #energy-drift #structural-test

## What

A closed walk $v_0 \to v_1 \to \ldots \to v_n \to v_0$ in the FSM corresponds (under the dart-permutation machinery in [Dart Permutations](./dart-permutations.md) and [Monodromy Group](./monodromy-group.md)) to a group element of the monodromy group $\langle \rho, \tau \rangle$ that fixes the starting dart's vertex. If the world model's energy field respects the same group structure, then traversing such a walk should leave the system invariant — i.e., the cumulative energy drift around the loop is zero. The monodromy-consistency atom is the structural test that measures this drift and flags non-zero deviations as evidence that the world model is *not* group-equivariant.

## Why

The architecture commits to group-equivariant attention (orbit-pair) and a typed graph carrying a monodromy group. If the energy function is not equivariant under that group, the architecture's whole story breaks: σ_loop_risk is supposed to be holonomy under the group action, and a non-equivariant energy makes the holonomy meaningless. Monodromy consistency is the diagnostic that says "we *do* respect the group structure we claim to respect." Without it, the symmetry story is rhetorical, not verified.

## Interface

- **Input:** a closed walk (list of vertex IDs starting and ending at $v_0$), the current world model's energy function, the dart-permutation context.
- **Output:** the energy drift $\Delta E_{\text{loop}}$ around the loop, the corresponding monodromy group element, and a boolean / scalar `is_consistent` based on a tolerance threshold.
- **Used in:** training-time diagnostic (high drift triggers a structural alarm); eval-time σ_loop_risk computation; the recursive-TPN holonomy check.

## Build steps

- Resolve the dart sequence for the closed walk via [Dart Permutations](./dart-permutations.md).
- Compute the monodromy group element as the product $\rho^{a_1} \tau^{b_1} \rho^{a_2} \tau^{b_2} \ldots$ that traverses the dart sequence.
- Compute the energy at each step's prototype; accumulate signed drift.
- Compare $|\Delta E_{\text{loop}}|$ to a tolerance (default $10^{-6}$); emit a boolean and the raw drift.
- Optional: emit per-edge contribution decomposition for debugging non-equivariant regions.

## Links

- **See also:** [Dart Permutations](./dart-permutations.md), [Monodromy Group](./monodromy-group.md), [Singularity Detector](./singularity-detector.md) (the σ_loop_risk component reads this).
- **Drives:** σ_loop_risk in the singularity detector; the structural-AUROC signal on grammars with cyclic structure.
- **Driven by:** the FSM's dart-permutation context (which is fixed at runner start).
- **Math:** the monodromy group $\langle \rho, \tau \rangle$ acts on the dart set; equivariance of the energy is the condition $E(g \cdot x) = E(x)$ for all $g$ in the group; non-zero drift around a closed loop = non-trivial holonomy = a curvature signal.
- **Open:** how to handle approximate equivariance — the trained heads will never be exactly equivariant; the right threshold for `is_consistent` is grammar-dependent.
