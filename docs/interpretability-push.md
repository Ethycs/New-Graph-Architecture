# A push toward complete structural interpretability

A focused write-up of the conversation thread that asked: **can a TPN + a frozen LLM + some procedure deliver complete interpretability?** Answer in one paragraph, then unpacked: structurally yes, semantically no, and computationally tractable if a single empirical assumption about effective dimension holds.

This document records the conceptual move so future sessions inherit it cleanly. Companion to [`docs/insights.md`](insights.md) (the broader engineering reflection) and [`docs/results.md`](results.md) (the empirical synthesis).

## The one-paragraph version

A TPN over a frozen pretrained substrate delivers state-and-decision-level interpretability natively. Combined with the project's master theorem (Whitney stratification + the stratified partition function from `Mathematics.md`), this extends to **complete structural interpretability** at the resolution of the substrate's intrinsic polyhedral stratification — exactly affine within each stratum for ReLU, smoothly approximate for GELU, with σ as the geometric distance to the singular set. The naive obstacle ("polyhedral cell count is exponential in depth × width") dissolves under an empirical assumption: if the substrate's effective dimension $d_{\text{eff}}$ is small (~2.5 for GPT-2 by Patterson-Sullivan / Krylov measurement), then the interpretively-relevant structure lives in a low-dimensional essential subspace where marching simplices / explicit boundary reconstruction is computationally trivial. The semantic gap (model representations ↔ human concepts) is unchanged by any of this; that's a separate problem the framework does not pretend to solve.

## The starting question and the four levels

The question: given an LLM (e.g. GPT-2 frozen) plus a TPN wrapper plus some procedure, can we get *complete* interpretability per decision?

"Complete" splits into four levels:

| Level | Question | Tooling |
|---|---|---|
| **State / behavioral** | What is the model doing right now? | TPN regime graph, σ, control trace |
| **Mechanistic** | What internal feature / circuit fires when? | SAEs, dictionary learning, circuit discovery |
| **Causal** | What component is *necessary* for the output? | Activation patching, ablation, Phase 23c interventions |
| **Attentional / token-level** | Which input tokens drove the output? | Attention patterns, attribution, integrated gradients |

TPN cleanly delivers level 1 and provides a structural skeleton on which the other three can hang. **It does not deliver levels 2-4 by itself.** Concrete deliverables per token: regime ID, σ decomposition, control verdict (NORMAL/RECOVERY/ABSTAIN), full audit trail in the decision_trace.jsonl schema, and a `regime_unknown` sentinel for OOD detection at the regime-graph level.

## The master theorem (what I was under-using)

`Mathematics.md §"Whitney Stratification and Group-Action Orbit Types"` is the project's master theorem. Whitney stratification decomposes a singular space into a finite union of smooth strata glued at lower-dimensional singular sets, satisfying Whitney's conditions A and B (tangent-space-limit and secant-line-limit regularity). The stratified partition function carries this to probability:

$$Z = \sum_\lambda \int_{S_\lambda / H} e^{-E(x) / T} \, d\mu_\lambda(x)$$

Each stratum $S_\lambda$ is a polyhedral neighborhood where the substrate is locally smooth (exactly affine for ReLU), $H$ is a group action that quotients away symmetries, the integral is well-defined on each stratum independently, and stratum boundaries are exactly the singular set $\Sigma$.

The PCG-X regime graph (Phase 23 onward) is the discrete shadow of this stratification:

- **Each regime ↔ one stratum (or a behavioural quotient of several).**
- **Each edge in the regime graph ↔ a stratum-boundary transition the substrate actually traverses on the deployment corpus.**
- **σ fires at $\Sigma$ crossings by construction.** Stabilizer-jump, decision-tie, illegal-transition, loop-pressure are all detectors of "we are on or crossing a stratum boundary."
- **Beta(α, β) posterior calibrates transition probabilities** at each boundary observation.

This gives the project a much stronger interpretability claim than "we extract a discrete graph":

**Complete local description within each stratum.** For ReLU, the substrate is exactly an affine map $f(x) = A_\lambda x + b_\lambda$ on $S_\lambda$. Not approximated, not probed — the substrate IS that affine map locally. Knowing $\lambda$ tells you exactly what the substrate computes at $x$.

**Well-defined boundary semantics.** Whitney's regularity conditions geometrically constrain the local behaviour as you cross a boundary. σ firing isn't a heuristic — it's a detector of a genuinely singular event.

**The partition function is the right probability primitive.** $P(\lambda) = Z_\lambda / Z$ directly quantifies "which polyhedral cell is the model in."

**Group-action refinement.** The quotient $S_\lambda / H$ respects the substrate's natural symmetries. Translation in time, permutation invariance in attention, etc. fold into the stratification automatically.

**Finite determinacy makes interpretability tractable in principle.** Singularity theory says a singularity is determined up to equivalence by a finite jet — finitely many Taylor coefficients. The *interpretively meaningful* refinement is bounded by the substrate's finite jet, not arbitrary continuous resolution.

## What "arbitrarily fine partitions" actually means

The naive idea — partition arbitrarily finely — collapses to a degenerate limit (each unique activation becomes its own regime; the graph becomes the activation trace; no quotienting happens). The useful refinement is along the substrate's intrinsic structural axes:

- **TPN at K = V (Phase 24):** ~10-40 cells, FSM-state-aligned, coarse.
- **TPN at K = state × token (Phase 21's natural equivalence):** hundreds to thousands of cells, substrate-natural.
- **TPN refined by SAE features:** combinatorial in dictionary size, feature-aligned, lives inside the substrate.
- **Hypothetical limit (continuous):** degenerate.

The second-to-last is meaningful. SAE-refined regimes ARE fine partitions that carry meaning, because the basis is the substrate's intrinsic feature structure. Adding SAE feature decomposition on top of TPN gives mechanistic content within each regime.

## What survives even arbitrary refinement (the irreducible limits)

Three obstacles do not yield to finer partitions:

1. **Compositional opacity.** Knowing the active features tells you the signature; it doesn't tell you HOW features compose under attention + MLP. The composition is where the deep mystery lives and there is no analytic theory.
2. **Semantic gap.** The model has no canonical English description; we project our concepts onto its features. Refinement doesn't shrink this — it persists because the gap is between the model's representation and human-natural concepts.
3. **Basis identifiability.** Different training seeds produce different polyhedral subdivisions that compute equivalent functions. The function is well-defined; the partition isn't canonical.

These are real limits but they are not limits of the framework. They are limits of computation, of natural language, and of representational equivalence classes respectively. **The math is complete; the engineering and semantics around it are not.**

## The marching-cubes move

Marching cubes is the classical algorithm: sample a scalar field on a grid, detect threshold crossings, reconstruct the isosurface piecewise via lookup tables. Naive application doesn't generalize to 768-d activation space (lookup table is exponential in dimension). But the *idea* — trajectory-driven local geometric reconstruction of stratum boundaries — translates cleanly:

- The **scalar field** is σ (already defined on activation space).
- The **isosurface** is the singular set $\Sigma$ — the boundary between strata. This is what we want.
- The **sampling** is not a regular grid; it's the corpus trajectory through activation space. Dense where the model spends time, sparse where it doesn't.
- The **per-cell reconstruction** is a hyperplane (for ReLU) or smooth hypersurface (for GELU) fit between adjacent regime supports.

Implementation is straightforward:

```python
# Per regime: fit local affine map (exact for ReLU, approximate for GELU)
for r in regimes:
    A, b = LinearRegression().fit(acts_in[r], outputs[r]).params
    local_maps[r] = AffineMap(A, b, residual=...)

# Per adjacent-regime edge: fit boundary hyperplane via linear SVM
for (src, dst) in observed_regime_edges:
    svm = LinearSVC().fit(
        np.vstack([acts_in[src], acts_in[dst]]),
        np.r_[np.zeros(len(acts_in[src])), np.ones(len(acts_in[dst]))],
    )
    boundaries[(src, dst)] = Hyperplane(svm.coef_, svm.intercept_)

# σ as geometric boundary distance
def sigma_geometric(x):
    r = predict_regime(x)
    return min(abs(boundaries[(r, other)].signed_distance(x))
               for other in adjacent_regimes[r])
```

What this delivers beyond plain PCG-X:

- **σ becomes a geometric quantity, not a signal blend.** "Distance to nearest stratum boundary."
- **Decision trace can include `distance_to_nearest_boundary` and `boundary_normal` per step.**
- **Recovery becomes geometric.** Geodesic on the polyhedral mesh instead of BFS over a discrete graph.
- **Counterfactuals are analytically affine within stratum.** "What would the model do at $x + \delta$" is exactly $A_\lambda(x + \delta) + b_\lambda$ inside the stratum, which makes Phase 23c interventions much sharper.

What's still tricky:

- GELU/SiLU residual on the affine fit (small, measurable per stratum).
- High-dim hyperplane fits need ~1000+ samples per adjacent regime pair to be stable.
- Boundary identifiability through the projection's MLP nonlinearity: not exactly polyhedral in substrate space, but piecewise smooth.

## The $d_{\text{eff}}$ collapse (the move that makes this all tractable)

The user's adjacent work (Cat Scanner / Patterson-Sullivan / catastrophe-theoretic analysis) provides a single empirical fact that changes everything: **the substrate's effective dimension is small.**

Three equivalent definitions of $d_{\text{eff}}$:

1. **Hyperbolic / Sullivan log-law:** For a trajectory $\theta(t)$ in parameter space, with depth $M(t) = \max_{s \leq t} -\log \Delta(\theta(s))$ where $\Delta$ is the distance to the catastrophe surface, the Sullivan log-law gives $M(t) / \log t \to 2 / d_{\text{eff}}$ almost surely. $d_{\text{eff}}$ is the Hausdorff dimension of the limit set of the geodesic flow on the substrate's hyperbolic structure.

2. **Catastrophe / splitting lemma:** $d_{\text{eff}}$ is the codimension of the catastrophe stratum in the subspace the optimizer actually explores. By the splitting lemma, the substrate decomposes locally as $f(x) = Q(x_1, \ldots, x_{n-k}) + g(x_{n-k+1}, \ldots, x_n)$ where $Q$ is a non-degenerate quadratic (inert) and $g$ is the genuinely singular part. $k = d_{\text{eff}}$ = corank of the Hessian.

3. **Depth-law / Butterfly hazard:** $d_{\text{eff}}$ is the effective number of independent catastrophe layers in the depth-width hazard inequality. Layer-wise correlation reduces the count below the literal layer total.

**For GPT-2: Sullivan log-law measurement gives $r_\infty = 0.95$, hence $d_{\text{eff}} = 2.1$. Krylov rank (gradient + HVP) gives 2. Depth-law fit on layer-wise hazard gives ~2.5.** All three converge to ~2-2.5.

**Three immediate consequences for the TPN program:**

1. **Marching cubes / simplices is now computationally trivial.** Not 768-d; 2.5-d. Standard 8-corner lookup or marching simplices in 3-d covers the boundary geometry. The "exponential cell count" objection dissolves.

2. **The polyhedral stratification compresses to a measurably small structure.** Away from the catastrophe directions the substrate is a non-degenerate quadratic (smooth, predictable). The genuinely singular directions — where σ fires, where transitions matter — are bounded by 2.5 dimensions. **The regime graph for any task has a fundamentally low-dimensional embedding regardless of substrate width.**

3. **Patterson-Sullivan gives the regime graph's hyperbolic structure directly.** The project has had hyperbolic prototypes from Phase 3. The regime graph embedded in hyperbolic space has an ideal-boundary limit set of Hausdorff dimension $d_{\text{eff}}$. Recovery becomes a hyperbolic geodesic on this 2.5-d limit set, not a heuristic BFS.

## The integrated story

Combining the master theorem + polyhedral / marching-cubes + $d_{\text{eff}}$:

1. **Find the essential subspace.** Krylov method (gradient + HVP) on the substrate produces ~2 directions. Extend the Krylov basis slightly to capture $d_{\text{eff}} \approx 2.5$ worth of structure.
2. **Project activation trajectories there.** 768-d → 3-d projection. Cheap. Reproducible.
3. **Compute σ on a regular grid in essential subspace.** σ is computable per activation; the projection gives σ at each grid cell.
4. **Marching simplices on σ.** Extract isosurfaces $\{x : \sigma(x) = \theta_{\text{normal}}\}$ and $\{x : \sigma(x) = \theta_{\text{abstain}}\}$. These ARE the regime boundaries in geometric form.
5. **Per stratum: fit the local affine map** (or smooth local model for GELU). Within each cell the substrate is exactly described.
6. **Result: a fully-geometric, visualizable, mathematically complete description of the substrate's phase structure in the essential subspace.**

The unification: **the TPN's discrete regime graph is the discrete shadow of a low-dimensional Patterson-Sullivan limit set; σ is geometric distance to the singular set; recovery is geodesic flow on the limit set; marching cubes reconstructs the geometry explicitly because the relevant structure is low-dimensional by the master theorem combined with the empirical $d_{\text{eff}}$ measurement.**

## Where this leaves "complete interpretability"

**Structural interpretability — yes, computationally tractable.** Pick a deployment corpus, project to essential subspace, partition the strata visited, fit local affine maps, marching-cubes the boundaries, emit per-step trace. Every distinguishable computation is captured; every transition is detected and quantified; every local mechanism is exactly described; the picture is drawable in 2-3 dimensions.

**Semantic interpretability — no, unchanged.** Labels on regions ("this is the `S_op` regime") are human-supplied projections. Refinement doesn't fix this.

**Computational scaling — solved if $d_{\text{eff}}$ remains low for the deployment substrate.** GPT-2 small at 2.1-2.5 is the existing measurement. Whether $d_{\text{eff}}$ stays small for larger substrates (TinyLlama-1.1B, Qwen-1.5B) is an open empirical question; the framework predicts it should stay bounded by the catastrophe corank, not by substrate width.

## Phase 27 — the validation experiment

Concrete plan to validate the framework on the existing TPN/PCG-X artefacts:

**Step 1 (½ day): Measure $d_{\text{eff}}$ from existing decision-trace outputs.**
For each Phase 24 run's `decision_trace.jsonl`, compute the Sullivan log-law:
- For each step, $-\log|\sigma(t)|$ (using σ as the discriminant proxy)
- Running max $M(t)$
- Ratio $r(t) = M(t) / \log t$
- Asymptote $r_\infty$
- $d_{\text{eff}} = 2 / r_\infty$

Acceptance: $r_\infty$ converges with std < 1% of mean over the last window, $d_{\text{eff}} \in [2, 4]$ across all 5 grammars. Predicted value from the user's framework: ~2.5.

**Step 2 (½ day): Compute the Krylov essential subspace at layer L10.**
Gradient of σ with respect to substrate activations + HVP of the projection's loss. Extract top-3 directions. Project all corpus activations into the 3-d subspace.

**Step 3 (1 day): Project + visualize.**
UMAP / PCA-fallback into 3-d. Plot the regime graph as a 3-d structure. Visually check whether regimes form a coherent Hausdorff-2.5 pattern (fractal boundary in 3-d) rather than diffuse high-dim noise.

**Step 4 (1-2 days): Marching simplices on σ in essential subspace.**
Compute σ on a regular grid in the 3-d essential subspace. Apply marching simplices to extract the isosurfaces $\sigma = \theta_{\text{normal}}$ and $\sigma = \theta_{\text{abstain}}$. Compare the reconstructed boundary geometry to the discrete regime-edge structure: do edges in the discrete graph correspond to boundary crossings in the geometric reconstruction?

**Step 5 (½ day): Local affine map fit per regime.**
Linear regression of substrate output on substrate input within each regime support. Report fit residual per regime. ReLU substrate: residual ≈ 0. GELU substrate (GPT-2): residual > 0 but bounded.

**Step 6 (1 day): Validation against the existing discrete regime graph.**
Match boundary hyperplanes (Step 4) to regime edges (existing PCG-X output). Match local affine maps (Step 5) to per-regime stats (existing `control_graph.json`). Discrepancies indicate either (a) the framework's assumptions don't hold for this substrate, or (b) the discrete extraction was under-resolved.

**End-to-end: ~one week. Output: `polyhedral_mesh.json` artefact + `phase27_d_eff_validation.json` summary + visualization PNGs.**

Outcomes:

- **If $d_{\text{eff}} \approx 2.5$ converges and the marching reconstruction matches the discrete graph:** the framework lands. Phase 28+ builds production tooling on top (geometric audit trace, geodesic recovery, analytical counterfactuals).
- **If $d_{\text{eff}}$ is higher than estimated:** the framework holds but needs more dimensions. Investigate why (substrate scaling? task complexity? layer choice?).
- **If the marching reconstruction doesn't match the discrete graph:** the discrete extraction is under-resolved OR the framework's assumptions break on synthetic grammars. Both are interesting and publishable.

## Open questions worth flagging

1. **Does $d_{\text{eff}}$ scale with substrate size?** GPT-2 small at 2.1-2.5; what about Qwen-1.5B (DVC-tracked)? Llama-class? The framework predicts $d_{\text{eff}}$ is governed by catastrophe corank, not parameter count, but this needs measurement.
2. **Does $d_{\text{eff}}$ depend on layer choice?** Phase 25 found L10 was the peak harvest layer. Is $d_{\text{eff}}$ minimal at L10? At an even later layer? The two might be related.
3. **Does the essential subspace align with SAE features?** If $d_{\text{eff}} = 2.5$, then either (a) ~2-3 SAE features dominate per regime, or (b) the essential subspace is a non-axis-aligned combination of many SAE features. (a) would be remarkable; (b) is more likely.
4. **Does the framework survive GELU non-linearity?** The Whitney stratification is robust to smooth nonlinearity; the polyhedral cell count is not exact under GELU. Need to measure the affine-fit residual per stratum on actual GPT-2 activations to see how badly the piecewise-linear approximation degrades.
5. **What's the semantic projection?** Even granted complete structural interpretability, labelling regimes with human concepts is a separate inferential step. The closest existing tool is SAE feature labelling (Anthropic-style); integrating it with the regime structure is Phase 28+.

## The honest framing for publication

**The publishable claim:** "PCG-X is structurally complete for a frozen pretrained substrate at the resolution of its Whitney stratification; combined with the catastrophe-theoretic measurement of effective dimension (Sullivan log-law / Krylov rank ≈ 2.5 on GPT-2), the regime graph admits an explicit low-dimensional geometric reconstruction via marching simplices, with per-stratum local affine maps and per-boundary hyperplane fits. Decision-by-decision audit at this resolution is computationally tractable and mathematically complete."

**What this is not:** A solution to mechanistic interpretability. A theory of feature composition. A semantic labelling system. A replacement for SAE / circuit work.

**What this is:** A bridge from the project's master theorem (Whitney + stratified partition) to a measurable, drawable, auditable interpretability artefact for any frozen pretrained substrate, valid up to the substrate's intrinsic determinacy and dependent on the empirical assumption that $d_{\text{eff}}$ is small for the substrate in question.

## Cross-references

- Master theorem: `Mathematics.md §"Whitney Stratification and Group-Action Orbit Types"` (line 1287+).
- PCG-X regime extraction: `docs/exp/e28-pcg-extractor.md`, `docs/exp/e30-pcg-extractor-pretrained.md`.
- σ ensemble: `docs/arch/singularity-detector.md`.
- Control surface: `docs/arch/control-policy.md`.
- Layer ablation (Phase 25): `research_log2.md` Phase 25 entry.
- $d_{\text{eff}}$ framework: external to this repo; lives in the user's adjacent `gpt2_attack_notions` / Cat Scanner / Patterson-Sullivan work.
- Engineering reflection: `docs/insights.md` for the broader "what is the project for" framing.

## Compaction-safe summary (for future sessions)

Read this section first if context has been compacted:

- The project's master theorem is **Whitney stratification + the stratified partition function** (`Mathematics.md §"Whitney Stratification and Group-Action Orbit Types"`). Use it; don't under-credit it.
- The PCG-X regime graph (Phase 23+) is the **discrete shadow of the stratification**. Each regime ↔ stratum; each edge ↔ stratum-boundary transition; σ ↔ distance to singular set.
- **Marching simplices in a low-dimensional essential subspace** is the natural extension that gives geometric (not just discrete) interpretability. Implementable in ~one week.
- $d_{\text{eff}}$ (Patterson-Sullivan critical exponent / Krylov rank / catastrophe corank, three equivalent definitions) is **measured at ~2.1-2.5 for GPT-2** from the user's adjacent work. If this holds, the marching-cubes approach is computationally tractable, not just architecturally clean.
- The Phase 27 experiment plan (step-by-step above) validates the framework on existing TPN artefacts in ~one week.
- The semantic gap (model representation ↔ human concepts) is **unchanged by any of this**. Marching cubes gives geometry, not concepts.
- **Bridge to be built next:** Phase 27's $d_{\text{eff}}$ measurement + marching-simplices reconstruction on the existing Phase 24/25 outputs. Three substantive deliverables: dimensional measurement, geometric reconstruction, per-stratum affine fit. Either lands as a publishable framework or honestly falsifies one of its components.
