# Experiments

> Source: chat-export-2026-05-05T04-35-07.md, owned chunks of the Experiments piece (6 chunks of 72 total).

## Experiment Register and Ablation Plan

*This section introduces the planned experimental register, opening with the classifier-to-graph sanity check and stepping through the early entries that motivate the rest of the program.*

### E0–E4: Sanity Check Through Singularity Detection

#### E0 — Classifier-to-Graph Sanity Check

**Purpose.** Show that ordinary classifier output can be lifted into:

$$x \to \text{typed label} + \text{distribution} + \text{confusion graph} + \text{singularity flag}$$

**Dataset.** MNIST or sklearn digits.

**Baseline.** Plain classifier:

$$x \to p(y \mid x)$$

**Structured version.**

$$x \to p(y) \to \text{typed digit state} \to \text{margin singularity} \to \text{confusion graph}$$

**Metrics.**

| Metric | Meaning |
| --- | --- |
| Accuracy | raw classifier quality |
| Low-margin accuracy | accuracy near decision wall |
| Confusion graph density | which classes are geometrically adjacent |
| Failure localization | whether ambiguous cases are flagged |

**Expected result.** Not novel, but confirms:

> classifier uncertainty induces useful graph/singularity structure

---

#### E1 — Synthetic BabyAI-like Graph Mask Test

**Purpose.** Test whether graph legality constraints improve prediction efficiency.

**Baseline.** Classifier predicts next state from features:

$$x_t \to q_{t+1}$$

without knowing legal transitions.

**Structured version.** Classifier predicts scores, then graph masks illegal moves:

$$p(q_{t+1} \mid x_t) \quad \text{masked by} \quad (q_t, q_{t+1}) \in E$$

**Synthetic result already observed.**

| Setup | Accuracy | Illegal transition rate |
| --- | --- | --- |
| Classifier only | 86.5% | 10.5% |
| Classifier + graph mask | 96.8% | 0.0% |
| Classifier with current state | 99.0% | 0.0% |

**Interpretation.** This validates the basic efficiency claim:

> known graph constraints reduce action search and eliminate invalid transitions

---

#### E2 — Real BabyAI / MiniGrid

**Purpose.** Test the same idea on a real grammar/action benchmark.

**Hypothesis.** A typed graph controller reaches target performance with fewer samples than a flat policy.

**Conditions.**

| Condition | Description |
| --- | --- |
| Flat policy | observation → action |
| Concept bottleneck | observation → typed fields → action |
| Graph FSM | observation → typed fields → legal graph transition |
| Graph FSM + uncertainty | same plus distribution and singularity flag |

**Metrics.**

- success rate
- sample complexity
- illegal action rate
- recovery rate after ambiguity

**Load-bearing question.** Does the graph reduce learning burden?

---

#### E3 — Hyperbolic Graph Extrusion

**Purpose.** Test whether the graph should extrude into hyperbolic geometry.

**Setup.** Embed task nodes into $\mathbb{R}^d$ versus $\mathbb{H}^d$. Then classify states by nearest prototype or edge tube.

**Hypothesis.** For hierarchical tasks, $\mathbb{H}^d$ achieves equal or better performance with fewer dimensions.

**Metrics.**

| Metric | Meaning |
| --- | --- |
| Accuracy vs dimension | compression efficiency |
| Distance-to-goal correlation | whether geometry reflects progress |
| Radius-depth correlation | whether hyperbolic radius tracks task specificity |
| Boundary-escape score | rabbit-hole detection |

**Expected efficiency gain.**

> hyperbolic geometry compresses hierarchy

---

#### E4 — Singularity Detector

**Purpose.** Determine whether singularity flags predict failures better than raw confidence.

**Singularity definitions.**

$$\Sigma = \{\text{low margin}, \text{decision tie}, \text{contradiction collision}, \text{illegal transition pressure}, \text{loop risk}, \text{stabilizer jump}\}$$

**Baselines.**

| Baseline | Comparator |
| --- | --- |
| max softmax confidence | ordinary classifier uncertainty |
| entropy | distributional uncertainty |
| margin | top-1 / top-2 distance |

**Structured detector.**

$$\sigma(x) = w_1 \cdot \text{margin}^{-1} + w_2 \cdot \text{contradiction} + w_3 \cdot \text{loop pressure} + w_4 \cdot \text{illegal transition pressure}$$

**Metrics.**

- AUROC for failure
- precision at top-k risky states
- cost saved by early intervention

**Load-bearing question.** Does singularity theory add a useful failure detector, or is margin enough?

---

## Ablation Descriptions E4–E7

*This section continues the experimental register from IDF rare-stratum weighting through the full agent trace benchmark, then specifies the ablation matrix, efficiency targets, and the minimal next experiment.*

### E5–E9 and the Ablation Matrix

#### E5 — IDF Rare-Stratum Weighting

**Purpose.** Test whether rare grammar states and rare transitions deserve more training/control weight.

**Weight.**

$$\mathrm{idf}(\lambda) = \log \frac{N + \alpha}{n_\lambda + \alpha}$$

**Apply to.**

| Object | IDF target |
| --- | --- |
| grammar objects | rare rules/types |
| transitions | rare state changes |
| strata | rare behavioral regions |
| singularities | rare failure modes |

**Training loss.**

$$\mathcal{L} = \sum_i \mathrm{idf}(\lambda_i) \, \ell(f(x_i), \lambda_i)$$

**Metrics.**

| Metric | Meaning |
| --- | --- |
| rare-state recall | catches uncommon regimes |
| rare-failure recall | catches important edge cases |
| overall accuracy | ensure no collapse |
| false alarm rate | avoid overreacting to noise |

**Load-bearing question.** Does IDF improve meaningful rare-event handling?

---

#### E6 — Group-Action Quotient Attention

**Purpose.** Test whether group actions decimate all-to-all attention efficiently.

**Baseline.** Dense object attention: $O(n^2)$.

**Structured version.** Group equivalent objects into orbits $V/H$, then attend over orbit pairs: $O(|V/H|^2)$. Then re-expand high-IDF or singular orbits.

**Metrics.**

| Metric | Meaning |
| --- | --- |
| FLOPs | compute reduction |
| latency | runtime reduction |
| success rate | task quality |
| re-expansion accuracy | whether compressed orbits recover details |
| orbit purity | whether grouped objects are truly equivalent |

**Load-bearing question.** Can semantic quotienting reduce attention without losing task-relevant information?

---

#### E7 — Reservoir Readout Experiment

**Purpose.** Test the reservoir-computing interpretation.

**Baseline.** Train or finetune a policy model.

**Structured reservoir version.** Freeze encoder/LLM:

$$x_t \to h_t$$

Train only readout:

$$h_t \to Y_t \to P(Y_t) \to q_{t+1}$$

**Metrics.**

| Metric | Meaning |
| --- | --- |
| trainable parameters | efficiency |
| samples to convergence | sample efficiency |
| accuracy | control quality |
| calibration | score reliability |
| transfer | generalization |

**Hypothesis.**

> frozen reservoir + typed graph readout gives high efficiency per parameter

---

#### E8 — Transfer Experiment

**Purpose.** Test whether the graph abstraction transfers across task families.

**Source.** Train on one set of tasks.

**Target.** Evaluate on related but different tasks.

**Compare.**

| Model | Transfer mechanism |
| --- | --- |
| flat policy | raw behavior transfer |
| concept bottleneck | typed-score transfer |
| graph FSM | state-transition transfer |
| hyperbolic graph | geometric hierarchy transfer |

**Metrics.**

- $R_t$ — target risk
- $R_t - R_s$ — transfer gap
- $\operatorname{Div}(P_s^\phi, P_t^\phi)$ — structured-domain divergence

**Load-bearing question.** Does the structured abstraction reduce domain shift?

---

#### E9 — Full Agent Trace Benchmark

**Purpose.** Run the complete architecture on multi-step agent traces.

**Task examples.**

| Task | Why useful |
| --- | --- |
| BabyAI | grammar/action control |
| ALFWorld | procedural transfer |
| ScienceWorld | reasoning and experimentation |
| research-agent traces | closest to target application |

**Full system.**

$$x_t \to h_t \to Y_t \to z_{\mathbb{H}} \to P(Y_t, z_t) \to \Gamma/H \to E \to p(q_{t+1}) \to \Sigma \to \phi(x_t)$$

**Metrics.**

| Metric | Meaning |
| --- | --- |
| success rate | solves task |
| invalid transition rate | graph discipline |
| average steps | efficiency |
| token/tool cost | resource efficiency |
| loop rate | avoids churn |
| failure prediction | singularity usefulness |
| transfer | abstraction quality |
| interpretability | debugging quality |

---

### Ablation Matrix

To prove efficiency, run ablations:

| Variant | Removed component |
| --- | --- |
| A0 | Full system |
| A1 | no graph mask |
| A2 | no typed grammar |
| A3 | no uncertainty distribution |
| A4 | no singularity detector |
| A5 | no IDF weighting |
| A6 | Euclidean instead of hyperbolic |
| A7 | no group quotienting |
| A8 | trained end-to-end instead of reservoir readout |
| A9 | no trace history |

The efficiency claim is supported only if removals hurt at least one of:

- compute
- sample efficiency
- illegal action rate
- failure prediction
- transfer
- debugging time

### Efficiency Targets

| Target | Desired outcome |
| --- | --- |
| Graph mask | 0 illegal transitions |
| Typed grammar | fewer samples to stable behavior |
| Hyperbolic geometry | same accuracy with lower dimension |
| Group quotient | lower attention cost |
| IDF | better rare-failure recall |
| Singularity detector | better failure prediction |
| Reservoir readout | fewer trainable parameters |
| Distribution | fewer brittle decisions |

### Current Evidence Level

| Claim | Current support |
| --- | --- |
| Graph masks improve valid prediction | Supported by synthetic toy test |
| Classifier uncertainty maps to singular regions | Supported by toy digits/MNIST-style slice |
| Hyperbolic geometry improves this system | Not yet tested |
| Group-action quotienting improves efficiency | Not yet tested |
| IDF improves rare-stratum handling | Not yet tested |
| Reservoir readout is enough | Not yet tested |
| Full architecture improves agent efficiency | Not yet tested |

### Minimal Next Experiment

The next best experiment is:

> BabyAI/MiniGrid with graph FSM vs flat policy

Minimum viable comparison:

| Model | Description |
| --- | --- |
| Flat classifier | observation → action |
| Typed classifier | observation → typed fields → action |
| Graph classifier | observation → typed fields → legal transition |
| Graph + singularity | same plus low-margin recovery |
| Hyperbolic graph | graph nodes embedded in $\mathbb{H}^d$ |

**Primary metrics.**

- success rate
- sample efficiency
- illegal transitions
- steps per episode
- failure prediction

That is the cleanest register entry to run next.

---

## MNIST Typed-State Classifier Run

*This section reports the concrete sklearn digits experiment, where the headline 97.78% accuracy is less interesting than the gap between high-margin and low-margin singular regions.*

### 97.78% on sklearn digits and the Singularity Slice

The classifier score we discussed was **97.78%** on the sklearn `load_digits` dataset, which is an 8×8 handwritten digit dataset, not full 28×28 MNIST.

Yes — that score is expected for a simple classical classifier on that dataset. It is good, but not surprising. `load_digits` is much smaller and easier than full MNIST, and logistic regression / small MLPs commonly reach the mid-to-high 90s.

The more interesting result was not the raw accuracy. It was this:

| Region | Accuracy |
| --- | --- |
| Non-singular / high-margin samples | ~98.2% |
| Singular / low-margin samples | ~66.7% |
| Overall | ~97.8% |

That is exactly what we would expect if the singularity detector is doing something meaningful. Low-margin samples are the ambiguous boundary cases where the classifier is unsure, such as:

$$1 \leftrightarrow 8, \quad 5 \leftrightarrow 9, \quad 7 \leftrightarrow 9$$

So the toy result supports the pipeline:

$$\text{classifier probabilities} \to \text{margin} \to \text{confusion graph} \to \text{singularity flag}$$

But it does not prove the full hyperbolic graph-agent architecture. It only validates a small piece: that classifier uncertainty can be turned into typed graph/singularity structure.

The honest conclusion:

> 97.78% was expected; the useful signal was that low-margin singular regions were much less reliable.

---

### User

> Was it the full architecture?

### Assistant

No. It was not the full architecture.

It was a toy slice of the architecture:

$$\text{image} \to \text{classifier probabilities} \to \text{typed label} \to \text{margin/singularity flag} \to \text{confusion graph}$$

What it included:

| Component | Included? |
| --- | --- |
| ML classifier | Yes |
| typed output labels | Yes |
| probability distribution | Yes |
| uncertainty/margin detector | Yes |
| confusion graph | Yes |
| singularity-like low-margin region | Yes |
| graph state machine | No |
| hyperbolic embedding | No |
| grammar DSL | No |
| group-action quotienting | No |
| IDF weighting | No |
| energy/partition function $Z$ | No |
| sheaf/local memory | No |
| agent trace dynamics | No |

So the MNIST/digits test only showed:

> classifier output can be converted into graph + uncertainty/singularity structure

It did not test:

> hyperbolic graph extrusion + grammar + state machine + distribution + singularity dynamics

A fuller toy version would use a hierarchical task dataset or synthetic agent traces, not digits. Digits are good for testing the classifier-to-graph bridge, but they are not a good test for the full research-agent architecture.

---

### User

> Do you have a data set that would be a good pic[k]

### Assistant

I'll pick a dataset based on what would actually test the full architecture, not just classifier accuracy. The key requirement is state/action traces with a grammar-like task structure, preferably hierarchical.

---

## BabyAI Task Selection and Setup

*This section documents the dataset choice — BabyAI as the first testbed, then ALFWorld and ScienceWorld — and the fallback to a synthetic grid benchmark when MiniGrid is unavailable.*

### Choosing BabyAI / ALFWorld / ScienceWorld

Yes — a better pick than MNIST would be **BabyAI** first, then **ALFWorld** or **ScienceWorld**.

#### Best first dataset: BabyAI / MiniGrid

BabyAI is a grounded language-learning platform built around compositional instructions in a gridworld. It is explicitly meant to study sample efficiency and language-conditioned agent learning, and the benchmark has levels of increasing difficulty.

Why it is a good pick for our architecture:

| Requirement | BabyAI fit |
| --- | --- |
| Typed grammar | Instructions are compositional and rule-like |
| Task graph | Gridworld states/actions form a clean transition graph |
| State machine | Agent has explicit actions and goals |
| Classifier labels | We can classify state as Navigate, PickUp, OpenDoor, GoToObject, etc. |
| Distribution | Predict next action or next subgoal |
| Singularity detector | Detect ambiguous instruction/state/action ties |
| Hyperbolic geometry | Test hierarchy: mission → subgoal → object → action |

So BabyAI is the cleanest controlled testbed for:

$$\text{grammar} \to \text{graph} \to \text{classifier} \to \text{state machine} \to \text{distribution}$$

It is small enough to debug but structured enough to matter.

#### Best "agentic research-like" dataset: ALFWorld

ALFWorld is stronger if we want something closer to real agent procedures. It aligns abstract text-based policies with embodied action environments, letting agents reason in a high-level text world before acting in a grounded environment.

Why it is good:

| Requirement | ALFWorld fit |
| --- | --- |
| High-level task graph | Household tasks decompose into subtasks |
| Procedures | find, take, open, put, clean, heat, etc. |
| State transitions | Explicit action sequences |
| Transfer | TextWorld abstraction ↔ embodied ALFRED-style tasks |
| Graph/geometry test | Can test whether graph abstraction transfers |

This is probably the best second-stage benchmark after BabyAI.

#### Best science/research-adjacent pick: ScienceWorld

ScienceWorld is an interactive text environment for testing agents' scientific reasoning abilities, built around elementary-school science tasks.

This is closer to a "research agent" because it involves experimentation, observation, reasoning, and explanations. It is less minimal than BabyAI, but more semantically relevant.

#### My recommendation

Use this progression:

> BabyAI → ALFWorld → ScienceWorld

- BabyAI tests the architecture in a clean grammar/action world.
- ALFWorld tests transfer from abstract graph policy to grounded action.
- ScienceWorld tests whether the same structure helps with reasoning-heavy agent behavior.

If I had to pick one dataset right now: **BabyAI** — because it gives us the fastest path to proving whether the typed classifier + graph state machine + distribution idea works before adding harder semantics.

---

### User

> Can you run it

### Assistant

I'll try to run a small version locally. If BabyAI/MiniGrid is not installed in this sandbox, I'll run a faithful synthetic grid-task benchmark that tests the same classifier → graph-state → distribution pipeline.

BabyAI/MiniGrid are not installed in this runtime, so I'm running a synthetic BabyAI-like grid-task benchmark instead. It will still exercise the pipeline: typed task states, legal graph transitions, classifier probabilities, masking by grammar, and singularity detection.

---

## Synthetic BabyAI Run and Results

*This section reports the synthetic BabyAI-like benchmark — first the headline result that the graph mask boosts accuracy from 86.5% to 96.8% with no novelty claim, then the deeper sweep showing the graph is load-bearing only when the classifier does not already see current state.*

### First Synthetic Sweep and Honest Reading

I tried to run BabyAI/MiniGrid directly, but those packages are not installed in this sandbox. So I ran a synthetic BabyAI-like grid-task benchmark instead.

It tested this slice of the architecture:

$$\text{typed task state} \to \text{classifier scores} \to \text{legal graph transition mask} \to \text{next-state distribution} \to \text{singularity flag}$$

**Synthetic task setup.**

- States: `Parse`, `Navigate`, `ResolveDoor`, `Pickup`, `Deliver`, `Interact`, `Done`
- Tasks: `goto object`, `pickup object`, `open door`, `put object near target`

The model had to predict the next task state from grid/task features.

**Result.**

| Model setup | Accuracy | Illegal transition rate |
| --- | --- | --- |
| Classifier without graph/current-state structure | 86.5% | 10.5% |
| Same classifier + graph legality mask | 96.8% | 0.0% |
| Classifier with current state included | 99.0% | 0.0% |

That is a good sign. The graph/state-machine layer was load-bearing when the scorer was weaker: it boosted accuracy from 86.5% to 96.8% and eliminated illegal transitions.

**Singularity detector.** I flagged the lowest-margin 10% of predictions as "singular / ambiguous."

| Region | Accuracy |
| --- | --- |
| Singular low-margin states | 81.7% |
| Non-singular states | 98.5% |

That is also what we wanted: the singularity detector identified the part of the state space where the classifier was much less reliable.

**What this run proves.** It supports the core pipeline:

> ML scorer + typed grammar + graph state machine + distribution + singularity flag

It does not yet test the full hyperbolic version. This run did not include learned Poincaré embeddings, group-action quotienting, IDF expansion, or a full partition function. It was the first sanity check: does the graph-constrained classifier work better than a free classifier? On this toy benchmark, yes.

---

### User

> Okay so nothing new so far

### Assistant

Mostly correct: nothing fundamentally new has been demonstrated yet.

What we have so far is a recombination of known pieces:

$$\text{classifier} + \text{typed features} + \text{finite state machine} + \text{graph transition constraints} + \text{uncertainty/margin flag}$$

That is not new by itself. It is close to:

- structured prediction
- probabilistic finite-state controllers
- constrained decoding
- concept bottleneck models
- graphical models
- hierarchical policies
- uncertainty-aware classifiers

The synthetic run only showed the obvious but important sanity check:

$$\text{classifier + legal graph mask} > \text{free classifier}$$

That is expected. If you know the legal transition graph, masking illegal moves should improve accuracy and eliminate invalid transitions.

#### Where novelty could appear

The potentially interesting part is not the basic classifier/state-machine architecture. It is the full synthesis:

$$\text{LLM/ML outputs} \to \text{typed grammar} \to \text{hyperbolic graph extrusion} \to \text{group-action quotient attention} \to \text{stratified/singularity diagnostics} \to \text{IDF-weighted re-expansion} \to \text{probabilistic controller}$$

That combination is unusual. But we have not proven it is better yet.

#### What would make it new

It becomes new or at least research-worthy if we can show one of these:

1. Hyperbolic graph extrusion improves transfer over a flat graph controller.
2. Group-action quotienting reduces attention/computation without hurting performance.
3. Singularity flags predict failures better than ordinary confidence/margin.

---

### Ablation Sweep with Sample Counts

#### User

> Do it

#### Assistant

I'll run the next experiment rather than just outline it. I'll first check whether MiniGrid/BabyAI is available; if not, I'll run the closest reproducible local benchmark with the same ablation structure: flat classifier, typed classifier, graph-masked classifier, graph+singularity.

MiniGrid/BabyAI are not available here, but sklearn and PyTorch are. I'm running a controlled synthetic grid-agent trace benchmark with the same ablations so we get actual numbers rather than a design sketch.

I ran it. BabyAI/MiniGrid were not installed in this runtime, so I ran a controlled synthetic BabyAI-like grid-task benchmark with the same ablation structure.

The tested pipeline was:

$$\text{features} \to \text{classifier} \to \text{typed scores} \to \text{graph legality mask} \to \text{next-state distribution} \to \text{singularity flag}$$

- **States:** Parse, Navigate, ResolveDoor, Pickup, Deliver, Interact, Done
- **Tasks:** goto, pickup, open_door, put_near, toggle

#### Main result

When the scorer did not receive current-state identity, but the graph controller did, the graph mask was strongly load-bearing:

| Train samples | Typed features | Flat classifier | + graph mask | Illegal transition rate before mask |
| ---: | :---: | ---: | ---: | ---: |
| 100 | no | 66.6% | 76.7% | 10.2% |
| 100 | yes | 67.1% | 77.3% | 10.4% |
| 500 | no | 67.4% | 77.3% | 10.0% |
| 500 | yes | 67.8% | 78.4% | 10.9% |
| 1000 | no | 67.6% | 77.5% | 10.0% |
| 1000 | yes | 67.8% | 78.4% | 10.9% |

So the graph mask gives about +10 accuracy points and eliminates illegal transitions. That supports the efficiency claim:

> external graph structure can remove search space the classifier would otherwise have to learn.

#### When the classifier already saw current state

When the classifier received current-state identity as a feature, it mostly learned the transition legality itself. Then the graph mask added almost nothing:

| Train samples | Typed features | Flat classifier | + graph mask |
| ---: | :---: | ---: | ---: |
| 100 | no | 79.8% | 79.8% |
| 100 | yes | 79.4% | 80.0% |
| 500 | no | 83.9% | 83.9% |
| 500 | yes | 85.4% | 85.4% |
| 1000 | no | 84.1% | 84.1% |
| 1000 | yes | 86.5% | 86.5% |

Here the graph is less visible because the model already got enough information to infer the graph. The typed feature layer helped more: roughly +2 to +3 points at 500–1000 samples.

#### Singularity detector

I flagged the lowest-margin 10% of graph-masked predictions as "singular/ambiguous." With current-state features and typed fields:

| Train samples | Singular-region accuracy | Non-singular accuracy | Failure AUROC from margin |
| ---: | ---: | ---: | ---: |
| 100 | 47.2% | 83.7% | 0.81 |
| 500 | 56.0% | 88.7% | 0.84 |
| 1000 | 56.8% | 89.8% | 0.85 |

That is useful: low-margin regions were much less reliable, and margin predicted failure reasonably well.

#### Readout

This experiment says:

- Graph constraints are efficient when the classifier would otherwise waste capacity learning impossible transitions.
- Typed features help when the scorer has enough context to use them.
- Singularity flags are useful because low-margin states are much more failure-prone.

This was not a real BabyAI run and not the full hyperbolic architecture.

The clean takeaway:

> The graph is load-bearing when it externalizes structure the model has not already learned.

If the model already sees and learns the state-machine structure, the graph becomes less of an accuracy booster, but still valuable as a hard safety/validity constraint.
