"""Policy-Intent dataset adapter (Phase 28b).

Generates synthetic transition traces under the v1 or v2 policy-intent FSM
specs (`tests/fixtures/graphs/policy_intent_v{1,2}.fsm.yaml`). The dataset
shape matches the existing ListOps / Python / JSON adapters so it slots
into `e25_extraction_torch.GRAMMAR_DISPATCH` transparently and the entire
PCG-X pipeline runs end-to-end without further changes.

The adapter is the **Wave-A synthetic backend** of the proposal in
`docs/proposals/policy-intent-fsm-extraction.md`. A Wave-B real-trace
backend (reading `langgraph_servants` audit logs) will mirror this
interface so the experiment can swap data sources without touching the
runner.

Per-sample fields
-----------------
* ``sample_id``       Stable identifier ``policy{policy_idx}_step{step_idx}``.
* ``features``        One-hot vector over the input alphabet (3 dims for
                      v1, 5 dims for v2).
* ``observed_token``  Classifier-output equivalent class. v1:
                      ``{GRANT_EVENT, REVOKE_EVENT, NULL}``. v2:
                      ``{GRANT, REVOKE, CONDITION, REQUEST, ACKNOWLEDGE}``.
* ``prev_state``      Vertex id at the START of this transition.
* ``current_state``   Alias of ``prev_state`` (matches the ListOps adapter
                      convention).
* ``true_next_state`` Vertex id at the END of this transition, after the
                      reducer applies. For REJECTED authority-gate cases
                      this equals ``current_state`` (state unchanged).
* ``is_adversarial``  ``True`` iff the authority gate REJECTED this
                      modification. The token still appears in features,
                      but the FSM does not advance. PCG-X should learn
                      that ``(state, token, rejected)`` is observationally
                      distinct from ``(state, token, accepted)``.
* ``sequence_id``     Integer ``policy_id`` -- the index of the
                      ``(source, target, domain)`` instance this trace
                      belongs to.
* ``authority_mode``  v2 only. One of
                      ``{MASTER_OF_RELATIONSHIP, PEER_MUTUAL, SELF_ONLY, SYSTEM}``.
                      ``None`` for v1.
* ``anchor_present``  ``True`` iff the renderer would insert a ``## Speech
                      permission revoked`` (v1 WITHHELD; v2 WITHHELD or
                      CONDITIONAL) anchor block for the *current* state.

Run shape
---------
A single ``policy`` is one ``(source_id, target_id, domain)`` instance
running an independent FSM. A ``dataset`` is a flat concatenation of
multiple policy traces; the ``sequence_id`` distinguishes them.

Default: ``n_policies = 80`` (matches Phase 24 default), trace length
sampled from a small range so the total step count lands in the
~1000-2000 range typical of our other adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..arch.graph_fsm import GraphFSM

# ---------------------------------------------------------------------------
# Alphabets
# ---------------------------------------------------------------------------

POLICY_V1_TOKENS: tuple[str, ...] = ("GRANT_EVENT", "REVOKE_EVENT", "NULL")
POLICY_V2_TOKENS: tuple[str, ...] = ("GRANT", "REVOKE", "CONDITION", "REQUEST", "ACKNOWLEDGE")

# Templated natural-language renderings for the abstract event tokens. Used
# when the harvest substrate is a pretrained LM whose token priors on the
# abstract labels are weak (Phase 28b Wave-A v2 follow-up). The template
# slot is the dialogue-zone first-token of a short turn that *would* have
# elicited the corresponding policy intent under the runtime's authority
# gate; the rendering is intentionally short and unambiguous.
POLICY_V1_TEMPLATES: dict[str, str] = {
    "GRANT_EVENT": "user grants speech permission",
    "REVOKE_EVENT": "user revokes speech permission",
    "NULL": "user makes a neutral comment",
}
POLICY_V2_TEMPLATES: dict[str, str] = {
    "GRANT": "user grants speech permission",
    "REVOKE": "user revokes speech permission",
    "CONDITION": "user grants conditional speech permission",
    "REQUEST": "servant requests speech permission",
    "ACKNOWLEDGE": "servant acknowledges the directive",
}


def template_for(token: str, version: str) -> str:
    """Return the natural-language template for an abstract event token."""
    table = POLICY_V1_TEMPLATES if version == "v1" else POLICY_V2_TEMPLATES
    if token not in table:
        raise KeyError(f"unknown {version} token {token!r}")
    return table[token]

AUTHORITY_MODES_V2: tuple[str, ...] = (
    "MASTER_OF_RELATIONSHIP",
    "PEER_MUTUAL",
    "SELF_ONLY",
    "SYSTEM",
)

# Renderer anchor: WITHHELD always shows the "## Speech permission revoked"
# block; CONDITIONAL (v2 only) shows a conditional grant anchor.
ANCHOR_STATES: dict[str, frozenset[str]] = {
    "v1": frozenset({"WITHHELD"}),
    "v2": frozenset({"WITHHELD", "CONDITIONAL"}),
}


@dataclass
class PolicySample:
    """One transition sample in a policy-intent trace.

    Notes
    -----
    The classifier sees ``features`` (one-hot over the input alphabet) and
    must predict ``true_next_state`` given ``current_state``. The authority
    gate is *not* visible to the classifier as a separate input -- it
    only affects which transitions are observable in the trace
    (REJECT --> state stays at ``current_state``). PCG-X regime extraction
    should surface "the LLM treats (state, token, rejected) as a distinct
    regime" if and only if the LLM internalizes the gate.
    """

    sample_id: str
    features: np.ndarray
    observed_token: str
    prev_state: str
    current_state: str
    true_next_state: str
    is_adversarial: bool  # REJECT by authority gate
    sequence_id: int
    authority_mode: str | None
    anchor_present: bool


@dataclass
class PolicyDataset:
    """Materialised dataset matching the ListOps / Python adapter shape."""

    samples: list[PolicySample]
    feature_dim: int
    fsm: GraphFSM
    version: str  # "v1" or "v2"
    X: np.ndarray
    y_next: np.ndarray
    prev_states: np.ndarray
    current_states: np.ndarray
    is_adversarial: np.ndarray
    authority_mode_ids: np.ndarray  # int64, -1 for v1 (mode = None)
    anchor_present: np.ndarray  # bool


# ---------------------------------------------------------------------------
# Reducer (deterministic transition function)
# ---------------------------------------------------------------------------


def _reducer(current_state: str, token: str, version: str) -> str:
    """Pure deterministic next-state function under the policy-intent FSM.

    Mirrors the langgraph_servants reducer exactly. The authority gate is
    applied OUTSIDE this function: if REJECTED, the caller keeps
    ``current_state`` and never invokes the reducer.
    """
    if version == "v1":
        if token == "GRANT_EVENT":
            return "ALLOWED"
        if token == "REVOKE_EVENT":
            return "WITHHELD"
        if token == "NULL":
            return current_state
        raise ValueError(f"unknown v1 token {token!r}")
    if version == "v2":
        if token == "GRANT":
            return "GRANTED"
        if token == "REVOKE":
            return "WITHHELD"
        if token == "CONDITION":
            return "CONDITIONAL"
        if token in {"REQUEST", "ACKNOWLEDGE"}:
            return current_state
        raise ValueError(f"unknown v2 token {token!r}")
    raise ValueError(f"unknown version {version!r}")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _one_hot(token: str, alphabet: Sequence[str]) -> np.ndarray:
    vec = np.zeros(len(alphabet), dtype=np.float64)
    vec[alphabet.index(token)] = 1.0
    return vec


def generate_policy_intent_dataset(
    fsm: GraphFSM,
    n_policies: int = 80,
    seed: int = 42,
    *,
    version: str = "v2",
    min_steps_per_policy: int = 12,
    max_steps_per_policy: int = 28,
    reject_rate: float = 0.15,
    token_weights: dict[str, float] | None = None,
    rendering: str = "abstract",
) -> PolicyDataset:
    """Generate synthetic ``(source, target, domain)`` traces under the FSM.

    Parameters
    ----------
    fsm:
        Loaded ``GraphFSM`` from ``policy_intent_v{1,2}.fsm.yaml``.
    n_policies:
        Number of independent ``(source, target, domain)`` instances
        (analog of ``n_programs`` in the grammar adapters).
    seed:
        RNG seed for reproducibility.
    version:
        ``"v1"`` (2-state binary) or ``"v2"`` (4-state).
    min_steps_per_policy, max_steps_per_policy:
        Per-trace length is uniformly sampled in this inclusive range. The
        defaults give ~ 12 * 80 = 960 to 28 * 80 = 2240 steps total --
        comparable to the smaller grammar sweeps (listops 144,
        python_expr 1612).
    reject_rate:
        Probability that the authority gate REJECTS a sampled intent.
        Rejected steps appear in the trace with ``is_adversarial=True``
        and ``true_next_state == current_state``.
    token_weights:
        Optional non-uniform prior over the input alphabet. Defaults to
        uniform. Useful for stress-testing the regime extraction under
        skewed distributions (e.g. mostly NULL with rare flips).
    rendering:
        How to materialize each step's ``observed_token`` for downstream
        harvest:

        * ``"abstract"`` (default) -- the abstract event label
          (``"GRANT"``, ``"REVOKE"``, ...). Matches the original Phase 28b
          Wave-A baseline.
        * ``"templated"`` -- a short natural-language sentence rendering
          (e.g. ``"user grants speech permission"``). Useful when the
          harvest substrate is a pretrained LM with weak priors on the
          abstract labels; gives the LLM contextual signal it has been
          trained on. The ``features`` one-hot is unchanged (still over
          the 3-token v1 / 5-token v2 alphabet); only the
          ``observed_token`` string differs.

    Returns
    -------
    PolicyDataset
        Concatenated samples across all ``n_policies`` instances.
    """
    if rendering not in {"abstract", "templated"}:
        raise ValueError(f"rendering must be 'abstract' or 'templated', got {rendering!r}")
    if version not in {"v1", "v2"}:
        raise ValueError(f"version must be 'v1' or 'v2', got {version!r}")
    alphabet = POLICY_V1_TOKENS if version == "v1" else POLICY_V2_TOKENS
    feature_dim = len(alphabet)
    rng = np.random.default_rng(int(seed))

    # Token sampler: uniform unless overridden.
    if token_weights is None:
        probs = np.ones(len(alphabet)) / len(alphabet)
    else:
        probs = np.array([token_weights.get(t, 0.0) for t in alphabet], dtype=np.float64)
        if probs.sum() <= 0.0:
            raise ValueError("token_weights must have a positive sum")
        probs = probs / probs.sum()

    start_state = "ALLOWED" if version == "v1" else "UNESTABLISHED"
    anchor_states = ANCHOR_STATES[version]

    samples: list[PolicySample] = []
    for policy_idx in range(int(n_policies)):
        n_steps = int(rng.integers(min_steps_per_policy, max_steps_per_policy + 1))
        # One authority mode per policy in v2 (deployment-realistic: the
        # gate is set by the relationship configuration, not per turn).
        if version == "v2":
            authority_mode = AUTHORITY_MODES_V2[int(rng.integers(len(AUTHORITY_MODES_V2)))]
        else:
            authority_mode = None

        current_state = start_state
        for step_idx in range(n_steps):
            token = alphabet[int(rng.choice(len(alphabet), p=probs))]
            rejected = bool(rng.random() < reject_rate)
            if rejected:
                true_next_state = current_state
            else:
                true_next_state = _reducer(current_state, token, version)
            anchor_present = current_state in anchor_states

            rendered = token if rendering == "abstract" else template_for(token, version)

            samples.append(
                PolicySample(
                    sample_id=f"policy{policy_idx}_step{step_idx}",
                    features=_one_hot(token, alphabet),
                    observed_token=rendered,
                    prev_state=current_state,
                    current_state=current_state,
                    true_next_state=true_next_state,
                    is_adversarial=rejected,
                    sequence_id=int(policy_idx),
                    authority_mode=authority_mode,
                    anchor_present=anchor_present,
                )
            )

            current_state = true_next_state

    # Materialise numpy arrays.
    vertex_ids = fsm.vertex_ids
    vidx = {v: i for i, v in enumerate(vertex_ids)}
    n = len(samples)
    X = np.stack([s.features for s in samples], axis=0)
    y_next = np.asarray([vidx[s.true_next_state] for s in samples], dtype=np.int64)
    prev_states = np.asarray([vidx[s.prev_state] for s in samples], dtype=np.int64)
    current_states = prev_states.copy()
    is_adversarial = np.asarray([s.is_adversarial for s in samples], dtype=bool)
    anchor_present_arr = np.asarray([s.anchor_present for s in samples], dtype=bool)

    if version == "v2":
        amap = {m: i for i, m in enumerate(AUTHORITY_MODES_V2)}
        authority_mode_ids = np.asarray(
            [amap[s.authority_mode] for s in samples], dtype=np.int64
        )
    else:
        authority_mode_ids = -np.ones(n, dtype=np.int64)

    return PolicyDataset(
        samples=samples,
        feature_dim=feature_dim,
        fsm=fsm,
        version=version,
        X=X,
        y_next=y_next,
        prev_states=prev_states,
        current_states=current_states,
        is_adversarial=is_adversarial,
        authority_mode_ids=authority_mode_ids,
        anchor_present=anchor_present_arr,
    )


# ---------------------------------------------------------------------------
# Audit helpers (kept lightweight; the proposal's anchor-ablation control
# and v1->v2 migration audit are implemented in scripts/, not here)
# ---------------------------------------------------------------------------


def transition_table(version: str) -> dict[tuple[str, str], str]:
    """Return the full ``(prev_state, token) -> next_state`` table.

    Used by unit tests to verify the reducer is consistent with the YAML
    FSM and the policy-intent docs. Also useful for the v1->v2 migration
    audit script to compare expected vs. observed transition matrices.
    """
    if version == "v1":
        states = ("ALLOWED", "WITHHELD")
        tokens = POLICY_V1_TOKENS
    elif version == "v2":
        states = ("UNESTABLISHED", "GRANTED", "WITHHELD", "CONDITIONAL")
        tokens = POLICY_V2_TOKENS
    else:
        raise ValueError(f"unknown version {version!r}")
    return {(s, t): _reducer(s, t, version) for s in states for t in tokens}


def alphabet_for(version: str) -> tuple[str, ...]:
    """Return the input alphabet for the given FSM version."""
    if version == "v1":
        return POLICY_V1_TOKENS
    if version == "v2":
        return POLICY_V2_TOKENS
    raise ValueError(f"unknown version {version!r}")


__all__: Iterable[str] = (
    "PolicySample",
    "PolicyDataset",
    "POLICY_V1_TOKENS",
    "POLICY_V2_TOKENS",
    "POLICY_V1_TEMPLATES",
    "POLICY_V2_TEMPLATES",
    "AUTHORITY_MODES_V2",
    "ANCHOR_STATES",
    "generate_policy_intent_dataset",
    "template_for",
    "transition_table",
    "alphabet_for",
)
