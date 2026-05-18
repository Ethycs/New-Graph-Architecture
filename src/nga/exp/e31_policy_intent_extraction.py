"""E31 - Policy-Intent FSM extraction on a frozen pretrained substrate.

Phase 28b runner. Thin wrapper around ``e30_pcg_extractor_pretrained.run_e30``
that pins the grammar to a policy-intent FSM (v1 or v2), renames
``n_programs -> n_policies`` for clarity, and centralises the FSM-YAML
path resolution.

The full PCG-X pipeline (predictive projection, argmax partition,
bisimulation merge, regime-graph emit, sigma + control bridge, decision
trace, optional held-out eval) runs unchanged via the existing
``GRAMMAR_DISPATCH`` entries ``policy_intent_v1`` / ``policy_intent_v2``
wired up in ``e25_extraction_torch``.

The pre-registered acceptance bars live in
``docs/proposals/policy-intent-fsm-extraction.md``:

    A1  - v1 binary sanity         eff_rank(grad) < 1.5 AND top-3 > 0.95
    A2  - v2 4-state headline      2.0 <= eff_rank(grad) <= 4.0 AND top-3 >= 0.85
    A3  - cluster purity vs gold   >= 0.75 averaged across policy instances
    A4  - ABSTAIN on disagreement  sigma > 0.7 on >= 70% of adversarial held-out turns
    A5  - anchor-block ablation    cluster purity drop >= 0.20 absolute
    A6  - v1->v2 migration audit   diagnostic (separate vs merged True-default-GRANTED)

This runner covers A1 + A2 directly via ``E30Result.mean_purity_against_current_state``
and the existing Phase-27 Step 4 Krylov measurement on the same harvested
activations. A3 is reported in the sweep summary; A4-A6 require the
``scripts/phase28b_policy_intent_sweep.py`` driver which composes this
runner with anchor-ablation replay and held-out adversarial turns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..arch.graph_fsm import GraphFSM
from ..drivers import graph_fsm_spec
from .e25_extraction_torch import GRAMMAR_DISPATCH
from .e30_pcg_extractor_pretrained import (
    _DEFAULT_HARVEST_LAYER,
    _DEFAULT_MODEL_ID,
    E30Result,
    run_e30,
)

PolicyVersion = Literal["v1", "v2"]


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _grammar_name_for_version(version: PolicyVersion, rendering: str = "abstract") -> str:
    if version not in {"v1", "v2"}:
        raise ValueError(f"unknown policy version {version!r}; expected 'v1' or 'v2'")
    if rendering not in {"abstract", "templated"}:
        raise ValueError(f"rendering must be 'abstract' or 'templated', got {rendering!r}")
    suffix = "" if rendering == "abstract" else "_templated"
    return f"policy_intent_{version}{suffix}"


def _load_policy_fsm(version: PolicyVersion) -> GraphFSM:
    """Load the v1 or v2 policy-intent FSM from the canonical YAML."""
    grammar = _grammar_name_for_version(version)
    dispatch = GRAMMAR_DISPATCH[grammar]
    yaml_path = _REPO_ROOT / dispatch["fsm_yaml_path"]
    spec = graph_fsm_spec.load(yaml_path)
    return GraphFSM(spec)


def run_e31(
    *,
    version: PolicyVersion,
    run_id: str,
    output_dir: Path,
    seed: int,
    n_policies: int | None = None,
    rendering: str = "abstract",
    model_id: str = _DEFAULT_MODEL_ID,
    harvest_layer: int = _DEFAULT_HARVEST_LAYER,
    device: str | None = None,
    projection_z_dim: int = 32,
    projection_hidden_dim: int = 64,
    n_projection_train_epochs: int = 30,
    target_n_regimes: int | None = None,
    adversarial_token_weight: float = 0.0,
    eval_n_policies: int | None = None,
    eval_seed: int | None = None,
) -> E30Result:
    """PCG-X applied to the policy-intent FSM on a frozen pretrained LM.

    Parameters
    ----------
    version:
        ``"v1"`` (2-state binary speech_permission gate) or ``"v2"``
        (4-state UNESTABLISHED / GRANTED / WITHHELD / CONDITIONAL).
    run_id:
        Run identifier for the output dir (matches the E30 convention).
    output_dir:
        Output directory; will be created if missing. Existing contents
        are overwritten.
    seed:
        RNG seed for the synthetic-trace generator + projection training.
    n_policies:
        Number of independent ``(source, target, domain)`` instances to
        sample. ``None`` falls back to the dispatch default (80).
    model_id, harvest_layer, device, projection_*, target_n_regimes,
    adversarial_token_weight:
        Forwarded to ``run_e30``. Defaults match the canonical Phase 24
        configuration (GPT-2 small, block 6, 32-d z, 64-hidden, 30 epochs).
        ``target_n_regimes`` defaults to V (= 2 for v1, = 4 for v2) so
        the bisimulation merge collapses the argmax cells to exactly the
        gold FSM cardinality.
    eval_n_policies, eval_seed:
        Optional held-out evaluation slice (same semantics as
        ``run_e30``'s ``eval_n_programs`` / ``eval_seed``). When set,
        emits a second ``decision_trace`` slice with ``ablation="A0_eval"``
        rows -- used by A4 (ABSTAIN-on-disagreement) downstream.

    Returns
    -------
    E30Result
        Standard E30 result record. ``E30Result.grammar`` is set to
        ``"policy_intent_v1"`` / ``"policy_intent_v2"`` for downstream
        identification.
    """
    grammar = _grammar_name_for_version(version, rendering=rendering)
    fsm = _load_policy_fsm(version)
    if target_n_regimes is None:
        target_n_regimes = fsm.vertex_count  # 2 for v1, 4 for v2

    return run_e30(
        fsm=fsm,
        run_id=run_id,
        output_dir=output_dir,
        seed=seed,
        grammar=grammar,
        n_programs=n_policies,
        model_id=model_id,
        harvest_layer=harvest_layer,
        device=device,
        projection_z_dim=projection_z_dim,
        projection_hidden_dim=projection_hidden_dim,
        n_projection_train_epochs=n_projection_train_epochs,
        target_n_regimes=target_n_regimes,
        adversarial_token_weight=adversarial_token_weight,
        eval_n_programs=eval_n_policies,
        eval_seed=eval_seed,
    )


__all__ = ("run_e31", "PolicyVersion", "_load_policy_fsm")
