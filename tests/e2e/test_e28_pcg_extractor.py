"""End-to-end test for E28 (Predictive Control Graph Extractor) σ + control trace.

Exercises the full ``run_e28`` pipeline on the ``python_expr`` grammar
(V=14, multiple steps per program — listops only emits 1 step per
program so it cannot exercise current_state continuity) with minimal
training so it is fast enough for the regular suite. The acceptance
focus is the σ + control + decision_trace.jsonl integration added on
top of the regime-graph extraction:

  - decision_trace.jsonl exists and has one row per harvested step.
  - Every row's mask is well-formed; pre/post argmax mention the regime
    naming convention (``regime_K``).
  - Every row's sigma_total is in [0, 1] and sigma_signals carries the
    documented keys.
  - Every row's control verdict is one of ROUTE_NORMAL / ROUTE_RECOVERY /
    ABSTAIN and is consistent with the σ thresholds the trace records.
  - At least one row has current_state populated (i.e. not the first step
    of every program).

This is the integration test for σ + control on the PCG-X regime graph,
the bug the Phase 23 series left open after extraction itself was wired.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from nga.arch.graph_fsm import GraphFSM  # noqa: E402
from nga.drivers import graph_fsm_spec as graph_fsm_spec_mod  # noqa: E402
from nga.drivers.decision_trace_jsonl import read_decision_trace  # noqa: E402
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH  # noqa: E402
from nga.exp.e28_pcg_extractor import run_e28  # noqa: E402


@pytest.fixture(scope="module")
def e28_run(tmp_path_factory) -> Path:
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e28_run") / "E28_A0_seed42_python_expr"
    dispatch = GRAMMAR_DISPATCH["python_expr"]
    fsm_spec = graph_fsm_spec_mod.load(repo_root / dispatch["fsm_yaml_path"])
    fsm = GraphFSM(fsm_spec)
    run_e28(
        fsm=fsm,
        run_id="E28_A0_seed42_python_expr",
        output_dir=run_dir,
        seed=42,
        grammar="python_expr",
        n_programs=8,
        n_encoder_train_epochs=4,
        n_projection_train_epochs=4,
    )
    return run_dir


def test_e28_emits_decision_trace(e28_run: Path) -> None:
    """decision_trace.jsonl exists and is non-empty."""
    trace_path = e28_run / "decision_trace.jsonl"
    assert trace_path.exists()
    records = read_decision_trace(trace_path)
    assert len(records) > 0


def test_e28_trace_rows_have_sigma_and_control(e28_run: Path) -> None:
    """Every row carries σ in [0, 1], sigma_signals, and a valid control verdict."""
    records = read_decision_trace(e28_run / "decision_trace.jsonl")

    valid_decisions = {"ROUTE_NORMAL", "ROUTE_RECOVERY", "ABSTAIN"}
    documented_signal_keys = {
        "margin",
        "decision_tie",
        "illegal",
        "loop",
        "stabilizer",
        "catastrophe_bias",
    }
    for r in records:
        assert r.sigma_total is not None and 0.0 <= r.sigma_total <= 1.0
        assert r.sigma_signals is not None
        assert documented_signal_keys.issubset(r.sigma_signals.keys())
        assert r.control.decision in valid_decisions
        # Sigma observed by the policy must match what's logged.
        assert r.control.sigma_observed == pytest.approx(r.sigma_total)
        # Thresholds round-trip.
        assert 0.0 <= r.control.sigma_threshold_used <= 1.0
        assert r.control.abstain_threshold_used is not None
        assert (
            r.control.sigma_threshold_used <= r.control.abstain_threshold_used
        )


def test_e28_trace_rows_use_regime_naming(e28_run: Path) -> None:
    """top1_state and mask argmaxes use the ``regime_K`` naming convention.

    This pins the audited graph identity: the trace is over the PCG-X
    regime graph, not the original typed FSM. ``top2_state`` may be None
    when the second-argmax FSM state never appeared as a regime cell.
    """
    records = read_decision_trace(e28_run / "decision_trace.jsonl")
    for r in records:
        assert r.top1_state.startswith("regime_")
        assert r.mask.pre_mask_argmax is not None
        assert r.mask.pre_mask_argmax.startswith("regime_")
        assert r.mask.post_mask_argmax is not None
        assert r.mask.post_mask_argmax.startswith("regime_")
        if r.top2_state is not None:
            assert r.top2_state.startswith("regime_")
        if r.mask.current_state is not None:
            assert r.mask.current_state.startswith("regime_")


def test_e28_trace_has_intra_program_current_state(e28_run: Path) -> None:
    """At least one row has a non-None current_state.

    The first step of every program has current_state=None (no prior
    regime); subsequent steps within the same program carry the previous
    regime. With multiple programs at non-trivial length this must fire
    at least once.
    """
    records = read_decision_trace(e28_run / "decision_trace.jsonl")
    intra_program = [r for r in records if r.mask.current_state is not None]
    assert len(intra_program) > 0


def test_e28_control_graph_still_emitted(e28_run: Path) -> None:
    """Adding the trace pass did not regress the existing control_graph.json output."""
    cg_path = e28_run / "control_graph.json"
    assert cg_path.exists()
    import json

    cg = json.loads(cg_path.read_text())
    assert cg["grammar"] == "python_expr"
    assert cg["n_regimes_after_merge"] > 0
    assert "regimes" in cg
    assert "edges" in cg


@pytest.fixture(scope="module")
def e28_eval_run(tmp_path_factory) -> Path:
    """Run E28 with a held-out eval slice (eval_seed != seed).

    Same training as ``e28_run`` (so the regime graph is the same shape),
    but the decision_trace.jsonl is emitted over a fresh dataset the
    projection never saw.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e28_eval") / "E28_A0_eval_python_expr"
    dispatch = GRAMMAR_DISPATCH["python_expr"]
    fsm_spec = graph_fsm_spec_mod.load(repo_root / dispatch["fsm_yaml_path"])
    fsm = GraphFSM(fsm_spec)
    run_e28(
        fsm=fsm,
        run_id="E28_A0_eval_python_expr",
        output_dir=run_dir,
        seed=42,
        grammar="python_expr",
        n_programs=8,
        n_encoder_train_epochs=4,
        n_projection_train_epochs=4,
        eval_n_programs=8,
        eval_seed=43,
    )
    return run_dir


def test_e28_eval_trace_ablation_label(e28_eval_run: Path) -> None:
    """Held-out-eval rows carry ablation label `A0_eval` so train vs. eval is joinable."""
    records = read_decision_trace(e28_eval_run / "decision_trace.jsonl")
    assert len(records) > 0
    assert all(r.ablation == "A0_eval" for r in records)


def test_e28_eval_trace_can_have_unknown_regime(e28_eval_run: Path) -> None:
    """The eval slice may produce predictions outside the extracted regime graph.

    Out-of-graph predictions show up as `regime_unknown` in `top1_state` and
    automatically fire the illegal signal (the predicted regime has no edges
    by construction). This pins the integration's handling of held-out
    drift: when the model predicts something the regime graph cannot name,
    the trace records that fact rather than silently bucketing it.
    """
    records = read_decision_trace(e28_eval_run / "decision_trace.jsonl")
    # Either some rows are unknown (illegal fires) or all rows are in-graph.
    # Either is operationally valid for this test; the assertion is that
    # the helper never silently relabels — when unknown, it is named.
    for r in records:
        if r.top1_state == "regime_unknown":
            assert r.sigma_signals is not None
            assert r.sigma_signals["illegal"] == pytest.approx(1.0)
