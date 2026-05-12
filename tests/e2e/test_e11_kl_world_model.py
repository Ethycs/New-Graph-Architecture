"""End-to-end tests for E11 (KL/Fisher/CRB world model -- Phase 7).

These tests assert STRUCTURE first and a single observed-numbers bar for
Phase A's classical cold-start convergence. Per the user's directive, if a
bar can't be hit on the actual numbers we LOOSEN it -- the tests do not
tune the runner.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nga.cli import main


@pytest.fixture(scope="module")
def e11_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e11_run") / "E11_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/e11_dyck_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main(
        [
            "--experiment",
            "E11",
            "--ablation",
            "A0",
            "--config",
            str(cfg),
            "--seed",
            "42",
            "--output",
            str(run_dir),
            "--ablation-file",
            str(abl),
        ]
    )
    assert rc == 0
    return run_dir


def _read_metrics(d: Path) -> dict[str, float]:
    return {
        json.loads(line)["metric_name"]: json.loads(line)["value"]
        for line in (d / "metrics.jsonl").read_text().splitlines()
        if line
    }


def _read_trace(d: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (d / "decision_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_e11_runs_end_to_end(e11_run):
    """rc == 0 (already asserted in fixture); standard four artefacts plus
    decision_trace.jsonl present."""
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "scores.jsonl",
        "decision_trace.jsonl",
    ):
        assert (e11_run / name).exists(), f"missing artefact: {name}"
    m = _read_metrics(e11_run)
    for required in (
        "accuracy",
        "accuracy_no_mask",
        "mask_accuracy_uplift",
        "illegal_transition_rate",
        "illegal_transition_rate_no_mask",
        "monodromy_consistency_loss",
        "hamming_distance_from_fsm",
        "hamming_normalised",
        "phase_a_hamming_normalised",
        "phase_b_hamming_normalised",
        "crb_satisfied_fraction",
        "mean_crb_confidence",
        "kl_progress_final",
        "mean_kl_surprise",
        "mean_quality_signal_kl",
        "mean_quality_signal_blended",
        "free_energy",
        "n_samples",
    ):
        assert required in m, f"missing metric {required!r}"


def test_e11_decision_trace_has_v11_fields(e11_run):
    """Every row populates the v1.1 node-tuple commitment fields; at least
    one row also has edge_traversed populated (every step after the first)."""
    rows = _read_trace(e11_run)
    assert len(rows) > 0
    n_with_edge = 0
    for r in rows:
        out = r.get("output_node_tuple")
        assert isinstance(out, list) and len(out) > 0, "output_node_tuple empty/missing"
        assert isinstance(r.get("mask_version_id"), str) and r["mask_version_id"], (
            "mask_version_id must be a non-empty string"
        )
        assert isinstance(r.get("axis_node_ids"), dict) and len(r["axis_node_ids"]) > 0, (
            "axis_node_ids must be a populated dict"
        )
        if r.get("edge_traversed") is not None:
            n_with_edge += 1
    assert n_with_edge > 0, "no row has edge_traversed populated"


def test_e11_phase_a_converges_via_classical(e11_run):
    """Phase A (forward-backward + Bayesian Beta-Dirichlet M-step) seeds the
    posterior mask from PURE pair-frequency evidence on observed
    transitions.

    Under the skeptical-prior threshold convention (legality_matrix uses
    strict ``>``, so an unobserved edge with posterior_mean = 0.5 stays
    illegal), the classical cold-start recovers the Dyck-k FSM EXACTLY
    on this synthetic dataset. The bar is therefore the original < 0.20
    proposal, which is comfortably met (observed values are exactly 0.0
    when every legal edge is sampled enough to push alpha > beta and
    every illegal edge is never observed).

    Bar < 0.20 documents that the architecture converges to the FSM via
    classical inference alone -- the math contract that Phase 7 set out
    to validate.
    """
    m = _read_metrics(e11_run)
    assert "phase_a_hamming_normalised" in m
    assert m["phase_a_hamming_normalised"] < 0.20, (
        f"Phase-A classical cold start must recover the FSM within 20% "
        f"hamming distance under the skeptical-prior threshold; got "
        f"{m['phase_a_hamming_normalised']:.4f}"
    )


def test_e11_crb_reports_per_edge_confidence(e11_run):
    """CRB confidence diagnostics are reported and finite."""
    m = _read_metrics(e11_run)
    assert "crb_satisfied_fraction" in m
    assert 0.0 <= m["crb_satisfied_fraction"] <= 1.0, (
        f"crb_satisfied_fraction out of [0, 1]: {m['crb_satisfied_fraction']}"
    )
    assert "mean_crb_confidence" in m
    assert m["mean_crb_confidence"] > 0.0, (
        f"mean_crb_confidence must be > 0 (some materialised edges); "
        f"got {m['mean_crb_confidence']}"
    )


def test_e11_kl_progress_recorded(e11_run):
    """kl_progress_final is reported and non-negative.

    KL is non-negative by construction; this is a sanity check on the
    Phase-A->Phase-B mask-difference diagnostic.
    """
    m = _read_metrics(e11_run)
    assert "kl_progress_final" in m
    assert m["kl_progress_final"] >= 0.0, (
        f"kl_progress_final must be >= 0; got {m['kl_progress_final']}"
    )


def test_e11_kl_surprise_signal_recorded(e11_run):
    """mean_kl_surprise is reported and non-negative.

    Each per-step kl_surprise lies in [0, 1) by construction
    (1 - exp(-KL)), so the mean is also non-negative.
    """
    m = _read_metrics(e11_run)
    assert "mean_kl_surprise" in m
    assert m["mean_kl_surprise"] >= 0.0, (
        f"mean_kl_surprise must be >= 0; got {m['mean_kl_surprise']}"
    )


def test_e11_phase_b_does_not_destroy_phase_a(e11_run):
    """KL-blended refinement must not catastrophically undo the classical
    cold-start.

    A small worsening (up to 0.05 normalised Hamming) is tolerated to
    account for stochastic exploration during refinement; the bar is that
    Phase B must not blow up Phase A's seed.
    """
    m = _read_metrics(e11_run)
    assert (
        m["phase_b_hamming_normalised"] <= m["phase_a_hamming_normalised"] + 0.05
    ), (
        f"Phase B hamming ({m['phase_b_hamming_normalised']:.4f}) destroyed "
        f"Phase A hamming ({m['phase_a_hamming_normalised']:.4f}) by more "
        f"than 0.05"
    )


def test_e11_monodromy_consistency_recorded(e11_run):
    """monodromy_consistency_loss exists and is a non-negative float."""
    m = _read_metrics(e11_run)
    assert "monodromy_consistency_loss" in m
    val = m["monodromy_consistency_loss"]
    assert isinstance(val, float)
    assert val >= 0.0, f"monodromy_consistency_loss must be >= 0, got {val}"
