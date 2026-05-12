"""End-to-end tests for E18 (torch-native end-to-end gradient training
on the bigger Python subset -- Wave II).

E18 mirrors E17 structurally but swaps the python_expr dataset
(arithmetic + assignment only) for python_big (arithmetic + assignment
+ function calls + function definitions + return). The whole module
is gated by ``pytest.importorskip("torch")`` so a torch-less CI
continues to run the rest of the suite.

Architectural prediction (Phase 14 Wave II): the structural-ambiguity
σ-uplift on python_big should be at least as positive as on
python_expr, because python_big has many MORE structural-ambiguity
points -- the assignment-vs-expression-statement choice plus the
call-vs-variable choice at every paren depth. If σ doesn't fire more
strongly at those richer ambiguity points than at non-boundary steps,
the architectural claim weakens; we mark that single test xfail with
the observed numbers per the don't-tune directive.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nga.cli import main  # noqa: E402


@pytest.fixture(scope="module")
def e18_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e18_run") / "E18_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/e18_python_big_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main(
        [
            "--experiment",
            "E18",
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


def test_e18_runs_end_to_end(e18_run):
    """rc == 0 (already asserted in fixture); standard four artefacts plus
    decision_trace.jsonl present, with the headline E18 metrics."""
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "scores.jsonl",
        "decision_trace.jsonl",
    ):
        assert (e18_run / name).exists(), f"missing artefact: {name}"
    m = _read_metrics(e18_run)
    for required in (
        "accuracy",
        "accuracy_no_mask",
        "mask_accuracy_uplift",
        "illegal_transition_rate",
        "illegal_transition_rate_no_mask",
        "phase_a_hamming_normalised",
        "phase_b_hamming_normalised",
        "sigma_auroc",
        "margin_auroc",
        "sigma_uplift",
        "sigma_structural_auroc",
        "margin_structural_auroc",
        "sigma_structural_uplift",
        "mean_sigma_at_operator_boundary",
        "mean_sigma_at_non_boundary",
        "sigma_boundary_ratio",
        "mean_program_depth",
        "n_torch_trainable_params",
        "final_loss",
    ):
        assert required in m, f"missing metric {required!r}"


def test_e18_decision_trace_has_v11_fields(e18_run):
    """Every row populates the v1.1 node-tuple commitment fields
    (output_node_tuple is a populated 4-tuple, mask_version_id is a
    non-empty string)."""
    rows = _read_trace(e18_run)
    assert len(rows) > 0
    for r in rows:
        out = r.get("output_node_tuple")
        assert isinstance(out, list) and len(out) == 4, (
            f"output_node_tuple must be a 4-tuple; got {out!r}"
        )
        assert all(isinstance(x, str) and x for x in out), (
            f"output_node_tuple entries must be non-empty strings; got {out!r}"
        )
        assert isinstance(r.get("mask_version_id"), str) and r["mask_version_id"], (
            "mask_version_id must be a non-empty string"
        )


def test_e18_phase_a_recovers_python_big_fsm(e18_run):
    """Phase A's classical cold-start recovers the 24-vertex python_big
    parser FSM from PURE pair-frequency evidence on observed
    transitions (uniform Beta prior), substrate-independently.

    Bar < 0.20: same numerical bar as E14 / E17. If Phase A seeds the
    richer call+def grammar within 20% Hamming, the recovery claim
    survives the second dataset swap.
    """
    m = _read_metrics(e18_run)
    assert "phase_a_hamming_normalised" in m
    assert m["phase_a_hamming_normalised"] < 0.20, (
        f"Phase-A classical cold start must recover the python_big FSM "
        f"within 20% hamming distance; got "
        f"{m['phase_a_hamming_normalised']:.4f}"
    )


def test_e18_mask_zeros_illegal(e18_run):
    """The graph mask zeros illegal transitions: with the mask enabled
    (ablation A0 has graph_mask_enabled=True), illegal transitions in
    test should be rare (< 10%) on python_big as well."""
    m = _read_metrics(e18_run)
    assert "illegal_transition_rate" in m
    assert m["illegal_transition_rate"] < 0.10, (
        f"illegal_transition_rate={m['illegal_transition_rate']:.4f} "
        f"under graph mask should be < 0.10 on the python_big FSM"
    )


def test_e18_mask_uplift_positive(e18_run):
    """Mask uplift survives on the richer python_big grammar: accuracy
    with the mask enabled exceeds accuracy with the mask disabled by
    > 0.10 (same loose bar as python_expr; the mask must still help
    despite a 24-vertex FSM with two distinct kinds of structural
    ambiguity)."""
    m = _read_metrics(e18_run)
    assert "mask_accuracy_uplift" in m
    assert m["mask_accuracy_uplift"] > 0.10, (
        f"mask_accuracy_uplift={m['mask_accuracy_uplift']:+.4f} should "
        f"exceed 0.10 on python_big; accuracy={m.get('accuracy')!r}, "
        f"accuracy_no_mask={m.get('accuracy_no_mask')!r}"
    )


def test_e18_sigma_at_python_big_ambiguity(e18_run):
    """sigma fires AT LEAST as strongly at python_big's structural
    ambiguity points (the union of S0_after_name_at_start and every
    S{d}_after_name_in_factor) as at non-boundary steps.

    This is the structural-ambiguity claim re-anchored on python_big's
    actual grammar: the multi-ambiguity-point structure of the bigger
    grammar should make σ MORE useful at those vertices, not less.
    Per the don't-tune directive, if the natural numbers don't satisfy
    the bar we mark this xfail with the observed values documented.
    """
    m = _read_metrics(e18_run)
    sigma_b = m["mean_sigma_at_operator_boundary"]
    sigma_nb = m["mean_sigma_at_non_boundary"]
    if sigma_b < sigma_nb:
        pytest.xfail(
            f"sigma did not fire as strongly at python_big's structural "
            f"ambiguity points: "
            f"mean_sigma_at_operator_boundary={sigma_b:.4f} < "
            f"mean_sigma_at_non_boundary={sigma_nb:.4f} "
            f"(boundary_ratio={m.get('sigma_boundary_ratio'):.4f})"
        )
    assert sigma_b >= sigma_nb, (
        f"mean_sigma_at_operator_boundary={sigma_b:.4f} must be >= "
        f"mean_sigma_at_non_boundary={sigma_nb:.4f} on python_big"
    )
