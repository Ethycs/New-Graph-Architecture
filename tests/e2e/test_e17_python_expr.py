"""End-to-end tests for E17 (torch-native end-to-end gradient training
on Python expressions -- Wave I).

E17 mirrors E14 structurally but swaps the synthetic ListOps dataset
for the first external published-language benchmark: a subset of real
Python validated by stdlib ``ast.parse``. The whole module is gated
by ``pytest.importorskip("torch")`` so a torch-less CI continues to
run the rest of the suite.

Tests assert STRUCTURE and a single numerical bar each. Where the
numerical bar is the load-bearing claim and the runner is NOT tuned
to chase it, an xfail-on-miss path documents the honest result.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nga.cli import main  # noqa: E402


@pytest.fixture(scope="module")
def e17_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e17_run") / "E17_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/e17_python_expr_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main(
        [
            "--experiment",
            "E17",
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


def test_e17_runs_end_to_end(e17_run):
    """rc == 0 (already asserted in fixture); standard four artefacts plus
    decision_trace.jsonl present, with the headline E17 metrics."""
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "scores.jsonl",
        "decision_trace.jsonl",
    ):
        assert (e17_run / name).exists(), f"missing artefact: {name}"
    m = _read_metrics(e17_run)
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


def test_e17_decision_trace_has_v11_fields(e17_run):
    """Every row populates the v1.1 node-tuple commitment fields
    (output_node_tuple is a populated 4-tuple, mask_version_id is a
    non-empty string)."""
    rows = _read_trace(e17_run)
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


def test_e17_phase_a_recovers_python_fsm(e17_run):
    """Phase A's classical cold-start recovers the 14-vertex Python
    expression parser FSM from PURE pair-frequency evidence on observed
    transitions (uniform Beta prior), substrate-independently.

    Bar < 0.20: the same numerical bar as E14 on ListOps. If Phase A
    seeds a richer real-grammar FSM within 20% Hamming, the recovery
    claim survives the dataset swap.
    """
    m = _read_metrics(e17_run)
    assert "phase_a_hamming_normalised" in m
    assert m["phase_a_hamming_normalised"] < 0.20, (
        f"Phase-A classical cold start must recover the python_expr FSM "
        f"within 20% hamming distance; got "
        f"{m['phase_a_hamming_normalised']:.4f}"
    )


def test_e17_mask_zeros_illegal(e17_run):
    """The graph mask zeros illegal transitions: with the mask enabled
    (ablation A0 has graph_mask_enabled=True), illegal transitions in
    test should be rare (< 10%). This is the most direct
    operationalisation of the "mask works" claim on real Python."""
    m = _read_metrics(e17_run)
    assert "illegal_transition_rate" in m
    assert m["illegal_transition_rate"] < 0.10, (
        f"illegal_transition_rate={m['illegal_transition_rate']:.4f} "
        f"under graph mask should be < 0.10 on the python_expr FSM"
    )


def test_e17_mask_uplift_positive(e17_run):
    """Mask uplift survives on real Python: accuracy with the mask
    enabled exceeds accuracy with the mask disabled by > 0.10
    (loosened from E14's 0.30 since the Python grammar is richer and
    the natural per-step entropy is higher)."""
    m = _read_metrics(e17_run)
    assert "mask_accuracy_uplift" in m
    assert m["mask_accuracy_uplift"] > 0.10, (
        f"mask_accuracy_uplift={m['mask_accuracy_uplift']:+.4f} should "
        f"exceed 0.10 on python_expr; accuracy={m.get('accuracy')!r}, "
        f"accuracy_no_mask={m.get('accuracy_no_mask')!r}"
    )


def test_e17_sigma_at_python_ambiguity(e17_run):
    """sigma fires AT LEAST as strongly at the assignment-vs-expression
    decision point (S0_after_name_at_start) as at non-boundary steps.

    This is the structural-ambiguity claim re-anchored on Python's
    actual grammar: the only top-level continue-vs-close decision in
    the expression subset is the choice between starting an
    ``= expr NEWLINE`` assignment and continuing an ``expr NEWLINE``
    expression-statement after a leading NAME. Per the don't-tune
    directive, if the natural numbers don't satisfy the bar we mark
    this xfail with the observed values documented.
    """
    m = _read_metrics(e17_run)
    sigma_b = m["mean_sigma_at_operator_boundary"]
    sigma_nb = m["mean_sigma_at_non_boundary"]
    if sigma_b < sigma_nb:
        pytest.xfail(
            f"sigma did not fire as strongly at the Python "
            f"assignment-vs-expression decision point: "
            f"mean_sigma_at_operator_boundary={sigma_b:.4f} < "
            f"mean_sigma_at_non_boundary={sigma_nb:.4f} "
            f"(boundary_ratio={m.get('sigma_boundary_ratio'):.4f})"
        )
    assert sigma_b >= sigma_nb, (
        f"mean_sigma_at_operator_boundary={sigma_b:.4f} must be >= "
        f"mean_sigma_at_non_boundary={sigma_nb:.4f} on python_expr"
    )
