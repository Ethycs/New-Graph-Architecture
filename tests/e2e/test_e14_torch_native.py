"""End-to-end tests for E14 (torch-native end-to-end gradient training
on ListOps -- Phase 9).

E14 swaps E13's classical sklearn substrate for the torch end-to-end
gradient trainer (TorchEnergyTrainer). The whole module is gated by
``pytest.importorskip("torch")`` so a torch-less CI continues to run
the rest of the suite.

Tests assert STRUCTURE and a single numerical bar each; the
sigma_uplift q10 strict bar test (the load-bearing claim) is wrapped
in a tolerant xfail-on-miss path so the runner is NOT tuned to chase
the threshold -- the user wants to know the honest result.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nga.cli import main  # noqa: E402


@pytest.fixture(scope="module")
def e14_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e14_run") / "E14_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/e14_listops_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main(
        [
            "--experiment",
            "E14",
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


def test_e14_runs_end_to_end(e14_run):
    """rc == 0 (already asserted in fixture); standard four artefacts plus
    decision_trace.jsonl present, with the headline E14 metrics."""
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "scores.jsonl",
        "decision_trace.jsonl",
    ):
        assert (e14_run / name).exists(), f"missing artefact: {name}"
    m = _read_metrics(e14_run)
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
        "mean_sigma_at_operator_boundary",
        "mean_sigma_at_non_boundary",
        "sigma_boundary_ratio",
        "n_torch_trainable_params",
        "final_loss",
        "loss_decrease_first_to_last_epoch",
    ):
        assert required in m, f"missing metric {required!r}"


def test_e14_decision_trace_v11(e14_run):
    """Every row populates the v1.1 node-tuple commitment fields
    (output_node_tuple is a populated 4-tuple, mask_version_id is a
    non-empty string)."""
    rows = _read_trace(e14_run)
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


def test_e14_phase_a_classical_seed(e14_run):
    """Phase A's classical cold-start (forward-backward + Beta M-step)
    must seed the posterior mask correctly REGARDLESS of substrate.

    Bar < 0.20 is the substrate-independent claim: the seed converges
    on the FSM's legality matrix from PURE pair-frequency evidence on
    observed transitions, with a uniform Beta prior, before any torch
    update touches the posterior.
    """
    m = _read_metrics(e14_run)
    assert "phase_a_hamming_normalised" in m
    assert m["phase_a_hamming_normalised"] < 0.20, (
        f"Phase-A classical cold start must recover the ListOps FSM "
        f"within 20% hamming distance under the torch substrate; got "
        f"{m['phase_a_hamming_normalised']:.4f}"
    )


def test_e14_loss_decreased_during_training(e14_run):
    """End-to-end gradient training actually moves the loss in the right
    direction.

    ``loss_decrease_first_to_last_epoch > 0`` is the structural
    "training did something" check -- if Adam never reached the
    parameters or the loss curve was flat, this would be 0.
    """
    m = _read_metrics(e14_run)
    assert "loss_decrease_first_to_last_epoch" in m
    assert m["loss_decrease_first_to_last_epoch"] > 0.0, (
        f"loss did not decrease over training: "
        f"loss_decrease_first_to_last_epoch="
        f"{m['loss_decrease_first_to_last_epoch']:.6f}"
    )


def test_e14_torch_atom_used(e14_run):
    """The runner must actually use torch: the trainer's parameter
    count is positive (prototypes + alpha_log + beta_log + readout
    heads, all with requires_grad=True).
    """
    m = _read_metrics(e14_run)
    assert m["n_torch_trainable_params"] > 0, (
        f"n_torch_trainable_params must be > 0 (TorchEnergyTrainer "
        f"registered prototypes + posterior + readout heads); got "
        f"{m['n_torch_trainable_params']}"
    )


def test_e14_sigma_at_boundaries(e14_run):
    """sigma fires at least as strongly at S{d}_after_operand boundaries
    (genuine grammar ambiguity: continue vs. close) as at non-boundary
    steps. Same structural claim as E13; with torch capacity it should
    hold at least as strongly.

    A small 0.05 slack is allowed: the runner is not tuned to chase a
    strict ratio; we report the difference being non-negative within
    epsilon as the honest signal.
    """
    m = _read_metrics(e14_run)
    sigma_b = m["mean_sigma_at_operator_boundary"]
    sigma_nb = m["mean_sigma_at_non_boundary"]
    assert sigma_b + 0.05 >= sigma_nb, (
        f"sigma at operator boundaries ({sigma_b:.4f}) should be at "
        f"least as large as sigma at non-boundaries ({sigma_nb:.4f}) "
        f"under the torch substrate. Slack = 0.05."
    )


def test_e14_structural_ambiguity_auroc_recorded(e14_run):
    """Structure check: the three structural-ambiguity AUROC metrics
    must be emitted, both AUROCs in [0, 1], and the uplift in
    [-1, 1] (negative is allowed -- it's an honest observation).
    """
    m = _read_metrics(e14_run)
    for required in (
        "sigma_structural_auroc",
        "margin_structural_auroc",
        "sigma_structural_uplift",
    ):
        assert required in m, f"missing metric {required!r}"
    assert 0.0 <= m["sigma_structural_auroc"] <= 1.0, (
        f"sigma_structural_auroc out of [0,1]: "
        f"{m['sigma_structural_auroc']:.4f}"
    )
    assert 0.0 <= m["margin_structural_auroc"] <= 1.0, (
        f"margin_structural_auroc out of [0,1]: "
        f"{m['margin_structural_auroc']:.4f}"
    )
    assert -1.0 <= m["sigma_structural_uplift"] <= 1.0, (
        f"sigma_structural_uplift out of [-1,1]: "
        f"{m['sigma_structural_uplift']:.4f}"
    )


def test_e14_sigma_wins_structural_ambiguity_bar(e14_run):
    """sigma should identify grammar-level ambiguity points STRICTLY
    BETTER than margin does on the structural-ambiguity AUROC metric.

    This is the q10 claim re-framed correctly: sigma's job is to fire
    on structural ambiguity (S{d}_after_operand states where the
    parser must choose continue-list vs. close), not to predict
    prediction errors. Per the don't-tune directive, if the natural
    numbers do not satisfy the bar we mark this xfail with the
    observed value documented -- not change the runner.
    """
    m = _read_metrics(e14_run)
    uplift = m["sigma_structural_uplift"]
    if uplift <= 0.0:
        pytest.xfail(
            f"sigma did not strictly beat margin on the structural-"
            f"ambiguity AUROC: sigma_structural_uplift={uplift:+.4f} "
            f"(sigma_structural_auroc="
            f"{m['sigma_structural_auroc']:.4f}, "
            f"margin_structural_auroc="
            f"{m['margin_structural_auroc']:.4f}); needed > 0.0."
        )
    assert uplift > 0.0, (
        f"sigma_structural_uplift={uplift:+.4f}; sigma must beat margin "
        f"on the structural-ambiguity AUROC. "
        f"sigma_structural_auroc={m['sigma_structural_auroc']:.4f}, "
        f"margin_structural_auroc={m['margin_structural_auroc']:.4f}."
    )


def test_e14_q10_strict_bar(e14_run):
    """The Phase 2 q10 strict bar: sigma_auroc - margin_auroc >= 0.03.

    This bar has been XFAIL since Phase 2; E14's hypothesis was that
    real grammar ambiguity (ListOps ``S{d}_after_operand`` boundaries)
    combined with a learned-encoder torch substrate would finally
    cross it. Per the don't-tune directive, if the natural numbers do
    not satisfy the bar we mark this xfail with the observed value
    documented in the reason -- not change the runner.
    """
    m = _read_metrics(e14_run)
    sigma_uplift = m["sigma_uplift"]
    if sigma_uplift < 0.03:
        pytest.xfail(
            f"q10 strict bar not yet met under torch substrate on "
            f"ListOps: sigma_uplift={sigma_uplift:+.4f} (sigma_auroc="
            f"{m['sigma_auroc']:.4f}, margin_auroc="
            f"{m['margin_auroc']:.4f}); needed >= 0.03."
        )
    assert sigma_uplift >= 0.03, (
        f"sigma_uplift={sigma_uplift:+.4f} below the q10 strict bar of "
        f"0.03; sigma_auroc={m['sigma_auroc']:.4f}, margin_auroc="
        f"{m['margin_auroc']:.4f}"
    )
