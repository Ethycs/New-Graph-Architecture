"""End-to-end tests for E12 (torch-substrate world model -- Phase 8).

E12 is E11 with the substrate swapped: a frozen torch encoder produces
the embedding, a torch typed-readout produces the predicted next-state
distribution. These tests assert structure (artefact shape, v1.1 trace
fields, torch parameter wiring) and one observed-numbers bar (Phase A
classical convergence, substrate-independent).

The whole module is gated by ``pytest.importorskip("torch")`` so a
torch-less CI continues to run the rest of the suite.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nga.cli import main  # noqa: E402


@pytest.fixture(scope="module")
def e12_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e12_run") / "E12_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/e12_dyck_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main(
        [
            "--experiment",
            "E12",
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


def test_e12_runs_end_to_end(e12_run):
    """rc == 0 (already asserted in fixture); standard four artefacts plus
    decision_trace.jsonl present, with the substrate-counter metrics
    reported alongside E11's headline metrics."""
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "scores.jsonl",
        "decision_trace.jsonl",
    ):
        assert (e12_run / name).exists(), f"missing artefact: {name}"
    m = _read_metrics(e12_run)
    for required in (
        "accuracy",
        "accuracy_no_mask",
        "mask_accuracy_uplift",
        "illegal_transition_rate",
        "illegal_transition_rate_no_mask",
        "phase_a_hamming_normalised",
        "phase_b_hamming_normalised",
        "crb_satisfied_fraction",
        "mean_crb_confidence",
        "kl_progress_final",
        "mean_kl_surprise",
        "n_torch_trainable_params",
        "n_torch_frozen_params",
    ):
        assert required in m, f"missing metric {required!r}"


def test_e12_decision_trace_has_v11_fields(e12_run):
    """Every row populates the v1.1 node-tuple commitment fields."""
    rows = _read_trace(e12_run)
    assert len(rows) > 0
    for r in rows:
        out = r.get("output_node_tuple")
        assert isinstance(out, list) and len(out) > 0, "output_node_tuple empty/missing"
        assert isinstance(r.get("mask_version_id"), str) and r["mask_version_id"], (
            "mask_version_id must be a non-empty string"
        )


def test_e12_phase_a_converges_via_classical(e12_run):
    """Substrate-independence claim: Phase A's classical cold-start (forward-
    backward + Beta M-step on observed transitions) recovers the FSM
    regardless of the embedding/prototype substrate.

    Bar < 0.20 documents that the torch substrate inherits the same
    cold-start convergence story as E11 -- the architecture's inference
    framework does not depend on the readout's machinery.
    """
    m = _read_metrics(e12_run)
    assert "phase_a_hamming_normalised" in m
    assert m["phase_a_hamming_normalised"] < 0.20, (
        f"Phase-A classical cold start must recover the FSM within 20% "
        f"hamming distance under the torch substrate; got "
        f"{m['phase_a_hamming_normalised']:.4f}"
    )


def test_e12_torch_atoms_used(e12_run):
    """The runner must actually use torch: both the trainable readout
    parameter count and the frozen encoder parameter count are positive.
    These metrics document the substrate wiring -- if the experiment
    silently fell back to a non-torch path either would be 0.
    """
    m = _read_metrics(e12_run)
    assert m["n_torch_trainable_params"] > 0, (
        f"n_torch_trainable_params must be > 0 (typed_readout fitted); "
        f"got {m['n_torch_trainable_params']}"
    )
    assert m["n_torch_frozen_params"] > 0, (
        f"n_torch_frozen_params must be > 0 (frozen encoder constructed); "
        f"got {m['n_torch_frozen_params']}"
    )


def test_e12_phase_b_does_not_destroy_phase_a(e12_run):
    """KL-blended refinement must not catastrophically undo the classical
    cold-start under the torch substrate.

    Mirrors E11's bar: a small worsening (up to 0.05 normalised Hamming)
    is tolerated to account for stochastic exploration during refinement.
    """
    m = _read_metrics(e12_run)
    assert (
        m["phase_b_hamming_normalised"] <= m["phase_a_hamming_normalised"] + 0.05
    ), (
        f"Phase B hamming ({m['phase_b_hamming_normalised']:.4f}) destroyed "
        f"Phase A hamming ({m['phase_a_hamming_normalised']:.4f}) by more "
        f"than 0.05"
    )


def test_e12_control_routing_records(e12_run):
    """Control policy routes / abstains on at least one step.

    The torch substrate's confidence profile differs from E11's Poincare
    one, but the control_policy still fires on high-sigma steps. This
    test guards the wiring -- a silent substitution where control never
    fires would suggest the readout is overconfident in a way that
    bypasses the policy.
    """
    m = _read_metrics(e12_run)
    n_recovery = float(m.get("n_routed_to_recovery", 0.0))
    n_abstain = float(m.get("n_abstained", 0.0))
    assert n_recovery + n_abstain > 0, (
        f"expected control_policy to route or abstain at least once; "
        f"n_routed_to_recovery={n_recovery}, n_abstained={n_abstain}"
    )
