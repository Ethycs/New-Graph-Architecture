"""End-to-end smoke tests for E9 (Dyck-k bracket-language run)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nga.cli import main


@pytest.fixture(scope="module")
def e9_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e9_run") / "E9_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/dyck_k_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main([
        "--experiment", "E9", "--ablation", "A0",
        "--config", str(cfg), "--seed", "42",
        "--output", str(run_dir), "--ablation-file", str(abl),
    ])
    assert rc == 0
    return run_dir


def _read_metrics(d: Path) -> dict[str, float]:
    return {
        json.loads(line)["metric_name"]: json.loads(line)["value"]
        for line in (d / "metrics.jsonl").read_text().splitlines()
        if line
    }


def test_e9_runs_end_to_end(e9_run):
    """rc == 0 (already asserted in fixture); metrics.jsonl has headline fields."""
    m = _read_metrics(e9_run)
    for required in (
        "accuracy",
        "accuracy_no_mask",
        "mask_accuracy_uplift",
        "illegal_transition_rate",
        "illegal_transition_rate_no_mask",
        "n_routed_to_recovery",
        "n_abstained",
        "n_mask_modified_predictions",
        "free_energy",
        "mean_energy_correct",
        "mean_energy_incorrect",
        "mean_depth",
        "n_monodromy_cycle_revisits",
        "n_samples",
        "n_features",
    ):
        assert required in m, f"missing metric {required!r} in metrics.jsonl"


def test_e9_mask_drives_illegal_to_zero(e9_run):
    """Mask zeroes illegal transitions; without it adversarial samples push >5%."""
    m = _read_metrics(e9_run)
    assert m["illegal_transition_rate"] == 0.0
    assert m["illegal_transition_rate_no_mask"] > 0.05


def test_e9_mask_uplift_positive(e9_run):
    """Adversarial samples make the mask load-bearing for accuracy."""
    m = _read_metrics(e9_run)
    assert m["mask_accuracy_uplift"] > 0.0


def test_e9_emits_decision_trace(e9_run):
    """decision_trace.jsonl is populated; first row has full schema."""
    trace_path = e9_run / "decision_trace.jsonl"
    assert trace_path.exists()
    lines = [
        line for line in trace_path.read_text().splitlines() if line.strip()
    ]
    assert len(lines) > 0
    first = json.loads(lines[0])
    # Spot-check: identity + scoring + mask + sigma + control all present.
    for key in (
        "schema_version",
        "run_id",
        "experiment",
        "ablation",
        "seed",
        "step",
        "sample_id",
        "confidence",
        "top1_state",
        "top1_prob",
        "margin",
        "mask",
        "sigma_total",
        "sigma_signals",
        "energy_total",
        "energy_breakdown",
        "monodromy_class",
        "control",
    ):
        assert key in first, f"decision_trace row missing {key!r}"
    # MaskAction sub-record fields.
    for key in (
        "enabled",
        "current_state",
        "illegal_indices_zeroed",
        "pre_mask_argmax",
        "post_mask_argmax",
        "mask_changed_prediction",
    ):
        assert key in first["mask"], f"mask sub-record missing {key!r}"
    # ControlAction sub-record fields.
    for key in (
        "decision",
        "reason",
        "sigma_threshold_used",
        "sigma_observed",
    ):
        assert key in first["control"], f"control sub-record missing {key!r}"


def test_e9_monodromy_cycle_revisits_recorded(e9_run):
    """Dyck has many open/close cycles; expect strictly positive revisits."""
    m = _read_metrics(e9_run)
    assert "n_monodromy_cycle_revisits" in m
    assert m["n_monodromy_cycle_revisits"] >= 0
    # Dyck creates many cycles -- expect strictly positive.
    assert m["n_monodromy_cycle_revisits"] > 0
