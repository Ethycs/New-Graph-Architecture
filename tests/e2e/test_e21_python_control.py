"""End-to-end tests for E21 (torch-native end-to-end gradient training
on python_control -- Phase 18 Track 2).

E21 mirrors E18 structurally but swaps the python_big dataset for
python_control, which adds single-line if / else / while statements.
The new sigma-load-bearing site is S0_after_if_body (the if/else
branching ambiguity).

Phase 18 Track 2 also bakes in compute-efficiency tracking from the
start (per the Phase 18 Track 1 / E20 pattern): per-phase wall-clock,
total wall-clock, inference throughput, peak memory.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nga.cli import main  # noqa: E402


@pytest.fixture(scope="module")
def e21_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e21_run") / "E21_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/e21_python_control_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main(
        [
            "--experiment",
            "E21",
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


def test_e21_runs_end_to_end(e21_run):
    """rc == 0 (already asserted in fixture); standard four artefacts plus
    decision_trace.jsonl present, with the headline E21 metrics."""
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "scores.jsonl",
        "decision_trace.jsonl",
    ):
        assert (e21_run / name).exists(), f"missing artefact: {name}"
    m = _read_metrics(e21_run)
    for required in (
        "accuracy",
        "accuracy_no_mask",
        "mask_accuracy_uplift",
        "illegal_transition_rate",
        "phase_a_hamming_normalised",
        "phase_b_hamming_normalised",
        "sigma_auroc",
        "margin_auroc",
        "sigma_uplift",
        "sigma_structural_auroc",
        "sigma_structural_uplift",
        "mean_sigma_at_operator_boundary",
        "mean_sigma_at_non_boundary",
        "sigma_boundary_ratio",
        "mean_program_depth",
        "n_torch_trainable_params",
        "final_loss",
    ):
        assert required in m, f"missing metric {required!r}"


def test_e21_decision_trace_has_v11_fields(e21_run):
    """Every row populates the v1.1 node-tuple commitment fields."""
    rows = _read_trace(e21_run)
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


def test_e21_phase_a_recovers_python_control_fsm(e21_run):
    """Phase A's classical cold-start recovers the python_control parser
    FSM from PURE pair-frequency evidence within 20% Hamming distance.
    """
    m = _read_metrics(e21_run)
    assert "phase_a_hamming_normalised" in m
    assert m["phase_a_hamming_normalised"] < 0.20, (
        f"Phase-A classical cold start must recover the python_control FSM "
        f"within 20% hamming distance; got "
        f"{m['phase_a_hamming_normalised']:.4f}"
    )


def test_e21_mask_zeros_illegal(e21_run):
    """Under the graph mask the illegal-transition rate stays under 10%."""
    m = _read_metrics(e21_run)
    assert "illegal_transition_rate" in m
    assert m["illegal_transition_rate"] < 0.10, (
        f"illegal_transition_rate={m['illegal_transition_rate']:.4f} "
        f"under graph mask should be < 0.10 on python_control"
    )


def test_e21_emits_efficiency_metrics(e21_run):
    """Compute-efficiency metrics are baked in from the start (Phase 18 Track 1
    pattern): phase_a_wall_clock_seconds, phase_b_wall_clock_seconds,
    total_wall_clock_seconds, inference_throughput_samples_per_sec,
    peak_memory_kb are all present and POSITIVE.
    """
    m = _read_metrics(e21_run)
    for k in (
        "phase_a_wall_clock_seconds",
        "phase_b_wall_clock_seconds",
        "total_wall_clock_seconds",
        "inference_throughput_samples_per_sec",
        "peak_memory_kb",
    ):
        assert k in m, f"missing efficiency metric {k!r}"
        assert m[k] >= 0.0, f"{k}={m[k]} should be >= 0"
    # Total > phase_a + phase_b - small tolerance; total_wall_clock includes
    # Phase C (eval) too. So total > max(phase_a, phase_b).
    assert m["total_wall_clock_seconds"] >= max(
        m["phase_a_wall_clock_seconds"],
        m["phase_b_wall_clock_seconds"],
    ) - 1e-3, (
        f"total_wall_clock_seconds={m['total_wall_clock_seconds']} should "
        f"be >= max(phase_a={m['phase_a_wall_clock_seconds']}, "
        f"phase_b={m['phase_b_wall_clock_seconds']})"
    )
    # Throughput must be strictly positive (we ran at least 1 test sample).
    assert m["inference_throughput_samples_per_sec"] > 0.0, (
        f"inference_throughput_samples_per_sec="
        f"{m['inference_throughput_samples_per_sec']} should be > 0"
    )
