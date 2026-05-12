"""End-to-end tests for E22 (Phase 19) -- the diagnostic experiment on
the Kaggle Disease-Prediction-from-Symptoms CSV.

E22 is the first experiment with a real-world (non-synthetic) input:
each patient's symptom set is linearised into a token stream
``[s_1, ..., s_|S|, d]`` and walked through a 3-state FSM
(OBSERVING_FEW / OBSERVING_ENOUGH / DIAGNOSED). OBSERVING_ENOUGH is the
sigma-load-bearing structural-ambiguity site (another symptom or commit?).

Compute-efficiency tracking is baked in (Phase 18 Track 1 pattern).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nga.cli import main  # noqa: E402


@pytest.fixture(scope="module")
def e22_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e22_run") / "E22_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/e22_diagnostic_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main(
        [
            "--experiment",
            "E22",
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


def test_e22_runs_end_to_end(e22_run):
    """rc == 0; standard four artefacts plus decision_trace.jsonl
    are present on disk."""
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "scores.jsonl",
        "decision_trace.jsonl",
    ):
        assert (e22_run / name).exists(), f"missing artefact: {name}"


def test_e22_decision_trace_has_v11_fields(e22_run):
    """Every decision_trace row populates the v1.1 node-tuple commitment
    fields (output_node_tuple as a 4-tuple of non-empty strings,
    mask_version_id present and non-empty)."""
    rows = _read_trace(e22_run)
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


def test_e22_emits_diagnostic_metrics(e22_run):
    """diagnostic_accuracy and mean_symptoms_observed_before_diagnosis
    are present in metrics.jsonl."""
    m = _read_metrics(e22_run)
    assert "diagnostic_accuracy" in m, "missing metric diagnostic_accuracy"
    assert "mean_symptoms_observed_before_diagnosis" in m, (
        "missing metric mean_symptoms_observed_before_diagnosis"
    )
    # mean symptoms observed should be > 0 (non-empty patients).
    assert m["mean_symptoms_observed_before_diagnosis"] > 0.0, (
        f"mean_symptoms={m['mean_symptoms_observed_before_diagnosis']} "
        f"should be > 0 on real patient data"
    )
    # diagnostic_accuracy is a fraction in [0, 1].
    assert 0.0 <= m["diagnostic_accuracy"] <= 1.0


def test_e22_emits_efficiency_metrics(e22_run):
    """Compute-efficiency metrics (per-phase wall-clock, total wall-clock,
    inference throughput, peak memory) are present and non-negative."""
    m = _read_metrics(e22_run)
    for k in (
        "phase_a_wall_clock_seconds",
        "phase_b_wall_clock_seconds",
        "total_wall_clock_seconds",
        "inference_throughput_samples_per_sec",
        "peak_memory_kb",
    ):
        assert k in m, f"missing efficiency metric {k!r}"
        assert m[k] >= 0.0, f"{k}={m[k]} should be >= 0"
    assert m["total_wall_clock_seconds"] >= max(
        m["phase_a_wall_clock_seconds"],
        m["phase_b_wall_clock_seconds"],
    ) - 1e-3, (
        "total_wall_clock_seconds must dominate the per-phase clocks"
    )
    assert m["inference_throughput_samples_per_sec"] > 0.0, (
        "inference_throughput must be strictly positive"
    )


def test_e22_phase_a_recovers_diagnostic_fsm(e22_run):
    """Phase A's classical cold-start recovers the diagnostic FSM from
    pair-frequency evidence within a 20% Hamming bound. (The 3-state
    FSM is small; this should be tight.)"""
    m = _read_metrics(e22_run)
    assert "phase_a_hamming_normalised" in m
    assert m["phase_a_hamming_normalised"] < 0.20, (
        f"Phase-A classical cold start must recover the diagnostic FSM "
        f"within 20% hamming distance; got "
        f"{m['phase_a_hamming_normalised']:.4f}"
    )


def test_e22_mask_zeros_illegal(e22_run):
    """Under the graph mask the illegal-transition rate stays under 10%."""
    m = _read_metrics(e22_run)
    assert "illegal_transition_rate" in m
    assert m["illegal_transition_rate"] < 0.10, (
        f"illegal_transition_rate={m['illegal_transition_rate']:.4f} "
        f"under graph mask should be < 0.10 on the diagnostic dataset"
    )
