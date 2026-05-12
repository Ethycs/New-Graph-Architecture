"""E2E tests for E20 (cross-grammar sigma-weight transfer + compute
efficiency). Gated by ``pytest.importorskip("torch")``."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nga.cli import main  # noqa: E402


@pytest.fixture(scope="module")
def e20_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e20_run") / "E20_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/e20_json_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main(
        [
            "--experiment",
            "E20",
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


def test_e20_runs_end_to_end(e20_run):
    """rc == 0 (asserted in fixture); the standard four artefacts plus
    decision_trace.jsonl plus tuned_weights.json are all present."""
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "scores.jsonl",
        "decision_trace.jsonl",
        "tuned_weights.json",
    ):
        assert (e20_run / name).exists(), f"missing artefact: {name}"
    # Tuned weights is a JSON object with the expected keys.
    tuned = json.loads((e20_run / "tuned_weights.json").read_text())
    assert isinstance(tuned, dict) and tuned, "tuned_weights.json must be a non-empty object"


def test_e20_emits_both_aurocs(e20_run):
    """``sigma_auroc_default`` and ``sigma_auroc_transferred`` both
    surface as metrics rows."""
    m = _read_metrics(e20_run)
    for required in (
        "sigma_auroc_default",
        "sigma_auroc_transferred",
        "margin_auroc",
        "sigma_uplift_default",
        "sigma_uplift_transferred",
        "transfer_lift",
    ):
        assert required in m, f"missing metric {required!r}"
    # AUROCs are in [0, 1].
    assert 0.0 <= m["sigma_auroc_default"] <= 1.0
    assert 0.0 <= m["sigma_auroc_transferred"] <= 1.0


def test_e20_emits_efficiency_metrics(e20_run):
    """Phase 0/1 wall-clock plus throughput surface as positive-valued metrics."""
    m = _read_metrics(e20_run)
    for required in (
        "phase0_wall_clock_seconds",
        "phase1_wall_clock_seconds",
        "total_wall_clock_seconds",
        "inference_throughput_samples_per_sec",
        "peak_memory_kb",
    ):
        assert required in m, f"missing efficiency metric {required!r}"
    assert m["phase0_wall_clock_seconds"] > 0.0
    assert m["phase1_wall_clock_seconds"] > 0.0
    assert m["total_wall_clock_seconds"] > 0.0
    assert m["inference_throughput_samples_per_sec"] > 0.0
    # peak_memory_kb may be 0 on systems where ru_maxrss is monotonic
    # and didn't change during this run; allow >= 0.
    assert m["peak_memory_kb"] >= 0.0


def test_e20_transfer_lift_is_finite(e20_run):
    """transfer_lift is recorded and is a finite real number (not NaN)."""
    m = _read_metrics(e20_run)
    assert "transfer_lift" in m
    val = m["transfer_lift"]
    assert math.isfinite(val), f"transfer_lift={val!r} must be finite"


def test_e20_decision_trace_has_v11_fields(e20_run):
    """Every decision-trace row carries the v1.1 node-tuple commitment
    fields (output_node_tuple is a populated 4-tuple, mask_version_id
    is non-empty)."""
    rows = _read_trace(e20_run)
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


def test_e20_phase_a_recovers_json_fsm(e20_run):
    """Phase A's classical cold start on the JSON FSM still recovers
    legality within 20% Hamming -- the same numerical bar as E14 / E17
    / E18 / E19. E20's twin-sigma evaluation must not perturb the
    phase-A recovery property."""
    m = _read_metrics(e20_run)
    assert "phase_a_hamming_normalised" in m
    assert m["phase_a_hamming_normalised"] < 0.20, (
        f"phase_a_hamming_normalised={m['phase_a_hamming_normalised']:.4f}"
        f" must be < 0.20 on the JSON FSM"
    )
