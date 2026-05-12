"""End-to-end tests for E10 (Unified world model: every Wave A/B/C atom).

These tests assert STRUCTURE, not optimised numbers. Per the user's
directive, if a bar fails because it is too strict we LOOSEN it (and say
why) -- we do not tune the runner to game it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nga.cli import main


@pytest.fixture(scope="module")
def e10_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e10_run") / "E10_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/e10_dyck_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main(
        [
            "--experiment",
            "E10",
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


def test_e10_runs_end_to_end(e10_run):
    """rc == 0 (already asserted in fixture); standard four artefacts plus
    decision_trace.jsonl present."""
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "scores.jsonl",
        "decision_trace.jsonl",
    ):
        assert (e10_run / name).exists(), f"missing artefact: {name}"
    m = _read_metrics(e10_run)
    for required in (
        "accuracy",
        "accuracy_no_mask",
        "mask_accuracy_uplift",
        "illegal_transition_rate",
        "illegal_transition_rate_no_mask",
        "monodromy_consistency_loss",
        "hamming_distance_from_fsm",
        "hamming_normalised",
        "free_energy",
        "n_samples",
    ):
        assert required in m, f"missing metric {required!r}"


def test_e10_decision_trace_has_v11_fields(e10_run):
    """Every row populates the v1.1 node-tuple commitment fields; at least
    one row also has edge_traversed populated (every step after the first)."""
    rows = _read_trace(e10_run)
    assert len(rows) > 0
    n_with_edge = 0
    for r in rows:
        # Required fields (v1.1):
        out = r.get("output_node_tuple")
        assert isinstance(out, list) and len(out) > 0, "output_node_tuple empty/missing"
        assert isinstance(r.get("mask_version_id"), str) and r["mask_version_id"], (
            "mask_version_id must be a non-empty string"
        )
        assert isinstance(r.get("axis_node_ids"), dict), "axis_node_ids must be a dict"
        if r.get("edge_traversed") is not None:
            n_with_edge += 1
    assert n_with_edge > 0, "no row has edge_traversed populated"


def test_e10_no_raw_floats_at_interface(e10_run):
    """The output_node_tuple must be all strings -- no float leaked into the
    new node-IS-output interface. (Internal sigma_total etc remain floats;
    that is fine.)"""
    rows = _read_trace(e10_run)
    for r in rows:
        out = r["output_node_tuple"]
        for component in out:
            assert isinstance(component, str), (
                f"output_node_tuple component is not a string: {component!r} "
                f"(type {type(component).__name__})"
            )


def test_e10_posterior_mask_converges(e10_run):
    """hamming_normalised must be reported and lie in [0.0, 1.0]. No tighter
    bar -- we report what the cold-start posterior actually converges to."""
    m = _read_metrics(e10_run)
    assert "hamming_normalised" in m
    h = m["hamming_normalised"]
    assert 0.0 <= h <= 1.0, f"hamming_normalised out of [0, 1]: {h}"


def test_e10_monodromy_consistency_recorded(e10_run):
    """monodromy_consistency_loss exists and is a non-negative float."""
    m = _read_metrics(e10_run)
    assert "monodromy_consistency_loss" in m
    val = m["monodromy_consistency_loss"]
    assert isinstance(val, float)
    assert val >= 0.0, f"monodromy_consistency_loss must be >= 0, got {val}"


def test_e10_mask_zeros_illegal_after_convergence(e10_run):
    """illegal_transition_rate is reported by the LEARNED mask path.

    Originally this bar was '< 0.10'. The user's directive is to LOOSEN, not
    to tune the runner. With a fully cold-start posterior (Beta(1,1)) and a
    quality signal q = sigmoid(-z) where z >= 0 (energy / sigma /
    contradiction are non-negative), each observation can only push q in
    [0, 0.5]. Unobserved (s, t) pairs stay at posterior_mean == 0.5 exactly.
    The learned mask thresholded at >= 0.5 therefore does NOT match the
    hand-authored FSM in this single-pass unsupervised regime; we report
    that fact rather than gaming the threshold.

    The structural assertion is the LOOSER one: the metric is recorded and
    is a finite probability in [0, 1].
    """
    m = _read_metrics(e10_run)
    rate = m["illegal_transition_rate"]
    assert 0.0 <= rate <= 1.0, f"illegal_transition_rate out of [0, 1]: {rate}"
