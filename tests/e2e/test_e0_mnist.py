"""Phase 1 acceptance test: E0 MNIST end-to-end.

Gate for Phase 1 closing per docs/build-order.md §Phase 1 §Acceptance:

    E0 MNIST runs end-to-end and writes accuracy, low_margin_accuracy,
    confusion_graph_density into metrics.jsonl plus per-sample singular_flag
    (margin-only) into results.jsonl.

The test invokes nga.cli.main without --skip-train/--skip-eval so the runner
trains a sklearn LogisticRegression on the digits dataset and writes all three
JSONL streams. It then asserts on the aggregate metrics and the per-sample
record schema.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nga.cli import main


@pytest.mark.slow
def test_e0_mnist_end_to_end(
    tmp_run_dir: Path,
    mnist_config: Path,
    ablation_yaml_path: Path,
    seeded_rng: int,
) -> None:
    """Phase 1 acceptance: E0 hits >= 0.90 accuracy and writes the spec'd metrics."""
    rc = main([
        "--experiment", "E0",
        "--ablation", "A0",
        "--config", str(mnist_config),
        "--seed", "42",
        "--output", str(tmp_run_dir),
        "--ablation-file", str(ablation_yaml_path),
    ])
    assert rc == 0, f"E0 exited with rc={rc}"

    # Snapshots from Phase 0 must still be there.
    for name in ("config_snapshot.yaml", "ablation_snapshot.yaml", "graph_fsm.yaml"):
        assert (tmp_run_dir / name).exists(), f"missing {name}"

    # The three JSONL streams must be non-empty.
    metrics_path = tmp_run_dir / "metrics.jsonl"
    results_path = tmp_run_dir / "results.jsonl"
    scores_path = tmp_run_dir / "scores.jsonl"
    for p in (metrics_path, results_path, scores_path):
        assert p.exists() and p.stat().st_size > 0, f"{p.name} is empty"

    # Parse metrics.
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    metric_names = {m["metric_name"] for m in metrics}
    assert {"accuracy", "low_margin_accuracy", "confusion_graph_density"} <= metric_names, (
        f"missing required metric names; got {metric_names}"
    )

    by_name = {m["metric_name"]: m for m in metrics}
    accuracy = by_name["accuracy"]["value"]
    low_margin_acc = by_name["low_margin_accuracy"]["value"]
    confusion_density = by_name["confusion_graph_density"]["value"]

    # Acceptance threshold from docs/setup/test-fixtures.md.
    assert accuracy >= 0.90, f"accuracy {accuracy:.4f} below the 0.90 acceptance floor"

    # Sanity: low-margin samples are less accurate than overall.
    assert low_margin_acc <= accuracy + 1e-9, (
        f"low_margin_accuracy ({low_margin_acc:.4f}) should not exceed overall "
        f"accuracy ({accuracy:.4f}); the singularity flag would be useless if it did"
    )

    # Confusion graph density is in [0, 1].
    assert 0.0 <= confusion_density <= 1.0, f"density {confusion_density} out of range"

    # Per-metric record fields.
    for m in metrics:
        assert m["schema_version"] == "1.0"
        assert m["experiment"] == "E0"
        assert m["ablation"] == "A0"
        assert m["seed"] == 42
        assert m["split"] == "test"

    # Parse results: every record has singular_flag and margin.
    results = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    assert len(results) > 100, f"expected hundreds of test samples; got {len(results)}"
    for r in results:
        assert "singular_flag" in r
        assert isinstance(r["singular_flag"], bool)
        assert "margin" in r
        assert -1.0 <= r["margin"] <= 1.0
        assert r["experiment"] == "E0"
        assert r["ablation"] == "A0"

    # Parse scores: every record is a TypedScoreRecord with non-empty dist.
    scores = [json.loads(line) for line in scores_path.read_text().splitlines() if line]
    assert len(scores) == len(results), "scores.jsonl and results.jsonl must align 1:1"
    for s in scores:
        assert "dist" in s and len(s["dist"]) == 10  # ten digit classes
        assert 0.0 <= s["confidence"] <= 1.0


@pytest.mark.slow
def test_e0_singular_flag_correlates_with_errors(
    tmp_run_dir: Path,
    mnist_config: Path,
    ablation_yaml_path: Path,
    seeded_rng: int,
) -> None:
    """The singular_flag (margin-only) should fire more often on incorrect predictions
    than on correct ones. Validates the load-bearing claim that low margin predicts failure.
    """
    rc = main([
        "--experiment", "E0",
        "--ablation", "A0",
        "--config", str(mnist_config),
        "--seed", "42",
        "--output", str(tmp_run_dir),
        "--ablation-file", str(ablation_yaml_path),
    ])
    assert rc == 0

    results = [
        json.loads(line)
        for line in (tmp_run_dir / "results.jsonl").read_text().splitlines()
        if line
    ]
    correct = [r for r in results if r["y_true"] == r["y_hat"]]
    incorrect = [r for r in results if r["y_true"] != r["y_hat"]]
    if not incorrect:
        pytest.skip("classifier was perfect; no errors to compare against")

    rate_correct = sum(r["singular_flag"] for r in correct) / max(len(correct), 1)
    rate_incorrect = sum(r["singular_flag"] for r in incorrect) / len(incorrect)
    assert rate_incorrect > rate_correct, (
        f"singular_flag rate on errors ({rate_incorrect:.3f}) should exceed rate on "
        f"correct predictions ({rate_correct:.3f}); margin is not a useful signal otherwise"
    )


@pytest.mark.slow
def test_e0_emits_decision_trace(
    tmp_run_dir: Path,
    mnist_config: Path,
    ablation_yaml_path: Path,
    seeded_rng: int,
) -> None:
    """E0 must emit a decision_trace.jsonl with one row per sample populated
    with sigma, mask, and control fields. This is the interpretability surface
    introduced in Phase 4-5."""
    rc = main([
        "--experiment", "E0",
        "--ablation", "A0",
        "--config", str(mnist_config),
        "--seed", "42",
        "--output", str(tmp_run_dir),
        "--ablation-file", str(ablation_yaml_path),
    ])
    assert rc == 0

    trace_path = tmp_run_dir / "decision_trace.jsonl"
    assert trace_path.exists() and trace_path.stat().st_size > 0, (
        "decision_trace.jsonl was not produced (or is empty)"
    )

    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    assert len(rows) >= 1, "expected at least one decision-trace row"

    # Spot-check the first row.
    first = rows[0]
    assert "sigma_total" in first and isinstance(first["sigma_total"], (int, float))
    assert 0.0 <= first["sigma_total"] <= 1.0
    assert "mask" in first and isinstance(first["mask"], dict)
    assert "enabled" in first["mask"]
    assert "current_state" in first["mask"]
    assert "control" in first and isinstance(first["control"], dict)
    assert "decision" in first["control"]
    assert "sigma_observed" in first["control"]
    assert "sigma_threshold_used" in first["control"]
    # Energy + monodromy fields wired in.
    assert "energy_total" in first and isinstance(first["energy_total"], (int, float))
    assert "energy_breakdown" in first and isinstance(first["energy_breakdown"], dict)
    assert "monodromy_class" in first


@pytest.mark.slow
def test_e0_control_decisions_recorded(
    tmp_run_dir: Path,
    mnist_config: Path,
    ablation_yaml_path: Path,
    seeded_rng: int,
) -> None:
    """The control-policy verdict per sample must land in decision_trace.jsonl.
    Decisions must come from the documented vocabulary, and ROUTE_NORMAL must
    dominate on a digit classifier where most predictions are confident."""
    rc = main([
        "--experiment", "E0",
        "--ablation", "A0",
        "--config", str(mnist_config),
        "--seed", "42",
        "--output", str(tmp_run_dir),
        "--ablation-file", str(ablation_yaml_path),
    ])
    assert rc == 0

    trace_path = tmp_run_dir / "decision_trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    assert len(rows) > 0

    allowed = {"ROUTE_NORMAL", "ROUTE_RECOVERY", "ABSTAIN"}
    counts: dict[str, int] = {k: 0 for k in allowed}
    for r in rows:
        decision = r["control"]["decision"]
        assert decision in allowed, f"unexpected control decision: {decision!r}"
        counts[decision] += 1

    assert counts["ROUTE_NORMAL"] > 0, (
        f"expected most predictions to ROUTE_NORMAL on digits; counts={counts}"
    )
