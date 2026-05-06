"""Phase 0 acceptance test. See docs/build-order.md §Phase 0 §Acceptance.

This test gates Phase 0 closing: a noop CLI invocation must load a config,
validate the graph FSM, resolve the A0 ablation tuple, and write four snapshot
files plus three byte-empty JSONL streams to the output directory.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nga.cli import derive_run_id, main


# ---------------------------------------------------------------------------
# test_phase0_noop_writes_empty_streams
# ---------------------------------------------------------------------------

def test_phase0_noop_writes_empty_streams(
    tmp_run_dir: Path,
    mnist_config: Path,
    ablation_yaml_path: Path,
    seeded_rng: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Noop CLI invocation writes four snapshots and three byte-empty JSONL files."""
    rc = main([
        "--experiment", "E0",
        "--ablation", "A0",
        "--config", str(mnist_config),
        "--seed", "42",
        "--output", str(tmp_run_dir),
        "--ablation-file", str(ablation_yaml_path),
        "--skip-train",
        "--skip-eval",
    ])
    assert rc == 0, f"expected exit code 0, got {rc}"

    # All six expected output files must exist.
    for name in (
        "config_snapshot.yaml",
        "ablation_snapshot.yaml",
        "graph_fsm.yaml",
        "metrics.jsonl",
        "results.jsonl",
        "scores.jsonl",
    ):
        assert (tmp_run_dir / name).exists(), f"missing {name}"

    # The three JSONL sinks must be byte-empty for a noop run.
    for name in ("metrics.jsonl", "results.jsonl", "scores.jsonl"):
        path = tmp_run_dir / name
        assert path.read_bytes() == b"", f"{name} should be empty for noop"

    # ablation_snapshot.yaml must parse and contain the A0 defaults.
    ablation_snap = yaml.safe_load(
        (tmp_run_dir / "ablation_snapshot.yaml").read_text(encoding="utf-8")
    )
    assert ablation_snap["graph_mask_enabled"] is True
    assert ablation_snap["typed_scores_enabled"] is True

    # config_snapshot.yaml must parse and reflect CLI-merged values.
    config_snap = yaml.safe_load(
        (tmp_run_dir / "config_snapshot.yaml").read_text(encoding="utf-8")
    )
    assert config_snap["seed"] == 42
    assert config_snap["ablation"] == "A0"
    assert config_snap["dataset"] == "mnist"

    # graph_fsm.yaml must parse and report the expected vertex count.
    fsm_snap = yaml.safe_load(
        (tmp_run_dir / "graph_fsm.yaml").read_text(encoding="utf-8")
    )
    assert fsm_snap["vertex_count"] == 10


# ---------------------------------------------------------------------------
# test_derive_run_id
# ---------------------------------------------------------------------------

def test_derive_run_id() -> None:
    """derive_run_id formats the canonical run-id string correctly."""
    assert derive_run_id("E0", "A0", 42) == "E0_A0_seed42"
