"""End-to-end tests for E25 (Phase 20 Wave B Tier 1 sanity).

Extract a TPN's typed graph from the hidden-state stream of our own
``FrozenEncoderTorch`` substrate on the python_big grammar, and compare
the extracted legality matrix against the hand-authored
``python_big.fsm.yaml`` legality matrix by best-permutation Hamming.

The Tier 1 sanity bar in ``docs/proposals/graph-extraction.md`` calls
for Hamming <= 0.05 on >= 3/5 grammars; this runner exercises one
grammar (python_big), so these tests check that:

  1. The pipeline runs to completion and emits the expected artefacts.
  2. K-selection picks a value within a reasonable band of V (= n_states).
  3. The reported Hamming and cluster_purity are recorded (no strict
     bar on the absolute numerical values at this tier -- the
     architectural finding lives in research_log.md when all 5 grammars
     are run).
  4. Compute-efficiency metrics are present.
  5. extracted_graph.json is well-formed and carries both the extracted
     and the gold legality matrices for downstream inspection.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nga.cli import main  # noqa: E402


@pytest.fixture(scope="module")
def e25_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e25_run") / "E25_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/e25_extraction_torch_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main(
        [
            "--experiment",
            "E25",
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
    assert rc == 0, f"E25 CLI returned non-zero: {rc}"
    return run_dir


def _load_metrics(run_dir: Path) -> dict[str, float]:
    metrics = {}
    text = (run_dir / "metrics.jsonl").read_text()
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metrics[row["metric_name"]] = float(row["value"])
    return metrics


def test_e25_emits_artefacts(e25_run: Path) -> None:
    """All standard artefacts emitted, plus extracted_graph.json."""
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "config_snapshot.yaml",
        "ablation_snapshot.yaml",
        "extracted_graph.json",
    ):
        assert (e25_run / name).exists(), f"missing artefact: {name}"


def test_e25_K_star_in_reasonable_band(e25_run: Path) -> None:
    """K_star lands within +/- K_range_pad of V (the FSM vertex count)."""
    metrics = _load_metrics(e25_run)
    V = int(metrics["V_ground_truth"])
    K_star = int(metrics["K_star"])
    # Runner's default K_range_pad=5 means K_range is [V-5, V+5].
    assert V - 5 <= K_star <= V + 5, (
        f"K_star={K_star} outside the K_range +/- 5 of V={V}"
    )


def test_e25_hamming_and_purity_recorded(e25_run: Path) -> None:
    """Hamming and cluster_purity are recorded as floats in [0, 1].

    No strict numerical bar at this tier; the architectural finding
    (whether Hamming meets the 0.05 sanity bar on python_big) goes to
    research_log.md as an observation, not a gating assertion.
    """
    metrics = _load_metrics(e25_run)
    h = metrics["extracted_hamming_normalised"]
    p = metrics["cluster_purity"]
    assert 0.0 <= h <= 1.0, f"hamming out of range: {h}"
    assert 0.0 <= p <= 1.0, f"cluster_purity out of range: {p}"


def test_e25_cluster_purity_beats_chance(e25_run: Path) -> None:
    """Cluster purity is materially above 1/V (chance for a random partition)."""
    metrics = _load_metrics(e25_run)
    V = int(metrics["V_ground_truth"])
    chance = 1.0 / V
    purity = metrics["cluster_purity"]
    assert purity > 2.0 * chance, (
        f"cluster_purity={purity:.4f} not materially above chance "
        f"({chance:.4f}); the frozen-encoder substrate carries no "
        f"structural information at all"
    )


def test_e25_compute_metrics_recorded(e25_run: Path) -> None:
    """All five compute-efficiency metrics present and non-negative."""
    metrics = _load_metrics(e25_run)
    for name in (
        "phase_1_wall_clock_seconds",
        "phase_2_wall_clock_seconds",
        "phase_3_wall_clock_seconds",
        "total_wall_clock_seconds",
        "extraction_throughput_steps_per_sec",
        "peak_memory_kb",
    ):
        v = metrics.get(name)
        assert v is not None, f"missing metric: {name}"
        assert v >= 0.0, f"metric {name}={v} should be non-negative"


def test_e25_extracted_graph_json_well_formed(e25_run: Path) -> None:
    """extracted_graph.json carries the extracted and gold legality matrices
    plus the FSM vertex IDs in canonical order."""
    data = json.loads((e25_run / "extracted_graph.json").read_text())
    assert data["grammar"] == "python_big"
    assert "V_ground_truth" in data
    assert "K_star" in data
    assert "extracted_mask" in data
    assert "gold_legality" in data
    assert "fsm_vertex_ids" in data
    K = data["K_star"]
    assert len(data["extracted_mask"]) == K
    assert all(len(row) == K for row in data["extracted_mask"])
    V = data["V_ground_truth"]
    assert len(data["gold_legality"]) == V
    assert len(data["fsm_vertex_ids"]) == V
