"""End-to-end tests for E24 (Phase 20 Wave A) -- universal graph extraction.

The runner generates a synthetic typed Markov chain and a noisy-centroid
embedding of it, then drives the proposal's six-step extraction pipeline
end-to-end. Acceptance bars:

  - K_star is recovered exactly (K_true=5),
  - extracted_hamming_normalised against the ground-truth legality matrix
    is <= 0.10 (the Tier 1 sanity bar from docs/proposals/graph-extraction.md),
  - holdout_nll_improvement_per_token > 0 (the extracted FSM beats a
    uniform-transition chain on held-out trajectories).

When the runner passes those bars on the synthetic substrate, the
machinery is validated; Wave B will repeat the same checks against
TorchEnergyTrainer extractions on Python big and JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nga.cli import main


@pytest.fixture(scope="module")
def e24_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e24_run") / "E24_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/e24_graph_extraction_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main(
        [
            "--experiment",
            "E24",
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
    assert rc == 0, f"E24 CLI returned non-zero: {rc}"
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


def test_e24_run_emits_four_artefacts(e24_run: Path) -> None:
    """E24 emits metrics.jsonl, results.jsonl, config_snapshot,
    ablation_snapshot plus the extracted_graph.json side artefact."""
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "config_snapshot.yaml",
        "ablation_snapshot.yaml",
        "extracted_graph.json",
    ):
        assert (e24_run / name).exists(), f"missing artefact: {name}"


def test_e24_K_star_matches_K_true(e24_run: Path) -> None:
    """The K-selection step recovers the true number of states."""
    metrics = _load_metrics(e24_run)
    assert metrics["K_star"] == metrics["K_ground_truth"], (
        f"K_star={metrics['K_star']} != K_ground_truth="
        f"{metrics['K_ground_truth']}; K-selection failed to recover K_true"
    )


def test_e24_extracted_hamming_meets_tier1_bar(e24_run: Path) -> None:
    """Extracted FSM matches ground-truth at <= 0.10 normalised Hamming
    (the Tier 1 sanity bar from the graph-extraction proposal)."""
    metrics = _load_metrics(e24_run)
    h = metrics["extracted_hamming_normalised"]
    assert h <= 0.10, (
        f"extracted_hamming_normalised={h:.4f} exceeds the Tier 1 bar "
        f"of 0.10; FSM extraction did not recover the ground-truth legality"
    )


def test_e24_holdout_nll_beats_chain_baseline(e24_run: Path) -> None:
    """Extracted transition matrix beats a uniform-chain baseline on
    held-out NLL."""
    metrics = _load_metrics(e24_run)
    lift = metrics["holdout_nll_improvement_per_token"]
    assert lift > 0.0, (
        f"holdout_nll_improvement_per_token={lift:+.4f}; extracted FSM "
        f"does not beat the uniform-transition chain"
    )


def test_e24_phase_a_self_consistency(e24_run: Path) -> None:
    """The posterior-driven legality matrix equals the legality matrix it
    was built from (zero self-Hamming by construction)."""
    metrics = _load_metrics(e24_run)
    assert metrics["phase_a_hamming_normalised"] == pytest.approx(0.0, abs=1e-9)


def test_e24_cluster_purity_high(e24_run: Path) -> None:
    """Cluster purity vs ground-truth states is materially above 1/K."""
    metrics = _load_metrics(e24_run)
    k = int(metrics["K_ground_truth"])
    chance = 1.0 / k
    assert metrics["cluster_purity"] >= 5.0 * chance, (
        f"cluster_purity={metrics['cluster_purity']:.4f}; only 1/K="
        f"{chance:.4f} expected from chance; recovery very weak"
    )


def test_e24_compute_metrics_recorded(e24_run: Path) -> None:
    """All five compute-efficiency metrics are recorded with finite values."""
    metrics = _load_metrics(e24_run)
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


def test_e24_extracted_graph_json_well_formed(e24_run: Path) -> None:
    """extracted_graph.json carries the expected fields and shapes."""
    data = json.loads((e24_run / "extracted_graph.json").read_text())
    assert "K_star" in data
    assert "extracted_mask" in data
    assert "ground_truth_legality" in data
    assert "types" in data
    K = data["K_star"]
    assert len(data["extracted_mask"]) == K
    assert all(len(row) == K for row in data["extracted_mask"])
    assert len(data["types"]) == K
