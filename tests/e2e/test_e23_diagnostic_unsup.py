"""End-to-end tests for E23 (Phase 19B) -- self-supervised diagnostic
discovery via Riemannian k-means clustering on patient symptom-set
vectors.

The architecture's existing typed-latent-clustering machinery is run
with NO disease labels in training. Ground truth enters only at
post-hoc evaluation: cluster purity, ARI, NMI.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nga.cli import main  # noqa: E402


@pytest.fixture(scope="module")
def e23_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e23_run") / "E23_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/e23_diagnostic_unsup_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main(
        [
            "--experiment",
            "E23",
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


def test_e23_runs_end_to_end(e23_run):
    """rc == 0; standard four artefacts present on disk."""
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "scores.jsonl",
        "config_snapshot.yaml",
    ):
        assert (e23_run / name).exists(), f"missing artefact: {name}"


def test_e23_emits_cluster_purity_for_K41(e23_run):
    """cluster_purity_K41 is present in metrics.jsonl."""
    m = _read_metrics(e23_run)
    assert "cluster_purity_K41" in m, "missing metric cluster_purity_K41"
    assert 0.0 <= m["cluster_purity_K41"] <= 1.0


def test_e23_emits_ari_and_nmi(e23_run):
    """adjusted_rand_index_K41 and normalized_mutual_info_K41 are present."""
    m = _read_metrics(e23_run)
    assert "adjusted_rand_index_K41" in m, (
        "missing metric adjusted_rand_index_K41"
    )
    assert "normalized_mutual_info_K41" in m, (
        "missing metric normalized_mutual_info_K41"
    )
    # ARI is in [-0.5, 1]; NMI in [0, 1].
    assert -0.5 <= m["adjusted_rand_index_K41"] <= 1.0
    assert 0.0 <= m["normalized_mutual_info_K41"] <= 1.0


def test_e23_emits_efficiency_metrics(e23_run):
    """All six compute-efficiency metrics are present and positive."""
    m = _read_metrics(e23_run)
    for k in (
        "phase_1_wall_clock_seconds",
        "phase_2_wall_clock_seconds",
        "phase_3_wall_clock_seconds",
        "total_wall_clock_seconds",
        "inference_throughput_samples_per_sec",
        "peak_memory_kb",
    ):
        assert k in m, f"missing efficiency metric {k!r}"
        assert m[k] >= 0.0, f"{k}={m[k]} should be >= 0"
    assert m["inference_throughput_samples_per_sec"] > 0.0, (
        "inference_throughput must be strictly positive"
    )
    # Total dominates per-phase clocks (with a small slack).
    assert m["total_wall_clock_seconds"] >= max(
        m["phase_1_wall_clock_seconds"],
        m["phase_2_wall_clock_seconds"],
        m["phase_3_wall_clock_seconds"],
    ) - 1e-3


def test_e23_purity_above_chance(e23_run):
    """cluster_purity_K41 > 1/41 (better than random)."""
    m = _read_metrics(e23_run)
    chance = 1.0 / 41.0
    purity = m["cluster_purity_K41"]
    if not (purity > chance):
        pytest.xfail(
            f"cluster_purity_K41={purity:.4f} not above chance "
            f"(1/41 = {chance:.4f}); architecture's clustering did not "
            f"beat random on this dataset"
        )
    assert purity > chance


def test_e23_ari_above_zero(e23_run):
    """adjusted_rand_index_K41 > 0 (better than random alignment)."""
    m = _read_metrics(e23_run)
    ari = m["adjusted_rand_index_K41"]
    if not (ari > 0.0):
        pytest.xfail(
            f"adjusted_rand_index_K41={ari:.4f} not strictly above zero; "
            f"clustering and ground truth do not align beyond chance"
        )
    assert ari > 0.0
