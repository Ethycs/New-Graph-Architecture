from __future__ import annotations
import json
from pathlib import Path
import pytest
from nga.cli import main

@pytest.fixture(scope="module")
def e8_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e8_run") / "E8_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/babyai_synthetic_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main([
        "--experiment", "E8", "--ablation", "A0",
        "--config", str(cfg), "--seed", "42",
        "--output", str(run_dir), "--ablation-file", str(abl),
    ])
    assert rc == 0
    return run_dir

def _read_metrics(d):
    return {json.loads(l)["metric_name"]: json.loads(l)["value"] for l in (d / "metrics.jsonl").read_text().splitlines() if l}

@pytest.mark.slow
def test_e8_reports_transfer_metrics(e8_run):
    m = _read_metrics(e8_run)
    assert "accuracy_train" in m
    assert "accuracy_test" in m
    assert "transfer_gap" in m
    assert 0.0 <= m["accuracy_train"] <= 1.0
    assert 0.0 <= m["accuracy_test"] <= 1.0

@pytest.mark.slow
def test_e8_transfer_gap_consistent(e8_run):
    m = _read_metrics(e8_run)
    assert abs(m["transfer_gap"] - (m["accuracy_train"] - m["accuracy_test"])) < 1e-9
