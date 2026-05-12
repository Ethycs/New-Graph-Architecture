"""End-to-end tests for E15 (training-maturity sweep on E14's pipeline).

E15 trains the same TorchEnergyTrainer pipeline E14 trains, but breaks
the run into checkpoints (default ``[1, 3, 5, 10]``) and evaluates the
four AUROCs (sigma_failure / margin_failure / sigma_structural /
margin_structural) at each one. The maturity hypothesis is that
sigma's lead over margin should DECAY (or flip) as training matures.

Per the don't-tune directive: if either uplift curve does NOT decay
the test marks XFAIL with the observed values documented as a load-
bearing finding rather than chasing the threshold from the runner.

The whole module is gated by ``pytest.importorskip("torch")`` so a
torch-less CI continues to run the rest of the suite.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nga.cli import main  # noqa: E402

E15_CHECKPOINT_NAMES = (
    "sigma_failure_auroc",
    "margin_failure_auroc",
    "sigma_failure_uplift",
    "sigma_structural_auroc",
    "margin_structural_auroc",
    "sigma_structural_uplift",
)


@pytest.fixture(scope="module")
def e15_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e15_run") / "E15_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/e15_listops_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main(
        [
            "--experiment",
            "E15",
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


def _read_metrics_rows(d: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (d / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _read_metrics_headline(d: Path) -> dict[str, float]:
    """Headline (legacy E14-style) metrics: ``step == 0`` rows only."""
    return {
        row["metric_name"]: row["value"]
        for row in _read_metrics_rows(d)
        if int(row["step"]) == 0
    }


def _read_checkpoint_table(d: Path) -> dict[int, dict[str, float]]:
    """Per-checkpoint AUROC table indexed by epoch (step > 0)."""
    table: dict[int, dict[str, float]] = {}
    for row in _read_metrics_rows(d):
        step = int(row["step"])
        if step <= 0:
            continue
        if row["metric_name"] not in E15_CHECKPOINT_NAMES:
            continue
        table.setdefault(step, {})[row["metric_name"]] = float(row["value"])
    return table


def _read_trace(d: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (d / "decision_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_e15_runs_end_to_end(e15_run):
    """rc == 0 (already asserted in fixture); standard four artefacts plus
    decision_trace.jsonl present."""
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "scores.jsonl",
        "decision_trace.jsonl",
    ):
        assert (e15_run / name).exists(), f"missing artefact: {name}"
    headline = _read_metrics_headline(e15_run)
    # Headline keys mirror E14 so aggregate.py readers keep working.
    for required in (
        "phase_a_hamming_normalised",
        "phase_b_hamming_normalised",
        "sigma_auroc",
        "margin_auroc",
        "sigma_uplift",
        "sigma_structural_auroc",
        "margin_structural_auroc",
        "sigma_structural_uplift",
        "n_torch_trainable_params",
        "final_loss",
    ):
        assert required in headline, f"missing headline metric {required!r}"


def test_e15_emits_checkpoint_metrics(e15_run):
    """metrics.jsonl contains per-checkpoint AUROC rows at MULTIPLE
    distinct ``step`` values (the checkpoint epochs).
    """
    table = _read_checkpoint_table(e15_run)
    # We expect at least 2 distinct checkpoint steps; default config
    # uses [1, 3, 5, 10] so there should be 4.
    assert len(table) >= 2, (
        f"expected at least 2 distinct checkpoint steps; got "
        f"{sorted(table.keys())}"
    )
    # Every checkpoint must populate all six per-checkpoint metrics.
    for step, row in table.items():
        for name in E15_CHECKPOINT_NAMES:
            assert name in row, (
                f"checkpoint step={step} is missing metric {name!r}; "
                f"got {sorted(row.keys())}"
            )


def test_e15_uplift_curve_decays(e15_run):
    """The maturity hypothesis: sigma_failure_uplift at the FIRST
    checkpoint should be >= sigma_failure_uplift at the LAST checkpoint
    (i.e. the gap shrinks or stays flat as training matures).

    Per the don't-tune directive: if the natural numbers violate this
    we mark XFAIL with the observed values documented. This is a
    load-bearing finding: a violation means the maturity hypothesis is
    wrong and we have a different research direction.
    """
    table = _read_checkpoint_table(e15_run)
    epochs = sorted(table.keys())
    assert len(epochs) >= 2, (
        f"need at least 2 checkpoints to test decay; got {epochs}"
    )
    first, last = epochs[0], epochs[-1]
    u_first = table[first]["sigma_failure_uplift"]
    u_last = table[last]["sigma_failure_uplift"]
    if u_first < u_last:
        pytest.xfail(
            f"sigma_failure_uplift did NOT decay with training maturity: "
            f"epoch {first} uplift={u_first:+.4f}; "
            f"epoch {last} uplift={u_last:+.4f}. "
            f"This is a load-bearing finding -- the maturity hypothesis "
            f"appears to be wrong on this configuration. Full curve: "
            f"{[(e, table[e]['sigma_failure_uplift']) for e in epochs]}"
        )
    assert u_first >= u_last, (
        f"sigma_failure_uplift must decay (or stay flat) with training; "
        f"epoch {first}={u_first:+.4f} vs epoch {last}={u_last:+.4f}"
    )


def test_e15_structural_uplift_curve_decays(e15_run):
    """The structural-AUROC counterpart of the maturity hypothesis:
    sigma_structural_uplift at the first checkpoint should be >= the
    last. Same xfail-on-miss policy as the failure-AUROC variant.
    """
    table = _read_checkpoint_table(e15_run)
    epochs = sorted(table.keys())
    assert len(epochs) >= 2, (
        f"need at least 2 checkpoints to test decay; got {epochs}"
    )
    first, last = epochs[0], epochs[-1]
    u_first = table[first]["sigma_structural_uplift"]
    u_last = table[last]["sigma_structural_uplift"]
    if u_first < u_last:
        pytest.xfail(
            f"sigma_structural_uplift did NOT decay with training "
            f"maturity: epoch {first} uplift={u_first:+.4f}; "
            f"epoch {last} uplift={u_last:+.4f}. "
            f"This is a load-bearing finding -- the maturity hypothesis "
            f"appears to be wrong on this configuration. Full curve: "
            f"{[(e, table[e]['sigma_structural_uplift']) for e in epochs]}"
        )
    assert u_first >= u_last, (
        f"sigma_structural_uplift must decay (or stay flat) with training; "
        f"epoch {first}={u_first:+.4f} vs epoch {last}={u_last:+.4f}"
    )


def test_e15_phase_a_seeded_correctly(e15_run):
    """Phase A's classical cold-start (forward-backward + Beta M-step)
    seeds the posterior mask correctly under E15's chunked training,
    just as it does under E14. Substrate-independent, so the same
    < 0.20 bar applies.
    """
    headline = _read_metrics_headline(e15_run)
    assert "phase_a_hamming_normalised" in headline
    assert headline["phase_a_hamming_normalised"] < 0.20, (
        f"Phase-A classical cold start must recover the ListOps FSM "
        f"within 20% hamming distance; got "
        f"{headline['phase_a_hamming_normalised']:.4f}"
    )


def test_e15_decision_trace_present_on_last_checkpoint(e15_run):
    """decision_trace.jsonl exists with v1.1 fields populated. E15
    emits trace ROWS only from the final checkpoint (to avoid 4x trace
    bloat); per-row v1.1 commitment fields must still populate.
    """
    rows = _read_trace(e15_run)
    assert len(rows) > 0, "decision_trace.jsonl is empty"
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
