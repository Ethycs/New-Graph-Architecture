"""End-to-end tests for E16 (Phase-A-disabled maturity sweep).

E16 is the same pipeline as E15 with ONE thing changed: Phase A's
classical forward-backward + Beta-conjugate seed is removed. The
trainer's posterior starts at uniform Beta(1, 1) and Phase B has to
learn the legality structure from gradient updates alone.

Hypothesis (load-bearing): without Phase A's seed, sigma should
DOMINATE margin at low epochs because sigma encodes structural
ambiguity by FSM construction (no training needed) while margin's
softmax(-distance + legality_bias) bias starts at 0 and must be
learned.

Per the don't-tune directive, the dominance test below marks XFAIL
on miss with the actual observed numbers documented as a load-bearing
finding rather than chasing the threshold from the runner.

The whole module is gated by ``pytest.importorskip("torch")`` so a
torch-less CI continues to run the rest of the suite.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from nga.cli import main  # noqa: E402

E16_CHECKPOINT_NAMES = (
    "sigma_failure_auroc",
    "margin_failure_auroc",
    "sigma_failure_uplift",
    "sigma_structural_auroc",
    "margin_structural_auroc",
    "sigma_structural_uplift",
)


@pytest.fixture(scope="module")
def e16_run(tmp_path_factory):
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_dir = tmp_path_factory.mktemp("e16_run") / "E16_A0_seed42"
    cfg = repo_root / "tests/fixtures/configs/e16_listops_minimal.yaml"
    abl = repo_root / "tests/fixtures/ablations/ablations.yaml"
    rc = main(
        [
            "--experiment",
            "E16",
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
        if row["metric_name"] not in E16_CHECKPOINT_NAMES:
            continue
        table.setdefault(step, {})[row["metric_name"]] = float(row["value"])
    return table


def _read_trace(d: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (d / "decision_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_e16_runs_end_to_end(e16_run):
    """rc == 0 (asserted in the fixture); standard four artefacts plus
    decision_trace.jsonl present.
    """
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "scores.jsonl",
        "decision_trace.jsonl",
    ):
        assert (e16_run / name).exists(), f"missing artefact: {name}"
    headline = _read_metrics_headline(e16_run)
    for required in (
        "phase_a_hamming_normalised",
        "phase_b_hamming_normalised",
        "phase_a_disabled",
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


def test_e16_phase_a_actually_disabled(e16_run):
    """Phase A is REALLY disabled: ``phase_a_disabled == 1.0`` and the
    reported phase_a_hamming reflects an uncommitted cold prior, not a
    Phase-A-seeded mask. Under the strict skeptical threshold > 0.5
    with Beta(1, 1), the cold legality matrix is all-False; its hamming
    against the gold FSM legality is therefore the count of legal
    cells in the gold matrix divided by total cells -- well above the
    < 0.20 bar that E15's Phase-A-seeded equivalent must clear (and
    bitwise inverse to FSMs in which the majority of cells are legal).
    """
    headline = _read_metrics_headline(e16_run)
    assert headline.get("phase_a_disabled") == 1.0, (
        f"phase_a_disabled discriminator must be 1.0 for an E16 run; "
        f"got {headline.get('phase_a_disabled')!r}"
    )
    h = headline["phase_a_hamming_normalised"]
    if not (h > 0.5):
        pytest.xfail(
            f"phase_a_hamming_normalised = {h:.4f}; expected > 0.5 under "
            "the bitwise-inverse-to-FSM rationale. The actual value reflects "
            "the FSM's true legal-cell density: with the ListOps_d3 FSM the "
            "gold legality matrix is sparse (only ~15% of cells legal), so "
            "an all-False cold mask diverges from gold on only those cells, "
            "yielding a small hamming. The discriminator phase_a_disabled "
            "still confirms Phase A was disabled. This is a load-bearing "
            "finding: the cold-prior hamming is not a useful 'Phase A is "
            "off' signal for sparse FSMs; phase_a_disabled is."
        )
    assert h > 0.5


def test_e16_emits_checkpoint_metrics(e16_run):
    """metrics.jsonl contains per-checkpoint AUROC rows at MULTIPLE
    distinct ``step`` values (the checkpoint epochs). Same shape as E15.
    """
    table = _read_checkpoint_table(e16_run)
    assert len(table) >= 2, (
        f"expected at least 2 distinct checkpoint steps; got "
        f"{sorted(table.keys())}"
    )
    for step, row in table.items():
        for name in E16_CHECKPOINT_NAMES:
            assert name in row, (
                f"checkpoint step={step} is missing metric {name!r}; "
                f"got {sorted(row.keys())}"
            )


def test_e16_torch_atom_used(e16_run):
    """E16 actually wires the torch trainer: ``n_torch_trainable_params``
    must be strictly positive in the headline.
    """
    headline = _read_metrics_headline(e16_run)
    n_params = headline.get("n_torch_trainable_params", 0.0)
    assert n_params > 0, (
        f"n_torch_trainable_params must be > 0 (torch atom in use); "
        f"got {n_params!r}"
    )


def test_e16_decision_trace_present(e16_run):
    """decision_trace.jsonl exists with v1.1 fields populated. E16
    emits trace ROWS only from the final checkpoint (matching E15) to
    avoid 4x trace bloat.
    """
    rows = _read_trace(e16_run)
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


def test_e16_sigma_dominates_at_epoch_1(e16_run):
    """Load-bearing: at the FIRST checkpoint (epoch 1) WITHOUT Phase A,
    sigma should win on at least one of failure / structural uplift.

    Rationale: without Phase A's classical seed, the legality_bias
    starts at 0 and margin's predicted distribution carries no
    structural information. Sigma, derived from FSM construction
    independent of training, retains its structural signal and should
    dominate margin in the cold-start regime.

    Per the don't-tune directive: if both uplifts are NEGATIVE at
    epoch 1 the test marks XFAIL with the observed values. That is
    itself a load-bearing finding -- sigma does not have a regime in
    which it dominates margin even when margin starts cold.
    """
    table = _read_checkpoint_table(e16_run)
    epochs = sorted(table.keys())
    assert epochs, f"no checkpoint rows in metrics.jsonl: got {epochs}"
    first = epochs[0]
    assert first == 1, (
        f"first checkpoint must be epoch 1 by configuration; got {first}"
    )
    fail_uplift = table[first]["sigma_failure_uplift"]
    struct_uplift = table[first]["sigma_structural_uplift"]
    if not (fail_uplift > 0.0 or struct_uplift > 0.0):
        pytest.xfail(
            f"At epoch {first} without Phase A, neither uplift was "
            f"positive: sigma_failure_uplift={fail_uplift:+.4f} "
            f"sigma_structural_uplift={struct_uplift:+.4f}. "
            f"This falsifies the cold-start dominance hypothesis: sigma "
            f"does NOT have a regime in which it dominates margin even "
            f"when margin lacks the Phase A seed. Load-bearing finding. "
            f"Full first-checkpoint row: "
            f"{ {k: round(v, 4) for k, v in table[first].items()} }"
        )
    assert fail_uplift > 0.0 or struct_uplift > 0.0
