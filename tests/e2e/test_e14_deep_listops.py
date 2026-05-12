"""End-to-end tests for E14 at deeper grammar (Phase 11 -- max_depth = 6).

These tests READ from ``runs/E14_A0_seed42_deep/metrics.jsonl`` directly:
the deep run is performed once outside the test (the long torch training
is not re-executed by the test fixture). The whole module is gated by
``pytest.importorskip("torch")`` so a torch-less CI continues to run the
rest of the suite.

The bars below are the depth-6 survival bars. They are LOOSER than
depth-3 (Phase 10) on purpose -- the question is whether the
architectural claims survive scale, not whether scale-invariant numbers
fall out. Specifically:

* ``mask_accuracy_uplift > 0.20`` (depth-3 measured +0.484; depth-6
  ought to land near there but a +0.20 floor absorbs scale variation).
* ``phase_a_hamming_normalised < 0.20`` (depth-3 measured 0.0248;
  depth-6 ought to stay well under 0.20 even if the larger transition
  matrix shifts the normalised distance up).
* ``illegal_transition_rate < 0.10`` (depth-3 measured 0.0; the mask
  must continue to zero out illegal next-state predictions).
* ``sigma_boundary_ratio > 1.0`` (depth-3 measured 2.10x; loosened to
  >1x at depth 6 to absorb scale variation in the boundary ratio).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEEP_RUN_DIR = REPO_ROOT / "runs" / "E14_A0_seed42_deep"


def _read_metrics(d: Path) -> dict[str, float]:
    return {
        json.loads(line)["metric_name"]: json.loads(line)["value"]
        for line in (d / "metrics.jsonl").read_text().splitlines()
        if line
    }


@pytest.fixture(scope="module")
def deep_metrics() -> dict[str, float]:
    if not (DEEP_RUN_DIR / "metrics.jsonl").exists():
        pytest.skip(
            "runs/E14_A0_seed42_deep/metrics.jsonl is missing; run E14 on "
            "tests/fixtures/configs/e14_listops_deep_minimal.yaml first."
        )
    return _read_metrics(DEEP_RUN_DIR)


def test_e14_deep_runs_end_to_end(deep_metrics):
    """The four standard run artefacts plus decision_trace.jsonl exist,
    and the headline E14 metrics are populated."""
    for name in (
        "metrics.jsonl",
        "results.jsonl",
        "scores.jsonl",
        "decision_trace.jsonl",
    ):
        assert (DEEP_RUN_DIR / name).exists(), f"missing artefact: {name}"

    for required in (
        "accuracy",
        "accuracy_no_mask",
        "mask_accuracy_uplift",
        "illegal_transition_rate",
        "phase_a_hamming_normalised",
        "phase_b_hamming_normalised",
        "sigma_auroc",
        "margin_auroc",
        "mean_sigma_at_operator_boundary",
        "mean_sigma_at_non_boundary",
        "sigma_boundary_ratio",
        "n_torch_trainable_params",
        "final_loss",
        "loss_decrease_first_to_last_epoch",
    ):
        assert required in deep_metrics, f"missing metric {required!r}"

    # Reasonable headline accuracy at depth 6 -- substrate must still
    # train, not collapse to chance under the larger state space.
    assert 0.5 <= deep_metrics["accuracy"] <= 1.0, (
        f"accuracy out of plausible range at depth 6: "
        f"{deep_metrics['accuracy']:.4f}"
    )


def test_e14_deep_phase_a_recovers_fsm(deep_metrics):
    """Phase A's classical forward-backward + Beta M-step still converges
    on the FSM legality matrix at depth 6 (20 vertices x 16 tokens)."""
    h = deep_metrics["phase_a_hamming_normalised"]
    assert h < 0.20, (
        f"Phase-A classical cold start did not recover the depth-6 "
        f"ListOps FSM within 20% hamming distance under the torch "
        f"substrate; got {h:.4f}"
    )


def test_e14_deep_mask_uplift_holds(deep_metrics):
    """Mask uplift survives at depth 6 (relaxed from depth-3's +0.48 to
    >+0.20 to absorb scale variation; the bar is "the mask still does
    most of the work", not the exact magnitude)."""
    uplift = deep_metrics["mask_accuracy_uplift"]
    assert uplift > 0.20, (
        f"mask_accuracy_uplift {uplift:+.4f} below the depth-6 survival "
        f"bar of +0.20; depth-3 baseline measured +0.4842"
    )


def test_e14_deep_illegal_rate_low(deep_metrics):
    """Mask still zeros out illegal next-state predictions at depth 6
    (depth-3 measured 0.0; the bar is <0.10 to absorb deeper-grammar
    edge-case noise)."""
    rate = deep_metrics["illegal_transition_rate"]
    assert rate < 0.10, (
        f"illegal_transition_rate {rate:.4f} above the bar of 0.10 at "
        f"depth 6; the mask must zero illegal transitions"
    )


def test_e14_deep_sigma_boundary_ratio_survives(deep_metrics):
    """sigma fires at least 1x more strongly at S{d}_after_operand
    boundaries than at non-boundary states (depth-3 measured 2.10x;
    loosened to >1x at depth 6 because absolute sigma magnitudes vary
    with state-space size). The architectural claim is "sigma is a
    structural property of grammar ambiguity"; the ratio must remain
    above 1 even at depth 6."""
    ratio = deep_metrics["sigma_boundary_ratio"]
    assert ratio > 1.0, (
        f"sigma_boundary_ratio {ratio:.4f} below the depth-6 survival "
        f"bar of 1.0; depth-3 baseline measured 2.10x. mean_sigma at "
        f"boundary "
        f"{deep_metrics['mean_sigma_at_operator_boundary']:.4f} vs "
        f"non-boundary "
        f"{deep_metrics['mean_sigma_at_non_boundary']:.4f}."
    )
