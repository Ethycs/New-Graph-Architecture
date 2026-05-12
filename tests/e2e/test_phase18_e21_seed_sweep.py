"""End-to-end tests for Phase 18 -- multi-seed bootstrap of E21
(control flow Python: arithmetic + assignment + single-line if /
else / while), Track 2 of Phase 18.

These tests READ ``runs/phase18_e21_seed_sweep_summary.json`` written
by ``scripts/phase18_e21_seed_sweep.py``. The sweep itself is NOT
executed inside the test (5 seeds * ~1-2s per Phase B + Phase A + eval
is still slow enough to dominate a unit test); this module just
asserts that, when the summary is on disk, the load-bearing claims --
including Phase 18's new compute-efficiency observables -- survive
across the five seeds.

If the summary file is missing the entire module ``pytest.skip``s --
the sweep is run-on-demand, not part of the standard CI loop.

Bars (Phase 18 / Track 2):

* ``n_seeds == 5``: the canonical sweep ran the canonical five seeds.
* ``mean_mask_accuracy_uplift > 0.20`` AND ``std < 0.20``: the
  ~+46pp mask uplift seen at seed 42 holds above a meaningful floor
  across seeds; control flow Python keeps mask load-bearing.
* ``mean_phase_a_hamming < 0.20`` AND ``std < 0.10``: classical
  Phase A FSM recovery (the bar inherited from Phase 7/8/9/14/15/16)
  holds across seeds on the 37-vertex python_control FSM (the
  largest grammar yet).
* ``mean_illegal_transition_rate (mask) < 0.05``: mask zeros
  illegal transitions on control flow Python, robustly.
* ``mean_sigma_structural_uplift > 0``: the architectural
  prediction was that control flow's branching ambiguity
  (``S0_after_if_body``: next token is either ``else`` or a new
  statement, no stack disambiguation) shifts σ_structural_uplift
  POSITIVE on average. Single-seed showed +0.0255 (UP from python_big
  +0.0184); this test asserts the positive sign survives multi-seed.
* ``mean_sigma_boundary_ratio > 1.0``: σ fires more at branching
  points than at non-boundary states on average.
* All five compute-efficiency metrics are present in the summary AND
  total wall-clock per seed is reasonable (< 60s) -- the new Phase 18
  efficiency baseline is measurable and stable enough to publish.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY_PATH = REPO_ROOT / "runs" / "phase18_e21_seed_sweep_summary.json"


def _load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        pytest.skip(
            f"Phase 18 sweep summary not present at {SUMMARY_PATH}; run "
            f"`pixi run -e dev python scripts/phase18_e21_seed_sweep.py` first."
        )
    return json.loads(SUMMARY_PATH.read_text())


def test_phase18_summary_exists() -> None:
    """The summary JSON must be on disk for any of the other claims to
    be checkable."""
    assert SUMMARY_PATH.exists(), (
        f"Phase 18 summary missing at {SUMMARY_PATH}"
    )


def test_phase18_n_seeds() -> None:
    """The canonical sweep runs five seeds; smaller runs are not
    enough to bound variance."""
    s = _load_summary()
    assert s["n_seeds"] == 5, (
        f"expected n_seeds=5, got {s['n_seeds']}"
    )
    assert len(s["seeds"]) == 5
    assert len(s["per_seed"]) == 5


def test_phase18_mask_uplift_robust() -> None:
    """E21 mask uplift on control flow Python is materially positive
    and bounded in variance. Bar: mean > 0.20 AND std < 0.20."""
    s = _load_summary()
    u = s["summary"]["mask_accuracy_uplift"]
    assert u["mean"] > 0.20, (
        f"mean mask_accuracy_uplift {u['mean']:+.4f} <= 0.20 across seeds; "
        f"the mask-uplift claim does not hold robustly on control flow Python"
    )
    assert u["std"] < 0.20, (
        f"std mask_accuracy_uplift {u['std']:.4f} >= 0.20; "
        f"variance too large to call the claim robust"
    )


def test_phase18_phase_a_hamming_robust() -> None:
    """Phase 7/8/9/14/15/16 claim re-checked on control flow Python:
    classical Phase A cold-start recovers the 37-vertex python_control
    FSM. Bar: mean < 0.20 AND std < 0.10."""
    s = _load_summary()
    h = s["summary"]["phase_a_hamming_normalised"]
    assert h["mean"] < 0.20, (
        f"mean phase_a_hamming {h['mean']:.4f} >= 0.20; "
        f"Phase A FSM recovery does not survive multi-seed on control "
        f"flow Python"
    )
    assert h["std"] < 0.10, (
        f"std phase_a_hamming {h['std']:.4f} >= 0.10; "
        f"Phase A recovery is too seed-sensitive"
    )


def test_phase18_illegal_rate_zero_robust() -> None:
    """Mask zeros illegal transitions on control flow Python,
    robustly. Bar: mean illegal_transition_rate (mask on) < 0.05."""
    s = _load_summary()
    r = s["summary"]["illegal_transition_rate"]
    assert r["mean"] < 0.05, (
        f"mean illegal_transition_rate (mask) {r['mean']:.4f} >= 0.05; "
        f"mask is not zeroing illegal transitions robustly on control "
        f"flow Python"
    )


def test_phase18_sigma_structural_uplift_grows_with_control_flow() -> None:
    """Architectural prediction: control flow's branching site
    (``S0_after_if_body`` -- next token is either ``else`` or a new
    statement, no stack/depth disambiguation) shifts σ_structural_uplift
    MORE positive than python_big's call-vs-variable did. The bar is
    just MEAN > 0 -- the prediction is directional, not magnitude.

    Cross-grammar context (n=5 multi-seed where available):
    ListOps -0.088, python_expr +0.059, python_big -0.012,
    JSON +0.067, control flow python (this) -- expected positive.
    """
    s = _load_summary()
    su = s["summary"]["sigma_structural_uplift"]
    assert su["mean"] > 0.0, (
        f"mean sigma_structural_uplift {su['mean']:+.4f} <= 0 across seeds; "
        f"the architectural prediction (control flow shifts σ structural "
        f"uplift positive) did NOT survive multi-seed "
        f"(std {su['std']:.4f}, min {su['min']:+.4f}, max {su['max']:+.4f})"
    )


def test_phase18_sigma_boundary_ratio_robust() -> None:
    """σ fires more at structural-ambiguity points than at
    non-boundary states ON AVERAGE on control flow Python. Bar: mean > 1.0.

    Control flow adds a 2-way branch at S0_after_if_body (either
    ``else`` or a new statement), so the boundary ratio should stay
    above 1.0. python_big multi-seed showed 1.08; control flow's extra
    branching site should push it higher.
    """
    s = _load_summary()
    r = s["summary"]["sigma_boundary_ratio"]
    assert r["mean"] > 1.0, (
        f"mean sigma_boundary_ratio {r['mean']:.4f} <= 1.0; "
        f"σ is not firing more at control flow Python's branching "
        f"points than at non-boundary states on average across seeds "
        f"(std {r['std']:.4f}, min {r['min']:.4f}, max {r['max']:.4f})"
    )


def test_phase18_efficiency_metrics_recorded() -> None:
    """Phase 18's new compute-efficiency observables (introduced in
    Track 1 / E20 and baked into E21) are all present and reasonable
    across all five seeds.

    The efficiency tracking is the publication-grade addition of
    Phase 18: TPN runs at hundreds of samples/sec on CPU with sub-100MB
    memory footprint, even on the largest grammar yet (37-vertex
    python_control FSM).
    """
    s = _load_summary()
    summary = s["summary"]
    for k in (
        "phase_a_wall_clock_seconds",
        "phase_b_wall_clock_seconds",
        "total_wall_clock_seconds",
        "inference_throughput_samples_per_sec",
        "peak_memory_kb",
    ):
        assert k in summary, f"missing efficiency metric in summary: {k!r}"
        assert summary[k].get("n", 0) == 5, (
            f"efficiency metric {k} has n={summary[k].get('n')} (expected 5)"
        )
    # Total wall-clock per seed should be < 60s; control flow E21 ran
    # in ~2s/seed in the single-seed validation, so 60s is a comfortable
    # ceiling that catches any pathological regression while leaving
    # plenty of room for shared-host noise.
    total = summary["total_wall_clock_seconds"]
    assert total["max"] < 60.0, (
        f"max total_wall_clock_seconds {total['max']:.2f} >= 60s; "
        f"E21 should run in seconds per seed on the 37-vertex python_control"
    )
    # Throughput strictly positive on every seed.
    thru = summary["inference_throughput_samples_per_sec"]
    assert thru["min"] > 0.0, (
        f"min inference_throughput_samples_per_sec {thru['min']:.2f} <= 0; "
        f"every seed should record positive throughput"
    )
