"""End-to-end tests for Phase 15 -- multi-seed bootstrap of E18
(bigger Python: arithmetic + assignment + function calls + function
definitions + return).

These tests READ ``runs/phase15_e18_seed_sweep_summary.json`` written
by ``scripts/phase15_e18_seed_sweep.py``. The sweep itself is NOT
executed inside the test (5 seeds * ~60-90s each is too slow for a
unit test); this module just asserts that, when the summary is on
disk, the load-bearing claims survive across the five seeds.

If the summary file is missing the entire module ``pytest.skip``s --
the sweep is run-on-demand, not part of the standard CI loop.

Bars (Phase 15):

* ``n_seeds == 5``: the sweep ran the canonical five seeds.
* ``mean_mask_accuracy_uplift > 0.20`` AND ``std < 0.20``: the
  ~+48pp mask uplift seen at seed 42 holds above a meaningful floor
  across seeds; richer python_big grammar amplifies mask vs python_expr.
* ``mean_phase_a_hamming < 0.20`` AND ``std < 0.10``: classical
  Phase A FSM recovery (the bar inherited from Phase 7/8/9/14)
  holds across seeds on the 24-vertex python_big FSM.
* ``mean_illegal_transition_rate (mask) < 0.05``: mask zeros
  illegal transitions on bigger Python, robustly.
* ``mean_sigma_boundary_ratio > 1.0``: sigma fires more at
  call-vs-variable / assignment-vs-expression structural-ambiguity
  points than at non-boundary states on average; boundary-ratio
  inverts back from python_expr (0.65) towards ListOps (2.0+).
* ``test_phase15_sigma_uplift_sign_documented``: NO ASSERTION.
  Just record mean +/- std so the cross-grammar story is visible
  in the test log -- let the data speak about whether the seed-42
  -0.042 was within seed variance of zero or genuinely negative.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY_PATH = REPO_ROOT / "runs" / "phase15_e18_seed_sweep_summary.json"


def _load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        pytest.skip(
            f"Phase 15 sweep summary not present at {SUMMARY_PATH}; run "
            f"`pixi run -e dev python scripts/phase15_e18_seed_sweep.py` first."
        )
    return json.loads(SUMMARY_PATH.read_text())


def test_phase15_summary_exists() -> None:
    """The summary JSON must be on disk for any of the other claims to
    be checkable."""
    assert SUMMARY_PATH.exists(), (
        f"Phase 15 summary missing at {SUMMARY_PATH}"
    )


def test_phase15_n_seeds() -> None:
    """The canonical sweep runs five seeds; smaller runs are not
    enough to bound variance."""
    s = _load_summary()
    assert s["n_seeds"] == 5, (
        f"expected n_seeds=5, got {s['n_seeds']}"
    )
    assert len(s["seeds"]) == 5
    assert len(s["per_seed"]) == 5


def test_phase15_mask_uplift_robust() -> None:
    """E18 mask uplift on bigger Python is materially positive and
    bounded in variance. Bar: mean > 0.20 AND std < 0.20."""
    s = _load_summary()
    u = s["summary"]["mask_accuracy_uplift"]
    assert u["mean"] > 0.20, (
        f"mean mask_accuracy_uplift {u['mean']:+.4f} <= 0.20 across seeds; "
        f"the mask-uplift claim does not hold robustly on bigger Python"
    )
    assert u["std"] < 0.20, (
        f"std mask_accuracy_uplift {u['std']:.4f} >= 0.20; "
        f"variance too large to call the claim robust"
    )


def test_phase15_phase_a_hamming_robust() -> None:
    """Phase 7/8/9/14 claim re-checked on bigger Python: classical
    Phase A cold-start recovers the 24-vertex python_big FSM. Bar:
    mean < 0.20 AND std < 0.10."""
    s = _load_summary()
    h = s["summary"]["phase_a_hamming_normalised"]
    assert h["mean"] < 0.20, (
        f"mean phase_a_hamming {h['mean']:.4f} >= 0.20; "
        f"Phase A FSM recovery does not survive multi-seed on bigger Python"
    )
    assert h["std"] < 0.10, (
        f"std phase_a_hamming {h['std']:.4f} >= 0.10; "
        f"Phase A recovery is too seed-sensitive"
    )


def test_phase15_illegal_rate_zero_robust() -> None:
    """Mask zeros illegal transitions on bigger Python, robustly.
    Bar: mean illegal_transition_rate (mask on) < 0.05."""
    s = _load_summary()
    r = s["summary"]["illegal_transition_rate"]
    assert r["mean"] < 0.05, (
        f"mean illegal_transition_rate (mask) {r['mean']:.4f} >= 0.05; "
        f"mask is not zeroing illegal transitions robustly on bigger Python"
    )


def test_phase15_sigma_boundary_ratio_robust() -> None:
    """sigma fires more at structural-ambiguity points than at
    non-boundary states ON AVERAGE on bigger Python. Bar: mean > 1.0.

    On python_expr the boundary ratio inverted to 0.65 because the
    only ambiguity point is a single 2-way state; on python_big the
    grammar adds call-vs-variable choices at every paren depth, so
    the ratio should swing back above 1.0 even if it doesn't reach
    ListOps's 2.0+.
    """
    s = _load_summary()
    r = s["summary"]["sigma_boundary_ratio"]
    assert r["mean"] > 1.0, (
        f"mean sigma_boundary_ratio {r['mean']:.4f} <= 1.0; "
        f"sigma is not firing more at python_big's call-ambiguity points "
        f"than at non-boundary states on average across seeds "
        f"(std {r['std']:.4f}, min {r['min']:.4f}, max {r['max']:.4f})"
    )


def test_phase15_sigma_uplift_sign_documented() -> None:
    """REPORT-only test for the cross-grammar comparison.

    NO assertion on sign. The seed-42 sigma_uplift = -0.042 (DOWN
    from E17's +0.022) is the surprising finding that motivated this
    sweep. The question is whether bigger Python genuinely flipped
    the sign or whether the seed-42 negative value is within seed
    variance of zero. This test just records the multi-seed mean
    +/- std and passes -- the data speaks via the test log.
    """
    s = _load_summary()
    u = s["summary"]["sigma_uplift"]
    mean = u["mean"]
    std = u["std"]
    n = int(u.get("n", 0))
    if mean >= 0.03:
        status = "PASS (>= +0.03)"
    elif mean > 0.0:
        status = "PARTIAL (positive but < +0.03)"
    else:
        status = "FAIL (non-positive; sign-flip vs E17 looks genuine)"
    print(
        f"\n[phase15] sigma_uplift across n={n} seeds on bigger Python: "
        f"{status}. mean = {mean:+.4f}, std = {std:.4f}, "
        f"min = {u.get('min', float('nan')):+.4f}, "
        f"max = {u.get('max', float('nan')):+.4f}. "
        f"Compare E17 (Python expr) mean ~= +0.022. "
        f"Compare ListOps (E14, depth 3) mean ~= -0.098."
    )
    # No assertion on sign. The test passes by construction so the
    # cross-grammar story is recorded without imposing a directional
    # claim that may not be true.
    assert isinstance(mean, float)
    assert isinstance(std, float)
