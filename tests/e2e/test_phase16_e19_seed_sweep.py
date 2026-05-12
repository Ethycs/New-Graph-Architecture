"""End-to-end tests for Phase 16 -- multi-seed bootstrap of E19
(JSON external benchmark: 26-vertex FSM, 99 edges, json.loads-validated,
7 value-position ambiguity sites with 7+ legal continuations each).

These tests READ ``runs/phase16_e19_seed_sweep_summary.json`` written
by ``scripts/phase16_e19_seed_sweep.py``. The sweep itself is NOT
executed inside the test (5 seeds * ~60-90s each is too slow for a
unit test); this module just asserts that, when the summary is on
disk, the load-bearing claims survive across the five seeds.

If the summary file is missing the entire module ``pytest.skip``s --
the sweep is run-on-demand, not part of the standard CI loop.

Bars (Phase 16):

* ``n_seeds == 5``: the sweep ran the canonical five seeds.
* ``mean_mask_accuracy_uplift > 0.20`` AND ``std < 0.20``: the
  ~+48pp mask uplift seen at seed 42 holds above a meaningful floor
  across seeds; JSON keeps the "mask_uplift > +0.40 on every external
  grammar" claim alive.
* ``mean_phase_a_hamming < 0.20`` AND ``std < 0.10``: classical
  Phase A FSM recovery (the bar inherited from Phase 7/8/9/14/15)
  holds across seeds on the 26-vertex JSON FSM.
* ``mean_illegal_transition_rate (mask) < 0.05``: mask zeros
  illegal transitions on JSON, robustly.
* ``mean_sigma_boundary_ratio > 1.0``: sigma fires more at JSON's
  value-position ambiguity points than at non-boundary states on
  average.
* ``test_phase16_sigma_structural_uplift_documented``: NO
  ASSERTION. Just record mean +/- std so the cross-grammar story
  is visible in the test log -- the seed-42 +0.114 was the largest
  structural uplift across all four external benchmarks; let the
  data speak about whether the mean stays positive.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY_PATH = REPO_ROOT / "runs" / "phase16_e19_seed_sweep_summary.json"


def _load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        pytest.skip(
            f"Phase 16 sweep summary not present at {SUMMARY_PATH}; run "
            f"`pixi run -e dev python scripts/phase16_e19_seed_sweep.py` first."
        )
    return json.loads(SUMMARY_PATH.read_text())


def test_phase16_summary_exists() -> None:
    """The summary JSON must be on disk for any of the other claims to
    be checkable."""
    assert SUMMARY_PATH.exists(), (
        f"Phase 16 summary missing at {SUMMARY_PATH}"
    )


def test_phase16_n_seeds() -> None:
    """The canonical sweep runs five seeds; smaller runs are not
    enough to bound variance."""
    s = _load_summary()
    assert s["n_seeds"] == 5, (
        f"expected n_seeds=5, got {s['n_seeds']}"
    )
    assert len(s["seeds"]) == 5
    assert len(s["per_seed"]) == 5


def test_phase16_mask_uplift_robust() -> None:
    """E19 mask uplift on JSON is materially positive and bounded
    in variance. Bar: mean > 0.20 AND std < 0.20."""
    s = _load_summary()
    u = s["summary"]["mask_accuracy_uplift"]
    assert u["mean"] > 0.20, (
        f"mean mask_accuracy_uplift {u['mean']:+.4f} <= 0.20 across seeds; "
        f"the mask-uplift claim does not hold robustly on JSON"
    )
    assert u["std"] < 0.20, (
        f"std mask_accuracy_uplift {u['std']:.4f} >= 0.20; "
        f"variance too large to call the claim robust"
    )


def test_phase16_phase_a_hamming_robust() -> None:
    """Phase 7/8/9/14/15 claim re-checked on JSON: classical Phase A
    cold-start recovers the 26-vertex JSON FSM. Bar: mean < 0.20 AND
    std < 0.10."""
    s = _load_summary()
    h = s["summary"]["phase_a_hamming_normalised"]
    assert h["mean"] < 0.20, (
        f"mean phase_a_hamming {h['mean']:.4f} >= 0.20; "
        f"Phase A FSM recovery does not survive multi-seed on JSON"
    )
    assert h["std"] < 0.10, (
        f"std phase_a_hamming {h['std']:.4f} >= 0.10; "
        f"Phase A recovery is too seed-sensitive"
    )


def test_phase16_illegal_rate_zero_robust() -> None:
    """Mask zeros illegal transitions on JSON, robustly.
    Bar: mean illegal_transition_rate (mask on) < 0.05."""
    s = _load_summary()
    r = s["summary"]["illegal_transition_rate"]
    assert r["mean"] < 0.05, (
        f"mean illegal_transition_rate (mask) {r['mean']:.4f} >= 0.05; "
        f"mask is not zeroing illegal transitions robustly on JSON"
    )


def test_phase16_sigma_boundary_ratio_robust() -> None:
    """sigma fires more at value-position ambiguity points than at
    non-boundary states ON AVERAGE on JSON. Bar: mean > 1.0.

    JSON's value-position points each admit 7+ legal continuations
    (string, number, true, false, null, [, {); the prior prediction
    is that sigma localises here. Seed-42 ratio was 1.43 -- between
    python_big's 1.08 and ListOps's 2.10.
    """
    s = _load_summary()
    r = s["summary"]["sigma_boundary_ratio"]
    assert r["mean"] > 1.0, (
        f"mean sigma_boundary_ratio {r['mean']:.4f} <= 1.0; "
        f"sigma is not firing more at JSON's value-position ambiguity "
        f"points than at non-boundary states on average across seeds "
        f"(std {r['std']:.4f}, min {r['min']:.4f}, max {r['max']:.4f})"
    )


def test_phase16_sigma_structural_uplift_documented() -> None:
    """REPORT-only test for the cross-grammar comparison.

    NO assertion on sign. The seed-42 sigma_structural_uplift = +0.114
    was the LARGEST positive structural uplift across all four
    external benchmarks (ListOps -0.088; python_expr +0.096/+0.059;
    python_big +0.018/-0.012). The question is whether JSON's
    value-position ambiguity (multi-way and uniformly distributed
    along the trajectory) produces a robustly large structural uplift,
    or whether the seed-42 +0.114 was within seed variance. This
    test just records the multi-seed mean +/- std and passes -- the
    data speaks via the test log.
    """
    s = _load_summary()
    u = s["summary"]["sigma_structural_uplift"]
    mean = u["mean"]
    std = u["std"]
    n = int(u.get("n", 0))
    if mean >= 0.10:
        status = "STRONG POSITIVE (>= +0.10; largest across grammars)"
    elif mean >= 0.05:
        status = "MODERATE POSITIVE (>= +0.05)"
    elif mean > 0.0:
        status = "WEAK POSITIVE (positive but < +0.05)"
    else:
        status = "NON-POSITIVE (seed-42 +0.114 did not survive multi-seed)"
    print(
        f"\n[phase16] sigma_structural_uplift across n={n} seeds on JSON: "
        f"{status}. mean = {mean:+.4f}, std = {std:.4f}, "
        f"min = {u.get('min', float('nan')):+.4f}, "
        f"max = {u.get('max', float('nan')):+.4f}. "
        f"Cross-grammar reference: ListOps -0.088, "
        f"python_expr +0.059 (5-seed), python_big -0.012 (5-seed)."
    )
    # No assertion on sign or magnitude. The test passes by
    # construction so the cross-grammar story is recorded without
    # imposing a directional claim that may not be true.
    assert isinstance(mean, float)
    assert isinstance(std, float)
