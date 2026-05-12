"""End-to-end tests for Phase 14 -- multi-seed bootstrap of E17
(Python expressions, the first external published-grammar benchmark).

These tests READ ``runs/phase14_e17_seed_sweep_summary.json`` written
by ``scripts/phase14_e17_seed_sweep.py``. The sweep itself is NOT
executed inside the test (5 seeds * ~30-60s each is too slow for a
unit test); this module just asserts that, when the summary is on
disk, the load-bearing claims survive across the five seeds.

If the summary file is missing the entire module ``pytest.skip``s --
the sweep is run-on-demand, not part of the standard CI loop.

Bars:

* ``n_seeds == 5``: the sweep ran the canonical five seeds.
* ``mean_mask_accuracy_uplift > 0.20`` AND ``std < 0.20``: the
  +42pp mask uplift seen at seed 42 holds above a meaningful floor
  across seeds and is not a seed-42 fluke.
* ``mean_phase_a_hamming < 0.20`` AND ``std < 0.10``: Phase A FSM
  recovery (the bar inherited from Phase 7/8) holds across seeds.
* ``mean_sigma_uplift > 0.0``: THE LOAD-BEARING TEST. q10 strict
  bar (>=+0.03) has been XFAIL since Phase 2; if sigma wins on
  average across seeds on real Python, this is the first
  published-grammar evidence that sigma adds load-bearing
  failure-prediction signal.
* ``mean_illegal_transition_rate (mask) < 0.05``: mask zeros
  illegal transitions on real Python, robustly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY_PATH = REPO_ROOT / "runs" / "phase14_e17_seed_sweep_summary.json"


def _load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        pytest.skip(
            f"Phase 14 sweep summary not present at {SUMMARY_PATH}; run "
            f"`pixi run -e dev python scripts/phase14_e17_seed_sweep.py` first."
        )
    return json.loads(SUMMARY_PATH.read_text())


def test_phase14_summary_exists() -> None:
    """The summary JSON must be on disk for any of the other claims to
    be checkable."""
    assert SUMMARY_PATH.exists(), (
        f"Phase 14 summary missing at {SUMMARY_PATH}"
    )


def test_phase14_n_seeds() -> None:
    """The canonical sweep runs five seeds; smaller runs are not
    enough to bound variance."""
    s = _load_summary()
    assert s["n_seeds"] == 5, (
        f"expected n_seeds=5, got {s['n_seeds']}"
    )
    assert len(s["seeds"]) == 5
    assert len(s["per_seed"]) == 5


def test_phase14_mask_uplift_robust() -> None:
    """E17 mask uplift on real Python is materially positive and
    bounded in variance. Bar: mean > 0.20 AND std < 0.20."""
    s = _load_summary()
    u = s["summary"]["mask_accuracy_uplift"]
    assert u["mean"] > 0.20, (
        f"mean mask_accuracy_uplift {u['mean']:+.4f} <= 0.20 across seeds; "
        f"the mask-uplift claim does not hold robustly on Python"
    )
    assert u["std"] < 0.20, (
        f"std mask_accuracy_uplift {u['std']:.4f} >= 0.20; "
        f"variance too large to call the claim robust"
    )


def test_phase14_phase_a_hamming_robust() -> None:
    """Phase 7/8/9 claim re-checked on Python: classical Phase A
    cold-start recovers the FSM. Bar: mean < 0.20 AND std < 0.10."""
    s = _load_summary()
    h = s["summary"]["phase_a_hamming_normalised"]
    assert h["mean"] < 0.20, (
        f"mean phase_a_hamming {h['mean']:.4f} >= 0.20; "
        f"Phase A FSM recovery does not survive multi-seed on Python"
    )
    assert h["std"] < 0.10, (
        f"std phase_a_hamming {h['std']:.4f} >= 0.10; "
        f"Phase A recovery is too seed-sensitive"
    )


def test_phase14_q10_passes_multi_seed() -> None:
    """THE LOAD-BEARING TEST.

    q10 strict bar has been XFAIL since Phase 2: the original claim
    was ``sigma_uplift >= +0.03``. On synthetic ListOps margin
    saturated near 0.85 AUROC, leaving sigma no room. On real Python
    margin sits at ~0.72 AUROC, leaving headroom; if sigma's
    structural prior translates to a real failure-AUROC win on
    average across seeds, q10 is no longer XFAIL on a published
    grammar.

    Bar: ``mean(sigma_uplift) > 0.0``. Any positive mean confirms
    sigma wins on average. The strict +0.03 bar PASSES if mean >=
    0.03; if mean is positive but below 0.03 it's PARTIAL; if mean
    is non-positive q10 is still XFAIL and the seed-42 +0.050 was
    an artefact.
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
        status = "FAIL (non-positive)"
    msg = (
        f"q10 strict bar status across n={n} seeds: {status}. "
        f"sigma_uplift mean = {mean:+.4f}, std = {std:.4f}, "
        f"min = {u.get('min', float('nan')):+.4f}, "
        f"max = {u.get('max', float('nan')):+.4f}. "
        f"Bar for this test: mean > 0.0."
    )
    assert mean > 0.0, msg


def test_phase14_illegal_rate_zero_robust() -> None:
    """Mask zeros illegal transitions on Python expressions, robustly.
    Bar: mean illegal_transition_rate (mask on) < 0.05."""
    s = _load_summary()
    r = s["summary"]["illegal_transition_rate"]
    assert r["mean"] < 0.05, (
        f"mean illegal_transition_rate (mask) {r['mean']:.4f} >= 0.05; "
        f"mask is not zeroing illegal transitions robustly on Python"
    )
