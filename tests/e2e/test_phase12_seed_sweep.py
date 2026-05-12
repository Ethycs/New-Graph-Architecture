"""End-to-end tests for Phase 12 -- multi-seed bootstrap.

These tests READ ``runs/phase12_seed_sweep_summary.json`` written by
``scripts/phase12_seed_sweep.py``. The sweep itself is NOT executed
inside the test (5 seeds * ~30-60s each is too slow for a unit test);
this module just asserts that, when the summary is on disk, the
load-bearing Phase 8/9/11 claims survive across the five seeds.

If the summary file is missing the entire module ``pytest.skip``s --
the sweep is run-on-demand, not part of the standard CI loop.

Bars:

* ``n_seeds == 5``: the sweep ran the canonical five seeds.
* ``mean_accuracy_with_mask > 0.7``: mask uplift claim from Phase 9
  must hold at a meaningful level on average.
* ``std_accuracy_with_mask < 0.15``: variance bounded -- the +48pp
  mask uplift is not a seed-42 fluke.
* ``mean_phase_a_hamming < 0.20``: Phase A FSM recovery (the bar
  inherited from Phase 7/8) holds across seeds.
* ``std_phase_a_hamming < 0.10``: Phase A recovery is consistent
  across seeds, not riding seed-42 luck.
* ``mean_sigma_boundary_ratio > 1.0``: σ-at-operator-boundary
  structural claim from Phase 9/11 survives multi-seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY_PATH = REPO_ROOT / "runs" / "phase12_seed_sweep_summary.json"


def _load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        pytest.skip(
            f"Phase 12 sweep summary not present at {SUMMARY_PATH}; run "
            f"`pixi run -e dev python scripts/phase12_seed_sweep.py` first."
        )
    return json.loads(SUMMARY_PATH.read_text())


def test_summary_file_exists() -> None:
    """The summary JSON must be on disk for any of the other claims to
    be checkable."""
    assert SUMMARY_PATH.exists(), (
        f"Phase 12 summary missing at {SUMMARY_PATH}"
    )


def test_n_seeds_is_five() -> None:
    """The canonical sweep runs five seeds; smaller runs are not
    enough to bound variance."""
    s = _load_summary()
    assert s["n_seeds"] == 5, (
        f"expected n_seeds=5, got {s['n_seeds']}"
    )
    assert len(s["seeds"]) == 5
    assert len(s["per_seed"]) == 5


def test_mask_uplift_robust_across_seeds() -> None:
    """Phase 9 claim: with-mask accuracy is materially higher than
    no-mask. Bar: mean accuracy with mask > 0.7 AND std < 0.15.
    The +48pp mask uplift seen at seed 42 must hold above a
    meaningful floor across seeds."""
    s = _load_summary()
    acc = s["summary"]["accuracy"]
    assert acc["mean"] > 0.7, (
        f"mean accuracy with mask {acc['mean']:.4f} <= 0.7 across seeds; "
        f"the +48pp mask uplift claim does not hold robustly"
    )
    assert acc["std"] < 0.15, (
        f"std accuracy with mask {acc['std']:.4f} >= 0.15; "
        f"variance too large to call the claim robust"
    )


def test_phase_a_hamming_robust_across_seeds() -> None:
    """Phase 7/8 claim: classical Phase A cold-start recovers the FSM
    well below the 0.20 hamming bar. Across seeds we want
    mean < 0.20 AND std < 0.10."""
    s = _load_summary()
    h = s["summary"]["phase_a_hamming_normalised"]
    assert h["mean"] < 0.20, (
        f"mean phase_a_hamming {h['mean']:.4f} >= 0.20; "
        f"Phase A FSM recovery does not survive multi-seed"
    )
    assert h["std"] < 0.10, (
        f"std phase_a_hamming {h['std']:.4f} >= 0.10; "
        f"Phase A recovery is too seed-sensitive"
    )


def test_sigma_boundary_ratio_robust_across_seeds() -> None:
    """Phase 9/11 claim: σ fires more strongly at S{d}_after_operand
    operator boundaries than elsewhere. Bar: mean ratio > 1.0
    across seeds."""
    s = _load_summary()
    r = s["summary"]["sigma_boundary_ratio"]
    assert r["mean"] > 1.0, (
        f"mean sigma_boundary_ratio {r['mean']:.4f} <= 1.0; "
        f"σ-at-operator-boundary structural claim does not survive multi-seed"
    )
