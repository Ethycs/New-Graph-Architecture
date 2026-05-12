"""End-to-end tests for Phase 13 -- ablation matrix.

These tests READ ``runs/phase13_ablation_summary.json`` written by
``scripts/phase13_ablation_matrix.py``. The sweep itself is NOT
executed inside the test (10 runs * ~30s each is too slow for a unit
test); this module just asserts the sweep ran across enough ablations
and produced the expected structure.

If the summary file is missing the entire module ``pytest.skip``s --
the sweep is run-on-demand, not part of the standard CI loop.

Bars:

* ``n_ablations >= 8``: the matrix covers most of the A0..A9 tuples
  successfully (some ablations may legitimately crash if a required
  atom is disabled; we tolerate up to 2 failures).
* A0 must be present in ``per_ablation_metrics`` -- without the
  baseline the deltas are meaningless.
* ``delta_from_A0`` has entries for at least 3 non-A0 ablations.
* ``errors`` is a dict (possibly empty); failed ablations are recorded
  as data, not silently dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY_PATH = REPO_ROOT / "runs" / "phase13_ablation_summary.json"


def _load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        pytest.skip(
            f"Phase 13 ablation summary not present at {SUMMARY_PATH}; run "
            f"`pixi run -e dev python scripts/phase13_ablation_matrix.py` first."
        )
    return json.loads(SUMMARY_PATH.read_text())


def test_summary_file_exists() -> None:
    """The summary JSON must be on disk for any of the other claims to
    be checkable."""
    assert SUMMARY_PATH.exists(), (
        f"Phase 13 summary missing at {SUMMARY_PATH}"
    )


def test_n_ablations_at_least_eight() -> None:
    """The matrix covers A0..A9 (10 tuples). Tolerate up to 2 crashes:
    some ablations may legitimately fail at runtime if a required atom
    is disabled, but a useful matrix needs at least 8 successful runs."""
    s = _load_summary()
    succeeded = len(s["per_ablation_metrics"])
    assert succeeded >= 8, (
        f"only {succeeded} ablations succeeded; need at least 8 of 10 "
        f"for a useful matrix. errors={list(s.get('errors', {}).keys())}"
    )


def test_a0_baseline_present() -> None:
    """A0 is the baseline; without it the delta-from-A0 table is
    meaningless."""
    s = _load_summary()
    assert "A0" in s["per_ablation_metrics"], (
        "A0 (full system) baseline missing from per_ablation_metrics; "
        "the delta-from-A0 table cannot be computed without it"
    )
    a0 = s["per_ablation_metrics"]["A0"]
    assert "accuracy" in a0, (
        "A0 baseline does not contain 'accuracy' metric; "
        "summary structure is malformed"
    )


def test_delta_from_a0_has_entries() -> None:
    """At least 3 non-A0 ablations must have delta-from-A0 entries.
    This is a structural check: fewer than 3 means the matrix produced
    almost no comparison data."""
    s = _load_summary()
    deltas = s["delta_from_A0"]
    assert isinstance(deltas, dict), (
        f"delta_from_A0 should be a dict, got {type(deltas)}"
    )
    assert len(deltas) >= 3, (
        f"delta_from_A0 has only {len(deltas)} entries; need >=3 "
        f"non-A0 ablations to have any cross-ablation signal"
    )


def test_errors_is_dict() -> None:
    """``errors`` must be a dict (possibly empty). Crashed ablations
    are recorded as data, not silently dropped."""
    s = _load_summary()
    errors = s.get("errors")
    assert isinstance(errors, dict), (
        f"errors must be a dict (possibly empty), got {type(errors)}"
    )
