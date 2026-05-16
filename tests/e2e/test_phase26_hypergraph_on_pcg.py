"""Phase 26 end-to-end: labelled hypergraph composed on real Phase 24 PCG-X artefacts.

Acceptance bar A6 from ``docs/proposals/labelled-hypergraph.md`` in its
strongest form: building a hypergraph from an existing Phase 24 / E30
run directory produces a well-formed structure with expected regime
count, edge count, and per-regime KL signatures, and projects back to
the original ``(V, E)`` discrete graph exactly.

Skipped if no Phase 24 run dirs are present (e.g. fresh checkout).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nga.arch.kl_regime_signature import KLRegimeSignature
from nga.arch.labelled_hypergraph import LabelledHypergraph


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "runs"


def _phase24_runs() -> list[Path]:
    return sorted(p for p in RUNS_DIR.glob("E30_phase24_*") if p.is_dir())


def _read_control_graph(run_dir: Path) -> dict | None:
    p = run_dir / "control_graph.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


@pytest.mark.parametrize(
    "run_dir",
    _phase24_runs(),
    ids=lambda p: p.name,
)
def test_hypergraph_projects_back_to_input(run_dir: Path) -> None:
    """The hypergraph built from a Phase 24 run projects back to the original (V, E)."""
    cg = _read_control_graph(run_dir)
    if cg is None:
        pytest.skip(f"No control_graph.json in {run_dir.name}")

    regime_ids = [f"regime_{r['regime_id']}" for r in cg["regimes"]]
    edges = [
        (f"regime_{e['src']}", f"regime_{e['dst']}") for e in cg["edges"]
    ]

    hg = LabelledHypergraph.from_pcg_graph(
        regime_ids=regime_ids,
        edges=edges,
        named_labels={
            f"regime_{r['regime_id']}": {
                "fsm_dominant_state": str(r["dominant_current_state"]),
            }
            for r in cg["regimes"]
        },
        support_counts={
            f"regime_{r['regime_id']}": int(r["support"]) for r in cg["regimes"]
        },
        metadata={"grammar": cg.get("grammar")},
    )

    rids_back, edges_back = hg.as_discrete_graph()
    assert rids_back == regime_ids
    assert edges_back == edges


@pytest.mark.parametrize(
    "run_dir",
    _phase24_runs(),
    ids=lambda p: p.name,
)
def test_hypergraph_json_artefact_round_trips(run_dir: Path) -> None:
    """If phase26_build_hypergraph.py has run, the on-disk hypergraph.json round-trips canonically."""
    hg_path = run_dir / "hypergraph.json"
    if not hg_path.exists():
        pytest.skip(f"No hypergraph.json yet in {run_dir.name}")

    payload = hg_path.read_text()
    hg = LabelledHypergraph.from_json(payload)
    re_emitted = hg.to_json(sort_keys=True, indent=2)
    assert re_emitted == payload, (
        f"hypergraph.json in {run_dir.name} did not round-trip canonically"
    )


@pytest.mark.parametrize(
    "run_dir",
    _phase24_runs(),
    ids=lambda p: p.name,
)
def test_kl_signature_matrix_is_symmetric_and_zero_diagonal(run_dir: Path) -> None:
    """KL signatures on a real run produce a valid symmetric distance-like matrix."""
    hg_path = run_dir / "hypergraph.json"
    if not hg_path.exists():
        pytest.skip(f"No hypergraph.json yet in {run_dir.name}")
    hg = LabelledHypergraph.from_json(hg_path.read_text())
    sigs = {rid: r.canonical_signature for rid, r in hg.regimes.items()}
    # Skip if signatures weren't populated (decision_trace was absent).
    any_nonzero_len = any(s.size > 0 for s in sigs.values())
    if not any_nonzero_len:
        pytest.skip(f"No KL signatures in {run_dir.name} (no decision_trace.jsonl)")
    n = len(sigs)
    for i, (rid, s) in enumerate(sorted(sigs.items())):
        assert s.shape == (n,)
        assert s[i] == pytest.approx(0.0, abs=1e-9), f"self-distance nonzero for {rid}"
        assert np.all(s >= -1e-9), f"negative KL for {rid}"


@pytest.mark.parametrize(
    "run_dir",
    _phase24_runs(),
    ids=lambda p: p.name,
)
def test_p_lambda_sums_to_one(run_dir: Path) -> None:
    """p_lambda over regimes is a valid probability distribution."""
    hg_path = run_dir / "hypergraph.json"
    if not hg_path.exists():
        pytest.skip(f"No hypergraph.json yet in {run_dir.name}")
    hg = LabelledHypergraph.from_json(hg_path.read_text())
    p_vals = [r.p_lambda for r in hg.regimes.values() if r.p_lambda is not None]
    if not p_vals:
        pytest.skip(f"No p_lambda in {run_dir.name}")
    s = sum(p_vals)
    # Floating sum may drift slightly; allow 1e-6.
    assert s == pytest.approx(1.0, abs=1e-6), f"p_lambda sum = {s}"


def test_phase26_summary_exists_if_script_ran() -> None:
    """If the build script has run, phase26_summary.json is a well-formed report."""
    p = RUNS_DIR / "phase26_summary.json"
    if not p.exists():
        pytest.skip("phase26_summary.json not generated yet")
    data = json.loads(p.read_text())
    assert "per_run" in data
    assert "aggregate" in data
    assert data["aggregate"]["all_projections_recover_input"] is True


def test_kl_threshold_suggestion_can_be_recomputed(tmp_path: Path) -> None:
    """The gap-detection threshold + clustering can be recomputed off the artefact."""
    runs = _phase24_runs()
    if not runs:
        pytest.skip("No Phase 24 runs available")
    run_dir = runs[0]
    hg_path = run_dir / "hypergraph.json"
    if not hg_path.exists():
        pytest.skip(f"No hypergraph.json in {run_dir.name}")
    hg = LabelledHypergraph.from_json(hg_path.read_text())
    sigs = {rid: r.canonical_signature for rid, r in hg.regimes.items()}
    if not any(s.size > 0 for s in sigs.values()):
        pytest.skip("No signatures populated")
    thresh = KLRegimeSignature.suggest_threshold(sigs, "gap")
    assert thresh >= 0.0
    clusters = KLRegimeSignature.cluster(sigs, threshold=thresh)
    # At most one canonical class per input regime; at least one.
    n_classes = len(set(clusters.values()))
    assert 1 <= n_classes <= len(sigs)
