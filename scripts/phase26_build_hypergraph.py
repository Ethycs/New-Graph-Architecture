"""Phase 26 -- compose the labelled hypergraph on existing PCG-X run artefacts.

Reads each Phase 24 GPT-2 run directory (``runs/E30_phase24_*``) and produces
a ``hypergraph.json`` next to its existing ``control_graph.json`` and
``decision_trace.jsonl``. The output is the labelled hypergraph with:

- canonical KL signature per regime (from each regime's empirical conditional
  over the projection-head argmax distribution observed in decision_trace);
- named coordinates (FSM dominant_current_state from control_graph; no SAE
  features yet since Phase 28 is deferred);
- residual = empty list (no SAE plugged in; identity adapter would put every
  active dim in residual, which is uninformative on its own without a real
  SAE — we leave the field empty so coverage = 1.0 reads honestly as
  "named only");
- feature_delta per edge = empty (depends on SAE adapter, Phase 28);
- Beta(alpha, beta) posterior per edge (lifted from existing
  control_graph.json edges);
- p_lambda per regime (computed from regime support / total).

The script also computes the gap-detection threshold per grammar and reports
how many regimes would merge under KL-canonicalisation. This is the
information-geometric K-choice question from the proposal answered
empirically on the existing artefacts.

The artefacts are bitwise-additive: nothing in the existing run dir is
modified; ``hypergraph.json`` is the only new file.

Usage: ``pixi run python scripts/phase26_build_hypergraph.py``
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from nga.arch.kl_regime_signature import KLRegimeSignature  # noqa: E402
from nga.arch.labelled_hypergraph import LabelledHypergraph  # noqa: E402


def _read_decision_trace_rows(path: Path) -> list[dict]:
    """Read raw decision-trace rows as dicts (no pydantic validation; faster)."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_conditional_per_regime(
    rows: list[dict],
    regime_ids: list[str],
) -> dict[str, np.ndarray]:
    """For each regime, count next-regime transitions to form an empirical conditional.

    The "output" support is the regime alphabet itself; ``p_lambda(r' | r)``
    is the empirical fraction of times that the next observed top1_state
    after landing in ``r`` was ``r'``. This is the natural conditional for
    KL-canonicalisation in a regime graph.
    """
    n = len(regime_ids)
    id_to_idx = {rid: i for i, rid in enumerate(regime_ids)}
    counts = np.zeros((n, n), dtype=float)
    prev_idx: int | None = None
    for row in rows:
        rid = row.get("top1_state")
        if rid not in id_to_idx:
            prev_idx = None
            continue
        idx = id_to_idx[rid]
        if prev_idx is not None:
            counts[prev_idx, idx] += 1.0
        prev_idx = idx
    # Convert to conditionals row-by-row; rows with zero support are uniform.
    cond: dict[str, np.ndarray] = {}
    for i, rid in enumerate(regime_ids):
        row = counts[i]
        s = row.sum()
        if s <= 0:
            cond[rid] = np.full(n, 1.0 / n)
        else:
            cond[rid] = row / s
    return cond


def _process_run(run_dir: Path) -> dict:
    cg_path = run_dir / "control_graph.json"
    dt_path = run_dir / "decision_trace.jsonl"
    if not cg_path.exists():
        return {"skipped": True, "reason": "no control_graph.json"}
    with cg_path.open() as f:
        cg = json.load(f)

    regimes_raw = cg.get("regimes", [])
    edges_raw = cg.get("edges", [])
    regime_ids = [f"regime_{r['regime_id']}" for r in regimes_raw]
    n_total = max(1, int(cg.get("n_total_steps", sum(r["support"] for r in regimes_raw))))

    # Named labels: lift the FSM dominant_current_state per regime.
    named: dict[str, dict[str, str]] = {
        f"regime_{r['regime_id']}": {
            "fsm_dominant_state": str(r.get("dominant_current_state", "?")),
        }
        for r in regimes_raw
    }
    residual: dict[str, list[str]] = {rid: [] for rid in regime_ids}
    support: dict[str, int] = {
        f"regime_{r['regime_id']}": int(r["support"]) for r in regimes_raw
    }
    p_lambda: dict[str, float] = {
        rid: float(support[rid]) / float(n_total) for rid in regime_ids
    }

    # Edges -> hyperedge skeletons with Beta posteriors lifted.
    edges: list[tuple[str, str]] = []
    edge_counts: dict[tuple[str, str], int] = {}
    beta_posteriors: dict[tuple[str, str], tuple[float, float]] = {}
    for e in edges_raw:
        src = f"regime_{e['src']}"
        dst = f"regime_{e['dst']}"
        edges.append((src, dst))
        edge_counts[(src, dst)] = int(e.get("count", 1))
        beta_posteriors[(src, dst)] = (
            float(e.get("confidence_alpha", 1.0)),
            float(e.get("confidence_beta", 1.0)),
        )

    # Canonical signatures: KL distances from per-regime next-regime conditionals.
    signatures: dict[str, np.ndarray] = {}
    suggested_threshold: float | None = None
    kl_cluster_count: int | None = None
    if dt_path.exists():
        rows = _read_decision_trace_rows(dt_path)
        cond = _build_conditional_per_regime(rows, regime_ids)
        signatures = KLRegimeSignature.compute(cond, symmetric=True)
        suggested_threshold = KLRegimeSignature.suggest_threshold(signatures, "gap")
        clusters = KLRegimeSignature.cluster(signatures, threshold=suggested_threshold)
        kl_cluster_count = len(set(clusters.values()))

    # Build the hypergraph.
    hg = LabelledHypergraph.from_pcg_graph(
        regime_ids=regime_ids,
        edges=edges,
        edge_counts=edge_counts,
        canonical_signatures=signatures if signatures else None,
        named_labels=named,
        residual_features=residual,
        p_lambda=p_lambda,
        support_counts=support,
        metadata={
            "source_run_dir": run_dir.name,
            "grammar": cg.get("grammar"),
            "substrate": cg.get("substrate"),
            "n_total_steps": cg.get("n_total_steps"),
            "n_regimes_after_merge": cg.get("n_regimes_after_merge"),
            "kl_suggested_threshold": suggested_threshold,
            "kl_cluster_count": kl_cluster_count,
        },
    )
    # Lift Beta posteriors onto the corresponding Hyperedge objects.
    for he in hg.hyperedges:
        key = (he.src_regime_id, he.dst_regime_id)
        if key in beta_posteriors:
            he.beta_posterior = beta_posteriors[key]

    # Write hypergraph.json next to the existing artefacts.
    out_path = run_dir / "hypergraph.json"
    out_path.write_text(hg.to_json(sort_keys=True, indent=2))

    # Acceptance check: projection back to (V, E) recovers the original input.
    rids_back, edges_back = hg.as_discrete_graph()
    projection_ok = (rids_back == regime_ids) and (edges_back == edges)

    # Coverage check.
    coverage = hg.interpretation_coverage()
    coverage_finite = [v for v in coverage.values() if v == v]  # drop NaNs
    mean_coverage = float(np.mean(coverage_finite)) if coverage_finite else float("nan")

    return {
        "skipped": False,
        "grammar": cg.get("grammar"),
        "n_regimes": len(regime_ids),
        "n_edges": len(edges),
        "n_total_steps": cg.get("n_total_steps"),
        "projection_recovers_input": projection_ok,
        "hypergraph_hash": hg.hash(),
        "kl_suggested_threshold": suggested_threshold,
        "kl_cluster_count": kl_cluster_count,
        "mean_interpretation_coverage": mean_coverage,
        "hypergraph_json_size_kb": out_path.stat().st_size / 1024.0,
    }


def main() -> int:
    targets = sorted(p for p in RUNS_DIR.glob("E30_phase24_*") if p.is_dir())
    if not targets:
        print("No E30_phase24_* run dirs found.", file=sys.stderr)
        return 1

    print("=" * 100)
    print("Phase 26 -- compose labelled hypergraph on existing PCG-X (Phase 24 / GPT-2) runs")
    print("=" * 100)
    summary: dict = {"per_run": {}}

    for run_dir in targets:
        result = _process_run(run_dir)
        summary["per_run"][run_dir.name] = result
        if result["skipped"]:
            print(f"  {run_dir.name:35s} SKIPPED ({result['reason']})")
            continue
        print(
            f"  {run_dir.name:35s} "
            f"R={result['n_regimes']:3d}  E={result['n_edges']:3d}  "
            f"steps={result['n_total_steps']:5d}  "
            f"thr={result['kl_suggested_threshold']:.4f}  "
            f"clusters={result['kl_cluster_count']:2d}  "
            f"cov={result['mean_interpretation_coverage']:.2f}  "
            f"hash={result['hypergraph_hash']}  "
            f"proj_ok={result['projection_recovers_input']}"
        )

    # Aggregate.
    finite = [r for r in summary["per_run"].values() if not r["skipped"]]
    if finite:
        n_unique_clusters = Counter(r["kl_cluster_count"] for r in finite)
        n_regimes_dist = Counter(r["n_regimes"] for r in finite)
        all_projections_ok = all(r["projection_recovers_input"] for r in finite)
        print("-" * 100)
        print(f"  runs processed:        {len(finite)}")
        print(f"  regime-count distribution:   {dict(n_regimes_dist)}")
        print(f"  KL-cluster distribution:     {dict(n_unique_clusters)}")
        print(f"  all projections recover (V,E): {all_projections_ok}")
        summary["aggregate"] = {
            "runs_processed": len(finite),
            "regime_count_distribution": dict(n_regimes_dist),
            "kl_cluster_count_distribution": dict(n_unique_clusters),
            "all_projections_recover_input": all_projections_ok,
        }

    out_summary = RUNS_DIR / "phase26_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print("=" * 100)
    print(f"Wrote per-run hypergraph.json + {out_summary.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
