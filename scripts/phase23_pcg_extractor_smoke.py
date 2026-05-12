#!/usr/bin/env python3
"""Phase 23 -- Predictive Control Graph Extractor MVP smoke test.

Runs E28 on each of the 5 grammars and prints a compact table plus
exemplar regime / edge entries from python_big to give a feel for the
control-graph artefact PCG-X emits.

Run:
    pixi run -e dev python scripts/phase23_pcg_extractor_smoke.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
sys.path.insert(0, str(REPO_ROOT / "src"))

from nga.arch.graph_fsm import GraphFSM  # noqa: E402
from nga.drivers import graph_fsm_spec as graph_fsm_spec_mod  # noqa: E402
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH  # noqa: E402
from nga.exp.e28_pcg_extractor import run_e28  # noqa: E402

SEED = 42
GRAMMARS = ["listops", "python_expr", "python_big", "json", "python_control"]


def main() -> int:
    summary: dict = {"seed": SEED, "grammars": {}}
    print("=" * 110)
    print(
        f"Phase 23 -- Predictive Control Graph Extractor, 5-grammar MVP smoke, seed={SEED}"
    )
    print("=" * 110)
    print(
        f"{'grammar':<18s} {'V':>3s} {'argmax':>7s} {'regimes':>7s}  "
        f"{'fail_mean':>10s} {'ent_mean':>9s} {'margin':>8s} "
        f"{'purity':>7s} {'align_H':>8s} {'tot_s':>7s}"
    )
    print("-" * 110)
    t_start = time.perf_counter()
    for grammar in GRAMMARS:
        dispatch = GRAMMAR_DISPATCH[grammar]
        fsm_yaml = REPO_ROOT / dispatch["fsm_yaml_path"]
        fsm_spec = graph_fsm_spec_mod.load(fsm_yaml)
        fsm = GraphFSM(fsm_spec)
        V = fsm.vertex_count
        run_dir = RUNS_DIR / f"E28_A0_seed{SEED}_{grammar}"
        if run_dir.exists():
            import shutil

            shutil.rmtree(run_dir)
        result = run_e28(
            fsm=fsm,
            run_id=f"E28_A0_seed{SEED}_{grammar}",
            output_dir=run_dir,
            seed=SEED,
            grammar=grammar,
        )
        summary["grammars"][grammar] = {
            "V": V,
            "n_argmax_cells": result.n_argmax_cells,
            "n_regimes_after_merge": result.n_regimes_after_merge,
            "mean_failure_rate": result.mean_failure_rate,
            "mean_entropy": result.mean_entropy,
            "mean_margin_to_tie": result.mean_margin_to_tie,
            "mean_purity_against_current_state": (
                result.mean_purity_against_current_state
            ),
            "aligned_hamming_at_target_V": result.aligned_hamming_at_target_V,
            "encoder_train_accuracy": result.encoder_train_accuracy,
            "projection_train_accuracy": result.projection_train_accuracy,
            "projection_failure_accuracy": result.projection_failure_accuracy,
            "total_wall_clock_seconds": result.total_wall_clock_seconds,
        }
        print(
            f"{grammar:<18s} {V:>3d} {result.n_argmax_cells:>7d} "
            f"{result.n_regimes_after_merge:>7d}  "
            f"{result.mean_failure_rate:>10.4f} "
            f"{result.mean_entropy:>9.4f} "
            f"{result.mean_margin_to_tie:>8.3f} "
            f"{result.mean_purity_against_current_state:>7.4f} "
            f"{result.aligned_hamming_at_target_V:>8.4f} "
            f"{result.total_wall_clock_seconds:>7.2f}"
        )

    elapsed = time.perf_counter() - t_start
    summary["total_elapsed_seconds"] = elapsed
    print("-" * 110)
    print(f"total sweep wall-clock: {elapsed:.2f}s")

    out_path = RUNS_DIR / "phase23_pcg_extractor_smoke_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {out_path}")

    # Exemplar: dump the top-3 regimes from python_big by support.
    py_big_graph = json.loads(
        (RUNS_DIR / f"E28_A0_seed{SEED}_python_big" / "control_graph.json").read_text()
    )
    print(
        "\n--- python_big exemplar (top 3 regimes by support, top 3 edges) ---"
    )
    regimes = sorted(
        py_big_graph["regimes"], key=lambda r: r["support"], reverse=True
    )
    for r in regimes[:3]:
        print(
            f"  Node {r['regime_id']:>2d}  support={r['support']:>4d}  "
            f"failure_rate={r['failure_rate']:.3f}  "
            f"entropy_mean={r['entropy_mean']:.3f}  "
            f"margin={r['mean_margin_to_tie']:.3f}  "
            f"dom_state={r['dominant_current_state']:<28s} "
            f"purity={r['purity_against_current_state']:.3f}"
        )
    edges = sorted(
        py_big_graph["edges"], key=lambda e: e["count"], reverse=True
    )
    print()
    for e in edges[:3]:
        print(
            f"  Edge {e['src']:>2d} -> {e['dst']:>2d}  "
            f"prob={e['probability']:.3f}  "
            f"count={e['count']:>4d}  "
            f"Beta(a={e['confidence_alpha']:.1f}, b={e['confidence_beta']:.1f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
