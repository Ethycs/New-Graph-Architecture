"""Phase 24 -- PCG-X on a frozen pretrained GPT-2 substrate, 5-grammar sweep.

The deferred Tier-3 test from the original graph-extraction proposal:
extract regimes from activations of a model that was never trained on
the grammar. GPT-2 is loaded from the DVC-tracked ``~/models/hf`` cache;
harvest is mid-layer (block 6 of 12) of the small (124M) variant.

Reports the standard PCG-X table (purity / failure / entropy / margin /
argmax cells / merged regimes) so this row drops directly into Phase 23
comparisons.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
os.environ["HF_HOME"] = str(Path.home() / "models" / "hf")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(REPO_ROOT / "src"))

from nga.arch.graph_fsm import GraphFSM  # noqa: E402
from nga.drivers import graph_fsm_spec as graph_fsm_spec_mod  # noqa: E402
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH  # noqa: E402
from nga.exp.e30_pcg_extractor_pretrained import run_e30  # noqa: E402

SEED = 42
GRAMMARS = ["listops", "python_expr", "python_big", "json", "python_control"]


def main() -> int:
    summary: dict = {"seed": SEED, "model_id": "gpt2", "grammars": {}}
    print("=" * 115)
    print(
        f"Phase 24 -- PCG-X over frozen pretrained GPT-2 (124M, layer 6/12), 5-grammar sweep, seed={SEED}"
    )
    print("=" * 115)
    print(
        f"{'grammar':<16s} {'V':>3s} {'argmax':>7s} {'regimes':>7s}  "
        f"{'purity':>7s} {'fail':>7s} {'ent':>7s} {'margin':>7s}  "
        f"{'proj_acc':>9s}  {'trunc':>5s}  {'tot_s':>7s}"
    )
    print("-" * 115)
    t_start = time.perf_counter()
    for grammar in GRAMMARS:
        dispatch = GRAMMAR_DISPATCH[grammar]
        fsm = GraphFSM(
            graph_fsm_spec_mod.load(REPO_ROOT / dispatch["fsm_yaml_path"])
        )
        run_dir = RUNS_DIR / f"E30_phase24_{grammar}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        result = run_e30(
            fsm=fsm,
            run_id=f"E30_phase24_{grammar}",
            output_dir=run_dir,
            seed=SEED,
            grammar=grammar,
        )
        summary["grammars"][grammar] = {
            "V": int(fsm.vertex_count),
            "n_total_steps": result.n_total_steps,
            "n_argmax_cells": result.n_argmax_cells,
            "n_regimes_after_merge": result.n_regimes_after_merge,
            "mean_purity_against_current_state": (
                result.mean_purity_against_current_state
            ),
            "mean_failure_rate": result.mean_failure_rate,
            "mean_entropy": result.mean_entropy,
            "mean_margin_to_tie": result.mean_margin_to_tie,
            "aligned_hamming_at_target_V": result.aligned_hamming_at_target_V,
            "projection_next_state_accuracy": (
                result.projection_next_state_accuracy
            ),
            "n_programs_truncated_to_context": (
                result.n_programs_truncated_to_context
            ),
            "harvest_layer": result.harvest_layer,
            "total_wall_clock_seconds": result.total_wall_clock_seconds,
        }
        print(
            f"{grammar:<16s} {result.V_ground_truth:>3d} "
            f"{result.n_argmax_cells:>7d} {result.n_regimes_after_merge:>7d}  "
            f"{result.mean_purity_against_current_state:>7.3f} "
            f"{result.mean_failure_rate:>7.3f} "
            f"{result.mean_entropy:>7.3f} "
            f"{result.mean_margin_to_tie:>7.3f}  "
            f"{result.projection_next_state_accuracy:>9.3f}  "
            f"{result.n_programs_truncated_to_context:>5d}  "
            f"{result.total_wall_clock_seconds:>7.2f}"
        )

    print("-" * 115)
    total = time.perf_counter() - t_start
    print(f"total wall-clock: {total:.1f}s")
    summary["total_wall_clock_seconds"] = total

    out_path = RUNS_DIR / "phase24_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
