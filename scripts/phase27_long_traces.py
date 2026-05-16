"""Phase 27 Step 2 -- 10x-longer PCG-X traces to disambiguate (A) PS-in-burn-in vs (B) finite-mode saturation.

The Phase 27 Step 1.5 visualization showed M(t)/log(t+1) still drifting in
the tail at trajectory lengths 144-2066. Two readings are consistent:
  (A) Patterson-Sullivan in burn-in; asymptote exists at large t.
  (B) Finite-mode saturation; M plateaus, r(t) decays to 0; no asymptote.

A 10x trace-length extension visually distinguishes them.

Runs E30 with n_programs = 800 per grammar (default is 80) and writes
artefacts to runs/E30_phase27_long_{grammar}/.
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
N_PROGRAMS = 800   # 10x the default of 80


def main() -> int:
    summary: dict = {"seed": SEED, "n_programs": N_PROGRAMS, "grammars": {}}
    print("=" * 100)
    print(
        f"Phase 27 Step 2 -- 10x-longer PCG-X traces on GPT-2 substrate (n_programs={N_PROGRAMS} per grammar)"
    )
    print("=" * 100)
    t_start = time.perf_counter()
    for grammar in GRAMMARS:
        dispatch = GRAMMAR_DISPATCH[grammar]
        fsm = GraphFSM(
            graph_fsm_spec_mod.load(REPO_ROOT / dispatch["fsm_yaml_path"])
        )
        run_dir = RUNS_DIR / f"E30_phase27_long_{grammar}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        result = run_e30(
            fsm=fsm,
            run_id=f"E30_phase27_long_{grammar}",
            output_dir=run_dir,
            seed=SEED,
            grammar=grammar,
            n_programs=N_PROGRAMS,
        )
        summary["grammars"][grammar] = {
            "n_total_steps": result.n_total_steps,
            "n_argmax_cells": result.n_argmax_cells,
            "n_regimes_after_merge": result.n_regimes_after_merge,
            "mean_purity_against_current_state": (
                result.mean_purity_against_current_state
            ),
            "harvest_layer": result.harvest_layer,
            "total_wall_clock_seconds": result.total_wall_clock_seconds,
        }
        print(
            f"  {grammar:18s} "
            f"steps={result.n_total_steps:6d}  "
            f"R={result.n_regimes_after_merge:3d}  "
            f"purity={result.mean_purity_against_current_state:.3f}  "
            f"wall={result.total_wall_clock_seconds:.1f}s"
        )

    total = time.perf_counter() - t_start
    print("-" * 100)
    print(f"  total wall-clock: {total:.1f}s")
    summary["total_wall_clock_seconds"] = total
    out = RUNS_DIR / "phase27_long_traces_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
