"""Phase 25 -- Layer-ablation sweep for PCG-X on a frozen GPT-2 substrate.

Where in the encoder does the typed-graph structure live?

Phase 24 harvested at GPT-2 layer 6/12 because mid-layer was a defensible
default. This script answers the harder version: run E30 at multiple
harvest_layer values (embedding output, early, mid, late, final block)
on all 5 grammars and record how regime purity and projection accuracy
evolve through the network.

GPT-2 small: 12 transformer blocks; ``hidden_states`` tuple has 13
entries (index 0 = embedding output; index k ∈ [1, 12] = block k output).
Sweep over {0, 2, 6, 10, 12}: embedding baseline + early + mid + late +
final.

Output: ``runs/phase25_summary.json`` plus a per-layer per-grammar
table printed to stdout.
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
LAYERS = [0, 2, 6, 10, 12]  # embedding, early, mid (Phase 24 default), late, final


def main() -> int:
    summary: dict = {
        "seed": SEED,
        "model_id": "gpt2",
        "layers": LAYERS,
        "grammars": {g: {} for g in GRAMMARS},
    }
    print("=" * 130)
    print(
        f"Phase 25 -- Layer ablation: PCG-X over frozen GPT-2, layers={LAYERS}, seed={SEED}"
    )
    print("=" * 130)

    header_layers = "  ".join(f"L{ly:>2d}" for ly in LAYERS)
    print(f"{'grammar':<16s} {'V':>3s}  {'metric':<12s} {header_layers}")
    print("-" * 130)
    t_start = time.perf_counter()

    for grammar in GRAMMARS:
        dispatch = GRAMMAR_DISPATCH[grammar]
        fsm = GraphFSM(
            graph_fsm_spec_mod.load(REPO_ROOT / dispatch["fsm_yaml_path"])
        )
        per_layer_purity: list[float] = []
        per_layer_proj_acc: list[float] = []
        per_layer_regimes: list[int] = []
        per_layer_wall: list[float] = []
        for ly in LAYERS:
            run_dir = RUNS_DIR / f"E30_phase25_{grammar}_L{ly}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            result = run_e30(
                fsm=fsm,
                run_id=f"E30_phase25_{grammar}_L{ly}",
                output_dir=run_dir,
                seed=SEED,
                grammar=grammar,
                harvest_layer=ly,
            )
            per_layer_purity.append(result.mean_purity_against_current_state)
            per_layer_proj_acc.append(result.projection_next_state_accuracy)
            per_layer_regimes.append(result.n_regimes_after_merge)
            per_layer_wall.append(result.total_wall_clock_seconds)
            summary["grammars"][grammar][f"L{ly}"] = {
                "mean_purity": result.mean_purity_against_current_state,
                "projection_next_state_accuracy": (
                    result.projection_next_state_accuracy
                ),
                "n_regimes_after_merge": result.n_regimes_after_merge,
                "n_argmax_cells": result.n_argmax_cells,
                "wall_clock_s": result.total_wall_clock_seconds,
            }

        # Print this grammar's three rows.
        def _row(label: str, vals: list, fmt: str) -> str:
            cells = "  ".join(fmt.format(v) for v in vals)
            return f"{grammar:<16s} {fsm.vertex_count:>3d}  {label:<12s} {cells}"

        print(_row("purity", per_layer_purity, "{:>5.3f}"))
        print(_row("proj_acc", per_layer_proj_acc, "{:>5.3f}"))
        print(_row("regimes/V", per_layer_regimes, "{:>5d}"))

    print("-" * 130)
    total = time.perf_counter() - t_start
    print(f"total wall-clock: {total:.1f}s")
    summary["total_wall_clock_seconds"] = total

    # Per-layer mean across grammars.
    print()
    print("Per-layer mean across 5 grammars:")
    for label, key in (
        ("mean_purity", "mean_purity"),
        ("proj_next_state_acc", "projection_next_state_accuracy"),
    ):
        cells = []
        for ly in LAYERS:
            vals = [
                summary["grammars"][g][f"L{ly}"][key] for g in GRAMMARS
            ]
            cells.append(f"L{ly:>2d}={sum(vals)/len(vals):.3f}")
        print(f"  {label:<22s} " + "  ".join(cells))
    summary["per_layer_means"] = {
        f"L{ly}": {
            "mean_purity": sum(
                summary["grammars"][g][f"L{ly}"]["mean_purity"] for g in GRAMMARS
            )
            / len(GRAMMARS),
            "projection_next_state_accuracy": sum(
                summary["grammars"][g][f"L{ly}"][
                    "projection_next_state_accuracy"
                ]
                for g in GRAMMARS
            )
            / len(GRAMMARS),
        }
        for ly in LAYERS
    }

    out_path = RUNS_DIR / "phase25_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
