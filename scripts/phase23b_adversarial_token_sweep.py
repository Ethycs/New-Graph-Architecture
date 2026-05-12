#!/usr/bin/env python3
"""Phase 23b -- PCG-X with adversarial token head.

Runs the PCG-X MVP on each of the 5 grammars twice -- once with
``adversarial_token_weight=0.0`` (the Phase 23 baseline) and once with
a non-zero weight that turns on the gradient-reversed token-prediction
head. The decisive comparison is whether stripping nuisance content
from the projection ``z`` (a) shifts the number of argmax cells, (b)
raises per-regime purity against current_state (the FSM partition we
care about), or (c) sharpens the failure / entropy heads.

Writes ``runs/phase23b_adversarial_token_sweep_summary.json`` and
prints an adv-off / adv-on comparison table.

Run:
    pixi run -e dev python scripts/phase23b_adversarial_token_sweep.py
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
WEIGHTS = (0.0, 0.5, 1.0)  # 0.0 = Phase 23 baseline


def main() -> int:
    summary: dict = {"seed": SEED, "weights": list(WEIGHTS), "grammars": {}}
    print("=" * 122)
    print(
        "Phase 23b -- PCG-X + adversarial token head sweep "
        f"(seed={SEED}; weights {WEIGHTS})"
    )
    print("=" * 122)
    print(
        f"{'grammar':<18s} {'V':>3s} {'adv_w':>6s} {'argmax':>7s} {'regimes':>7s}  "
        f"{'purity':>8s} {'fail':>8s} {'entropy':>8s} "
        f"{'margin':>8s} {'tok_acc':>8s} {'align_H':>8s} {'tot_s':>7s}"
    )
    print("-" * 122)
    t_start = time.perf_counter()
    for grammar in GRAMMARS:
        dispatch = GRAMMAR_DISPATCH[grammar]
        fsm_yaml = REPO_ROOT / dispatch["fsm_yaml_path"]
        fsm_spec = graph_fsm_spec_mod.load(fsm_yaml)
        fsm = GraphFSM(fsm_spec)
        V = fsm.vertex_count
        summary["grammars"][grammar] = {"V": V, "runs": {}}
        for w in WEIGHTS:
            run_dir = RUNS_DIR / f"E28_A0_seed{SEED}_{grammar}_advw{w:.1f}"
            if run_dir.exists():
                import shutil

                shutil.rmtree(run_dir)
            result = run_e28(
                fsm=fsm,
                run_id=f"E28_A0_seed{SEED}_{grammar}_advw{w:.1f}",
                output_dir=run_dir,
                seed=SEED,
                grammar=grammar,
                adversarial_token_weight=w,
            )
            summary["grammars"][grammar]["runs"][f"adv_w_{w:.1f}"] = {
                "n_argmax_cells": result.n_argmax_cells,
                "n_regimes_after_merge": result.n_regimes_after_merge,
                "mean_purity_against_current_state": (
                    result.mean_purity_against_current_state
                ),
                "mean_failure_rate": result.mean_failure_rate,
                "mean_entropy": result.mean_entropy,
                "mean_margin_to_tie": result.mean_margin_to_tie,
                "projection_token_accuracy": (
                    result.projection_token_accuracy
                ),
                "aligned_hamming_at_target_V": (
                    result.aligned_hamming_at_target_V
                ),
                "total_wall_clock_seconds": result.total_wall_clock_seconds,
            }
            tok_acc_str = (
                f"{result.projection_token_accuracy:.4f}"
                if not (
                    result.projection_token_accuracy != result.projection_token_accuracy
                )
                else "      --"
            )
            align_str = (
                f"{result.aligned_hamming_at_target_V:.4f}"
                if not (
                    result.aligned_hamming_at_target_V != result.aligned_hamming_at_target_V
                )
                else "      --"
            )
            print(
                f"{grammar:<18s} {V:>3d} {w:>6.1f} {result.n_argmax_cells:>7d} "
                f"{result.n_regimes_after_merge:>7d}  "
                f"{result.mean_purity_against_current_state:>8.4f} "
                f"{result.mean_failure_rate:>8.4f} "
                f"{result.mean_entropy:>8.4f} "
                f"{result.mean_margin_to_tie:>8.3f} "
                f"{tok_acc_str:>8s} {align_str:>8s} "
                f"{result.total_wall_clock_seconds:>7.2f}"
            )
        print("-" * 122)

    elapsed = time.perf_counter() - t_start
    summary["total_elapsed_seconds"] = elapsed

    # Aggregate: per-weight mean purity, mean argmax cells, mean failure.
    for w in WEIGHTS:
        key = f"adv_w_{w:.1f}"
        mean_purity = sum(
            g["runs"][key]["mean_purity_against_current_state"]
            for g in summary["grammars"].values()
        ) / len(GRAMMARS)
        mean_cells = sum(
            g["runs"][key]["n_argmax_cells"]
            for g in summary["grammars"].values()
        ) / len(GRAMMARS)
        mean_fail = sum(
            g["runs"][key]["mean_failure_rate"]
            for g in summary["grammars"].values()
        ) / len(GRAMMARS)
        mean_margin = sum(
            g["runs"][key]["mean_margin_to_tie"]
            for g in summary["grammars"].values()
        ) / len(GRAMMARS)
        summary[f"mean_purity_{key}"] = mean_purity
        summary[f"mean_argmax_cells_{key}"] = mean_cells
        summary[f"mean_failure_{key}"] = mean_fail
        summary[f"mean_margin_{key}"] = mean_margin
        print(
            f"adv_w={w:>4.1f}: mean purity {mean_purity:.4f}  "
            f"mean argmax cells {mean_cells:.1f}  "
            f"mean failure rate {mean_fail:.4f}  "
            f"mean margin {mean_margin:.3f}"
        )

    out_path = RUNS_DIR / "phase23b_adversarial_token_sweep_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\ntotal sweep wall-clock: {elapsed:.2f}s")
    print(f"Summary written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
