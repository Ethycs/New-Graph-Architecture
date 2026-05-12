#!/usr/bin/env python3
"""Phase 23d -- PCG-X over a small-transformer substrate, adv-off vs adv-on.

Runs E29 on each of the 5 grammars at two adversarial-token weights:
0.0 (the substrate-control) and 1.0 (the predicted-positive setting
from Phase 23b's analysis). Headline question: with the encoder no
longer state-conditioned, does the adversarial-token head now improve
per-regime state-purity?

Writes ``runs/phase23d_transformer_pcg_sweep_summary.json``.

Run:
    pixi run -e dev python scripts/phase23d_transformer_pcg_sweep.py
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
from nga.exp.e29_pcg_extractor_transformer import run_e29  # noqa: E402

SEED = 42
GRAMMARS = ["listops", "python_expr", "python_big", "json", "python_control"]
WEIGHTS = (0.0, 1.0)
HARVEST_LAYER = 0  # mid-layer of a 2-layer transformer
N_XFM_EPOCHS = 12


def main() -> int:
    summary: dict = {
        "seed": SEED,
        "weights": list(WEIGHTS),
        "harvest_layer": HARVEST_LAYER,
        "n_transformer_train_epochs": N_XFM_EPOCHS,
        "grammars": {},
    }
    print("=" * 130)
    print(
        "Phase 23d -- PCG-X on small-transformer substrate, "
        f"5-grammar sweep (seed={SEED}, harvest_layer={HARVEST_LAYER}, "
        f"transformer_epochs={N_XFM_EPOCHS})"
    )
    print("=" * 130)
    print(
        f"{'grammar':<18s} {'V':>3s} {'adv_w':>6s} {'argmax':>7s} {'regimes':>7s}  "
        f"{'purity':>8s} {'failure':>8s} {'entropy':>8s} "
        f"{'margin':>8s} {'xfm_acc':>8s} {'tok_acc':>8s} {'align_H':>8s} {'tot_s':>7s}"
    )
    print("-" * 130)
    t_start = time.perf_counter()
    for grammar in GRAMMARS:
        dispatch = GRAMMAR_DISPATCH[grammar]
        fsm_yaml = REPO_ROOT / dispatch["fsm_yaml_path"]
        fsm_spec = graph_fsm_spec_mod.load(fsm_yaml)
        fsm = GraphFSM(fsm_spec)
        V = fsm.vertex_count
        summary["grammars"][grammar] = {"V": V, "runs": {}}
        for w in WEIGHTS:
            run_dir = RUNS_DIR / f"E29_A0_seed{SEED}_{grammar}_advw{w:.1f}"
            if run_dir.exists():
                import shutil

                shutil.rmtree(run_dir)
            result = run_e29(
                fsm=fsm,
                run_id=f"E29_A0_seed{SEED}_{grammar}_advw{w:.1f}",
                output_dir=run_dir,
                seed=SEED,
                grammar=grammar,
                harvest_layer=HARVEST_LAYER,
                n_transformer_train_epochs=N_XFM_EPOCHS,
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
                "transformer_token_accuracy": (
                    result.transformer_token_accuracy
                ),
                "projection_token_accuracy": (
                    result.projection_token_accuracy
                ),
                "aligned_hamming_at_target_V": (
                    result.aligned_hamming_at_target_V
                ),
                "total_wall_clock_seconds": result.total_wall_clock_seconds,
            }
            tok_acc = result.projection_token_accuracy
            tok_acc_str = (
                f"{tok_acc:.4f}" if tok_acc == tok_acc else "      --"
            )
            align = result.aligned_hamming_at_target_V
            align_str = (
                f"{align:.4f}" if align == align else "      --"
            )
            print(
                f"{grammar:<18s} {V:>3d} {w:>6.1f} {result.n_argmax_cells:>7d} "
                f"{result.n_regimes_after_merge:>7d}  "
                f"{result.mean_purity_against_current_state:>8.4f} "
                f"{result.mean_failure_rate:>8.4f} "
                f"{result.mean_entropy:>8.4f} "
                f"{result.mean_margin_to_tie:>8.3f} "
                f"{result.transformer_token_accuracy:>8.4f} "
                f"{tok_acc_str:>8s} {align_str:>8s} "
                f"{result.total_wall_clock_seconds:>7.2f}"
            )
        print("-" * 130)
    elapsed = time.perf_counter() - t_start
    summary["total_elapsed_seconds"] = elapsed

    for w in WEIGHTS:
        key = f"adv_w_{w:.1f}"
        n = len(GRAMMARS)
        mp = sum(
            g["runs"][key]["mean_purity_against_current_state"]
            for g in summary["grammars"].values()
        ) / n
        mc = sum(
            g["runs"][key]["n_argmax_cells"] for g in summary["grammars"].values()
        ) / n
        mm = sum(
            g["runs"][key]["mean_margin_to_tie"]
            for g in summary["grammars"].values()
        ) / n
        mf = sum(
            g["runs"][key]["mean_failure_rate"]
            for g in summary["grammars"].values()
        ) / n
        summary[f"mean_purity_{key}"] = mp
        summary[f"mean_argmax_{key}"] = mc
        summary[f"mean_margin_{key}"] = mm
        summary[f"mean_failure_{key}"] = mf
        print(
            f"adv_w={w:>4.1f}: mean purity {mp:.4f}  "
            f"mean argmax cells {mc:.1f}  "
            f"mean margin {mm:.3f}  mean failure {mf:.4f}"
        )

    print(f"\ntotal sweep wall-clock: {elapsed:.2f}s")
    out_path = RUNS_DIR / "phase23d_transformer_pcg_sweep_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
