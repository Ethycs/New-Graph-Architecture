#!/usr/bin/env python3
"""Phase 20 Wave C -- 5-grammar Tier 2 sweep on a trained-from-scratch encoder.

Runs E26 on each of the 5 grammars (listops, python_expr, python_big,
json, python_control) with a small MLP encoder trained from scratch on
next-FSM-state prediction, then extracts. Headline question: does any
grammar's extracted_hamming_normalised cross the strict 0.05 sanity
bar? If so, the universality claim from
``docs/proposals/graph-extraction.md`` earns its keep at the
trained-substrate level.

Writes ``runs/phase20_wave_c_sweep_summary.json`` and prints a compact
comparison table to stdout.

Run:
    pixi run -e dev python scripts/phase20_wave_c_sweep.py
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
from nga.exp.e26_extraction_trained_encoder import run_e26  # noqa: E402

SEED = 42
N_TRAIN_EPOCHS = 50
GRAMMARS = ["listops", "python_expr", "python_big", "json", "python_control"]


def main() -> int:
    summary: dict = {
        "seed": SEED,
        "n_train_epochs": N_TRAIN_EPOCHS,
        "grammars": {},
    }
    print("=" * 100)
    print(
        "Phase 20 Wave C -- trained-from-scratch encoder, 5-grammar Tier 2 sweep, "
        f"seed={SEED}, n_train_epochs={N_TRAIN_EPOCHS}"
    )
    print("=" * 100)
    header = (
        f"{'grammar':<18s} {'V':>3s} {'K*':>3s} "
        f"{'purity':>8s} {'hamming':>9s} {'nll_lift':>10s} "
        f"{'enc_acc':>8s} {'enc_loss':>8s} {'tot_s':>7s}"
    )
    print(header)
    print("-" * 100)

    for grammar in GRAMMARS:
        dispatch = GRAMMAR_DISPATCH[grammar]
        fsm_yaml = REPO_ROOT / dispatch["fsm_yaml_path"]
        fsm_spec = graph_fsm_spec_mod.load(fsm_yaml)
        fsm = GraphFSM(fsm_spec)
        V = fsm.vertex_count

        run_dir_free = RUNS_DIR / f"E26_A0_seed{SEED}_{grammar}"
        run_dir_forced = RUNS_DIR / f"E26_A0_seed{SEED}_{grammar}_forced"
        for rd in (run_dir_free, run_dir_forced):
            if rd.exists():
                import shutil

                shutil.rmtree(rd)

        # Free-K variant (BIC over K_range = [V-pad, V+pad]).
        t0 = time.perf_counter()
        result = run_e26(
            fsm=fsm,
            run_id=f"E26_A0_seed{SEED}_{grammar}",
            output_dir=run_dir_free,
            seed=SEED,
            grammar=grammar,
            n_train_epochs=N_TRAIN_EPOCHS,
        )
        elapsed = time.perf_counter() - t0

        # Force-K=V variant (isolate extraction quality from K-selection).
        result_forced = run_e26(
            fsm=fsm,
            run_id=f"E26_A0_seed{SEED}_{grammar}_forced",
            output_dir=run_dir_forced,
            seed=SEED,
            grammar=grammar,
            n_train_epochs=N_TRAIN_EPOCHS,
            force_K_equals_V=True,
        )

        summary["grammars"][grammar] = {
            "V_ground_truth": V,
            "free_K": {
                "K_star": result.K_star,
                "extracted_hamming_normalised": (
                    result.extracted_hamming_normalised
                ),
                "cluster_purity": result.cluster_purity,
                "holdout_nll_improvement_per_token": (
                    result.holdout_nll_improvement_per_token
                ),
            },
            "forced_K_equals_V": {
                "K_star": result_forced.K_star,
                "extracted_hamming_normalised": (
                    result_forced.extracted_hamming_normalised
                ),
                "cluster_purity": result_forced.cluster_purity,
                "holdout_nll_improvement_per_token": (
                    result_forced.holdout_nll_improvement_per_token
                ),
            },
            "encoder_train_accuracy": result.encoder_train_accuracy,
            "encoder_train_final_loss": result.encoder_train_final_loss,
            "n_total_steps": result.n_total_steps,
            "n_samples": result.n_samples,
            "total_wall_clock_seconds": result.total_wall_clock_seconds,
            "elapsed_wall_clock_seconds": elapsed,
        }
        # Print free-K row, then forced-K row.
        print(
            f"{grammar:<18s} {V:>3d} {result.K_star:>3d} "
            f"{result.cluster_purity:>8.4f} "
            f"{result.extracted_hamming_normalised:>9.4f} "
            f"{result.holdout_nll_improvement_per_token:>+10.4f} "
            f"{result.encoder_train_accuracy:>8.4f} "
            f"{result.encoder_train_final_loss:>8.4f} "
            f"{result.total_wall_clock_seconds:>7.2f}"
        )
        print(
            f"{'  (forced K=V)':<18s} {V:>3d} {result_forced.K_star:>3d} "
            f"{result_forced.cluster_purity:>8.4f} "
            f"{result_forced.extracted_hamming_normalised:>9.4f} "
            f"{result_forced.holdout_nll_improvement_per_token:>+10.4f} "
            f"{'-':>8s} {'-':>8s} {'-':>7s}"
        )

    print("-" * 100)
    n_free_strict = sum(
        1
        for g in summary["grammars"].values()
        if g["free_K"]["extracted_hamming_normalised"] <= 0.05
    )
    n_forced_strict = sum(
        1
        for g in summary["grammars"].values()
        if g["forced_K_equals_V"]["extracted_hamming_normalised"] <= 0.05
    )
    summary["n_free_meets_strict_hamming"] = n_free_strict
    summary["n_forced_meets_strict_hamming"] = n_forced_strict
    summary["n_grammars"] = len(GRAMMARS)
    mean_purity_free = sum(
        g["free_K"]["cluster_purity"] for g in summary["grammars"].values()
    ) / len(GRAMMARS)
    mean_purity_forced = sum(
        g["forced_K_equals_V"]["cluster_purity"]
        for g in summary["grammars"].values()
    ) / len(GRAMMARS)
    mean_hamming_forced = sum(
        g["forced_K_equals_V"]["extracted_hamming_normalised"]
        for g in summary["grammars"].values()
    ) / len(GRAMMARS)
    summary["mean_purity_free_K"] = mean_purity_free
    summary["mean_purity_forced_K_equals_V"] = mean_purity_forced
    summary["mean_hamming_forced_K_equals_V"] = mean_hamming_forced
    summary["mean_encoder_train_accuracy"] = sum(
        g["encoder_train_accuracy"] for g in summary["grammars"].values()
    ) / len(GRAMMARS)

    print(
        f"Tier 2 strict Hamming<=0.05 bar (free K): "
        f"{n_free_strict} / {len(GRAMMARS)} grammars met"
    )
    print(
        f"Tier 2 strict Hamming<=0.05 bar (forced K=V): "
        f"{n_forced_strict} / {len(GRAMMARS)} grammars met"
    )
    print(
        f"Mean cluster purity: free_K {mean_purity_free:.4f}, "
        f"forced_K_equals_V {mean_purity_forced:.4f}"
    )
    print(
        f"Mean Hamming (forced K=V): {mean_hamming_forced:.4f}"
    )
    print(
        f"Mean encoder train accuracy: "
        f"{summary['mean_encoder_train_accuracy']:.4f}"
    )

    out_path = RUNS_DIR / "phase20_wave_c_sweep_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
