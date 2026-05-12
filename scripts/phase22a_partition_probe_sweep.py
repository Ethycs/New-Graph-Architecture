#!/usr/bin/env python3
"""Phase 22a -- partition-function-probe encoder, 5-grammar sweep.

The decisive test of Condition (1) reframing: does a partition-function
probe attached to the trained-from-scratch encoder reshape its natural
equivalence classes from ``(state, token)`` toward ``state`` alone,
bringing forced-K=V Hamming from Wave C's 0.242 toward the oracle
floor of 0.026?

Runs E27 at force_K_equals_V=True with the gold-legality partition
target on each of the 5 grammars and reports:

  - Hamming under the probe-augmented encoder
  - cluster purity at K = V
  - encoder train accuracy + final probe KL

against Wave C's baseline (forced K = V, no probe) and the oracle
ceiling from Phase 21.

Writes ``runs/phase22a_partition_probe_sweep_summary.json`` and prints
a compact comparison table to stdout.

Run:
    pixi run -e dev python scripts/phase22a_partition_probe_sweep.py
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
from nga.exp.e27_extraction_partition_probe import run_e27  # noqa: E402

SEED = 42
N_TRAIN_EPOCHS = 50
PROBE_WEIGHTS_TO_SWEEP = (0.0, 1.0, 5.0)  # 0.0 = Wave-C-equivalent control
GRAMMARS = ["listops", "python_expr", "python_big", "json", "python_control"]


def main() -> int:
    summary: dict = {
        "seed": SEED,
        "n_train_epochs": N_TRAIN_EPOCHS,
        "probe_weights": list(PROBE_WEIGHTS_TO_SWEEP),
        "grammars": {},
    }
    print("=" * 108)
    print(
        "Phase 22a -- partition-function-probe encoder, 5-grammar forced K=V sweep, "
        f"seed={SEED}, n_train_epochs={N_TRAIN_EPOCHS}"
    )
    print("=" * 108)
    print(
        f"{'grammar':<18s} {'V':>3s} {'wt':>5s} {'enc_acc':>8s} "
        f"{'purity':>8s} {'hamming':>9s} {'probe_kl':>10s} {'tot_s':>7s}"
    )
    print("-" * 108)

    t_sweep_start = time.perf_counter()
    for grammar in GRAMMARS:
        dispatch = GRAMMAR_DISPATCH[grammar]
        fsm_yaml = REPO_ROOT / dispatch["fsm_yaml_path"]
        fsm_spec = graph_fsm_spec_mod.load(fsm_yaml)
        fsm = GraphFSM(fsm_spec)
        V = fsm.vertex_count
        summary["grammars"][grammar] = {"V": V, "runs": {}}
        for pw in PROBE_WEIGHTS_TO_SWEEP:
            run_dir = (
                RUNS_DIR / f"E27_A0_seed{SEED}_{grammar}_pw{pw:.1f}"
            )
            if run_dir.exists():
                import shutil

                shutil.rmtree(run_dir)
            t0 = time.perf_counter()
            result = run_e27(
                fsm=fsm,
                run_id=f"E27_A0_seed{SEED}_{grammar}_pw{pw:.1f}",
                output_dir=run_dir,
                seed=SEED,
                grammar=grammar,
                n_train_epochs=N_TRAIN_EPOCHS,
                probe_weight=pw,
                force_K_equals_V=True,
            )
            elapsed = time.perf_counter() - t0
            summary["grammars"][grammar]["runs"][f"probe_weight_{pw:.1f}"] = {
                "K_star": result.K_star,
                "extracted_hamming_normalised": (
                    result.extracted_hamming_normalised
                ),
                "cluster_purity": result.cluster_purity,
                "encoder_train_accuracy": result.encoder_train_accuracy,
                "encoder_train_final_probe_kl": (
                    result.encoder_train_final_probe_kl
                ),
                "holdout_nll_improvement_per_token": (
                    result.holdout_nll_improvement_per_token
                ),
                "total_wall_clock_seconds": result.total_wall_clock_seconds,
                "elapsed_wall_clock_seconds": elapsed,
            }
            print(
                f"{grammar:<18s} {V:>3d} {pw:>5.1f} "
                f"{result.encoder_train_accuracy:>8.4f} "
                f"{result.cluster_purity:>8.4f} "
                f"{result.extracted_hamming_normalised:>9.4f} "
                f"{result.encoder_train_final_probe_kl:>10.4f} "
                f"{result.total_wall_clock_seconds:>7.2f}"
            )
        print("-" * 108)
    total_elapsed = time.perf_counter() - t_sweep_start

    # Aggregate: mean Hamming and purity per probe weight.
    for pw in PROBE_WEIGHTS_TO_SWEEP:
        key = f"probe_weight_{pw:.1f}"
        mean_h = sum(
            g["runs"][key]["extracted_hamming_normalised"]
            for g in summary["grammars"].values()
        ) / len(GRAMMARS)
        mean_p = sum(
            g["runs"][key]["cluster_purity"]
            for g in summary["grammars"].values()
        ) / len(GRAMMARS)
        mean_acc = sum(
            g["runs"][key]["encoder_train_accuracy"]
            for g in summary["grammars"].values()
        ) / len(GRAMMARS)
        n_meets = sum(
            1
            for g in summary["grammars"].values()
            if g["runs"][key]["extracted_hamming_normalised"] <= 0.05
        )
        summary[f"mean_hamming_{key}"] = mean_h
        summary[f"mean_purity_{key}"] = mean_p
        summary[f"mean_train_acc_{key}"] = mean_acc
        summary[f"n_meets_strict_bar_{key}"] = n_meets
        print(
            f"probe_weight={pw:>4.1f}: mean Hamming {mean_h:.4f}  "
            f"mean purity {mean_p:.4f}  mean enc_acc {mean_acc:.4f}  "
            f"strict bar (<=0.05) on {n_meets}/{len(GRAMMARS)} grammars"
        )
    summary["total_elapsed_seconds"] = total_elapsed
    print(f"total sweep wall-clock: {total_elapsed:.2f}s")

    out_path = RUNS_DIR / "phase22a_partition_probe_sweep_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
