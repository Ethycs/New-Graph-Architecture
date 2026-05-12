#!/usr/bin/env python3
"""Phase 20 Wave-B-with-trained-encoder -- the decisive sweep.

Runs E25 on each of the 5 grammars *twice*: once with the frozen
random-projection encoder (the Wave-B baseline) and once with the
trained substrate (the readout's first-Linear pre-ReLU output after
``n_train_epochs`` of TorchEnergyTrainer gradient training). The
load-bearing comparison is the cross-grammar **frozen-vs-trained** delta
on K_star, cluster purity, normalised Hamming, and held-out NLL lift.

The proposal's strict Hamming <= 0.05 bar on Tier 1 sanity is reserved
for the trained substrate; if it survives there, the universality
hypothesis earns its keep at the strict bar. If not, the architecture's
claim is bounded by K-selection's small-N behaviour on this substrate.

Writes ``runs/phase20_e25_frozen_vs_trained_sweep_summary.json`` and
prints a compact comparison table.

Run:
    pixi run -e dev python scripts/phase20_e25_frozen_vs_trained_sweep.py
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
from nga.exp.e25_extraction_torch import (  # noqa: E402
    GRAMMAR_DISPATCH,
    run_e25,
)

SEED = 42
N_TRAIN_EPOCHS = 10
GRAMMARS = ["listops", "python_expr", "python_big", "json", "python_control"]


def _run_one(grammar: str, *, trained: bool) -> dict:
    dispatch = GRAMMAR_DISPATCH[grammar]
    fsm_yaml = REPO_ROOT / dispatch["fsm_yaml_path"]
    fsm_spec = graph_fsm_spec_mod.load(fsm_yaml)
    fsm = GraphFSM(fsm_spec)
    V = fsm.vertex_count

    variant = "trained" if trained else "frozen"
    run_dir = RUNS_DIR / f"E25_A0_seed{SEED}_{grammar}_{variant}"
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)

    t0 = time.perf_counter()
    result = run_e25(
        config=None,
        ablation=None,
        fsm=fsm,
        run_id=f"E25_A0_seed{SEED}_{grammar}_{variant}",
        output_dir=run_dir,
        seed=SEED,
        grammar=grammar,
        use_trained_substrate=trained,
        n_train_epochs=N_TRAIN_EPOCHS if trained else 0,
    )
    elapsed = time.perf_counter() - t0
    return {
        "grammar": grammar,
        "variant": variant,
        "V_ground_truth": V,
        "K_star": result.K_star,
        "extracted_hamming_normalised": result.extracted_hamming_normalised,
        "cluster_purity": result.cluster_purity,
        "holdout_nll_per_token_extracted": result.holdout_nll_per_token_extracted,
        "holdout_nll_per_token_chain": result.holdout_nll_per_token_chain,
        "holdout_nll_improvement_per_token": (
            result.holdout_nll_improvement_per_token
        ),
        "n_total_steps": result.n_total_steps,
        "n_samples": result.n_samples,
        "total_wall_clock_seconds": result.total_wall_clock_seconds,
        "elapsed_wall_clock_seconds": elapsed,
    }


def main() -> int:
    summary: dict = {
        "seed": SEED,
        "n_train_epochs": N_TRAIN_EPOCHS,
        "grammars": {},
    }

    print("=" * 96)
    print(
        "Phase 20 Wave B (trained vs frozen) -- 5-grammar Tier 1, "
        f"seed={SEED}, n_train_epochs={N_TRAIN_EPOCHS}"
    )
    print("=" * 96)
    header = (
        f"{'grammar':<18s} {'V':>3s} {'variant':>9s} {'K*':>3s} "
        f"{'purity':>8s} {'hamming':>9s} {'nll_lift':>10s} "
        f"{'n_steps':>8s} {'tot_s':>7s}"
    )
    print(header)
    print("-" * 96)

    for grammar in GRAMMARS:
        row_frozen = _run_one(grammar, trained=False)
        row_trained = _run_one(grammar, trained=True)
        summary["grammars"][grammar] = {
            "frozen": row_frozen,
            "trained": row_trained,
        }
        for row in (row_frozen, row_trained):
            print(
                f"{row['grammar']:<18s} {row['V_ground_truth']:>3d} "
                f"{row['variant']:>9s} {row['K_star']:>3d} "
                f"{row['cluster_purity']:>8.4f} "
                f"{row['extracted_hamming_normalised']:>9.4f} "
                f"{row['holdout_nll_improvement_per_token']:>+10.4f} "
                f"{row['n_total_steps']:>8d} "
                f"{row['total_wall_clock_seconds']:>7.2f}"
            )
        print("-" * 96)

    # Aggregate.
    n_frozen_meets = sum(
        1
        for v in summary["grammars"].values()
        if v["frozen"]["extracted_hamming_normalised"] <= 0.05
    )
    n_trained_meets = sum(
        1
        for v in summary["grammars"].values()
        if v["trained"]["extracted_hamming_normalised"] <= 0.05
    )
    summary["n_frozen_meets_strict_hamming"] = n_frozen_meets
    summary["n_trained_meets_strict_hamming"] = n_trained_meets
    summary["n_grammars"] = len(GRAMMARS)

    mean_purity_frozen = sum(
        v["frozen"]["cluster_purity"]
        for v in summary["grammars"].values()
    ) / len(GRAMMARS)
    mean_purity_trained = sum(
        v["trained"]["cluster_purity"]
        for v in summary["grammars"].values()
    ) / len(GRAMMARS)
    mean_lift_frozen = sum(
        v["frozen"]["holdout_nll_improvement_per_token"]
        for v in summary["grammars"].values()
    ) / len(GRAMMARS)
    mean_lift_trained = sum(
        v["trained"]["holdout_nll_improvement_per_token"]
        for v in summary["grammars"].values()
    ) / len(GRAMMARS)
    summary["mean_purity_frozen"] = mean_purity_frozen
    summary["mean_purity_trained"] = mean_purity_trained
    summary["mean_nll_lift_frozen"] = mean_lift_frozen
    summary["mean_nll_lift_trained"] = mean_lift_trained

    print(
        f"Strict Hamming <= 0.05: frozen {n_frozen_meets} / "
        f"{len(GRAMMARS)}; trained {n_trained_meets} / {len(GRAMMARS)}"
    )
    print(
        f"Mean cluster purity: frozen {mean_purity_frozen:.4f}, "
        f"trained {mean_purity_trained:.4f} "
        f"(delta {mean_purity_trained - mean_purity_frozen:+.4f})"
    )
    print(
        f"Mean NLL lift per token: frozen {mean_lift_frozen:+.4f}, "
        f"trained {mean_lift_trained:+.4f} "
        f"(delta {mean_lift_trained - mean_lift_frozen:+.4f})"
    )

    out_path = RUNS_DIR / "phase20_e25_frozen_vs_trained_sweep_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
