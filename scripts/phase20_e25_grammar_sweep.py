#!/usr/bin/env python3
"""Phase 20 Wave B follow-up -- multi-grammar Tier 1 sweep.

Runs E25 on each of the 5 grammars in turn (listops, python_expr,
python_big, json, python_control) under the FrozenEncoderTorch
substrate. Same seed across grammars; the load-bearing comparison is
the *cross-grammar* table of Hamming, cluster purity, and held-out NLL
lift -- the architectural reading of the proposal's Tier 1 sanity bar.

Writes the aggregated summary to
``runs/phase20_e25_grammar_sweep_summary.json`` and prints a compact
table to stdout.

Run:
    pixi run -e dev python scripts/phase20_e25_grammar_sweep.py

The runner does NOT use the run.py CLI shell because that path requires
loading a per-grammar config / FSM via the strict Pydantic validator.
We invoke ``run_e25`` directly, loading each grammar's FSM from its
canonical fixture inside the dispatch table. This keeps the sweep
self-contained and avoids 5 redundant minimal config files.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"

# Add src to sys.path so this is runnable directly.
sys.path.insert(0, str(REPO_ROOT / "src"))

from nga.arch.graph_fsm import GraphFSM  # noqa: E402
from nga.drivers import graph_fsm_spec as graph_fsm_spec_mod  # noqa: E402
from nga.exp.e25_extraction_torch import (  # noqa: E402
    GRAMMAR_DISPATCH,
    run_e25,
)

SEED = 42

# Sweep order: small -> large in vertex count. listops 11 -> control 37.
GRAMMARS = ["listops", "python_expr", "python_big", "json", "python_control"]


def main() -> int:
    summary: dict = {
        "seed": SEED,
        "grammars": {},
    }
    print("=" * 78)
    print(f"Phase 20 Wave B follow-up -- 5-grammar Tier 1 sweep, seed={SEED}")
    print("=" * 78)
    header = (
        f"{'grammar':<18s} {'V':>3s} {'K*':>3s} "
        f"{'purity':>8s} {'hamming':>9s} {'nll_lift':>10s} "
        f"{'n_steps':>8s} {'tot_s':>7s}"
    )
    print(header)
    print("-" * 78)

    for grammar in GRAMMARS:
        dispatch = GRAMMAR_DISPATCH[grammar]
        fsm_yaml = REPO_ROOT / dispatch["fsm_yaml_path"]
        fsm_spec = graph_fsm_spec_mod.load(fsm_yaml)
        fsm = GraphFSM(fsm_spec)
        V = fsm.vertex_count

        run_dir = RUNS_DIR / f"E25_A0_seed{SEED}_{grammar}"
        if run_dir.exists():
            import shutil

            shutil.rmtree(run_dir)

        t0 = time.perf_counter()
        # config and ablation are CLI-symmetry only; pass minimal stubs.
        result = run_e25(
            config=None,
            ablation=None,
            fsm=fsm,
            run_id=f"E25_A0_seed{SEED}_{grammar}",
            output_dir=run_dir,
            seed=SEED,
            grammar=grammar,
        )
        elapsed = time.perf_counter() - t0

        summary["grammars"][grammar] = {
            "V_ground_truth": V,
            "K_star": result.K_star,
            "extracted_hamming_normalised": result.extracted_hamming_normalised,
            "cluster_purity": result.cluster_purity,
            "holdout_nll_per_token_extracted": (
                result.holdout_nll_per_token_extracted
            ),
            "holdout_nll_per_token_chain": result.holdout_nll_per_token_chain,
            "holdout_nll_improvement_per_token": (
                result.holdout_nll_improvement_per_token
            ),
            "n_total_steps": result.n_total_steps,
            "n_samples": result.n_samples,
            "total_wall_clock_seconds": result.total_wall_clock_seconds,
            "extraction_throughput_steps_per_sec": (
                result.extraction_throughput_steps_per_sec
            ),
            "elapsed_wall_clock_seconds": elapsed,
        }
        print(
            f"{grammar:<18s} {V:>3d} {result.K_star:>3d} "
            f"{result.cluster_purity:>8.4f} "
            f"{result.extracted_hamming_normalised:>9.4f} "
            f"{result.holdout_nll_improvement_per_token:>+10.4f} "
            f"{result.n_total_steps:>8d} "
            f"{result.total_wall_clock_seconds:>7.2f}"
        )

    print("-" * 78)
    # Headline: how many grammars met the strict Hamming <= 0.05 bar
    # (the proposal's Tier 1 sanity criterion).
    n_meets_strict = sum(
        1
        for g in summary["grammars"].values()
        if g["extracted_hamming_normalised"] <= 0.05
    )
    summary["n_meets_tier_1_strict_hamming_bar"] = n_meets_strict
    summary["n_grammars"] = len(GRAMMARS)
    print(
        f"Tier 1 strict Hamming<=0.05 bar: "
        f"{n_meets_strict} / {len(GRAMMARS)} grammars met"
    )
    # Compute mean cluster purity vs mean chance (1/V).
    mean_purity = sum(
        g["cluster_purity"] for g in summary["grammars"].values()
    ) / len(GRAMMARS)
    mean_chance = sum(
        1.0 / g["V_ground_truth"] for g in summary["grammars"].values()
    ) / len(GRAMMARS)
    mean_lift = sum(
        g["holdout_nll_improvement_per_token"]
        for g in summary["grammars"].values()
    ) / len(GRAMMARS)
    summary["mean_cluster_purity"] = mean_purity
    summary["mean_chance_inverse_V"] = mean_chance
    summary["mean_holdout_nll_improvement_per_token"] = mean_lift
    print(
        f"Mean cluster purity: {mean_purity:.4f} "
        f"(mean chance 1/V: {mean_chance:.4f}; "
        f"ratio: {mean_purity / mean_chance:.2f}x)"
    )
    print(
        f"Mean holdout NLL improvement: "
        f"{mean_lift:+.4f} nat/token vs uniform chain"
    )

    out_path = RUNS_DIR / "phase20_e25_grammar_sweep_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
