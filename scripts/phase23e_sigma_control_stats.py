"""Phase 23e -- σ + control stats from PCG-X decision traces, train vs held-out eval.

Runs E28 twice per grammar (training set, held-out-eval set sampled with a
different seed) and reports:
  - Verdict mix: % ROUTE_NORMAL / ROUTE_RECOVERY / ABSTAIN
  - Mean σ overall
  - % of steps whose predicted regime falls outside the trained regime
    graph (the unknown sentinel; auto-fires illegal)
  - Mean illegal-signal contribution

This exposes how σ + control behave on data the projection never saw —
the diagnostic the σ + control integration unlocks.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from nga.arch.graph_fsm import GraphFSM
from nga.drivers import graph_fsm_spec as graph_fsm_spec_mod
from nga.drivers.decision_trace_jsonl import read_decision_trace
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH
from nga.exp.e28_pcg_extractor import run_e28

TRAIN_SEED = 42
EVAL_SEED = 43
N_PROGRAMS = 40
GRAMMARS = ["listops", "python_expr", "python_big", "json", "python_control"]


def _summarise(records: list) -> dict[str, float]:
    n = len(records)
    if n == 0:
        return {}
    decisions = np.array([r.control.decision for r in records])
    sigmas = np.array([r.sigma_total or 0.0 for r in records])
    illegal = np.array(
        [(r.sigma_signals or {}).get("illegal", 0.0) for r in records]
    )
    unknown = np.array([r.top1_state == "regime_unknown" for r in records])
    return {
        "rows": n,
        "norm_pct": float(100 * (decisions == "ROUTE_NORMAL").mean()),
        "rec_pct": float(100 * (decisions == "ROUTE_RECOVERY").mean()),
        "abs_pct": float(100 * (decisions == "ABSTAIN").mean()),
        "mean_sigma": float(sigmas.mean()),
        "unknown_pct": float(100 * unknown.mean()),
        "mean_illegal_signal": float(illegal.mean()),
    }


def main() -> int:
    print("=" * 110)
    print(
        f"Phase 23e -- σ + control stats, train_seed={TRAIN_SEED} vs eval_seed={EVAL_SEED}, "
        f"N={N_PROGRAMS}"
    )
    print("=" * 110)
    print(
        f"{'grammar':<16s} {'slice':<6s} "
        f"{'rows':>5s}  {'NORM%':>6s} {'REC%':>5s} {'ABS%':>5s}  "
        f"{'mean_σ':>7s}  {'unkn%':>6s} {'ill_sig':>8s}"
    )
    print("-" * 110)
    t0 = time.perf_counter()
    for grammar in GRAMMARS:
        dispatch = GRAMMAR_DISPATCH[grammar]
        fsm = GraphFSM(graph_fsm_spec_mod.load(REPO_ROOT / dispatch["fsm_yaml_path"]))
        # Train trace.
        run_train = REPO_ROOT / "runs" / f"E28_phase23e_{grammar}_train"
        run_eval = REPO_ROOT / "runs" / f"E28_phase23e_{grammar}_eval"
        for d in (run_train, run_eval):
            if d.exists():
                import shutil

                shutil.rmtree(d)
        run_e28(
            fsm=fsm,
            run_id=f"E28_phase23e_{grammar}_train",
            output_dir=run_train,
            seed=TRAIN_SEED,
            grammar=grammar,
            n_programs=N_PROGRAMS,
        )
        run_e28(
            fsm=fsm,
            run_id=f"E28_phase23e_{grammar}_eval",
            output_dir=run_eval,
            seed=TRAIN_SEED,
            grammar=grammar,
            n_programs=N_PROGRAMS,
            eval_n_programs=N_PROGRAMS,
            eval_seed=EVAL_SEED,
        )

        train_stats = _summarise(read_decision_trace(run_train / "decision_trace.jsonl"))
        eval_stats = _summarise(read_decision_trace(run_eval / "decision_trace.jsonl"))
        for label, s in (("train", train_stats), ("eval", eval_stats)):
            if not s:
                continue
            print(
                f"{grammar:<16s} {label:<6s} "
                f"{int(s['rows']):>5d}  "
                f"{s['norm_pct']:>6.1f} {s['rec_pct']:>5.1f} {s['abs_pct']:>5.1f}  "
                f"{s['mean_sigma']:>7.3f}  "
                f"{s['unknown_pct']:>6.1f} {s['mean_illegal_signal']:>8.3f}"
            )
    print("-" * 110)
    print(f"total wall-clock: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
