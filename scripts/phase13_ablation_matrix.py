#!/usr/bin/env python3
"""Phase 13 — ablation matrix sweep for E14 ListOps.

Runs E14 across the full A0..A9 one-flag-flipped ablation tuples at
seed 42, parses each run's ``metrics.jsonl``, builds a delta-from-A0
table, and writes ``runs/phase13_ablation_summary.json``.

Run:
    pixi run -e dev python scripts/phase13_ablation_matrix.py

Notes
-----
* Output dirs use a ``_phase13`` suffix to keep them separate from the
  existing ``runs/E14_A0_seed42/`` baseline.
* If a sub-run fails (e.g. a required atom is disabled and the runner
  raises), we record the error in the summary and continue with the
  next ablation. **Crashes are part of the data**: they tell us a
  feature is genuinely load-bearing rather than a no-op.
* The first thing this sweep tells us is which flags actually change
  E14's behaviour and which don't. Flags whose delta-from-A0 is
  identically zero on every metric are no-ops in this runner.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"

ABLATIONS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
ABLATION_NAMES = {
    "A0": "Full system",
    "A1": "No graph mask",
    "A2": "No typed scores",
    "A3": "No singularity detector",
    "A4": "Singularity computed but not routed",
    "A5": "No IDF weighting",
    "A6": "No hyperbolic geometry",
    "A7": "No group quotient",
    "A8": "Reservoir unfrozen",
    "A9": "No trace history",
}

SEED = 42
CONFIG_PATH = "tests/fixtures/configs/e14_listops_minimal.yaml"
ABLATION_FILE = "tests/fixtures/ablations/ablations.yaml"

# Headline metrics tracked across the matrix. These are the load-bearing
# claims from Phases 8/9/10/11 plus the structural-AUROC trio from
# Phase 10. The whole point of this sweep is to show, per metric, which
# flag flipping breaks the claim.
HEADLINE_METRICS = [
    "accuracy",
    "accuracy_no_mask",
    "mask_accuracy_uplift",
    "illegal_transition_rate",
    "illegal_transition_rate_no_mask",
    "phase_a_hamming_normalised",
    "phase_b_hamming_normalised",
    "sigma_auroc",
    "margin_auroc",
    "sigma_uplift",
    "sigma_structural_auroc",
    "margin_structural_auroc",
    "sigma_structural_uplift",
    "mean_sigma_at_operator_boundary",
    "mean_sigma_at_non_boundary",
    "sigma_boundary_ratio",
]

# Compact stdout columns -- keep this short so the table fits a terminal.
COMPACT_COLS = [
    "accuracy",
    "accuracy_no_mask",
    "mask_accuracy_uplift",
    "illegal_transition_rate",
    "phase_a_hamming_normalised",
    "sigma_boundary_ratio",
    "sigma_structural_auroc",
]


def parse_metrics_jsonl(path: Path) -> dict[str, float]:
    """Read metrics.jsonl and return the latest value per metric_name."""
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["metric_name"]] = float(row["value"])
    return out


def run_one_ablation(ablation: str) -> tuple[dict[str, float] | None, str | None]:
    """Launch E14 at this ablation and return (metrics, error).

    Exactly one of the two will be non-None.
    """
    out_dir = RUNS_DIR / f"E14_{ablation}_seed{SEED}_phase13"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
        "pixi", "run", "-e", "dev", "python", "run.py",
        "--experiment", "E14",
        "--ablation", ablation,
        "--config", CONFIG_PATH,
        "--seed", str(SEED),
        "--output", str(out_dir),
        "--ablation-file", ABLATION_FILE,
    ]
    print(f"[phase13] {ablation} ({ABLATION_NAMES[ablation]}) launching: "
          f"{' '.join(cmd)}", flush=True)
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"

    if proc.returncode != 0:
        # Capture last few lines of stderr -- enough to identify the failure
        # mode without writing a wall of text into the summary.
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        return None, (
            f"non-zero exit {proc.returncode}; stderr tail:\n{stderr_tail}"
        )

    metrics_path = out_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None, f"metrics.jsonl missing at {metrics_path}"

    try:
        metrics = parse_metrics_jsonl(metrics_path)
    except (json.JSONDecodeError, OSError, KeyError, ValueError) as exc:
        return None, f"failed to parse metrics.jsonl: {type(exc).__name__}: {exc}"

    # Filter to headline metrics only -- ignore params counts etc.
    return {m: metrics[m] for m in HEADLINE_METRICS if m in metrics}, None


def _fmt(value: float | None, width: int = 9, prec: int = 4) -> str:
    if value is None:
        return f"{'-':>{width}s}"
    return f"{value:>+{width}.{prec}f}"


def main() -> int:
    per_ablation_metrics: dict[str, dict[str, float]] = {}
    errors: dict[str, str] = {}

    for ablation in ABLATIONS:
        metrics, err = run_one_ablation(ablation)
        if err is not None:
            print(f"[phase13] {ablation} FAILED: {err}", flush=True)
            errors[ablation] = err
        else:
            assert metrics is not None
            per_ablation_metrics[ablation] = metrics
            print(f"[phase13] {ablation} ok: "
                  f"acc={metrics.get('accuracy', float('nan')):.4f}, "
                  f"phase_a_h="
                  f"{metrics.get('phase_a_hamming_normalised', float('nan')):.4f}, "
                  f"sigma_ratio="
                  f"{metrics.get('sigma_boundary_ratio', float('nan')):.4f}",
                  flush=True)

    # Build delta-from-A0 table.
    a0 = per_ablation_metrics.get("A0", {})
    delta_from_a0: dict[str, dict[str, float]] = {}
    for ablation in ABLATIONS:
        if ablation == "A0" or ablation not in per_ablation_metrics:
            continue
        m = per_ablation_metrics[ablation]
        delta_from_a0[ablation] = {}
        for metric in HEADLINE_METRICS:
            if metric in a0 and metric in m:
                delta_from_a0[ablation][metric] = m[metric] - a0[metric]

    payload = {
        "ablations": ABLATIONS,
        "ablation_names": ABLATION_NAMES,
        "seed": SEED,
        "config": CONFIG_PATH,
        "experiment": "E14",
        "n_ablations": len(ABLATIONS),
        "n_succeeded": len(per_ablation_metrics),
        "n_failed": len(errors),
        "per_ablation_metrics": per_ablation_metrics,
        "delta_from_A0": delta_from_a0,
        "errors": errors,
    }

    out_path = RUNS_DIR / "phase13_ablation_summary.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[phase13] wrote {out_path}\n", flush=True)

    # Compact comparison table: rows = ablations, cols = key metrics.
    print("=== Phase 13 ablation matrix (rows = ablations, cols = key metrics) ===")
    header = (
        f"{'abl':<4s}  {'name':<38s}  "
        + "  ".join(f"{c[:14]:>14s}" for c in COMPACT_COLS)
    )
    print(header)
    for ablation in ABLATIONS:
        if ablation in errors:
            print(
                f"{ablation:<4s}  {ABLATION_NAMES[ablation]:<38s}  "
                f"FAILED: {errors[ablation].splitlines()[0][:70]}"
            )
            continue
        m = per_ablation_metrics[ablation]
        cells = "  ".join(
            f"{m.get(c, float('nan')):>14.4f}" for c in COMPACT_COLS
        )
        print(f"{ablation:<4s}  {ABLATION_NAMES[ablation]:<38s}  {cells}")

    # Delta-from-A0 table: rows = A1..A9, cols = key metrics.
    print("\n=== Phase 13 delta-from-A0 (rows = A1..A9, cols = key metrics) ===")
    print(header)
    for ablation in ABLATIONS:
        if ablation == "A0":
            continue
        if ablation in errors:
            continue
        d = delta_from_a0.get(ablation, {})
        cells = "  ".join(
            _fmt(d.get(c), width=14, prec=4) for c in COMPACT_COLS
        )
        print(f"{ablation:<4s}  {ABLATION_NAMES[ablation]:<38s}  {cells}")

    # Quick read: which ablations are no-ops on E14? (all deltas zero)
    print("\n=== No-op ablations (delta-from-A0 == 0 on every tracked metric) ===")
    no_ops: list[str] = []
    behaviour_changers: list[str] = []
    for ablation, d in delta_from_a0.items():
        if d and all(abs(v) < 1e-12 for v in d.values()):
            no_ops.append(ablation)
        else:
            behaviour_changers.append(ablation)
    print(f"  no-ops: {no_ops if no_ops else '(none)'}")
    print(f"  behaviour-changers: "
          f"{behaviour_changers if behaviour_changers else '(none)'}")
    print(f"  crashed: {sorted(errors.keys()) if errors else '(none)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
