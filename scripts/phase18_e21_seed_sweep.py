#!/usr/bin/env python3
"""Phase 18 -- multi-seed bootstrap for E21 (control flow Python: if /
else / while single-line bodies, Track 2 of Phase 18).

Runs E21/A0 on the python_control config at five seeds [42, 43, 44, 45,
46], parses each run's metrics.jsonl, and computes mean / std / min /
max for the headline Phase 14/15/16 metrics carried over to E21 PLUS
the new compute-efficiency metrics baked in from Phase 18 Track 1
(per-phase wall-clock, total wall-clock, inference throughput, peak
memory). Writes the aggregated summary to
``runs/phase18_e21_seed_sweep_summary.json`` and prints a compact
comparison table to stdout.

The headline questions under multi-seed test here:

* Does the seed-42 ``mask_accuracy_uplift = +0.464`` (control flow
  amplifies mask similarly to python_big) survive across seeds?
* Does the seed-42 ``sigma_structural_uplift = +0.0255`` (UP from E18's
  +0.0184 -- the architectural prediction was that branching ambiguity
  shifts σ_structural_uplift more positive) survive multi-seed, or was
  it within seed variance of zero?
* Does the seed-42 ``sigma_boundary_ratio = 1.27`` (UP from E18's 1.21)
  survive -- σ does fire more at the if-body branching point?
* Compute efficiency: how stable are the new wall-clock / throughput /
  memory metrics across seeds?

Run:
    pixi run -e dev python scripts/phase18_e21_seed_sweep.py

Notes
-----
* Each seed's output dir (``runs/E21_A0_seed{seed}/``) is rmtree'd
  before launching the run, because run.py refuses to overwrite a
  non-empty target. This is intentional: we want a clean per-seed
  metrics.jsonl, not appended rows from prior invocations.
* If a sub-run fails, the script raises and the summary is not written
  -- do not paper over failures.
* Mirrors ``scripts/phase15_e18_seed_sweep.py`` exactly; only the
  experiment id, config, output prefix, and the new compute-efficiency
  metrics differ.
"""
from __future__ import annotations

import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"

SEEDS = [42, 43, 44, 45, 46]

CONFIG_PATH = "tests/fixtures/configs/e21_python_control_minimal.yaml"
ABLATION_FILE = "tests/fixtures/ablations/ablations.yaml"

# Headline metrics tracked across seeds. Mirror Phase 15's set then add
# the new Phase 18 compute-efficiency metrics so the cross-grammar
# scoreboard is column-by-column comparable AND the new efficiency
# baseline is recorded at the same fidelity.
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
    # Phase 18 compute-efficiency metrics (baked into E21 from the start).
    "phase_a_wall_clock_seconds",
    "phase_b_wall_clock_seconds",
    "total_wall_clock_seconds",
    "inference_throughput_samples_per_sec",
    "peak_memory_kb",
]


def parse_metrics_jsonl(path: Path) -> dict[str, float]:
    """Read metrics.jsonl and return the latest value per metric_name.

    The file may contain appended rows from prior runs; later rows win.
    """
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["metric_name"]] = float(row["value"])
    return out


def run_one_seed(seed: int) -> dict[str, float]:
    """Launch one E21 run at the given seed and return its metrics."""
    out_dir = RUNS_DIR / f"E21_A0_seed{seed}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
        "pixi", "run", "-e", "dev", "python", "run.py",
        "--experiment", "E21",
        "--ablation", "A0",
        "--config", CONFIG_PATH,
        "--seed", str(seed),
        "--output", str(out_dir),
        "--ablation-file", ABLATION_FILE,
    ]
    print(f"[phase18] seed={seed} launching: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    metrics_path = out_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise RuntimeError(f"seed={seed}: metrics.jsonl missing at {metrics_path}")
    return parse_metrics_jsonl(metrics_path)


def summarise(values: list[float]) -> dict[str, float]:
    """Compute mean / std / min / max. Std uses sample stdev (n-1)."""
    if not values:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def main() -> int:
    per_seed: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        per_seed[str(seed)] = run_one_seed(seed)

    # Collect headline metrics across seeds.
    summary: dict[str, dict[str, float]] = {}
    for metric in HEADLINE_METRICS:
        values: list[float] = []
        for seed in SEEDS:
            v = per_seed[str(seed)].get(metric)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            values.append(float(v))
        summary[metric] = summarise(values)

    payload = {
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "config": CONFIG_PATH,
        "experiment": "E21",
        "ablation": "A0",
        "per_seed": {
            seed: {m: per_seed[seed].get(m) for m in HEADLINE_METRICS}
            for seed in per_seed
        },
        "summary": summary,
    }

    out_path = RUNS_DIR / "phase18_e21_seed_sweep_summary.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[phase18] wrote {out_path}\n", flush=True)

    # Compact comparison table: per-metric mean +/- std (min, max).
    print("=== Phase 18 E21 seed sweep summary ===")
    print(f"{'metric':<42s}  {'mean':>12s}  {'std':>10s}  "
          f"{'min':>12s}  {'max':>12s}  {'n':>3s}")
    for metric in HEADLINE_METRICS:
        s = summary[metric]
        print(f"{metric:<42s}  {s['mean']:>12.4f}  {s['std']:>10.4f}  "
              f"{s['min']:>12.4f}  {s['max']:>12.4f}  {int(s['n']):>3d}")

    print("\n=== Per-seed values (mask uplift, phase_a_hamming, "
          "sigma_uplift, sigma_struct_uplift, sigma_boundary_ratio) ===")
    print(f"{'seed':>5s}  {'acc':>8s}  {'acc_nomask':>10s}  {'uplift':>8s}  "
          f"{'phase_a_h':>10s}  {'sigma_upl':>10s}  {'sigma_st':>9s}  "
          f"{'sigma_ratio':>11s}")
    for seed in SEEDS:
        m = per_seed[str(seed)]
        print(
            f"{seed:>5d}  "
            f"{m.get('accuracy', float('nan')):>8.4f}  "
            f"{m.get('accuracy_no_mask', float('nan')):>10.4f}  "
            f"{m.get('mask_accuracy_uplift', float('nan')):>+8.4f}  "
            f"{m.get('phase_a_hamming_normalised', float('nan')):>10.4f}  "
            f"{m.get('sigma_uplift', float('nan')):>+10.4f}  "
            f"{m.get('sigma_structural_uplift', float('nan')):>+9.4f}  "
            f"{m.get('sigma_boundary_ratio', float('nan')):>11.4f}"
        )

    print("\n=== Per-seed compute efficiency ===")
    print(f"{'seed':>5s}  {'phase_a_s':>10s}  {'phase_b_s':>10s}  "
          f"{'total_s':>10s}  {'thru/s':>10s}  {'peak_kb':>10s}")
    for seed in SEEDS:
        m = per_seed[str(seed)]
        print(
            f"{seed:>5d}  "
            f"{m.get('phase_a_wall_clock_seconds', float('nan')):>10.4f}  "
            f"{m.get('phase_b_wall_clock_seconds', float('nan')):>10.4f}  "
            f"{m.get('total_wall_clock_seconds', float('nan')):>10.4f}  "
            f"{m.get('inference_throughput_samples_per_sec', float('nan')):>10.2f}  "
            f"{m.get('peak_memory_kb', float('nan')):>10.0f}"
        )

    # Headline structural-uplift verdict on control flow Python.
    s_struct = summary["sigma_structural_uplift"]
    print("\n=== Architectural prediction (sigma_structural_uplift > 0 on "
          "control flow Python) ===")
    print(f"  mean sigma_structural_uplift = {s_struct['mean']:+.4f}  "
          f"(std {s_struct['std']:.4f}; "
          f"min {s_struct['min']:+.4f}, max {s_struct['max']:+.4f})")
    if s_struct["mean"] > 0.0:
        verdict = (
            "PASS (control flow's branching ambiguity shifts sigma "
            "structural-uplift positive on average; architectural prediction held)"
        )
    else:
        verdict = (
            "FAIL (sigma_structural_uplift mean non-positive on control "
            "flow Python; the predicted shift towards positive structural "
            "uplift did NOT survive multi-seed)"
        )
    print(f"  verdict: {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
