#!/usr/bin/env python3
"""Phase 16 -- multi-seed bootstrap for E19 (JSON external benchmark).

Runs E19/A0 on the published JSON FSM config (26-vertex,
99-edge, json.loads-validated grammar with 7 value-position
ambiguity sites that each admit 7+ legal continuations) at the
canonical five seeds [42, 43, 44, 45, 46], parses each run's
metrics.jsonl, and computes mean / std / min / max for the
headline Phase 8/9/10/14/15 metrics carried over to E19. Writes
the aggregated summary to ``runs/phase16_e19_seed_sweep_summary.json``
and prints a compact comparison table.

The headline question under multi-seed test here:

* Does the seed-42 ``sigma_structural_uplift = +0.114`` (the
  largest positive structural uplift seen across all four
  external benchmarks so far) survive across seeds, or is it
  seed luck? JSON's value-position ambiguity is multi-way and
  uniformly distributed across the trajectory -- the prior
  prediction is that sigma should fire structurally hardest
  here.
* Does the seed-42 ``sigma_boundary_ratio = 1.43`` (UP from
  python_big's 1.08, DOWN from ListOps's 2.10) survive --
  sigma does fire more at value-position points than at
  punctuation states?
* Does the seed-42 ``mask_accuracy_uplift = +0.477`` survive
  across seeds -- mask-uplift remains > +0.40 on every
  external grammar so far (E14/E17/E18).

Run:
    pixi run -e dev python scripts/phase16_e19_seed_sweep.py

Notes
-----
* Each seed's output dir (``runs/E19_A0_seed{seed}/``) is
  rmtree'd before launching the run; run.py refuses to overwrite
  a non-empty target. This is intentional: we want a clean
  per-seed metrics.jsonl, not appended rows from prior runs.
* If a sub-run fails, the script raises and the summary is not
  written -- do not paper over failures.
* Mirrors ``scripts/phase15_e18_seed_sweep.py`` exactly; only the
  experiment id, config, and output prefix differ.
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

CONFIG_PATH = "tests/fixtures/configs/e19_json_minimal.yaml"
ABLATION_FILE = "tests/fixtures/ablations/ablations.yaml"

# Headline metrics tracked across seeds. Same set as Phase 14/15
# so the Phase 14 / 15 / 16 numbers are directly comparable
# column-by-column across all four external benchmarks.
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
    """Launch one E19 run at the given seed and return its metrics."""
    out_dir = RUNS_DIR / f"E19_A0_seed{seed}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
        "pixi", "run", "-e", "dev", "python", "run.py",
        "--experiment", "E19",
        "--ablation", "A0",
        "--config", CONFIG_PATH,
        "--seed", str(seed),
        "--output", str(out_dir),
        "--ablation-file", ABLATION_FILE,
    ]
    print(f"[phase16] seed={seed} launching: {' '.join(cmd)}", flush=True)
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
        "experiment": "E19",
        "ablation": "A0",
        "per_seed": {
            seed: {m: per_seed[seed].get(m) for m in HEADLINE_METRICS}
            for seed in per_seed
        },
        "summary": summary,
    }

    out_path = RUNS_DIR / "phase16_e19_seed_sweep_summary.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[phase16] wrote {out_path}\n", flush=True)

    # Compact comparison table: per-metric mean +/- std (min, max).
    print("=== Phase 16 E19 seed sweep summary (JSON external benchmark) ===")
    print(f"{'metric':<40s}  {'mean':>10s}  {'std':>10s}  "
          f"{'min':>10s}  {'max':>10s}  {'n':>3s}")
    for metric in HEADLINE_METRICS:
        s = summary[metric]
        print(f"{metric:<40s}  {s['mean']:>10.4f}  {s['std']:>10.4f}  "
              f"{s['min']:>10.4f}  {s['max']:>10.4f}  {int(s['n']):>3d}")

    print("\n=== Per-seed values (mask uplift, phase_a_hamming, "
          "sigma_uplift, sigma_struct_uplift, sigma_boundary_ratio) ===")
    print(f"{'seed':>5s}  {'acc':>8s}  {'acc_nomask':>10s}  {'uplift':>8s}  "
          f"{'phase_a_h':>10s}  {'sigma_upl':>10s}  "
          f"{'sigma_struct':>12s}  {'sigma_ratio':>11s}")
    for seed in SEEDS:
        m = per_seed[str(seed)]
        print(
            f"{seed:>5d}  "
            f"{m.get('accuracy', float('nan')):>8.4f}  "
            f"{m.get('accuracy_no_mask', float('nan')):>10.4f}  "
            f"{m.get('mask_accuracy_uplift', float('nan')):>+8.4f}  "
            f"{m.get('phase_a_hamming_normalised', float('nan')):>10.4f}  "
            f"{m.get('sigma_uplift', float('nan')):>+10.4f}  "
            f"{m.get('sigma_structural_uplift', float('nan')):>+12.4f}  "
            f"{m.get('sigma_boundary_ratio', float('nan')):>11.4f}"
        )

    # Headline structural-uplift verdict on JSON. The seed-42 value
    # (+0.114) was the largest positive across all four external
    # benchmarks; the multi-seed question is whether this holds.
    s_su = summary["sigma_structural_uplift"]
    print("\n=== sigma_structural_uplift on JSON under multi-seed ===")
    print(f"  mean = {s_su['mean']:+.4f}  (std {s_su['std']:.4f}; "
          f"min {s_su['min']:+.4f}, max {s_su['max']:+.4f})")
    print(f"  cross-grammar reference (seed 42 / multi-seed mean):")
    print(f"    ListOps (E14):     -0.088 (single-seed)")
    print(f"    Python expr (E17): +0.096 / +0.059 (5-seed)")
    print(f"    Python big (E18):  +0.018 / -0.012 (5-seed)")
    print(f"    JSON (E19):        +0.114 (seed 42) / "
          f"{s_su['mean']:+.4f} (5-seed)")

    s_upl = summary["sigma_uplift"]
    print("\n=== q10 strict bar (sigma_uplift >= +0.03) under multi-seed "
          "on JSON ===")
    print(f"  mean sigma_uplift = {s_upl['mean']:+.4f}  (std {s_upl['std']:.4f}; "
          f"min {s_upl['min']:+.4f}, max {s_upl['max']:+.4f})")
    if s_upl["mean"] >= 0.03:
        verdict = "PASS (q10 strict bar passes multi-seed on JSON)"
    elif s_upl["mean"] > 0.0:
        verdict = (
            "PARTIAL (sigma_uplift mean is positive but below +0.03; "
            "sigma is winning on average but not at the strict bar)"
        )
    else:
        verdict = (
            "FAIL (sigma_uplift mean is non-positive; q10 still XFAIL on "
            "JSON; failure-AUROC is grammar-conditional, not universal)"
        )
    print(f"  verdict: {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
