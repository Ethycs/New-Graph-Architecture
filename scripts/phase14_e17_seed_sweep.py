#!/usr/bin/env python3
"""Phase 14 -- multi-seed bootstrap for E17 (Python expressions).

Runs E17/A0 on the published-grammar Python expression config at
five seeds [42, 43, 44, 45, 46], parses each run's metrics.jsonl,
and computes mean / std / min / max for the headline Phase 8/9/10
metrics carried over to E17. Writes the aggregated summary to
``runs/phase14_e17_seed_sweep_summary.json`` and prints a compact
comparison table to stdout.

The headline finding under multi-seed test here is
``sigma_uplift``: at seed 42 E17 produced a +0.050 sigma_uplift,
the FIRST positive sigma_uplift across all phases. This sweep asks
whether the q10 strict bar (sigma_uplift >= +0.03), XFAIL since
Phase 2, holds when the seed changes.

Run:
    pixi run -e dev python scripts/phase14_e17_seed_sweep.py

Notes
-----
* Each seed's output dir (``runs/E17_A0_seed{seed}/``) is rmtree'd
  before launching the run, because run.py refuses to overwrite a
  non-empty target. This is intentional: we want a clean per-seed
  metrics.jsonl, not appended rows from prior invocations.
* If a sub-run fails, the script raises and the summary is not
  written -- do not paper over failures.
* Mirrors ``scripts/phase12_seed_sweep.py`` exactly; only the
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

CONFIG_PATH = "tests/fixtures/configs/e17_python_expr_minimal.yaml"
ABLATION_FILE = "tests/fixtures/ablations/ablations.yaml"

# Headline metrics tracked across seeds. The load-bearing one is
# sigma_uplift -- whether the q10 strict bar (>=+0.03) survives
# multi-seed on real Python.
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
    """Launch one E17 run at the given seed and return its metrics."""
    out_dir = RUNS_DIR / f"E17_A0_seed{seed}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
        "pixi", "run", "-e", "dev", "python", "run.py",
        "--experiment", "E17",
        "--ablation", "A0",
        "--config", CONFIG_PATH,
        "--seed", str(seed),
        "--output", str(out_dir),
        "--ablation-file", ABLATION_FILE,
    ]
    print(f"[phase14] seed={seed} launching: {' '.join(cmd)}", flush=True)
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
        "experiment": "E17",
        "ablation": "A0",
        "per_seed": {
            seed: {m: per_seed[seed].get(m) for m in HEADLINE_METRICS}
            for seed in per_seed
        },
        "summary": summary,
    }

    out_path = RUNS_DIR / "phase14_e17_seed_sweep_summary.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[phase14] wrote {out_path}\n", flush=True)

    # Compact comparison table: per-metric mean +/- std (min, max).
    print("=== Phase 14 E17 seed sweep summary ===")
    print(f"{'metric':<40s}  {'mean':>10s}  {'std':>10s}  "
          f"{'min':>10s}  {'max':>10s}  {'n':>3s}")
    for metric in HEADLINE_METRICS:
        s = summary[metric]
        print(f"{metric:<40s}  {s['mean']:>10.4f}  {s['std']:>10.4f}  "
              f"{s['min']:>10.4f}  {s['max']:>10.4f}  {int(s['n']):>3d}")

    print("\n=== Per-seed values (mask uplift, phase_a_hamming, "
          "sigma_uplift, sigma_boundary_ratio) ===")
    print(f"{'seed':>5s}  {'acc':>8s}  {'acc_nomask':>10s}  {'uplift':>8s}  "
          f"{'phase_a_h':>10s}  {'sigma_upl':>10s}  {'sigma_ratio':>11s}")
    for seed in SEEDS:
        m = per_seed[str(seed)]
        print(
            f"{seed:>5d}  "
            f"{m.get('accuracy', float('nan')):>8.4f}  "
            f"{m.get('accuracy_no_mask', float('nan')):>10.4f}  "
            f"{m.get('mask_accuracy_uplift', float('nan')):>+8.4f}  "
            f"{m.get('phase_a_hamming_normalised', float('nan')):>10.4f}  "
            f"{m.get('sigma_uplift', float('nan')):>+10.4f}  "
            f"{m.get('sigma_boundary_ratio', float('nan')):>11.4f}"
        )

    # Headline q10 strict bar verdict.
    s_upl = summary["sigma_uplift"]
    print("\n=== q10 strict bar (sigma_uplift >= +0.03) under multi-seed ===")
    print(f"  mean sigma_uplift = {s_upl['mean']:+.4f}  (std {s_upl['std']:.4f}; "
          f"min {s_upl['min']:+.4f}, max {s_upl['max']:+.4f})")
    if s_upl["mean"] >= 0.03:
        verdict = "PASS (q10 strict bar passes multi-seed on real Python)"
    elif s_upl["mean"] > 0.0:
        verdict = (
            "PARTIAL (sigma_uplift mean is positive but below +0.03; "
            "sigma is winning on average but not at the strict bar)"
        )
    else:
        verdict = (
            "FAIL (sigma_uplift mean is non-positive; q10 still XFAIL; "
            "the seed-42 result was an artefact)"
        )
    print(f"  verdict: {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
