"""Phase 27 Step 1 -- measure d_eff via the Sullivan log-law on existing decision traces.

For each Phase 24 GPT-2 run, treat the σ trajectory as a discrete proxy for
a geodesic flow on the substrate's hyperbolic structure and compute the
Sullivan log-law:

    M(t) = max_{s <= t} -log Delta(sigma(s))
    r(t) = M(t) / log(t + 1)
    r_inf = mean of r(t) over the last window
    d_eff = 2 / r_inf

Two candidate discriminants are computed in parallel because the user's
framework places sigma in two distinct geometric positions:

  - DELTA_BOUNDARY  = 1 - sigma + eps      ("distance to boundary";
                       deep excursions are sigma -> 1).
  - DELTA_NORMAL    = sigma + eps          ("distance from boundary";
                       deep excursions are sigma -> 0).

Both should obey the same scaling law if the substrate's catastrophe
structure is genuinely Patterson-Sullivan-like; the *value* of d_eff under
each convention has different physical meaning but the *existence of a
clean asymptote* is what the framework predicts.

Acceptance bars (from docs/interpretability-push.md):
  - last-window std / mean < 0.05 (convergence; loosened from doc's 1%
    because n is small per grammar)
  - d_eff in [2, 4] across the 5 grammars
  - reference value from user's adjacent project: ~2.5 on GPT-2

The script writes runs/phase27_d_eff.json with per-grammar measurements +
diagnostics, and prints a single summary table. It does not modify any
existing artefact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

EPS = 1.0e-9
LAST_WINDOW_FRAC = 0.20    # fit r_inf over the last 20% of the trajectory
LAST_WINDOW_MIN = 50       # but at least this many points if available


def _load_sigma(
    run_dir: Path,
    *,
    smooth_only: bool = False,
) -> np.ndarray:
    """Load the σ trajectory.

    With ``smooth_only=True`` use only the smooth subset of the σ ensemble
    (margin + decision_tie + stabilizer), excluding the discrete signals
    (loop + illegal + catastrophe_bias). This matches the smooth-jet
    requirement for Patterson-Sullivan: the discriminant must be a smooth
    scalar field, not a discrete-saturated detector blend.
    """
    path = run_dir / "decision_trace.jsonl"
    if not path.exists():
        return np.array([], dtype=float)
    smooth_keys = ("margin", "decision_tie", "stabilizer")
    vals: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if smooth_only:
                sigs = rec.get("sigma_signals") or {}
                v = sum(float(sigs.get(k, 0.0)) for k in smooth_keys)
                # Clip to [0, 1] for log-law convention compatibility.
                v = max(0.0, min(1.0, v))
            else:
                s = rec.get("sigma_total")
                if s is None:
                    continue
                v = float(s)
            vals.append(v)
    return np.asarray(vals, dtype=float)


def _running_max(x: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(x)


def _log_law(sigma: np.ndarray, mode: str) -> dict:
    """Compute the Sullivan log-law trajectory under one of the two conventions.

    Returns dict with: M, r, r_inf, r_inf_std, d_eff, n, window_size, converged.
    """
    n = sigma.shape[0]
    if n < 10:
        return {
            "n": n,
            "skipped": True,
            "reason": "trajectory too short (n < 10)",
        }
    if mode == "boundary":
        delta = np.maximum(EPS, 1.0 - sigma)
    elif mode == "normal":
        delta = np.maximum(EPS, sigma)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    depth = -np.log(delta)               # +1 because t starts at 1 below
    M = _running_max(depth)
    t = np.arange(1, n + 1, dtype=float)
    log_t = np.log(t + 1.0)
    r = M / log_t                        # ratio trajectory

    win = max(LAST_WINDOW_MIN, int(np.ceil(LAST_WINDOW_FRAC * n)))
    win = min(win, n - 1) if n > 1 else 1
    r_tail = r[-win:]
    r_inf = float(np.mean(r_tail))
    r_inf_std = float(np.std(r_tail, ddof=0))
    converged = (
        r_inf > 0
        and (r_inf_std / max(EPS, r_inf)) < 0.05
    )
    d_eff = 2.0 / r_inf if r_inf > EPS else float("inf")
    return {
        "mode": mode,
        "n": int(n),
        "window_size": int(win),
        "r_inf": r_inf,
        "r_inf_std": r_inf_std,
        "r_inf_rel_std": float(r_inf_std / max(EPS, r_inf)),
        "d_eff": d_eff,
        "M_final": float(M[-1]),
        "sigma_min": float(sigma.min()),
        "sigma_max": float(sigma.max()),
        "sigma_mean": float(sigma.mean()),
        "converged": bool(converged),
        "skipped": False,
    }


def _process_run(run_dir: Path) -> dict:
    sigma_full = _load_sigma(run_dir, smooth_only=False)
    sigma_smooth = _load_sigma(run_dir, smooth_only=True)
    if sigma_full.size == 0:
        return {"skipped": True, "reason": "no decision_trace.jsonl"}
    grammar = run_dir.name
    for prefix in ("E30_phase24_", "E30_phase27_long_"):
        grammar = grammar.replace(prefix, "")
    out = {
        "grammar": grammar,
        "n_steps": int(sigma_full.size),
        "boundary_full":   _log_law(sigma_full,   "boundary"),
        "normal_full":     _log_law(sigma_full,   "normal"),
        "boundary_smooth": _log_law(sigma_smooth, "boundary"),
        "normal_smooth":   _log_law(sigma_smooth, "normal"),
    }
    return out


def _fmt_d(x: float) -> str:
    if not np.isfinite(x):
        return "  inf"
    return f"{x:6.3f}"


def main() -> int:
    # Optional CLI: run-dir glob pattern + output filename.
    if len(sys.argv) >= 2:
        glob_pattern = sys.argv[1]
    else:
        glob_pattern = "E30_phase24_*"
    if len(sys.argv) >= 3:
        out_name = sys.argv[2]
    else:
        out_name = "phase27_d_eff.json"
    targets = sorted(p for p in RUNS_DIR.glob(glob_pattern) if p.is_dir())
    if not targets:
        print(f"No {glob_pattern} run dirs found.", file=sys.stderr)
        return 1

    print("=" * 110)
    print("Phase 27 Step 1 -- Sullivan log-law d_eff measurement on Phase 24 GPT-2 decision traces")
    print("=" * 110)
    print(
        f"  {'grammar':14s}  {'N':>5s}  "
        f"{'d_eff B/full':>12s} {'OK':>3s}  {'d_eff N/full':>12s} {'OK':>3s}  "
        f"{'d_eff B/smooth':>14s} {'OK':>3s}  {'d_eff N/smooth':>14s} {'OK':>3s}"
    )
    print(
        f"  (B = boundary; N = normal; full = σ_total; smooth = margin+decision_tie+stabilizer)"
    )
    print("-" * 130)

    summary: dict = {
        "method": "sullivan-log-law",
        "discriminants": {
            "boundary": "Delta = max(EPS, 1 - sigma); cusp excursions at sigma -> 1",
            "normal":   "Delta = max(EPS, sigma); cusp excursions at sigma -> 0",
        },
        "window_fraction": LAST_WINDOW_FRAC,
        "convergence_rel_std_threshold": 0.05,
        "per_run": {},
    }
    for run_dir in targets:
        result = _process_run(run_dir)
        summary["per_run"][run_dir.name] = result
        if result.get("skipped"):
            print(f"  {run_dir.name:14s} SKIPPED ({result.get('reason','')})")
            continue
        bf = result["boundary_full"]
        nf = result["normal_full"]
        bs = result["boundary_smooth"]
        ns = result["normal_smooth"]
        print(
            f"  {result['grammar']:14s}  {result['n_steps']:5d}  "
            f"{_fmt_d(bf['d_eff']):>12s} {('Y' if bf['converged'] else 'N'):>3s}  "
            f"{_fmt_d(nf['d_eff']):>12s} {('Y' if nf['converged'] else 'N'):>3s}  "
            f"{_fmt_d(bs['d_eff']):>14s} {('Y' if bs['converged'] else 'N'):>3s}  "
            f"{_fmt_d(ns['d_eff']):>14s} {('Y' if ns['converged'] else 'N'):>3s}"
        )

    # Aggregate per discriminant convention.
    def _collect(key: str) -> list[float]:
        return [
            v[key]["d_eff"]
            for v in summary["per_run"].values()
            if not v.get("skipped")
            and v[key].get("converged")
            and np.isfinite(v[key]["d_eff"])
        ]

    framework_pred = 2.5
    framework_range = (2.0, 4.0)

    print("-" * 130)
    print("  Aggregate per convention (converged-only; framework prediction d_eff ≈ 2.5):")
    for key, label in [
        ("boundary_full",   "boundary / σ_total                 "),
        ("normal_full",     "normal   / σ_total                 "),
        ("boundary_smooth", "boundary / smooth (margin+tie+stab)"),
        ("normal_smooth",   "normal   / smooth (margin+tie+stab)"),
    ]:
        ds = _collect(key)
        if not ds:
            print(f"    {label}: no converged grammars.")
            summary[f"aggregate_{key}"] = {"n": 0}
            continue
        in_range = sum(1 for d in ds if framework_range[0] <= d <= framework_range[1])
        ag = {
            "n": len(ds),
            "mean": float(np.mean(ds)),
            "std": float(np.std(ds)),
            "in_pred_range": int(in_range),
        }
        summary[f"aggregate_{key}"] = ag
        print(
            f"    {label}: converged {ag['n']}/5  "
            f"mean d_eff = {ag['mean']:.3f} ± {ag['std']:.3f}  "
            f"in [2, 4]: {in_range}/{ag['n']}"
        )

    out_path = RUNS_DIR / out_name
    out_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"  wrote {out_path.name}")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
