"""Phase 27 Step 1.5 -- visualize the M(t)/log(t) trajectories.

For each Phase 24 GPT-2 run, plot:
  - The σ trajectory itself (panel 1: raw and smooth subset)
  - M(t) = running max of -log Delta vs log(t)  (panel 2)
  - r(t) = M(t)/log(t+1) with r_inf horizontal line + fit window shaded (panel 3)

Across all 4 conventions (boundary/normal × full/smooth), saved as one PNG
per run.

Purpose: distinguish "clean log-asymptote" from "still drifting in the tail."
A clean asymptote shows r(t) flat in the last window. Drift shows r(t)
still rising or falling, in which case the measured r_inf is just a window
mean of a non-converged ratio.

Reads runs/phase27_d_eff.json for the measured r_inf and writes PNGs to
runs/phase27_d_eff_trajectories/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


EPS = 1.0e-9
LAST_WINDOW_FRAC = 0.20
LAST_WINDOW_MIN = 50

CONVENTIONS = [
    ("boundary_full",   "Δ = 1−σ_total",         "tab:red"),
    ("normal_full",     "Δ = σ_total",            "tab:blue"),
    ("boundary_smooth", "Δ = 1−σ_smooth",         "tab:orange"),
    ("normal_smooth",   "Δ = σ_smooth",           "tab:green"),
]


def _load_sigma(
    run_dir: Path,
    *,
    smooth_only: bool = False,
) -> np.ndarray:
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
                v = max(0.0, min(1.0, v))
            else:
                s = rec.get("sigma_total")
                if s is None:
                    continue
                v = float(s)
            vals.append(v)
    return np.asarray(vals, dtype=float)


def _compute_M_r(sigma: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "boundary":
        delta = np.maximum(EPS, 1.0 - sigma)
    elif mode == "normal":
        delta = np.maximum(EPS, sigma)
    else:
        raise ValueError(mode)
    depth = -np.log(delta)
    M = np.maximum.accumulate(depth)
    t = np.arange(1, sigma.size + 1, dtype=float)
    r = M / np.log(t + 1.0)
    return M, r


def _plot_run(run_dir: Path, summary: dict, out_dir: Path) -> None:
    per_run = summary["per_run"].get(run_dir.name, {})
    if per_run.get("skipped"):
        return

    sigma_full = _load_sigma(run_dir, smooth_only=False)
    sigma_smooth = _load_sigma(run_dir, smooth_only=True)
    if sigma_full.size == 0:
        return
    n = sigma_full.size
    win = max(LAST_WINDOW_MIN, int(np.ceil(LAST_WINDOW_FRAC * n)))
    win = min(win, n - 1) if n > 1 else 1

    t = np.arange(1, n + 1, dtype=float)
    log_t = np.log(t + 1.0)

    fig, axes = plt.subplots(3, 4, figsize=(20, 11), constrained_layout=True)
    fig.suptitle(
        f"{run_dir.name}  (N = {n}, fit window = last {win} steps)",
        fontsize=13,
    )

    for col, (key, label, color) in enumerate(CONVENTIONS):
        mode = "boundary" if key.startswith("boundary") else "normal"
        sigma = sigma_smooth if key.endswith("smooth") else sigma_full
        M, r = _compute_M_r(sigma, mode)
        result = per_run.get(key, {})
        r_inf = result.get("r_inf", float("nan"))
        d_eff = result.get("d_eff", float("nan"))
        converged = result.get("converged", False)

        # Row 0: σ trajectory.
        ax0 = axes[0, col]
        ax0.plot(t, sigma, color=color, lw=0.6, alpha=0.7)
        ax0.set_title(label, fontsize=10)
        ax0.set_ylabel("σ" if col == 0 else "")
        ax0.set_xlim(1, n)
        ax0.set_ylim(0, max(0.05, sigma.max() * 1.05))
        ax0.tick_params(labelsize=8)
        ax0.axvspan(n - win, n, color="grey", alpha=0.15)

        # Row 1: M(t) and r_inf * log(t+1) reference line.
        ax1 = axes[1, col]
        ax1.plot(t, M, color=color, lw=0.9, label="M(t)")
        ax1.plot(
            t,
            r_inf * log_t,
            color="black",
            lw=0.6,
            ls="--",
            label=f"r_inf · log(t+1)  (r_inf={r_inf:.3f})",
        )
        ax1.set_ylabel("M(t)" if col == 0 else "")
        ax1.set_xlim(1, n)
        ax1.set_xscale("log")
        ax1.tick_params(labelsize=8)
        ax1.axvspan(n - win, n, color="grey", alpha=0.15)
        ax1.legend(fontsize=7, loc="upper left")

        # Row 2: r(t) with r_inf horizontal line.
        ax2 = axes[2, col]
        ax2.plot(t, r, color=color, lw=0.9)
        ax2.axhline(r_inf, color="black", lw=0.7, ls="--", label=f"r_inf={r_inf:.3f}")
        ax2.set_xlabel("t (steps, log scale)")
        ax2.set_ylabel("r(t) = M/log(t+1)" if col == 0 else "")
        ax2.set_xlim(1, n)
        ax2.set_xscale("log")
        ax2.tick_params(labelsize=8)
        ax2.axvspan(n - win, n, color="grey", alpha=0.15)
        ok_str = "✓" if converged else "✗"
        ax2.set_title(
            f"d_eff = {d_eff:.2f}  (converged: {ok_str})",
            fontsize=9,
        )
        ax2.legend(fontsize=7, loc="best")

    out_path = out_dir / f"{run_dir.name}_d_eff.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def _plot_overview(targets: list[Path], summary: dict, out_dir: Path) -> None:
    """Cross-grammar overview: r(t) for all 5 grammars overlaid, one panel per convention."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    fig.suptitle(
        "Cross-grammar r(t) = M(t)/log(t+1) overlay (Phase 27 Step 1)",
        fontsize=13,
    )
    grammar_colors = {
        "json":           "tab:red",
        "listops":        "tab:orange",
        "python_big":     "tab:green",
        "python_control": "tab:blue",
        "python_expr":    "tab:purple",
    }

    for col, (key, label, _) in enumerate(CONVENTIONS):
        ax = axes[col]
        ax.set_title(label, fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel("t (log scale)")
        if col == 0:
            ax.set_ylabel("r(t)")
        ax.grid(True, alpha=0.3)
        for run_dir in targets:
            per_run = summary["per_run"].get(run_dir.name, {})
            if per_run.get("skipped"):
                continue
            mode = "boundary" if key.startswith("boundary") else "normal"
            sigma = _load_sigma(run_dir, smooth_only=key.endswith("smooth"))
            if sigma.size == 0:
                continue
            _, r = _compute_M_r(sigma, mode)
            t = np.arange(1, sigma.size + 1, dtype=float)
            grammar = run_dir.name
            for prefix in ("E30_phase24_", "E30_phase27_long_"):
                grammar = grammar.replace(prefix, "")
            color = grammar_colors.get(grammar, "grey")
            d_eff = per_run.get(key, {}).get("d_eff", float("nan"))
            ax.plot(
                t,
                r,
                color=color,
                lw=0.9,
                alpha=0.85,
                label=f"{grammar} (d_eff={d_eff:.2f})",
            )
        ax.legend(fontsize=7, loc="best")
        ax.tick_params(labelsize=8)

    out_path = out_dir / "overview_r_trajectories.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


def main() -> int:
    # Optional CLI: summary filename + glob pattern + output dir name.
    summary_name = sys.argv[1] if len(sys.argv) >= 2 else "phase27_d_eff.json"
    glob_pattern = sys.argv[2] if len(sys.argv) >= 3 else "E30_phase24_*"
    out_dir_name = sys.argv[3] if len(sys.argv) >= 4 else "phase27_d_eff_trajectories"
    summary_path = RUNS_DIR / summary_name
    if not summary_path.exists():
        print(
            f"{summary_name} not found; run scripts/phase27_d_eff_measurement.py first.",
            file=sys.stderr,
        )
        return 1
    summary = json.loads(summary_path.read_text())

    targets = sorted(p for p in RUNS_DIR.glob(glob_pattern) if p.is_dir())
    out_dir = RUNS_DIR / out_dir_name
    out_dir.mkdir(exist_ok=True)

    print("=" * 100)
    print("Phase 27 Step 1.5 -- visualize Sullivan log-law trajectories")
    print("=" * 100)

    for run_dir in targets:
        _plot_run(run_dir, summary, out_dir)
    _plot_overview(targets, summary, out_dir)

    print("=" * 100)
    print(f"All figures in {out_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
