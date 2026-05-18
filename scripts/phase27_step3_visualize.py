"""Phase 27 Step 3 visualization -- spectrum + cumulative-variance curves + eff_rank vs V.

Reads runs/phase27_step3_pca.json and produces:
  * runs/phase27_step3_pca/{grammar}_spectrum.png -- per-grammar log-sigma scree
    plot + cumulative-variance ratio (with and without the rogue direction).
  * runs/phase27_step3_pca/overview_spectra.png  -- cross-grammar log-sigma
    overlay (denoised), highlighting the V-scaling.
  * runs/phase27_step3_pca/effective_rank_vs_V.png -- effective rank /
    participation / d_95 / d_99 plotted against the grammar's |V|.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
OUT_DIR = RUNS_DIR / "phase27_step3_pca"

V_GRAMMAR = {
    "listops": 11,
    "python_expr": 14,
    "python_big": 24,
    "json": 26,
    "python_control": 37,
}


def per_grammar_plot(grammar: str, stats: dict, out_path: Path) -> None:
    sigmas_raw = np.array(stats["sigma_all_raw"])
    sigmas_den = np.array(stats["sigma_all_denoised"])
    cum_raw = np.array(stats["raw"]["cumulative_variance_ratio"])
    cum_den = np.array(stats["denoised"]["cumulative_variance_ratio"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    # Panel 1: log-sigma scree.
    ax = axes[0]
    k_raw = np.arange(1, len(sigmas_raw) + 1)
    k_den = np.arange(1, len(sigmas_den) + 1)
    ax.semilogy(k_raw, sigmas_raw, "-o", ms=2, lw=0.8, label="raw (rogue dominates)", color="crimson")
    ax.semilogy(k_den, sigmas_den, "-o", ms=2, lw=0.8, label="denoised (top-1 removed)", color="steelblue")
    ax.set_xlabel("PC index k")
    ax.set_ylabel("σ_k  (log scale)")
    ax.set_title(f"{grammar}  N={stats['n_samples']}  rogue_frac={stats['rogue_variance_fraction']:.3f}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # Panel 2: cumulative variance ratio (denoised, since raw is dominated by PC1).
    ax = axes[1]
    ax.plot(k_den, cum_den, "-", lw=1.2, color="steelblue", label="denoised")
    ax.axhline(0.95, color="grey", ls=":", lw=0.8)
    ax.axhline(0.99, color="grey", ls="--", lw=0.8)
    den = stats["denoised"]
    ax.axvline(den["d_95"], color="orange", ls=":", lw=1, label=f"d_95={den['d_95']}")
    ax.axvline(den["d_99"], color="red", ls="--", lw=1, label=f"d_99={den['d_99']}")
    ax.set_xlabel("k")
    ax.set_ylabel("cumulative variance ratio (denoised)")
    ax.set_title(
        f"eff_rank={den['effective_rank']:.1f}  "
        f"part_ratio={den['participation_ratio']:.1f}"
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"Phase 27 Step 3 — GPT-2 block-6 activations on {grammar}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def overview_plot(per_grammar: dict, out_path: Path) -> None:
    """Cross-grammar denoised-spectrum overlay."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    grammars_sorted = sorted(per_grammar.keys(), key=lambda g: V_GRAMMAR[g])
    for i, g in enumerate(grammars_sorted):
        s = per_grammar[g]
        sigmas_den = np.array(s["sigma_all_denoised"])
        k = np.arange(1, len(sigmas_den) + 1)
        color = cmap(i / max(1, len(grammars_sorted) - 1))
        ax.semilogy(k, sigmas_den, "-", lw=1.2, color=color, label=f"{g} (V={V_GRAMMAR[g]})")
    ax.set_xlabel("PC index k (after top-1 rogue removal)")
    ax.set_ylabel("σ_k  (log scale)")
    ax.set_title("Phase 27 Step 3 — denoised activation spectra across grammars")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def eff_vs_v_plot(per_grammar: dict, out_path: Path) -> None:
    """Effective rank / participation / d_95 / d_99 against grammar V."""
    grammars_sorted = sorted(per_grammar.keys(), key=lambda g: V_GRAMMAR[g])
    V = np.array([V_GRAMMAR[g] for g in grammars_sorted])
    eff = np.array([per_grammar[g]["denoised"]["effective_rank"] for g in grammars_sorted])
    part = np.array([per_grammar[g]["denoised"]["participation_ratio"] for g in grammars_sorted])
    d95 = np.array([per_grammar[g]["denoised"]["d_95"] for g in grammars_sorted])
    d99 = np.array([per_grammar[g]["denoised"]["d_99"] for g in grammars_sorted])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(V, eff, "-o", color="steelblue", label="effective rank (exp H)")
    ax.plot(V, part, "-s", color="darkorange", label="participation ratio")
    # Linear V reference line.
    ax.plot(V, V, "--", color="grey", lw=0.8, label="V (linear)")
    for g, x, y in zip(grammars_sorted, V, eff):
        ax.annotate(g, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("|V| (FSM state count)")
    ax.set_ylabel("effective dimension")
    ax.set_title("Denoised effective dimension vs grammar |V|")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(V, d95, "-o", color="orange", label="d_95")
    ax.plot(V, d99, "-s", color="red", label="d_99")
    ax.plot(V, V, "--", color="grey", lw=0.8, label="V (linear)")
    for g, x, y in zip(grammars_sorted, V, d99):
        ax.annotate(g, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("|V| (FSM state count)")
    ax.set_ylabel("k for cumulative variance ≥ {0.95, 0.99}")
    ax.set_title("Denoised d_95 / d_99 vs grammar |V|")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 27 Step 3 — effective dimension scales with grammar V", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    data = json.loads((RUNS_DIR / "phase27_step3_pca.json").read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_grammar = data["per_grammar"]
    for g, stats in per_grammar.items():
        per_grammar_plot(g, stats, OUT_DIR / f"{g}_spectrum.png")
        print(f"  wrote {OUT_DIR / (g + '_spectrum.png')}")
    overview_plot(per_grammar, OUT_DIR / "overview_spectra.png")
    print(f"  wrote {OUT_DIR / 'overview_spectra.png'}")
    eff_vs_v_plot(per_grammar, OUT_DIR / "effective_rank_vs_V.png")
    print(f"  wrote {OUT_DIR / 'effective_rank_vs_V.png'}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
