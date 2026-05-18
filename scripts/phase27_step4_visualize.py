"""Phase 27 Step 4 visualization -- 3-d Krylov scatter + cross-grammar overlay.

For each grammar, plot:
  * panel 1: log-sigma scree of the gradient matrix (top-30)
  * panel 2: cumulative variance capture of ∇margin in top-k Krylov directions
  * panel 3: 3-d scatter of activations projected into the top-3 Krylov
             directions, colored by argmax regime ID

Plus a cross-grammar overlay of cumulative-capture curves.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers projection

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
DATA_DIR = RUNS_DIR / "phase27_step4_krylov"

V_GRAMMAR = {
    "listops": 11,
    "python_expr": 14,
    "python_big": 24,
    "json": 26,
    "python_control": 37,
}


def per_grammar_plot(grammar: str, stats: dict, out_path: Path) -> None:
    sigmas = np.array(stats["sigma_top_k"])
    npz = np.load(DATA_DIR / f"{grammar}_3d_data.npz")
    h_3d = npz["h_3d"]
    cells = npz["argmax_cells"]

    fig = plt.figure(figsize=(15, 4.5))

    # Panel 1: gradient scree.
    ax = fig.add_subplot(1, 3, 1)
    k = np.arange(1, len(sigmas) + 1)
    ax.semilogy(k, sigmas, "-o", ms=4, color="steelblue")
    ax.set_xlabel("singular index k")
    ax.set_ylabel("σ_k of ∇margin matrix (log)")
    ax.set_title(f"gradient scree (top {len(sigmas)})")
    ax.grid(True, which="both", alpha=0.3)

    # Panel 2: cumulative-capture vs k.
    ax = fig.add_subplot(1, 3, 2)
    variances = sigmas ** 2
    total = variances.sum()
    cum = np.cumsum(variances) / max(total, 1e-30)
    ax.plot(k, cum, "-", lw=1.5, color="steelblue")
    ax.axhline(0.95, color="grey", ls=":", lw=0.8)
    ax.axhline(0.99, color="grey", ls="--", lw=0.8)
    for x, label in [(3, "top-3"), (5, "top-5"), (10, "top-10")]:
        if x <= len(cum):
            ax.scatter([x], [cum[x - 1]], color="red", s=40, zorder=5)
            ax.annotate(
                f"{label}\n{cum[x - 1]:.3f}",
                (x, cum[x - 1]),
                fontsize=8,
                xytext=(6, -2),
                textcoords="offset points",
            )
    ax.set_xlabel("k (Krylov direction)")
    ax.set_ylabel("cumulative σ²_k / total")
    ax.set_title("fraction of ∇margin variance captured")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 3: 3-d scatter colored by argmax regime.
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    unique_cells = np.unique(cells)
    cmap = plt.get_cmap("tab20", max(20, unique_cells.size))
    for ci, c in enumerate(unique_cells):
        mask = cells == c
        ax.scatter(
            h_3d[mask, 0],
            h_3d[mask, 1],
            h_3d[mask, 2],
            s=8,
            color=cmap(ci % cmap.N),
            alpha=0.7,
            label=f"R{int(c)}",
        )
    ax.set_xlabel("Krylov 1")
    ax.set_ylabel("Krylov 2")
    ax.set_zlabel("Krylov 3")
    ax.set_title(f"activations in σ-Krylov 3-d (N={len(cells)}, {unique_cells.size} cells)")
    # No legend for grammars with many regimes — too noisy.
    if unique_cells.size <= 12:
        ax.legend(loc="upper left", fontsize=7, markerscale=2)

    fig.suptitle(
        f"Phase 27 Step 4 — Krylov subspace of ∇margin on {grammar}  "
        f"|V|={V_GRAMMAR[grammar]}  top-3={stats['var_frac_top3']:.3f}",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def overview_plot(per_grammar: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    grammars = sorted(per_grammar.keys(), key=lambda g: V_GRAMMAR[g])
    for i, g in enumerate(grammars):
        s = per_grammar[g]
        sigmas = np.array(s["sigma_top_k"])
        variances = sigmas ** 2
        cum = np.cumsum(variances) / max(variances.sum(), 1e-30)
        k = np.arange(1, len(cum) + 1)
        color = cmap(i / max(1, len(grammars) - 1))
        ax.plot(k, cum, "-o", ms=4, lw=1.4, color=color, label=f"{g} (V={V_GRAMMAR[g]})")
    ax.axhline(0.95, color="grey", ls=":", lw=0.8)
    ax.axhline(0.99, color="grey", ls="--", lw=0.8)
    ax.axvline(3, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("k (Krylov direction)")
    ax.set_ylabel("cumulative σ²_k / total")
    ax.set_title("Phase 27 Step 4 — ∇margin variance captured in top-k Krylov directions")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    data = json.loads((RUNS_DIR / "phase27_step4_krylov.json").read_text())
    per_grammar = data["per_grammar"]
    for g, s in per_grammar.items():
        out_path = DATA_DIR / f"{g}_krylov_3d.png"
        per_grammar_plot(g, s, out_path)
        print(f"  wrote {out_path}")
    overview_plot(per_grammar, DATA_DIR / "overview_capture.png")
    print(f"  wrote {DATA_DIR / 'overview_capture.png'}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
