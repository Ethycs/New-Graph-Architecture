"""Phase 27 Step 6 + 6b visualization -- binary sentiment + 5-class emotion 3-d Krylov scatter."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"


def plot_binary(out_dir: Path) -> None:
    npz = np.load(out_dir / "sentiment_3d_data.npz")
    h_3d = npz["h_3d"]
    labels = npz["labels"]
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    for label, color, name in [(1, "tab:green", "positive"), (0, "tab:red", "negative")]:
        mask = labels == label
        ax.scatter(h_3d[mask, 0], h_3d[mask, 1], h_3d[mask, 2],
                   s=30, color=color, alpha=0.8, label=name, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Krylov 1")
    ax.set_ylabel("Krylov 2")
    ax.set_zlabel("Krylov 3")
    ax.set_title("Phase 27 Step 6 -- binary sentiment in 3-d ∇margin Krylov subspace\n"
                 "(eff_rank=1.03, top-3=0.998 -- collapsed to 1-d as predicted for binary)")
    ax.legend(loc="upper left", fontsize=10, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out_dir / "sentiment_3d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_emotion(out_dir: Path) -> None:
    npz = np.load(out_dir / "emotion_3d_data.npz")
    h_3d = npz["h_3d"]
    labels = npz["labels"]
    classes = npz["classes"].tolist()
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    for label_id in range(len(classes)):
        mask = labels == label_id
        ax.scatter(h_3d[mask, 0], h_3d[mask, 1], h_3d[mask, 2],
                   s=42, color=cmap(label_id), alpha=0.85,
                   label=classes[label_id], edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Krylov 1")
    ax.set_ylabel("Krylov 2")
    ax.set_zlabel("Krylov 3")
    ax.set_title("Phase 27 Step 6b -- 5-class emotion in 3-d ∇margin Krylov subspace\n"
                 "(eff_rank=3.94, top-3=0.816, top-5=0.999)")
    ax.legend(loc="upper left", fontsize=10, markerscale=1.2)
    fig.tight_layout()
    fig.savefig(out_dir / "emotion_3d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    plot_binary(RUNS_DIR / "phase27_step6_realtask")
    print(f"  wrote {RUNS_DIR / 'phase27_step6_realtask' / 'sentiment_3d.png'}")
    plot_emotion(RUNS_DIR / "phase27_step6b_emotion")
    print(f"  wrote {RUNS_DIR / 'phase27_step6b_emotion' / 'emotion_3d.png'}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
