"""Phase 28 GAN-leader scout visualization -- side-by-side GPT-2 vs Qwen2.5-1.5B emotion clusters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"


def main() -> int:
    gpt2_npz = np.load(RUNS_DIR / "phase27_step6b_emotion" / "emotion_3d_data.npz")
    qwen_npz = np.load(RUNS_DIR / "phase28_gan_leader_scout" / "emotion_3d_data.npz")
    classes = gpt2_npz["classes"].tolist()
    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(15, 6.5))

    for ax_i, (npz, title) in enumerate([
        (gpt2_npz, "GPT-2 small (124M, block 6/12, hidden=768)\n"
                   "eff_rank=3.94  top-3=0.816  top-5=0.999"),
        (qwen_npz, "Qwen2.5-1.5B (1.5B, block 14/28, hidden=1536)\n"
                   "eff_rank=2.93  top-3=0.974  top-5=0.999"),
    ]):
        ax = fig.add_subplot(1, 2, ax_i + 1, projection="3d")
        h_3d = npz["h_3d"]
        labels = npz["labels"]
        for label_id in range(len(classes)):
            mask = labels == label_id
            ax.scatter(h_3d[mask, 0], h_3d[mask, 1], h_3d[mask, 2],
                       s=40, color=cmap(label_id), alpha=0.85,
                       label=classes[label_id],
                       edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Krylov 1")
        ax.set_ylabel("Krylov 2")
        ax.set_zlabel("Krylov 3")
        ax.set_title(title, fontsize=11)
        if ax_i == 0:
            ax.legend(loc="upper left", fontsize=9, markerscale=1.2)

    fig.suptitle("Phase 28 GAN-leader scout — 5-class emotion in 3-d ∇margin Krylov subspace\n"
                 "Qwen finds a sharper concept subspace than GPT-2 on the same 100-sentence corpus", y=1.02)
    fig.tight_layout()
    out_path = RUNS_DIR / "phase28_gan_leader_scout" / "gpt2_vs_qwen_emotion_3d.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
