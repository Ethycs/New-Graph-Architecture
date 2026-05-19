"""Phase 28 -- auto-label SAE features by their top-activating examples.

For each SAE feature, find the corpus samples that activate it most and
look at their gold class. If a feature concentrates on one class above
threshold, label it ``<class>:f<id>`` ("activates predominantly on joy",
etc.). Otherwise leave it unlabelled (it becomes part of `residual`).

This is the standard SAE-feature-interpretation move (Anthropic's
released SAEs come with handcrafted labels derived from a similar
process at scale; we do a tiny automated version here).

Outputs:
  * runs/phase28_sae/labels.json -- {feature_id_str: label} dict
                                    consumable by PretrainedSAEAdapter.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from nga.arch.sae_adapter import PretrainedSAEAdapter  # noqa: E402

from phase27_step6b_emotion_krylov import CLASSES  # noqa: E402

PURITY_THRESHOLD = 0.65  # fraction of top-k activations from one class to label
TOP_K = 5  # examine top-5 activating sentences per feature
MIN_TOTAL_ACTIVATIONS = 3  # feature must fire on at least this many samples


def main() -> int:
    out_dir = RUNS_DIR / "phase28_sae"
    sae = PretrainedSAEAdapter.from_checkpoint(
        out_dir / "emotion_block6.npz",
        activation_threshold=0.0,
    )
    harvest = np.load(out_dir / "emotion_harvest.npz", allow_pickle=True)
    H = harvest["H"]
    labels = harvest["labels"]
    sentences = harvest["sentences"]

    print("=" * 100)
    print("Phase 28 -- auto-label SAE features by top-activating-class purity")
    print(f"  SAE features      : {sae.n_features}")
    print(f"  n_samples         : {H.shape[0]}")
    print(f"  classes           : {CLASSES}")
    print(f"  purity_threshold  : {PURITY_THRESHOLD}")
    print(f"  top_k             : {TOP_K}")
    print(f"  min_total_active  : {MIN_TOTAL_ACTIVATIONS}")
    print("=" * 100)

    # Encode every sample -> (n_samples, n_features) dense activation matrix.
    Z = np.zeros((H.shape[0], sae.n_features), dtype=np.float32)
    for i in range(H.shape[0]):
        code = sae.encode(H[i])
        for fid, val in zip(code.active_features, code.activations):
            Z[i, fid] = val

    # Per-feature: top-K samples by activation, count class purity.
    auto_labels: dict[str, str] = {}
    per_class_count: Counter[str] = Counter()
    feature_descriptions: dict[int, dict] = {}
    for fid in range(sae.n_features):
        active_samples = np.where(Z[:, fid] > 0)[0]
        if active_samples.size < MIN_TOTAL_ACTIVATIONS:
            continue
        # Top-K activating samples.
        sorted_idx = active_samples[np.argsort(-Z[active_samples, fid])][:TOP_K]
        top_classes = labels[sorted_idx]
        if top_classes.size == 0:
            continue
        # Most common class among top activations.
        most_common_class, count = Counter(top_classes.tolist()).most_common(1)[0]
        purity = count / float(top_classes.size)
        if purity >= PURITY_THRESHOLD:
            cls_name = CLASSES[most_common_class]
            auto_labels[str(fid)] = f"{cls_name}:f{fid}"
            per_class_count[cls_name] += 1
            feature_descriptions[fid] = {
                "label": auto_labels[str(fid)],
                "purity": float(purity),
                "n_top_samples": int(top_classes.size),
                "n_total_active": int(active_samples.size),
                "top_examples": [
                    {
                        "class": CLASSES[int(labels[int(j)])],
                        "activation": float(Z[int(j), fid]),
                        "sentence": str(sentences[int(j)]),
                    }
                    for j in sorted_idx[:3]
                ],
            }

    (out_dir / "labels.json").write_text(json.dumps(auto_labels, indent=2))
    (out_dir / "feature_descriptions.json").write_text(
        json.dumps(feature_descriptions, indent=2)
    )

    print(f"  total labelled features         : {len(auto_labels)} / {sae.n_features}  "
          f"({100 * len(auto_labels) / sae.n_features:.2f}%)")
    print(f"  per-class label count:")
    for c in CLASSES:
        print(f"    {c:>10}: {per_class_count.get(c, 0):>4}")
    print()
    print("Example labelled features (highest-purity):")
    sorted_feats = sorted(
        feature_descriptions.items(),
        key=lambda kv: (-kv[1]["purity"], -kv[1]["n_total_active"]),
    )
    for fid, info in sorted_feats[:5]:
        print(f"  feature {fid} -> {info['label']}  "
              f"(purity {info['purity']:.2f}, active on {info['n_total_active']} samples)")
        for ex in info["top_examples"][:2]:
            short = ex["sentence"][:80] + ("..." if len(ex["sentence"]) > 80 else "")
            print(f"    [{ex['class']:>8} act={ex['activation']:5.2f}] {short}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
