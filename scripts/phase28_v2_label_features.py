"""Phase 28 v2 -- auto-label the WikiText-trained SAE features by emotion-corpus purity.

The v2 SAE was trained on WikiText (broad-distribution natural text). To
make a fair comparison against v1's audit performance, we label v2's
features using the SAME emotion corpus that v1 was labelled on. The
question this answers: does an SAE trained on a broader corpus produce
*more* features that fire informatively on emotion-classification, even
though it was never trained on emotion data?

Procedure (identical to phase28_label_top_features.py except for the
input SAE checkpoint):

  1. Load v2 SAE from runs/phase28_v2_sae/wikitext_block6.npz.
  2. Load the emotion-corpus harvest (re-use v1's harvest --
     runs/phase28_sae/emotion_harvest.npz, 100 sentences x 768 d).
  3. Encode every sentence through v2 SAE -> sparse code.
  4. For each feature, find top-K activating samples; if >= purity
     threshold from one class, label it ``<class>:f<id>``.
  5. Save labels.json + feature_descriptions.json.

Output:
  * runs/phase28_v2_sae/labels.json
  * runs/phase28_v2_sae/feature_descriptions.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from nga.arch.sae_adapter import PretrainedSAEAdapter  # noqa: E402

from phase27_step6b_emotion_krylov import CLASSES  # noqa: E402

PURITY_THRESHOLD = 0.65
TOP_K = 5
MIN_TOTAL_ACTIVATIONS = 3


def main() -> int:
    out_dir = RUNS_DIR / "phase28_v2_sae"
    sae = PretrainedSAEAdapter.from_checkpoint(
        out_dir / "wikitext_block6.npz",
        activation_threshold=0.0,
    )
    harvest = np.load(RUNS_DIR / "phase28_sae" / "emotion_harvest.npz", allow_pickle=True)
    H = harvest["H"]
    labels = harvest["labels"]
    sentences = harvest["sentences"]

    print("=" * 100)
    print("Phase 28 v2 -- auto-label v2 (WikiText-trained) SAE features on emotion corpus")
    print(f"  SAE features      : {sae.n_features}")
    print(f"  n_samples         : {H.shape[0]}")
    print(f"  classes           : {CLASSES}")
    print(f"  purity_threshold  : {PURITY_THRESHOLD}")
    print(f"  top_k             : {TOP_K}")
    print(f"  min_total_active  : {MIN_TOTAL_ACTIVATIONS}")
    print("=" * 100)

    Z = np.zeros((H.shape[0], sae.n_features), dtype=np.float32)
    for i in range(H.shape[0]):
        code = sae.encode(H[i])
        for fid, val in zip(code.active_features, code.activations):
            Z[i, fid] = val

    print(f"  total active features (any sample): "
          f"{int((Z > 0).any(axis=0).sum())}/{sae.n_features}")
    print(f"  mean active per sample            : "
          f"{float((Z > 0).sum(axis=1).mean()):.1f}")

    auto_labels: dict[str, str] = {}
    per_class_count: Counter[str] = Counter()
    feature_descriptions: dict[int, dict] = {}
    for fid in range(sae.n_features):
        active_samples = np.where(Z[:, fid] > 0)[0]
        if active_samples.size < MIN_TOTAL_ACTIVATIONS:
            continue
        sorted_idx = active_samples[np.argsort(-Z[active_samples, fid])][:TOP_K]
        top_classes = labels[sorted_idx]
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

    print(f"  total labelled features  : {len(auto_labels)} / {sae.n_features}  "
          f"({100 * len(auto_labels) / sae.n_features:.2f}%)")
    print(f"  per-class label count    :")
    for c in CLASSES:
        print(f"    {c:>10}: {per_class_count.get(c, 0):>4}")

    # v1 vs v2 quick comparison.
    v1_labels_path = RUNS_DIR / "phase28_sae" / "labels.json"
    if v1_labels_path.exists():
        v1_labels = json.loads(v1_labels_path.read_text())
        # Load v1 SAE feature count for the rate calc.
        v1_log = json.loads((RUNS_DIR / "phase28_sae" / "training_log.json").read_text())
        v1_n_features = v1_log["n_features"]
        v1_n_train = v1_log["n_samples"]
        print()
        print(f"{'SAE':>8} | {'n_features':>10} | {'n_train_activations':>20} | "
              f"{'features labelled':>17} | {'%':>6}")
        print(f"{'-' * 8} | {'-' * 10} | {'-' * 20} | {'-' * 17} | {'-' * 6}")
        print(f"{'v1':>8} | {v1_n_features:>10} | {v1_n_train:>20} | "
              f"{len(v1_labels):>17} | {100 * len(v1_labels) / v1_n_features:>5.2f}%")
        v2_log = json.loads((RUNS_DIR / "phase28_v2_sae" / "training_log.json").read_text())
        v2_n_train = v2_log["n_train_activations"]
        print(f"{'v2':>8} | {sae.n_features:>10} | {v2_n_train:>20} | "
              f"{len(auto_labels):>17} | {100 * len(auto_labels) / sae.n_features:>5.2f}%")
    print()
    sorted_feats = sorted(
        feature_descriptions.items(),
        key=lambda kv: (-kv[1]["purity"], -kv[1]["n_total_active"]),
    )
    print("Top-5 labelled features (v2):")
    for fid, info in sorted_feats[:5]:
        print(f"  feature {fid:>4} -> {info['label']}  "
              f"(purity {info['purity']:.2f}, active on {info['n_total_active']} samples)")
        for ex in info["top_examples"][:2]:
            short = ex["sentence"][:80] + ("..." if len(ex["sentence"]) > 80 else "")
            print(f"    [{ex['class']:>8} act={ex['activation']:5.2f}] {short}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
