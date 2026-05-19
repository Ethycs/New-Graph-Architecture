"""Phase 28 -- build a labelled hypergraph on emotion-corpus regimes using the trained SAE.

Closes the audit-by-construction loop end-to-end:

  1. Reuse the Phase 28b/Step 6b emotion-classification setup:
     harvest GPT-2 block-6 activations per sentence, train a 5-class
     predictive projection, partition by argmax into 5 regimes.
  2. Pass each sentence's activation through the trained SAE
     (`PretrainedSAEAdapter`), get sparse code + named/residual split.
  3. For each regime, aggregate:
       - canonical_signature  = mean activation over regime samples
       - named                = union of SAE-labelled features active in
                                 the regime (with frequency-of-activation)
       - residual             = union of unlabelled active features
  4. Build the `LabelledHypergraph` from the regime structure.
  5. Report the **interpretability fraction** per regime --
     `len(named) / (len(named) + len(residual))` -- the calibrated
     audit-by-construction signal Phase 26 was designed to deliver.

Outputs:
  * runs/phase28_audit/labelled_hypergraph.json -- the hypergraph
    artifact (round-trippable via LabelledHypergraph.from_json).
  * runs/phase28_audit/per_regime_audit.json -- per-regime named/residual
    + interpretability fraction.
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
os.environ["HF_HOME"] = str(Path.home() / "models" / "hf")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402

from nga.arch.labelled_hypergraph import LabelledHypergraph  # noqa: E402
from nga.arch.predictive_projection import (  # noqa: E402
    PredictiveProjection,
    PredictiveProjectionConfig,
    train_predictive_projection,
)
from nga.arch.sae_adapter import PretrainedSAEAdapter  # noqa: E402

from phase27_step6b_emotion_krylov import CLASSES  # noqa: E402

SEED = 42

# Threshold: a feature is "active in this regime" if it fires (post-ReLU > 0)
# on >= this fraction of the regime's samples. Filters noise from rare-firing
# features in small regimes.
REGIME_ACTIVATION_FRACTION_THRESHOLD = 0.20


def main() -> int:
    sae_dir = RUNS_DIR / "phase28_sae"
    out_dir = RUNS_DIR / "phase28_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Phase 28 -- labelled hypergraph end-to-end (real SAE)")
    print("=" * 100)

    # Load harvest + SAE + labels.
    harvest = np.load(sae_dir / "emotion_harvest.npz", allow_pickle=True)
    H = harvest["H"]
    labels = harvest["labels"]
    sentences = harvest["sentences"]
    print(f"  harvest: {H.shape}  classes: {CLASSES}")

    sae = PretrainedSAEAdapter.from_checkpoint(
        sae_dir / "emotion_block6.npz",
        labels_path=sae_dir / "labels.json",
        activation_threshold=0.0,
    )
    n_labelled = len(sae.feature_labels())
    print(f"  SAE: d_in={sae.d_in} n_features={sae.n_features} "
          f"labelled={n_labelled} ({100 * n_labelled / sae.n_features:.2f}%)")

    # Train the 5-class predictive projection on the harvested activations
    # (same as Phase 27 Step 6b -- this is the source of the argmax cells
    # that become regimes).
    cfg = PredictiveProjectionConfig(
        z_dim=32, hidden_dim=64, n_states=len(CLASSES),
        entropy_weight=0.0, failure_weight=0.0,
    )
    proj = PredictiveProjection(input_dim=H.shape[1], config=cfg)
    train_predictive_projection(
        proj, H, next_states=labels.astype(np.int64), entropy_targets=None,
        failure_targets=None, token_ids=None, epochs=60, seed=SEED,
    )
    proj.eval()
    with torch.no_grad():
        argmax_cells = proj.forward(torch.from_numpy(H))["next_state_logits"].argmax(dim=-1).cpu().numpy()
    print(f"  projection acc on train: "
          f"{(argmax_cells == labels).mean():.3f}")

    # Encode all samples through SAE -> dense (n, n_features) post-ReLU code.
    Z = np.zeros((H.shape[0], sae.n_features), dtype=np.float32)
    for i in range(H.shape[0]):
        code = sae.encode(H[i])
        for fid, val in zip(code.active_features, code.activations):
            Z[i, fid] = val

    # Per-regime audit.
    sae_labels_dict = sae.feature_labels()
    per_regime_audit: dict[str, dict] = {}
    canonical_signatures: dict[str, np.ndarray] = {}
    named_per_regime: dict[str, dict[str, str]] = {}
    residual_per_regime: dict[str, list[str]] = {}
    support_counts: dict[str, int] = {}

    regime_ids = [f"R{c}_{CLASSES[c]}" for c in range(len(CLASSES))]
    for c, rid in enumerate(regime_ids):
        mask = argmax_cells == c
        n_in_regime = int(mask.sum())
        if n_in_regime == 0:
            named_per_regime[rid] = {}
            residual_per_regime[rid] = []
            support_counts[rid] = 0
            canonical_signatures[rid] = np.zeros(sae.n_features, dtype=np.float32)
            per_regime_audit[rid] = {
                "support_count": 0,
                "n_named": 0, "n_residual": 0,
                "interpretability_fraction": float("nan"),
            }
            continue
        Z_regime = Z[mask]
        # Active-in-regime: feature fires (>0) on >= threshold fraction of regime samples.
        active_fraction = (Z_regime > 0).mean(axis=0)
        active_in_regime = np.where(active_fraction >= REGIME_ACTIVATION_FRACTION_THRESHOLD)[0]
        # Split into named / residual.
        named: dict[str, str] = {}
        residual: list[str] = []
        for fid in active_in_regime.tolist():
            label = sae_labels_dict.get(int(fid))
            if label is not None:
                named[f"sae_f{fid}"] = label
            else:
                residual.append(f"sae_f{fid}")
        # Also surface the dominant gold class as the FSM-state-like axis.
        regime_class_counts = Counter(labels[mask].tolist())
        dominant_class_id = regime_class_counts.most_common(1)[0][0]
        purity = regime_class_counts[dominant_class_id] / n_in_regime
        named["fsm_state"] = CLASSES[dominant_class_id]
        # Canonical signature: mean SAE activation vector for the regime.
        canonical_signatures[rid] = Z_regime.mean(axis=0).astype(np.float32)
        named_per_regime[rid] = named
        residual_per_regime[rid] = residual
        support_counts[rid] = n_in_regime
        interp_frac = (len(named) - 1) / max(1, len(named) - 1 + len(residual))
        per_regime_audit[rid] = {
            "support_count": n_in_regime,
            "dominant_class": CLASSES[dominant_class_id],
            "dominant_class_purity": float(purity),
            "n_named_sae_features": len(named) - 1,  # exclude fsm_state
            "n_residual_sae_features": len(residual),
            "interpretability_fraction_sae_only": float(interp_frac),
            "named_sample": dict(list(named.items())[:8]),
        }

    # Edges: count co-occurrences between consecutive regime visits per sample.
    # The emotion corpus is a flat 100-sentence set with no temporal order
    # beyond the shuffle, so this just demonstrates the edge-count plumbing.
    edge_counts: Counter[tuple[str, str]] = Counter()
    # Use original (shuffled) order as a stand-in for "trajectory".
    last_regime: str | None = None
    for i in range(H.shape[0]):
        cur = regime_ids[int(argmax_cells[i])]
        if last_regime is not None and last_regime != cur:
            edge_counts[(last_regime, cur)] += 1
        last_regime = cur
    edges = list(edge_counts.keys())
    edge_count_dict = dict(edge_counts)

    hyp = LabelledHypergraph.from_pcg_graph(
        regime_ids=regime_ids,
        edges=edges,
        edge_counts=edge_count_dict,
        canonical_signatures=canonical_signatures,
        named_labels=named_per_regime,
        residual_features=residual_per_regime,
        support_counts=support_counts,
        metadata={
            "phase": "28",
            "source": "scripts/phase28_build_labelled_hypergraph.py",
            "sae_checkpoint": str(sae_dir / "emotion_block6.npz"),
            "n_total_samples": int(H.shape[0]),
            "sae_n_features": int(sae.n_features),
            "sae_n_labelled": int(n_labelled),
            "regime_activation_fraction_threshold": REGIME_ACTIVATION_FRACTION_THRESHOLD,
        },
    )

    hyp_path = out_dir / "labelled_hypergraph.json"
    hyp_path.write_text(hyp.to_json())

    audit_summary = {
        "regime_ids": regime_ids,
        "n_regimes": len(regime_ids),
        "n_edges": len(edges),
        "per_regime": per_regime_audit,
        "sae_n_features": int(sae.n_features),
        "sae_n_labelled": int(n_labelled),
        "regime_activation_fraction_threshold": REGIME_ACTIVATION_FRACTION_THRESHOLD,
    }
    (out_dir / "per_regime_audit.json").write_text(json.dumps(audit_summary, indent=2))

    print(f"  wrote {hyp_path}")
    print(f"  wrote {out_dir / 'per_regime_audit.json'}")
    print()
    print(f"{'regime':>20} | {'support':>7} | {'dom class':>10} | {'purity':>7} | "
          f"{'named':>5} | {'residual':>8} | {'interp_frac':>11}")
    print(f"{'-' * 20} | {'-' * 7} | {'-' * 10} | {'-' * 7} | {'-' * 5} | {'-' * 8} | {'-' * 11}")
    for rid in regime_ids:
        info = per_regime_audit[rid]
        print(f"{rid:>20} | {info['support_count']:>7} | "
              f"{info.get('dominant_class', 'n/a'):>10} | "
              f"{info.get('dominant_class_purity', float('nan')):>7.3f} | "
              f"{info['n_named_sae_features']:>5} | "
              f"{info['n_residual_sae_features']:>8} | "
              f"{info['interpretability_fraction_sae_only']:>11.3f}")
    print()
    print(f"Headline: regime structure has named={any(info['n_named_sae_features'] > 0 for info in per_regime_audit.values())} "
          f"AND residual={any(info['n_residual_sae_features'] > 0 for info in per_regime_audit.values())} simultaneously.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
