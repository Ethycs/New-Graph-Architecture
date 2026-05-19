"""Phase 28 v2 -- build labelled hypergraph with v2 SAE and compare to v1.

Mirrors `phase28_build_labelled_hypergraph.py` but uses the v2 SAE
(WikiText-trained) instead of the v1 SAE (emotion-corpus-trained).
Saves the v2 audit artifact to `runs/phase28_v2_audit/` and prints a
side-by-side comparison of v1 vs v2 per-regime interpretability
fractions and total named/residual counts.

The central question: does a larger SAE trained on broader text deliver
a higher interpretability fraction per regime when audited on the same
emotion corpus? This is the structural payoff of "swap in a better SAE
without changing the rest of the pipeline."

Output:
  * runs/phase28_v2_audit/labelled_hypergraph.json
  * runs/phase28_v2_audit/per_regime_audit.json
  * stdout: side-by-side v1 vs v2 table
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
REGIME_ACTIVATION_FRACTION_THRESHOLD = 0.20


def _build_audit_for_sae(sae: PretrainedSAEAdapter, H, labels, regime_ids, argmax_cells):
    """Return per-regime audit dict + named/residual/canonical_sigs."""
    sae_labels_dict = sae.feature_labels()
    Z = np.zeros((H.shape[0], sae.n_features), dtype=np.float32)
    for i in range(H.shape[0]):
        code = sae.encode(H[i])
        for fid, val in zip(code.active_features, code.activations):
            Z[i, fid] = val

    per_regime: dict[str, dict] = {}
    canonical_signatures: dict[str, np.ndarray] = {}
    named_per_regime: dict[str, dict[str, str]] = {}
    residual_per_regime: dict[str, list[str]] = {}
    support_counts: dict[str, int] = {}

    for c, rid in enumerate(regime_ids):
        mask = argmax_cells == c
        n = int(mask.sum())
        if n == 0:
            per_regime[rid] = {"support_count": 0, "n_named": 0, "n_residual": 0,
                               "interp_frac": float("nan")}
            canonical_signatures[rid] = np.zeros(sae.n_features, dtype=np.float32)
            named_per_regime[rid] = {}
            residual_per_regime[rid] = []
            support_counts[rid] = 0
            continue
        Z_regime = Z[mask]
        active_fraction = (Z_regime > 0).mean(axis=0)
        active_in_regime = np.where(active_fraction >= REGIME_ACTIVATION_FRACTION_THRESHOLD)[0]
        named: dict[str, str] = {}
        residual: list[str] = []
        for fid in active_in_regime.tolist():
            label = sae_labels_dict.get(int(fid))
            if label is not None:
                named[f"sae_f{fid}"] = label
            else:
                residual.append(f"sae_f{fid}")
        regime_class_counts = Counter(labels[mask].tolist())
        dominant_class_id = regime_class_counts.most_common(1)[0][0]
        purity = regime_class_counts[dominant_class_id] / n
        named["fsm_state"] = CLASSES[dominant_class_id]
        canonical_signatures[rid] = Z_regime.mean(axis=0).astype(np.float32)
        named_per_regime[rid] = named
        residual_per_regime[rid] = residual
        support_counts[rid] = n
        n_named_sae = len(named) - 1
        n_residual = len(residual)
        interp_frac = n_named_sae / max(1, n_named_sae + n_residual)
        per_regime[rid] = {
            "support_count": n,
            "dominant_class": CLASSES[dominant_class_id],
            "dominant_class_purity": float(purity),
            "n_named_sae_features": n_named_sae,
            "n_residual_sae_features": n_residual,
            "interpretability_fraction_sae_only": float(interp_frac),
        }
    return per_regime, named_per_regime, residual_per_regime, canonical_signatures, support_counts


def main() -> int:
    out_dir = RUNS_DIR / "phase28_v2_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    sae_v1 = PretrainedSAEAdapter.from_checkpoint(
        RUNS_DIR / "phase28_sae" / "emotion_block6.npz",
        labels_path=RUNS_DIR / "phase28_sae" / "labels.json",
    )
    sae_v2 = PretrainedSAEAdapter.from_checkpoint(
        RUNS_DIR / "phase28_v2_sae" / "wikitext_block6.npz",
        labels_path=RUNS_DIR / "phase28_v2_sae" / "labels.json",
    )

    harvest = np.load(RUNS_DIR / "phase28_sae" / "emotion_harvest.npz", allow_pickle=True)
    H = harvest["H"]
    labels = harvest["labels"]
    print("=" * 100)
    print("Phase 28 v2 audit -- v1 SAE vs v2 SAE on the same emotion regimes")
    print(f"  v1 SAE: n_features={sae_v1.n_features}, labelled={len(sae_v1.feature_labels())}")
    print(f"  v2 SAE: n_features={sae_v2.n_features}, labelled={len(sae_v2.feature_labels())}")
    print(f"  emotion corpus: {H.shape}")
    print("=" * 100)

    # Re-train the predictive projection (same as Phase 27 Step 6b / v1 audit).
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
        argmax = proj.forward(torch.from_numpy(H))["next_state_logits"].argmax(dim=-1).cpu().numpy()

    regime_ids = [f"R{c}_{CLASSES[c]}" for c in range(len(CLASSES))]

    pr_v1, named_v1, residual_v1, sigs_v1, supp_v1 = _build_audit_for_sae(
        sae_v1, H, labels, regime_ids, argmax,
    )
    pr_v2, named_v2, residual_v2, sigs_v2, supp_v2 = _build_audit_for_sae(
        sae_v2, H, labels, regime_ids, argmax,
    )

    # Save v2 hypergraph artifact.
    hyp = LabelledHypergraph.from_pcg_graph(
        regime_ids=regime_ids,
        canonical_signatures=sigs_v2,
        named_labels=named_v2,
        residual_features=residual_v2,
        support_counts=supp_v2,
        metadata={
            "phase": "28-v2",
            "sae_checkpoint": str(RUNS_DIR / "phase28_v2_sae" / "wikitext_block6.npz"),
            "sae_labels_path": str(RUNS_DIR / "phase28_v2_sae" / "labels.json"),
            "regime_activation_fraction_threshold": REGIME_ACTIVATION_FRACTION_THRESHOLD,
        },
    )
    (out_dir / "labelled_hypergraph.json").write_text(hyp.to_json())
    (out_dir / "per_regime_audit.json").write_text(json.dumps({
        "v1": pr_v1, "v2": pr_v2, "regime_ids": regime_ids,
    }, indent=2))
    print(f"  wrote {out_dir / 'labelled_hypergraph.json'}")
    print(f"  wrote {out_dir / 'per_regime_audit.json'}")
    print()

    # Side-by-side table.
    print(f"{'regime':>20} | {'v1 named':>8} | {'v1 resid':>8} | {'v1 frac':>8} | "
          f"{'v2 named':>8} | {'v2 resid':>8} | {'v2 frac':>8} | {'Δ frac':>7}")
    print(f"{'-' * 20} | {'-' * 8} | {'-' * 8} | {'-' * 8} | "
          f"{'-' * 8} | {'-' * 8} | {'-' * 8} | {'-' * 7}")
    for rid in regime_ids:
        a = pr_v1[rid]; b = pr_v2[rid]
        delta = b["interpretability_fraction_sae_only"] - a["interpretability_fraction_sae_only"]
        print(f"{rid:>20} | "
              f"{a['n_named_sae_features']:>8} | {a['n_residual_sae_features']:>8} | "
              f"{a['interpretability_fraction_sae_only']:>8.3f} | "
              f"{b['n_named_sae_features']:>8} | {b['n_residual_sae_features']:>8} | "
              f"{b['interpretability_fraction_sae_only']:>8.3f} | "
              f"{delta:>+7.3f}")

    # Aggregates.
    v1_frac_mean = np.mean([pr_v1[r]["interpretability_fraction_sae_only"] for r in regime_ids])
    v2_frac_mean = np.mean([pr_v2[r]["interpretability_fraction_sae_only"] for r in regime_ids])
    v1_named_total = sum(pr_v1[r]["n_named_sae_features"] for r in regime_ids)
    v2_named_total = sum(pr_v2[r]["n_named_sae_features"] for r in regime_ids)
    v1_residual_total = sum(pr_v1[r]["n_residual_sae_features"] for r in regime_ids)
    v2_residual_total = sum(pr_v2[r]["n_residual_sae_features"] for r in regime_ids)
    print()
    print(f"{'aggregate':>20} | "
          f"{v1_named_total:>8} | {v1_residual_total:>8} | {v1_frac_mean:>8.3f} | "
          f"{v2_named_total:>8} | {v2_residual_total:>8} | {v2_frac_mean:>8.3f} | "
          f"{v2_frac_mean - v1_frac_mean:>+7.3f}")
    print()
    print(f"Headline: v2 mean interpretability fraction = {v2_frac_mean:.3f}  "
          f"(v1 = {v1_frac_mean:.3f}, Δ = {v2_frac_mean - v1_frac_mean:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
