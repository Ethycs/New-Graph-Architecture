"""Phase 28b - policy-intent FSM extraction sweep (v1 + v2).

Runs the E31 PCG-X pipeline on the v1 and v2 policy-intent FSMs at the
default Phase 24 configuration (GPT-2 small, block 6, 80 policies, 30
projection epochs) and reports the pre-registered acceptance bars from
``docs/proposals/policy-intent-fsm-extraction.md``:

    A1  -- v1 binary sanity         eff_rank(grad) < 1.5 AND top-3 > 0.95
    A2  -- v2 4-state headline      2.0 <= eff_rank(grad) <= 4.0
                                    AND top-3 >= 0.85
    A3  -- cluster purity vs gold   mean_purity_against_current_state
                                    >= 0.75 averaged across policies

A4 (ABSTAIN-on-disagreement), A5 (anchor-ablation), and A6 (v1->v2
migration) are downstream steps that compose this sweep's outputs;
they live in follow-up scripts. This sweep delivers A1+A2+A3.

Outputs:
  * ``runs/E31_policy_intent_v1/`` -- standard E30-shape run dir.
  * ``runs/E31_policy_intent_v2/`` -- standard E30-shape run dir.
  * ``runs/phase28b_policy_intent_sweep.json`` -- consolidated summary
    + acceptance-bar verdict + Krylov measurement on each run's
    harvested activations.

The gradient-Krylov measurement mirrors ``phase27_step4_krylov.py`` but
operates on the projection trained inside ``run_e31`` (rather than
training a fresh one). Since we don't currently persist the trained
projection to disk, this script re-runs the projection training on the
harvested activations -- ~5 seconds extra per grammar on CPU.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
os.environ["HF_HOME"] = str(Path.home() / "models" / "hf")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from nga.arch.predictive_projection import (  # noqa: E402
    PredictiveProjection,
    PredictiveProjectionConfig,
    train_predictive_projection,
)
from nga.exp.dataset_policy_intent import generate_policy_intent_dataset  # noqa: E402
from nga.exp.e30_pcg_extractor_pretrained import _PretrainedSubstrate  # noqa: E402
from nga.exp.e31_policy_intent_extraction import _load_policy_fsm, run_e31  # noqa: E402

SEED = 42
N_POLICIES = 80
PROJECTION_EPOCHS = 30
HARVEST_LAYER = 6


# ---------------------------------------------------------------------------
# Krylov measurement on the harvested activations
# (mirrors phase27_step4_krylov.py but locally scoped)
# ---------------------------------------------------------------------------


def _gradient_matrix(projection: PredictiveProjection, h: np.ndarray) -> np.ndarray:
    n, d = h.shape
    G = np.zeros((n, d), dtype=np.float32)
    projection.eval()
    for i in range(n):
        h_i = torch.from_numpy(h[i:i + 1]).requires_grad_(True)
        out = projection.forward(h_i)
        logits = out["next_state_logits"][0]
        top2 = torch.topk(logits, k=min(2, logits.numel()), largest=True, sorted=True)
        if top2.values.numel() < 2:
            margin = top2.values[0]
        else:
            margin = top2.values[0] - top2.values[1]
        margin.backward()
        G[i] = h_i.grad.detach().cpu().numpy()[0]
    return G


def _krylov_stats(G: np.ndarray) -> dict:
    _U, sigmas, _Vt = np.linalg.svd(G, full_matrices=False)
    variances = sigmas ** 2
    total = float(variances.sum())
    p = variances / max(total, 1e-30)
    nonzero = p > 0
    H_entropy = float(-(p[nonzero] * np.log(p[nonzero])).sum())
    return {
        "n_singular_values": int(sigmas.size),
        "total_variance": total,
        "effective_rank": float(np.exp(H_entropy)),
        "var_frac_top3": float(p[:3].sum()),
        "var_frac_top5": float(p[:5].sum()),
        "var_frac_top10": float(p[:10].sum()),
        "sigma_top_k": [float(s) for s in sigmas[:10]],
    }


def _measure_krylov(version: str, n_policies: int, seed: int, rendering: str = "abstract") -> dict:
    """Re-train a small projection on harvested activations and SVD the gradient.

    Independent harvest from run_e31 -- we just want the gradient subspace
    of the trained projection. Total wall-clock ~10 s for the 80-policy
    case on a 6 GB GPU.
    """
    fsm = _load_policy_fsm(version)
    ds = generate_policy_intent_dataset(
        fsm, n_policies=n_policies, seed=seed, version=version, rendering=rendering
    )
    substrate = _PretrainedSubstrate(harvest_layer=HARVEST_LAYER)

    # Group by sequence_id, harvest one h per step.
    seq_to_indices: dict[int, list[int]] = {}
    for i, s in enumerate(ds.samples):
        seq_to_indices.setdefault(s.sequence_id, []).append(i)

    H_all = np.zeros((len(ds.samples), substrate.hidden_size), dtype=np.float32)
    for seq, indices in seq_to_indices.items():
        ordered = sorted(indices)
        tokens = [ds.samples[i].observed_token for i in ordered]
        per_step, _truncated = substrate.harvest_program(tokens)
        for k, original in enumerate(ordered):
            H_all[original] = per_step[k].astype(np.float32)

    # Train the projection h -> z -> V_states.
    cfg = PredictiveProjectionConfig(
        z_dim=32,
        hidden_dim=64,
        n_states=fsm.vertex_count,
        entropy_weight=0.0,
        failure_weight=0.0,
    )
    projection = PredictiveProjection(input_dim=substrate.hidden_size, config=cfg)
    train_predictive_projection(
        projection,
        H_all,
        next_states=ds.y_next,
        entropy_targets=None,
        failure_targets=None,
        token_ids=None,
        epochs=PROJECTION_EPOCHS,
        seed=seed,
    )

    G = _gradient_matrix(projection, H_all)
    kry = _krylov_stats(G)
    kry["n_samples"] = int(H_all.shape[0])
    kry["V"] = int(fsm.vertex_count)
    return kry


# ---------------------------------------------------------------------------
# Acceptance bars
# ---------------------------------------------------------------------------


def _check_acceptance(v1: dict, v2: dict) -> dict:
    """Apply pre-registered A1/A2/A3 bars to the sweep summary."""
    a1_eff_rank = v1["kry"]["effective_rank"] < 1.5
    a1_top3 = v1["kry"]["var_frac_top3"] > 0.95
    a1_pass = a1_eff_rank and a1_top3

    a2_eff_rank = 2.0 <= v2["kry"]["effective_rank"] <= 4.0
    a2_top3 = v2["kry"]["var_frac_top3"] >= 0.85
    a2_pass = a2_eff_rank and a2_top3

    a3_v1 = v1["run"]["mean_purity_against_current_state"] >= 0.75
    a3_v2 = v2["run"]["mean_purity_against_current_state"] >= 0.75
    a3_pass = a3_v1 and a3_v2

    return {
        "A1_v1_binary_sanity": {
            "eff_rank_below_1.5": a1_eff_rank,
            "top3_above_0.95": a1_top3,
            "eff_rank_value": v1["kry"]["effective_rank"],
            "top3_value": v1["kry"]["var_frac_top3"],
            "PASS": a1_pass,
        },
        "A2_v2_4state_headline": {
            "eff_rank_in_[2.0,4.0]": a2_eff_rank,
            "top3_above_0.85": a2_top3,
            "eff_rank_value": v2["kry"]["effective_rank"],
            "top3_value": v2["kry"]["var_frac_top3"],
            "PASS": a2_pass,
        },
        "A3_cluster_purity_vs_gold": {
            "v1_purity_above_0.75": a3_v1,
            "v2_purity_above_0.75": a3_v2,
            "v1_purity": v1["run"]["mean_purity_against_current_state"],
            "v2_purity": v2["run"]["mean_purity_against_current_state"],
            "PASS": a3_pass,
        },
        "headline_passes_count": int(a1_pass) + int(a2_pass) + int(a3_pass),
    }


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def _run_one(version: str, rendering: str = "abstract") -> dict:
    suffix = "" if rendering == "abstract" else "_templated"
    out_dir = RUNS_DIR / f"E31_policy_intent_{version}{suffix}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    t = time.perf_counter()
    result = run_e31(
        version=version,
        rendering=rendering,
        run_id=f"E31_policy_intent_{version}{suffix}",
        output_dir=out_dir,
        seed=SEED,
        n_policies=N_POLICIES,
        n_projection_train_epochs=PROJECTION_EPOCHS,
    )
    run_time = time.perf_counter() - t

    t2 = time.perf_counter()
    kry = _measure_krylov(version, N_POLICIES, SEED, rendering=rendering)
    kry_time = time.perf_counter() - t2

    return {
        "version": version,
        "rendering": rendering,
        "out_dir": str(out_dir),
        "run": {
            "V_ground_truth": result.V_ground_truth,
            "n_argmax_cells": result.n_argmax_cells,
            "n_regimes_after_merge": result.n_regimes_after_merge,
            "n_total_steps": result.n_total_steps,
            "mean_purity_against_current_state": result.mean_purity_against_current_state,
            "mean_failure_rate": result.mean_failure_rate,
            "mean_entropy": result.mean_entropy,
            "mean_margin_to_tie": result.mean_margin_to_tie,
            "projection_next_state_accuracy": result.projection_next_state_accuracy,
            "total_wall_clock_seconds": run_time,
        },
        "kry": kry,
        "kry_wall_clock_seconds": kry_time,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rendering",
        choices=("abstract", "templated", "both"),
        default="both",
        help="abstract = raw event-label tokens; templated = natural-language renderings; "
        "both = run both and compare side-by-side (default).",
    )
    args = parser.parse_args()

    print("=" * 100)
    print(f"Phase 28b -- policy-intent FSM extraction (v1 + v2)")
    print(f"  substrate    : frozen GPT-2 small (block {HARVEST_LAYER}/12)")
    print(f"  n_policies   : {N_POLICIES} per version")
    print(f"  proj epochs  : {PROJECTION_EPOCHS}")
    print(f"  seed         : {SEED}")
    print(f"  rendering    : {args.rendering}")
    print("=" * 100)

    t_start = time.perf_counter()
    if args.rendering in {"abstract", "both"}:
        v1 = _run_one("v1", rendering="abstract")
        print()
        v2 = _run_one("v2", rendering="abstract")
        acceptance = _check_acceptance(v1, v2)
    else:
        v1 = None
        v2 = None
        acceptance = None

    if args.rendering in {"templated", "both"}:
        print()
        v1_t = _run_one("v1", rendering="templated")
        print()
        v2_t = _run_one("v2", rendering="templated")
        acceptance_templated = _check_acceptance(v1_t, v2_t)
    else:
        v1_t = None
        v2_t = None
        acceptance_templated = None
    total = time.perf_counter() - t_start

    summary = {
        "seed": SEED,
        "n_policies": N_POLICIES,
        "projection_epochs": PROJECTION_EPOCHS,
        "harvest_layer": HARVEST_LAYER,
        "rendering_mode": args.rendering,
        "abstract": {
            "v1": v1, "v2": v2, "acceptance": acceptance,
        } if v1 is not None else None,
        "templated": {
            "v1": v1_t, "v2": v2_t, "acceptance": acceptance_templated,
        } if v1_t is not None else None,
        "total_wall_clock_seconds": total,
    }
    suffix = f"_{args.rendering}" if args.rendering != "both" else ""
    out_json = RUNS_DIR / f"phase28b_policy_intent_sweep{suffix}.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("=" * 100)
    print(f"  wrote {out_json}")
    print(f"  total wall-clock: {total:.1f}s")
    print()

    def _print_block(title: str, blk: dict, accept: dict) -> None:
        if blk is None:
            return
        bv1 = blk["v1"]; bv2 = blk["v2"]
        print(f"== {title} ==")
        print(f"{'metric':>34} | {'v1 binary':>14} | {'v2 four-state':>14}")
        print(f"{'-' * 34} | {'-' * 14} | {'-' * 14}")
        print(f"{'V_ground_truth':>34} | {bv1['run']['V_ground_truth']:>14} | {bv2['run']['V_ground_truth']:>14}")
        print(f"{'argmax cells':>34} | {bv1['run']['n_argmax_cells']:>14} | {bv2['run']['n_argmax_cells']:>14}")
        print(f"{'regimes after merge':>34} | {bv1['run']['n_regimes_after_merge']:>14} | {bv2['run']['n_regimes_after_merge']:>14}")
        print(f"{'n_total_steps':>34} | {bv1['run']['n_total_steps']:>14} | {bv2['run']['n_total_steps']:>14}")
        print(f"{'mean_purity_vs_current_state':>34} | "
              f"{bv1['run']['mean_purity_against_current_state']:>14.4f} | "
              f"{bv2['run']['mean_purity_against_current_state']:>14.4f}")
        print(f"{'projection_next_state_acc':>34} | "
              f"{bv1['run']['projection_next_state_accuracy']:>14.4f} | "
              f"{bv2['run']['projection_next_state_accuracy']:>14.4f}")
        print(f"{'eff_rank(grad)':>34} | "
              f"{bv1['kry']['effective_rank']:>14.4f} | "
              f"{bv2['kry']['effective_rank']:>14.4f}")
        print(f"{'top-3 capture':>34} | "
              f"{bv1['kry']['var_frac_top3']:>14.4f} | "
              f"{bv2['kry']['var_frac_top3']:>14.4f}")
        print(f"{'top-5 capture':>34} | "
              f"{bv1['kry']['var_frac_top5']:>14.4f} | "
              f"{bv2['kry']['var_frac_top5']:>14.4f}")
        print()
        print(f"Acceptance bars ({title}):")
        for bar_name, bar in accept.items():
            if bar_name == "headline_passes_count":
                continue
            verdict = "PASS" if bar["PASS"] else "FAIL"
            print(f"  {bar_name:>32}: {verdict}  ({bar})")
        print(f"  Headline pass count: {accept['headline_passes_count']} / 3")
        print()

    if v1 is not None:
        _print_block("ABSTRACT tokens", summary["abstract"], acceptance)
    if v1_t is not None:
        _print_block("TEMPLATED tokens", summary["templated"], acceptance_templated)

    if v1 is not None and v1_t is not None:
        print("== ABSTRACT vs TEMPLATED side-by-side ==")
        for ver, abs_blk, tmp_blk in [("v1", v1, v1_t), ("v2", v2, v2_t)]:
            print(f"  {ver}:")
            print(f"    purity:    abstract={abs_blk['run']['mean_purity_against_current_state']:.4f}"
                  f"  templated={tmp_blk['run']['mean_purity_against_current_state']:.4f}"
                  f"  Δ={tmp_blk['run']['mean_purity_against_current_state'] - abs_blk['run']['mean_purity_against_current_state']:+.4f}")
            print(f"    eff_rank:  abstract={abs_blk['kry']['effective_rank']:.4f}"
                  f"  templated={tmp_blk['kry']['effective_rank']:.4f}"
                  f"  Δ={tmp_blk['kry']['effective_rank'] - abs_blk['kry']['effective_rank']:+.4f}")
            print(f"    top-3:     abstract={abs_blk['kry']['var_frac_top3']:.4f}"
                  f"  templated={tmp_blk['kry']['var_frac_top3']:.4f}"
                  f"  Δ={tmp_blk['kry']['var_frac_top3'] - abs_blk['kry']['var_frac_top3']:+.4f}")
            print(f"    next-acc:  abstract={abs_blk['run']['projection_next_state_accuracy']:.4f}"
                  f"  templated={tmp_blk['run']['projection_next_state_accuracy']:.4f}"
                  f"  Δ={tmp_blk['run']['projection_next_state_accuracy'] - abs_blk['run']['projection_next_state_accuracy']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
