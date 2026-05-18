"""Phase 27 Step 4 -- gradient-Krylov essential subspace of sigma at layer 6.

The framework's 3-d marching claim is about the **sigma-relevant subspace**,
not the raw activation-variance subspace (Phase 27 Step 3 measured the
wrong thing). This script implements the original Phase 27 Step 2 from
docs/interpretability-push.md:

  > Compute the Krylov essential subspace at layer L10. Gradient of sigma
  > with respect to substrate activations + HVP of the projection's loss.
  > Extract top-3 directions. Project all corpus activations into the 3-d
  > subspace.

Adapted for the actual deployment:

* Substrate is GPT-2 small at block 6 (matches Phase 24 / Phase 25 default).
* Sigma is replaced by its smooth, h-dependent component: the predictive
  projection's margin = top1_logit - top2_logit. The discrete pieces of
  sigma (illegal, loop) don't depend on h smoothly so they have no
  gradient signal.
* "Krylov subspace" is realised here as the top-3 right singular vectors
  of the gradient matrix G = [grad_h margin_i]_i. This is equivalent to
  3 steps of Lanczos on A = G^T G starting from a random unit vector,
  and is what "Krylov essential subspace" means in the optimization /
  loss-landscape literature when applied to gradients of a scalar loss.

Outputs:

* runs/phase27_step4_krylov.json -- per-grammar top-10 gradient singular
  values + variance captured in top-3 / top-5 / top-10 + 3-d coordinates
  for every harvested step + argmax regime label.
* runs/phase27_step4_krylov/{grammar}_grad_spectrum.png -- gradient
  scree plot + cumulative variance.
* runs/phase27_step4_krylov/{grammar}_3d_projection.png -- 3-d scatter of
  activations projected into the Krylov subspace, colored by argmax
  regime ID.
"""
from __future__ import annotations

import json
import os
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

from nga.arch.graph_fsm import GraphFSM  # noqa: E402
from nga.arch.predictive_projection import (  # noqa: E402
    PredictiveProjection,
    PredictiveProjectionConfig,
    train_predictive_projection,
)
from nga.drivers import graph_fsm_spec as graph_fsm_spec_mod  # noqa: E402
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH  # noqa: E402
from nga.exp.e30_pcg_extractor_pretrained import _PretrainedSubstrate  # noqa: E402

SEED = 42
GRAMMARS = ["listops", "python_expr", "python_big", "json", "python_control"]
N_PROGRAMS = 80
HARVEST_LAYER = 6
PROJECTION_EPOCHS = 30
Z_DIM = 32
PROJ_HIDDEN = 64


def harvest(grammar: str, fsm: GraphFSM, substrate) -> tuple[np.ndarray, np.ndarray, list]:
    dispatch = GRAMMAR_DISPATCH[grammar]
    train_ds = dispatch["loader"](fsm, N_PROGRAMS, SEED)
    seq_attr = dispatch["sequence_id_attr"]
    n_total = int(train_ds.X.shape[0])
    program_ids = np.asarray(
        [int(getattr(s, seq_attr)) for s in train_ds.samples], dtype=np.int64
    )
    program_to_indices: dict[int, list[int]] = {}
    for i, pid in enumerate(program_ids):
        program_to_indices.setdefault(int(pid), []).append(i)
    h_all = np.zeros((n_total, substrate.hidden_size), dtype=np.float32)
    for pid, indices in program_to_indices.items():
        ordered = sorted(indices)
        observed_tokens = [train_ds.samples[i].observed_token for i in ordered]
        per_step, _was_truncated = substrate.harvest_program(observed_tokens)
        for k, original_idx in enumerate(ordered):
            h_all[original_idx] = per_step[k].astype(np.float32)
    next_states = train_ds.y_next.astype(np.int64)
    return h_all, next_states, train_ds.samples


def gradient_matrix(projection: PredictiveProjection, h: np.ndarray, device) -> np.ndarray:
    """Per-row gradient of margin w.r.t. h (shape (N, D), float32, on cpu)."""
    n, d = h.shape
    G = np.zeros((n, d), dtype=np.float32)
    projection.eval()
    # Margin is the top1 - top2 logit gap of the next-state head. Compute it
    # one step at a time with grad enabled on h_i; this is O(N) forward+backward
    # passes through a tiny MLP, well under a minute even at N ~ 2000.
    for i in range(n):
        h_i = torch.from_numpy(h[i:i + 1]).to(device).requires_grad_(True)
        out = projection.forward(h_i)
        logits = out["next_state_logits"][0]
        top2 = torch.topk(logits, k=2, largest=True, sorted=True)
        margin = top2.values[0] - top2.values[1]
        margin.backward()
        G[i] = h_i.grad.detach().cpu().numpy()[0]
    return G


def krylov_stats(G: np.ndarray, k_top: int = 10) -> dict:
    """Top-k singular spectrum + variance ratio in top-3 / 5 / 10."""
    # Truncated SVD via full SVD on G.T @ G (D x D); cheaper than SVD on (N, D)
    # when N >> D. Here D = 768, so just use np.linalg.svd on G directly with
    # full_matrices=False so we get min(N, D) singular values.
    _U, sigmas, Vt = np.linalg.svd(G, full_matrices=False)
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
        "sigma_top_k": [float(s) for s in sigmas[:k_top]],
        "V_top3": Vt[:3].tolist(),  # (3, D) — the Krylov 3-d subspace
    }


def main() -> int:
    out_dir = RUNS_DIR / "phase27_step4_krylov"
    out_dir.mkdir(parents=True, exist_ok=True)

    substrate = _PretrainedSubstrate(harvest_layer=HARVEST_LAYER)
    device = substrate.device
    print("=" * 100)
    print(f"Phase 27 Step 4 -- gradient-Krylov subspace of margin at GPT-2 block {HARVEST_LAYER}")
    print(f"  N_programs       : {N_PROGRAMS}")
    print(f"  projection epochs: {PROJECTION_EPOCHS}")
    print("=" * 100)

    per_grammar: dict[str, dict] = {}
    t_start = time.perf_counter()
    for grammar in GRAMMARS:
        t_g = time.perf_counter()
        dispatch = GRAMMAR_DISPATCH[grammar]
        fsm = GraphFSM(
            graph_fsm_spec_mod.load(REPO_ROOT / dispatch["fsm_yaml_path"])
        )
        V = fsm.vertex_count

        h_all, next_states, _samples = harvest(grammar, fsm, substrate)
        n_total = h_all.shape[0]

        # Train the predictive projection — same shape as E30's default.
        cfg = PredictiveProjectionConfig(
            z_dim=Z_DIM,
            hidden_dim=PROJ_HIDDEN,
            n_states=V,
            entropy_weight=0.0,  # We only need margin / next-state for this measurement.
            failure_weight=0.0,
        )
        projection = PredictiveProjection(input_dim=substrate.hidden_size, config=cfg)
        # train_predictive_projection runs on CPU by default (matches E30's setup).
        train_predictive_projection(
            projection,
            h_all,
            next_states=next_states,
            entropy_targets=None,
            failure_targets=None,
            token_ids=None,
            epochs=PROJECTION_EPOCHS,
            seed=SEED,
        )

        # Argmax cells = the PCG-X partition (CPU since projection is on CPU).
        with torch.no_grad():
            h_t = torch.from_numpy(h_all)
            out = projection.forward(h_t)
            argmax_cells = out["next_state_logits"].argmax(dim=-1).cpu().numpy()

        # Gradient matrix and Krylov SVD (CPU since projection is on CPU).
        G = gradient_matrix(projection, h_all, torch.device("cpu"))
        kry = krylov_stats(G)

        # Project centered activations into the top-3 Krylov subspace.
        h_centered = h_all - h_all.mean(axis=0, keepdims=True)
        V_top3 = np.array(kry["V_top3"], dtype=np.float32)  # (3, D)
        h_3d = h_centered @ V_top3.T  # (N, 3)

        # Save coords + labels to disk for the visualizer (keep JSON small).
        np.savez_compressed(
            out_dir / f"{grammar}_3d_data.npz",
            h_3d=h_3d.astype(np.float32),
            argmax_cells=argmax_cells.astype(np.int32),
            next_states=next_states.astype(np.int32),
        )

        kry["V_top3"] = None  # don't pickle the V matrix into the JSON summary
        kry["n_samples"] = int(n_total)
        kry["V"] = int(V)
        kry["argmax_n_unique"] = int(np.unique(argmax_cells).size)
        kry["wall_clock_seconds"] = time.perf_counter() - t_g
        per_grammar[grammar] = kry

        print(
            f"  [{grammar:>15}] N={n_total:>5}  eff_rank(grad)={kry['effective_rank']:6.2f}  "
            f"top3={kry['var_frac_top3']:.3f}  top5={kry['var_frac_top5']:.3f}  "
            f"top10={kry['var_frac_top10']:.3f}  ({kry['wall_clock_seconds']:.1f}s)"
        )

    summary = {
        "seed": SEED,
        "n_programs": N_PROGRAMS,
        "harvest_layer": HARVEST_LAYER,
        "projection_epochs": PROJECTION_EPOCHS,
        "per_grammar": per_grammar,
        "total_wall_clock_seconds": time.perf_counter() - t_start,
    }
    out_json = RUNS_DIR / "phase27_step4_krylov.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print("=" * 100)
    print(f"  wrote {out_json}")
    print(f"  total wall-clock: {summary['total_wall_clock_seconds']:.1f}s")
    print("=" * 100)
    print()
    print("Summary — variance fraction of gradient ∇margin captured in top-k Krylov directions:")
    print(f"  {'grammar':>15} | {'N':>5} | {'top-3':>8} | {'top-5':>8} | {'top-10':>8} | {'eff_rank_grad':>14}")
    print(f"  {'-' * 15} | {'-' * 5} | {'-' * 8} | {'-' * 8} | {'-' * 8} | {'-' * 14}")
    for g in GRAMMARS:
        s = per_grammar[g]
        print(
            f"  {g:>15} | {s['n_samples']:>5} | {s['var_frac_top3']:>8.3f} | "
            f"{s['var_frac_top5']:>8.3f} | {s['var_frac_top10']:>8.3f} | "
            f"{s['effective_rank']:>14.2f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
