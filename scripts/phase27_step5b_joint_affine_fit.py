"""Phase 27 Step 5b -- per-cell affine-fit residual at V*|tokens| joint-cardinality regimes.

Tests the pre-registered prediction from Step 5's writeup:

  > Predicted: rerunning Step 5 at joint-cardinality regimes pushes
  > R^2 toward 0.90 across the board.

Step 5 found that GPT-2's block_6 -> block_7 map is locally affine at
V-cell granularity ONLY on the smaller grammars (policy_intent_v2 V=4
R^2 0.90; listops V=11 R^2 0.89; python_big V=24 R^2 0.13;
python_control V=37 R^2 -0.28). The architectural reading is the
Phase 21 dynamic: the substrate's natural equivalence is finer than V,
so at V cells the larger grammars aggregate multiple natural strata
into each cell and the affine fit breaks.

Step 5b refines the regime cardinality to V*|tokens| via the same
labelled-hypergraph reframe that fixed Phase 28b's A3. For each grammar:

  1. Harvest block_6 and block_7 hidden states (same as Step 5).
  2. Build the joint target `current_state * |tokens| + token_idx`.
  3. Train a projection h -> z -> joint_logits with n_states = V*|tokens|.
  4. Argmax cells: up to V*|tokens| cells, each ideally containing one
     (state, token) tuple.
  5. Fit block_7 = A * block_6 + b within each cell, 5-fold CV, PCA-32,
     ridge -- identical methodology to Step 5.

Pre-registered acceptance: weighted-mean R^2 >= 0.90 on all grammars.
The cleanest version of the master theorem's local-smoothness claim is
that smoothness holds at the substrate's natural equivalence cardinality,
not at the gold FSM's V.

Output:
  * runs/phase27_step5b_joint_affine.json -- per-grammar summary.
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

# Reuse the dual-layer harvester from Step 5.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from phase27_step5_affine_fit import _DualLayerHarvester, _per_cell_affine_r2  # noqa: E402

SEED = 42
N_PROGRAMS = 80
HARVEST_LAYER = 6
PROJECTION_EPOCHS = 30
Z_DIM = 32
PROJ_HIDDEN = 64
R2_ACCEPTANCE_THRESHOLD = 0.90


def _harvest(grammar_name: str, substrate):
    """Return (H_in, H_out, current_states, token_idx, V, |tokens|, n_samples)."""
    dispatch = GRAMMAR_DISPATCH[grammar_name]
    fsm = GraphFSM(graph_fsm_spec_mod.load(REPO_ROOT / dispatch["fsm_yaml_path"]))
    ds = dispatch["loader"](fsm, N_PROGRAMS, SEED)
    seq_attr = dispatch["sequence_id_attr"]
    seq_to_indices: dict[int, list[int]] = {}
    for i, s in enumerate(ds.samples):
        seq_to_indices.setdefault(int(getattr(s, seq_attr)), []).append(i)
    H_in = np.zeros((len(ds.samples), substrate.hidden_size), dtype=np.float32)
    H_out = np.zeros((len(ds.samples), substrate.hidden_size), dtype=np.float32)
    for seq, idx in seq_to_indices.items():
        ordered = sorted(idx)
        tokens = [ds.samples[i].observed_token for i in ordered]
        Hi, Ho = substrate.harvest_program(tokens)
        for k, original in enumerate(ordered):
            H_in[original] = Hi[k]
            H_out[original] = Ho[k]
    # Token index: argmax of the one-hot features. Works for every
    # adapter (X is one-hot over the per-grammar alphabet in all of them).
    token_idx = ds.X.argmax(axis=1).astype(np.int64)
    n_tokens = int(ds.X.shape[1])
    return H_in, H_out, ds.current_states.astype(np.int64), token_idx, fsm.vertex_count, n_tokens


def _train_joint_projection(
    H_in: np.ndarray, joint_target: np.ndarray, n_joint: int
) -> np.ndarray:
    """Train a projection on the joint (state, token) target and return argmax cells."""
    cfg = PredictiveProjectionConfig(
        z_dim=Z_DIM, hidden_dim=PROJ_HIDDEN, n_states=n_joint,
        entropy_weight=0.0, failure_weight=0.0,
    )
    proj = PredictiveProjection(input_dim=H_in.shape[1], config=cfg)
    train_predictive_projection(
        proj, H_in, next_states=joint_target, entropy_targets=None,
        failure_targets=None, token_ids=None,
        epochs=PROJECTION_EPOCHS, seed=SEED,
    )
    proj.eval()
    with torch.no_grad():
        argmax = proj.forward(torch.from_numpy(H_in))["next_state_logits"].argmax(dim=-1).cpu().numpy()
    return argmax


def run_one(grammar_name: str, substrate: _DualLayerHarvester) -> dict:
    t = time.perf_counter()
    H_in, H_out, current_states, token_idx, V, n_tokens = _harvest(grammar_name, substrate)
    n_joint = V * n_tokens
    joint_target = current_states * n_tokens + token_idx

    argmax = _train_joint_projection(H_in, joint_target, n_joint)
    affine = _per_cell_affine_r2(H_in, H_out, argmax)
    return {
        "grammar": grammar_name,
        "V": V,
        "n_tokens": n_tokens,
        "n_joint": n_joint,
        "n_samples": int(H_in.shape[0]),
        "argmax_cells_observed": int(len(np.unique(argmax))),
        "layer_in": substrate.layer_in,
        "layer_out": substrate.layer_out,
        **affine,
        "wall_clock_seconds": time.perf_counter() - t,
    }


def main() -> int:
    grammars = [
        "listops", "python_expr", "python_big", "json", "python_control",
        "policy_intent_v2",
    ]
    print("=" * 100)
    print("Phase 27 Step 5b -- per-cell affine-fit at V*|tokens| joint-cardinality regimes")
    print(f"  substrate         : frozen GPT-2 small (block-{HARVEST_LAYER} -> block-{HARVEST_LAYER + 1}, GELU)")
    print(f"  grammars          : {grammars}")
    print(f"  n_programs        : {N_PROGRAMS} per grammar")
    print(f"  projection        : z_dim={Z_DIM}, hidden={PROJ_HIDDEN}, epochs={PROJECTION_EPOCHS}")
    print(f"  acceptance bar    : weighted mean held-out R^2 >= {R2_ACCEPTANCE_THRESHOLD}")
    print(f"  prediction        : R^2 climbs to >= 0.90 on grammars that failed at V cells")
    print("=" * 100)

    substrate = _DualLayerHarvester(layer_in=HARVEST_LAYER, layer_out=HARVEST_LAYER + 1)
    per_grammar: dict[str, dict] = {}
    t_start = time.perf_counter()
    for g in grammars:
        r = run_one(g, substrate)
        per_grammar[g] = r
        print(
            f"  [{g:>18}] V={r['V']:>2}  |t|={r['n_tokens']:>2}  n_joint={r['n_joint']:>4}  "
            f"cells_seen={r['argmax_cells_observed']:>4}  cells_fit={r['n_cells_fit']:>3} "
            f"(skip {r['n_cells_skipped_too_small']:>3})  "
            f"weighted_R^2={r['weighted_mean_r2_cv']:7.4f}  "
            f"min={r['min_cell_r2_cv']:.3f}  max={r['max_cell_r2_cv']:.3f}  "
            f"({r['wall_clock_seconds']:.1f}s)"
        )

    bar = R2_ACCEPTANCE_THRESHOLD
    passes = {g: per_grammar[g]["weighted_mean_r2_cv"] >= bar for g in grammars}
    n_pass = sum(passes.values())

    summary = {
        "seed": SEED,
        "n_programs": N_PROGRAMS,
        "harvest_layer": HARVEST_LAYER,
        "z_dim": Z_DIM,
        "projection_epochs": PROJECTION_EPOCHS,
        "r2_acceptance_threshold": bar,
        "model_id": substrate.model_id,
        "per_grammar": per_grammar,
        "acceptance_per_grammar": passes,
        "n_grammars_pass": n_pass,
        "total_grammars": len(grammars),
        "total_wall_clock_seconds": time.perf_counter() - t_start,
    }
    out_json = RUNS_DIR / "phase27_step5b_joint_affine.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("=" * 100)
    print(f"  wrote {out_json}")
    print(f"  total wall-clock : {summary['total_wall_clock_seconds']:.1f}s")
    print()
    print("Pre-registered prediction: at V*|tokens| cells, R^2 climbs to >= 0.90.")
    print(f"{'grammar':>20} | {'V':>2} | {'|t|':>3} | {'n_joint':>7} | {'Step 5 V-cell':>13} | {'Step 5b joint':>13} | {'verdict':>8}")
    print(f"{'-' * 20} | {'-' * 2} | {'-' * 3} | {'-' * 7} | {'-' * 13} | {'-' * 13} | {'-' * 8}")
    # Load Step 5 V-cell R^2 for comparison.
    step5_path = RUNS_DIR / "phase27_step5_affine.json"
    step5 = json.loads(step5_path.read_text()) if step5_path.exists() else {"per_grammar": {}}
    for g in grammars:
        r5 = step5["per_grammar"].get(g, {}).get("weighted_mean_r2_cv", float("nan"))
        r5b = per_grammar[g]["weighted_mean_r2_cv"]
        verdict = "PASS" if passes[g] else "FAIL"
        print(f"{g:>20} | {per_grammar[g]['V']:>2} | {per_grammar[g]['n_tokens']:>3} | "
              f"{per_grammar[g]['n_joint']:>7} | "
              f"{r5:>13.4f} | {r5b:>13.4f} | {verdict:>8}")
    print()
    print(f"Headline: {n_pass} / {len(grammars)} grammars PASS at joint-cardinality regimes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
