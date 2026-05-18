"""Phase 27 Step 5 -- per-regime affine-fit residual on the SUBSTRATE itself.

The master theorem (Mathematics.md "Whitney Stratification + Stratified
Partition Function") assumes each stratum is **locally smooth** -- exactly
affine inside on ReLU substrates, smoothly approximate on GELU. The
gradient-Krylov measurement of Phase 27 Step 4 *implicitly assumes* this
smoothness when SVDing per-step gradients: gradients only define a
meaningful local subspace if the function is locally linear.

The canonical test from docs/interpretability-push.md §Step 5:

  > Linear regression of substrate output on substrate input within each
  > regime support. Report fit residual per regime. ReLU substrate:
  > residual ~= 0. GELU substrate (GPT-2): residual > 0 but bounded.

We harvest TWO consecutive transformer-block outputs (block 6 and block 7)
per step, then within each PCG-X regime (the argmax cells of a projection
trained on block-6 harvests for next-state) fit:

    h_{block_7} = A * h_{block_6} + b

and report per-cell R^2 (multi-output coefficient of determination).
Pre-registered acceptance: GPT-2 (GELU) within-stratum block-to-block
R^2 >= 0.90 -- "smooth but not exact."

Note on what this *does* and *does not* measure:

  * DOES measure: whether the block_6 -> block_7 map is locally affine
    when restricted to a PCG-X regime's support, which is exactly the
    Whitney-stratification claim for the substrate's intrinsic geometry.
  * DOES NOT measure: full-substrate end-to-end smoothness; ReLU-vs-GELU
    qualitative difference (we have no ReLU LM to compare against in
    this sweep -- separate experiment).

We run on all 5 synthetic grammars plus the policy_intent_v2 real-task
FSM, on frozen GPT-2 small (canonical configuration). Wall-clock ~30 s
total on a 6 GB GPU.

Outputs:
  * runs/phase27_step5_affine.json -- per-grammar summary.
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
from nga.exp.dataset_policy_intent import generate_policy_intent_dataset  # noqa: E402
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH  # noqa: E402
from nga.exp.e30_pcg_extractor_pretrained import _PretrainedSubstrate  # noqa: E402
from nga.exp.e31_policy_intent_extraction import _load_policy_fsm  # noqa: E402

SEED = 42
N_PROGRAMS = 80
HARVEST_LAYER = 6
PROJECTION_EPOCHS = 30
Z_DIM = 32
PROJ_HIDDEN = 64

# Acceptance threshold (pre-registered for GELU substrate; exactly-affine
# strata on ReLU would justify > 0.99; GELU smooth-approximate justifies > 0.90).
R2_ACCEPTANCE_THRESHOLD = 0.90


class _DualLayerHarvester:
    """Harvests TWO consecutive transformer blocks at the same token positions.

    Wraps the model + tokenizer from _PretrainedSubstrate; reuses the
    offset-mapping path to align grammar steps to BPE positions, then
    pulls hidden_states[layer_in] AND hidden_states[layer_out] at the
    same positions per step.
    """

    def __init__(self, model_id: str = "gpt2", layer_in: int = HARVEST_LAYER,
                 layer_out: int | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(model_id)
        self._model.eval(); self._model.to(self.device)
        for p in self._model.parameters():
            p.requires_grad_(False)
        self.hidden_size = int(self._model.config.hidden_size)
        self._n_layers = int(self._model.config.num_hidden_layers)
        self.layer_in = int(layer_in)
        self.layer_out = int(layer_out) if layer_out is not None else self.layer_in + 1
        if not (0 <= self.layer_in <= self._n_layers and 0 <= self.layer_out <= self._n_layers):
            raise ValueError(f"layer indices out of range; model has {self._n_layers} blocks")
        self.max_context = int(getattr(self._model.config, "n_positions", 1024))

    def harvest_program(self, observed_tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (H_in, H_out), each (n_steps, hidden_size) float32."""
        if not observed_tokens:
            return (np.zeros((0, self.hidden_size), dtype=np.float32),
                    np.zeros((0, self.hidden_size), dtype=np.float32))
        # Build text + per-step char ranges (same recipe as E30).
        text_parts: list[str] = []
        char_ranges: list[tuple[int, int]] = []
        cursor = 0
        for i, tok in enumerate(observed_tokens):
            if i > 0:
                text_parts.append(" "); cursor += 1
            start = cursor
            text_parts.append(tok); cursor += len(tok)
            char_ranges.append((start, cursor))
        text = "".join(text_parts)
        enc = self._tokenizer(text, return_tensors="pt", return_offsets_mapping=True,
                              truncation=True, max_length=self.max_context,
                              add_special_tokens=False)
        offsets = enc["offset_mapping"][0].tolist()
        n_bpe = len(offsets)
        bpe_indices: list[int] = []
        last_seen = 0
        for _cs, ce in char_ranges:
            last_idx = -1
            for k in range(last_seen, n_bpe):
                bs = offsets[k][0]
                if bs < ce:
                    last_idx = k
                else:
                    break
            if last_idx < 0:
                bpe_indices.append(-1)
            else:
                bpe_indices.append(last_idx); last_seen = last_idx
        input_ids = enc["input_ids"].to(self.device)
        with torch.no_grad():
            out = self._model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        h_in = out.hidden_states[self.layer_in][0].detach().cpu().numpy()
        h_out = out.hidden_states[self.layer_out][0].detach().cpu().numpy()
        H_in = np.zeros((len(observed_tokens), self.hidden_size), dtype=np.float32)
        H_out = np.zeros((len(observed_tokens), self.hidden_size), dtype=np.float32)
        for step_i, bpe_i in enumerate(bpe_indices):
            idx = bpe_i if bpe_i >= 0 else h_in.shape[0] - 1
            H_in[step_i] = h_in[idx].astype(np.float32)
            H_out[step_i] = h_out[idx].astype(np.float32)
        return H_in, H_out


def _harvest_grammar(
    grammar_name: str, substrate: _DualLayerHarvester
) -> tuple[np.ndarray, np.ndarray, np.ndarray, GraphFSM]:
    """Return (H_in, H_out, y_next, fsm). Both H_* are (N, D) float32."""
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
    return H_in, H_out, ds.y_next.astype(np.int64), fsm


def _fit_projection_and_get_cells(
    H_in: np.ndarray, y_next: np.ndarray, n_states: int
) -> np.ndarray:
    """Train the standard projection on block-in activations and return per-step argmax cell labels."""
    cfg = PredictiveProjectionConfig(
        z_dim=Z_DIM,
        hidden_dim=PROJ_HIDDEN,
        n_states=n_states,
        entropy_weight=0.0,
        failure_weight=0.0,
    )
    proj = PredictiveProjection(input_dim=H_in.shape[1], config=cfg)
    train_predictive_projection(
        proj, H_in, next_states=y_next, entropy_targets=None,
        failure_targets=None, token_ids=None,
        epochs=PROJECTION_EPOCHS, seed=SEED,
    )
    proj.eval()
    with torch.no_grad():
        out = proj.forward(torch.from_numpy(H_in))
        argmax = out["next_state_logits"].argmax(dim=-1).cpu().numpy()
    return argmax


def _per_cell_affine_r2(
    H_in: np.ndarray, H_out: np.ndarray, cells: np.ndarray,
    min_cell_size: int = 20, n_folds: int = 5,
    pca_dim: int | None = 32, ridge_alpha: float = 1.0,
    seed: int = SEED,
) -> dict:
    """Out-of-sample per-cell affine-fit R^2 on the substrate's block-to-block map.

    Two cancellations of trivial-fit pitfalls:

      1. **Held-out evaluation via k-fold CV.** Plain in-sample R^2 on
         high-dim regression with n_samples ~< n_features is uninformative
         (least-squares interpolates exactly). We split each cell's
         samples into k folds, fit affine on k-1, evaluate on the
         remaining fold, and average. The averaged R^2 is honest:
         R^2 ~= 1 ONLY IF the underlying map is genuinely affine on
         this cell's input distribution.

      2. **PCA reduction of inputs to ``pca_dim``** (default 32). With
         768-d inputs the L2 ridge needs heavy regularization to behave
         on small cells; projecting onto the top-32 PCs first
         standardizes the conditioning across cells and matches the
         z_dim our predictive projection targets. The interpretation:
         "is the block-6 -> block-7 map locally affine in the variance-
         dominant subspace of this cell's inputs?" -- the right question
         for a smoothness audit.

    Cells with fewer than ``min_cell_size`` samples are excluded.
    """
    H_in = H_in.astype(np.float64)
    H_out = H_out.astype(np.float64)
    rng = np.random.default_rng(int(seed))
    per_cell: dict[int, dict] = {}
    skipped: list[int] = []
    for c in np.unique(cells):
        mask = cells == c
        n = int(mask.sum())
        if n < min_cell_size:
            skipped.append(int(c))
            continue
        Hi = H_in[mask]
        Ho = H_out[mask]
        # Optional PCA: project inputs onto top-pca_dim variance directions
        # computed FROM THIS CELL ONLY (so we test local linearity).
        if pca_dim is not None and pca_dim < Hi.shape[1]:
            Hi_centered = Hi - Hi.mean(axis=0, keepdims=True)
            _U, _s, Vt = np.linalg.svd(Hi_centered, full_matrices=False)
            n_eff = min(pca_dim, n - 1)
            Hi_red = Hi_centered @ Vt[:n_eff].T  # (n, n_eff)
        else:
            Hi_red = Hi

        # k-fold CV.
        idx = np.arange(n); rng.shuffle(idx)
        folds = np.array_split(idx, min(n_folds, n))
        ss_res_total = 0.0
        ss_tot_total = 0.0
        for fold in folds:
            test_mask = np.zeros(n, dtype=bool); test_mask[fold] = True
            train_mask = ~test_mask
            X_tr = Hi_red[train_mask]; y_tr = Ho[train_mask]
            X_te = Hi_red[test_mask]; y_te = Ho[test_mask]
            if X_tr.shape[0] < 2 or X_te.shape[0] < 1:
                continue
            # Ridge regression (closed form): W = (X^T X + alpha I)^-1 X^T y.
            # Augment X with bias column.
            X_tr_aug = np.concatenate([X_tr, np.ones((X_tr.shape[0], 1))], axis=1)
            X_te_aug = np.concatenate([X_te, np.ones((X_te.shape[0], 1))], axis=1)
            D = X_tr_aug.shape[1]
            A = X_tr_aug.T @ X_tr_aug + ridge_alpha * np.eye(D)
            # Don't regularize the bias term (last row/col).
            A[-1, -1] -= ridge_alpha
            W = np.linalg.solve(A, X_tr_aug.T @ y_tr)
            y_hat = X_te_aug @ W
            ss_res_total += float(((y_te - y_hat) ** 2).sum())
            # SS_tot uses the held-out test slice's variance vs the TRAINING
            # set's mean (proper out-of-sample variance reference).
            y_train_mean = y_tr.mean(axis=0, keepdims=True)
            ss_tot_total += float(((y_te - y_train_mean) ** 2).sum())

        r2 = 1.0 - ss_res_total / max(ss_tot_total, 1e-30)
        per_cell[int(c)] = {
            "n_samples": n,
            "r2_cv": r2,
            "ss_res": ss_res_total,
            "ss_tot": ss_tot_total,
        }
    total_support = sum(v["n_samples"] for v in per_cell.values())
    weighted_r2 = (
        sum(v["r2_cv"] * v["n_samples"] for v in per_cell.values()) / total_support
        if total_support > 0 else float("nan")
    )
    mean_r2 = float(np.mean([v["r2_cv"] for v in per_cell.values()])) if per_cell else float("nan")
    return {
        "n_cells_fit": len(per_cell),
        "n_cells_skipped_too_small": len(skipped),
        "min_cell_size": min_cell_size,
        "n_folds": n_folds,
        "pca_dim": pca_dim,
        "ridge_alpha": ridge_alpha,
        "per_cell": per_cell,
        "weighted_mean_r2_cv": weighted_r2,
        "unweighted_mean_r2_cv": mean_r2,
        "min_cell_r2_cv": float(min(v["r2_cv"] for v in per_cell.values())) if per_cell else float("nan"),
        "max_cell_r2_cv": float(max(v["r2_cv"] for v in per_cell.values())) if per_cell else float("nan"),
    }


def run_one(grammar_name: str, substrate: _DualLayerHarvester) -> dict:
    t = time.perf_counter()
    H_in, H_out, y_next, fsm = _harvest_grammar(grammar_name, substrate)
    argmax = _fit_projection_and_get_cells(H_in, y_next, fsm.vertex_count)
    # The substrate-side claim: H_out = A * H_in + b within each regime.
    affine = _per_cell_affine_r2(H_in, H_out, argmax)
    return {
        "grammar": grammar_name,
        "V": fsm.vertex_count,
        "n_samples": int(H_in.shape[0]),
        "hidden_size": int(H_in.shape[1]),
        "layer_in": substrate.layer_in,
        "layer_out": substrate.layer_out,
        "argmax_cells_observed": int(len(np.unique(argmax))),
        **affine,
        "wall_clock_seconds": time.perf_counter() - t,
    }


def main() -> int:
    grammars = [
        "listops", "python_expr", "python_big", "json", "python_control",
        "policy_intent_v2",
    ]
    print("=" * 100)
    print("Phase 27 Step 5 -- per-regime affine-fit residual on the substrate")
    print(f"  substrate         : frozen GPT-2 small (block-{HARVEST_LAYER} -> block-{HARVEST_LAYER + 1}, GELU)")
    print(f"  grammars          : {grammars}")
    print(f"  n_programs        : {N_PROGRAMS} per grammar")
    print(f"  projection        : z_dim={Z_DIM}, hidden={PROJ_HIDDEN}, epochs={PROJECTION_EPOCHS}")
    print(f"  acceptance bar    : weighted mean per-cell R^2 >= {R2_ACCEPTANCE_THRESHOLD}")
    print("=" * 100)

    substrate = _DualLayerHarvester(layer_in=HARVEST_LAYER, layer_out=HARVEST_LAYER + 1)

    per_grammar: dict[str, dict] = {}
    t_start = time.perf_counter()
    for g in grammars:
        r = run_one(g, substrate)
        per_grammar[g] = r
        print(
            f"  [{g:>18}] V={r['V']:>2} N={r['n_samples']:>5} "
            f"cells_fit={r['n_cells_fit']:>2} (skip {r['n_cells_skipped_too_small']:>2})  "
            f"weighted_R^2_cv={r['weighted_mean_r2_cv']:7.4f}  "
            f"min={r['min_cell_r2_cv']:.4f}  max={r['max_cell_r2_cv']:.4f}  "
            f"({r['wall_clock_seconds']:.1f}s)"
        )

    # Acceptance check.
    bar = R2_ACCEPTANCE_THRESHOLD
    passes = {g: per_grammar[g]["weighted_mean_r2_cv"] >= bar for g in grammars}
    n_pass = sum(passes.values())

    summary = {
        "seed": SEED,
        "n_programs": N_PROGRAMS,
        "harvest_layer": HARVEST_LAYER,
        "z_dim": Z_DIM,
        "projection_epochs": PROJECTION_EPOCHS,
        "min_cell_size": 8,
        "r2_acceptance_threshold": bar,
        "model_id": substrate.model_id,
        "per_grammar": per_grammar,
        "acceptance_per_grammar": passes,
        "n_grammars_pass": n_pass,
        "total_grammars": len(grammars),
        "total_wall_clock_seconds": time.perf_counter() - t_start,
    }
    out_json = RUNS_DIR / "phase27_step5_affine.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("=" * 100)
    print(f"  wrote {out_json}")
    print(f"  total wall-clock : {summary['total_wall_clock_seconds']:.1f}s")
    print()
    print(f"Acceptance: weighted-mean held-out per-cell R^2 >= {bar} per grammar")
    print(f"{'grammar':>20} | {'weighted R^2 (cv)':>17} | {'verdict':>8}")
    print(f"{'-' * 20} | {'-' * 17} | {'-' * 8}")
    for g in grammars:
        r2 = per_grammar[g]["weighted_mean_r2_cv"]
        verdict = "PASS" if passes[g] else "FAIL"
        print(f"{g:>20} | {r2:>17.4f} | {verdict:>8}")
    print()
    print(f"Headline: {n_pass} / {len(grammars)} grammars PASS the affine-fit smoothness bar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
