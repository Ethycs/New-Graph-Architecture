"""Phase 27 Step 3 -- PCA / effective-rank on harvested GPT-2 activations.

Pivots from the Sullivan log-law (which didn't converge robustly on inference
traces, see research_log2 Phase 27 Step 1+2) to a direct empirical-dimension
measurement: harvest block-6 hidden states from frozen GPT-2 across the
5-grammar suite and measure how much of the activation variance lives in
the top-k principal components.

Three summary statistics per grammar:

  * effective_rank          exp(H) where H = -sum p_k log p_k for
                            p_k = sigma_k^2 / sum_k sigma_k^2.
                            ("entropy dimension" of the singular-value
                            distribution; coordinate-free.)
  * participation_ratio     (sum sigma_k^2)^2 / sum sigma_k^4.
                            ("how many directions carry comparable mass";
                            also coordinate-free.)
  * d_95, d_99              smallest k whose top-k cumulative variance
                            exceeds 0.95 / 0.99. ("how many directions
                            you actually have to keep to reconstruct the
                            trajectory to fixed fidelity.")

Outputs:

  * runs/phase27_step3_pca.json     -- summary table + per-grammar full
                                       singular-value spectrum.
  * runs/phase27_step3_pca/{grammar}_spectrum.png -- log-sigma scree
                                       plot + cumulative-variance curve.

The script reuses E30's substrate + dataset machinery. No PCG-X pipeline
is run; this is harvest + PCA only. Wall-clock ~3 min on GPU for the
default n_programs=80 sweep.
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

from nga.arch.graph_fsm import GraphFSM  # noqa: E402
from nga.drivers import graph_fsm_spec as graph_fsm_spec_mod  # noqa: E402
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH  # noqa: E402
from nga.exp.e30_pcg_extractor_pretrained import _PretrainedSubstrate  # noqa: E402

SEED = 42
GRAMMARS = ["listops", "python_expr", "python_big", "json", "python_control"]
N_PROGRAMS = 80
HARVEST_LAYER = 6


def harvest_activations(grammar: str, fsm: GraphFSM, substrate, n_programs: int, seed: int) -> tuple[np.ndarray, dict]:
    """Reuses E30's harvest path; returns (N, hidden_size) and basic stats."""
    dispatch = GRAMMAR_DISPATCH[grammar]
    train_ds = dispatch["loader"](fsm, n_programs, seed)
    seq_attr = dispatch["sequence_id_attr"]
    n_total = int(train_ds.X.shape[0])
    program_ids = np.asarray(
        [int(getattr(s, seq_attr)) for s in train_ds.samples], dtype=np.int64
    )
    program_to_indices: dict[int, list[int]] = {}
    for i, pid in enumerate(program_ids):
        program_to_indices.setdefault(int(pid), []).append(i)
    h_all = np.zeros((n_total, substrate.hidden_size), dtype=np.float64)
    n_truncated = 0
    for pid, indices in program_to_indices.items():
        ordered = sorted(indices)
        observed_tokens = [train_ds.samples[i].observed_token for i in ordered]
        per_step, was_truncated = substrate.harvest_program(observed_tokens)
        if was_truncated:
            n_truncated += 1
        for k, original_idx in enumerate(ordered):
            h_all[original_idx] = per_step[k]
    return h_all, {
        "n_total_steps": n_total,
        "n_programs_with_truncation": int(n_truncated),
        "hidden_size": int(substrate.hidden_size),
    }


def _summary_from_sigmas(sigmas: np.ndarray) -> dict:
    """effective_rank / participation / d_95 / d_99 from a singular-value vector."""
    variances = sigmas ** 2
    total = float(variances.sum())
    if total <= 0.0:
        return {
            "total_variance": 0.0,
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "d_95": 0,
            "d_99": 0,
            "cumulative_variance_ratio": [],
        }
    p = variances / total
    nonzero = p > 0
    H_entropy = float(-(p[nonzero] * np.log(p[nonzero])).sum())
    effective_rank = float(np.exp(H_entropy))
    participation = float((variances.sum() ** 2) / (variances ** 2).sum())
    cumulative = np.cumsum(p)
    d_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    d_99 = int(np.searchsorted(cumulative, 0.99) + 1)
    return {
        "total_variance": total,
        "effective_rank": effective_rank,
        "participation_ratio": participation,
        "d_95": d_95,
        "d_99": d_99,
        "cumulative_variance_ratio": [float(c) for c in cumulative],
    }


def pca_stats(H: np.ndarray, n_rogue: int = 1) -> dict:
    """PCA + effective rank + participation ratio on (N, D) activations.

    Centers H, computes singular values, and reports coordinate-free
    dimensional summaries.

    GPT-2 hidden states are well-known to be **anisotropic**: a single
    dominant ("rogue") direction in hidden-state space dwarfs all others
    in variance, behaving like a global magnitude carrying no per-step
    structure (Ethayarajh 2019, Mu & Viswanath 2018, Gao et al. 2019). We
    report two summaries:

      * **raw**           variance over all D = 768 directions, including
                          the rogue. Dominated by direction 1.
      * **denoised**      variance after projecting out the top
                          ``n_rogue`` directions. This is the
                          "interpretively-relevant" structure: the
                          per-step variance orthogonal to the global
                          magnitude.

    The denoised number is the one to compare against the framework's
    expected low-d substrate claim.
    """
    H_centered = H - H.mean(axis=0, keepdims=True)
    # Full SVD: U (N, k), sigmas (k,), Vt (k, D); k = min(N, D).
    U, sigmas, Vt = np.linalg.svd(H_centered, full_matrices=False)
    raw_summary = _summary_from_sigmas(sigmas)

    # Denoised: subtract top-n_rogue rank-1 components, re-SVD residual.
    rogue_top = sigmas[:n_rogue]
    H_denoised = H_centered - (U[:, :n_rogue] * sigmas[:n_rogue]) @ Vt[:n_rogue]
    sigmas_denoised = np.linalg.svd(H_denoised, compute_uv=False)
    denoised_summary = _summary_from_sigmas(sigmas_denoised)

    return {
        "n_samples": int(H.shape[0]),
        "n_features": int(H.shape[1]),
        "n_rogue_removed": int(n_rogue),
        "sigma_top10_raw": [float(s) for s in sigmas[:10]],
        "sigma_top10_denoised": [float(s) for s in sigmas_denoised[:10]],
        "sigma_all_raw": [float(s) for s in sigmas],
        "sigma_all_denoised": [float(s) for s in sigmas_denoised],
        "rogue_variance_fraction": float((rogue_top ** 2).sum() / (sigmas ** 2).sum()),
        "raw": raw_summary,
        "denoised": denoised_summary,
    }


def main() -> int:
    out_dir = RUNS_DIR / "phase27_step3_pca"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print(f"Phase 27 Step 3 -- PCA / effective rank on GPT-2 block-{HARVEST_LAYER} activations")
    print(f"  grammars       : {GRAMMARS}")
    print(f"  n_programs     : {N_PROGRAMS} per grammar")
    print(f"  harvest_layer  : {HARVEST_LAYER}")
    print(f"  seed           : {SEED}")
    print("=" * 100)

    substrate = _PretrainedSubstrate(harvest_layer=HARVEST_LAYER)
    print(f"  substrate ready: {substrate.model_id} on {substrate.device}, "
          f"hidden_size={substrate.hidden_size}")

    t_start = time.perf_counter()
    per_grammar: dict[str, dict] = {}
    for grammar in GRAMMARS:
        t_g = time.perf_counter()
        dispatch = GRAMMAR_DISPATCH[grammar]
        fsm = GraphFSM(
            graph_fsm_spec_mod.load(REPO_ROOT / dispatch["fsm_yaml_path"])
        )
        H, harvest_stats = harvest_activations(
            grammar, fsm, substrate, n_programs=N_PROGRAMS, seed=SEED
        )
        stats = pca_stats(H)
        stats.update(harvest_stats)
        stats["wall_clock_seconds"] = time.perf_counter() - t_g
        per_grammar[grammar] = stats
        raw, den = stats["raw"], stats["denoised"]
        print(
            f"  [{grammar:>15}] N={stats['n_samples']:>5}  rogue_frac={stats['rogue_variance_fraction']:.3f}  "
            f"raw[eff={raw['effective_rank']:5.1f} d95={raw['d_95']:>3}]  "
            f"denoised[eff={den['effective_rank']:5.1f} part={den['participation_ratio']:5.1f} "
            f"d95={den['d_95']:>3} d99={den['d_99']:>3}]  ({stats['wall_clock_seconds']:.1f}s)"
        )

    summary = {
        "seed": SEED,
        "n_programs": N_PROGRAMS,
        "harvest_layer": HARVEST_LAYER,
        "model_id": substrate.model_id,
        "hidden_size": int(substrate.hidden_size),
        "per_grammar": per_grammar,
        "total_wall_clock_seconds": time.perf_counter() - t_start,
    }

    out_json = RUNS_DIR / "phase27_step3_pca.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("=" * 100)
    print(f"  wrote {out_json}")
    print(f"  total wall-clock: {summary['total_wall_clock_seconds']:.1f}s")
    print("=" * 100)
    print()
    print("Summary (denoised — after removing the top-1 rogue direction):")
    print(f"  {'grammar':>15} | {'N':>5} | {'rogue_frac':>10} | {'eff_rank':>8} | {'part_ratio':>10} | {'d_95':>4} | {'d_99':>4}")
    print(f"  {'-' * 15} | {'-' * 5} | {'-' * 10} | {'-' * 8} | {'-' * 10} | {'-' * 4} | {'-' * 4}")
    for grammar in GRAMMARS:
        s = per_grammar[grammar]
        den = s["denoised"]
        print(
            f"  {grammar:>15} | {s['n_samples']:>5} | {s['rogue_variance_fraction']:>10.3f} | "
            f"{den['effective_rank']:>8.2f} | {den['participation_ratio']:>10.2f} | "
            f"{den['d_95']:>4} | {den['d_99']:>4}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
