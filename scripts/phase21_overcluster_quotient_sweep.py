#!/usr/bin/env python3
"""Phase 21 -- overcluster-then-quotient sweep.

The decisive test of the reframed extraction claim: not "exact graph
recovery from arbitrary hidden states" but "minimal quotient under
behavioural equivalence from a state-sufficient substrate."

Pipeline per grammar:

  1. Generate the grammar's dataset.
  2. Train the state-conditioned MLP encoder (same as Wave C / E26).
  3. Harvest second-hidden-layer activations.
  4. Overcluster at K* via BIC over [V, V+pad].
  5. Build observed transition counts AND emission counts.
  6. Compute baselines:
       - free-K Hamming (sentinel if K* != V)
       - forced K=V Hamming (Wave C result)
       - oracle Hamming (ground-truth state labels per step)
  7. For each merge criterion in MERGE_CRITERIA, apply
     ``quotient_by_bisimulation`` to merge down to target_K = V,
     then compute best-permutation Hamming on the quotient.
  8. Tabulate.

Writes ``runs/phase21_overcluster_quotient_sweep_summary.json`` and
prints a compact comparison table.

Run:
    pixi run -e dev python scripts/phase21_overcluster_quotient_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from nga.arch.bisimulation_quotient import (  # noqa: E402
    MERGE_CRITERIA,
    quotient_by_bisimulation,
)
from nga.arch.forward_backward import (  # noqa: E402
    bayesian_m_step_beta,
    expected_counts_observed,
)
from nga.arch.graph_fsm import GraphFSM  # noqa: E402
from nga.arch.posterior_mask import PosteriorMask  # noqa: E402
from nga.drivers import graph_fsm_spec as graph_fsm_spec_mod  # noqa: E402
from nga.exp.e24_graph_extraction import (  # noqa: E402
    _best_permutation_hamming,
    _compute_cluster_purity,
    extract_graph,
)
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH  # noqa: E402
from nga.exp.e26_extraction_trained_encoder import (  # noqa: E402
    _TrainableEncoder,
    _train_encoder,
)

SEED = 42
N_TRAIN_EPOCHS = 50
HIDDEN_DIM = 32
GRAMMARS = ["listops", "python_expr", "python_big", "json", "python_control"]
K_RANGE_PAD = 8  # overcluster cap: V + 8


def _hamming_at_K(extracted_mask: np.ndarray, gold_mask: np.ndarray) -> float:
    """Best-permutation Hamming when shapes match, sentinel otherwise."""
    if extracted_mask.shape != gold_mask.shape:
        return 1.0
    h, _ = _best_permutation_hamming(
        extracted_mask.astype(np.int64), gold_mask.astype(np.int64)
    )
    return h


def _legality_from_counts(counts: np.ndarray) -> np.ndarray:
    """Beta(1,1) + observed counts -> mean > 0.5 boolean legality matrix."""
    alpha, beta = bayesian_m_step_beta(
        counts.astype(np.float64),
        expected_counts_neg=None,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    K = counts.shape[0]
    pm = PosteriorMask(K, prior_alpha=1.0, prior_beta=1.0)
    pm._alpha = alpha
    pm._beta = beta
    return pm.legality_matrix(0.5)


def _emission_counts_per_cluster(
    labels: np.ndarray,
    tokens: list[str],
    K: int,
) -> tuple[np.ndarray, list[str]]:
    """Per-cluster token-emission counts. Returns (counts, token_vocab)."""
    vocab = sorted(set(tokens))
    tok_idx = {t: i for i, t in enumerate(vocab)}
    em = np.zeros((K, len(vocab)), dtype=np.int64)
    for lbl, tok in zip(labels, tokens, strict=True):
        em[int(lbl), tok_idx[tok]] += 1
    return em, vocab


def _build_per_program_transition_counts(
    labels: np.ndarray, program_ids: np.ndarray, K: int
) -> np.ndarray:
    """Transition counts respecting per-program boundaries."""
    counts = np.zeros((K, K), dtype=np.int64)
    n_progs = int(program_ids.max()) + 1
    for p in range(n_progs):
        rows = np.where(program_ids == p)[0]
        if rows.size < 2:
            continue
        seq = labels[rows]
        counts += expected_counts_observed(seq, K)
    return counts


def _oracle_legality(
    current_states: np.ndarray,
    program_ids: np.ndarray,
    V: int,
) -> np.ndarray:
    """Legality matrix built from ground-truth state transitions."""
    counts = _build_per_program_transition_counts(current_states, program_ids, V)
    return _legality_from_counts(counts)


def _one_grammar(grammar: str) -> dict:
    dispatch = GRAMMAR_DISPATCH[grammar]
    fsm_yaml = REPO_ROOT / dispatch["fsm_yaml_path"]
    fsm_spec = graph_fsm_spec_mod.load(fsm_yaml)
    fsm = GraphFSM(fsm_spec)
    V = fsm.vertex_count
    gold_legality = fsm.legality_matrix

    n = int(dispatch["default_n"])
    train_ds = dispatch["loader"](fsm, n, SEED)
    seq_attr = dispatch["sequence_id_attr"]
    program_ids = np.asarray(
        [int(getattr(s, seq_attr)) for s in train_ds.samples], dtype=np.int64
    )
    current_states = train_ds.current_states.astype(np.int64)

    # State-conditioned input.
    input_dim = int(train_ds.X.shape[1])
    state_onehot = np.zeros((train_ds.X.shape[0], V), dtype=np.float64)
    state_onehot[np.arange(train_ds.X.shape[0]), current_states] = 1.0
    X_conditioned = np.concatenate(
        [train_ds.X.astype(np.float64), state_onehot], axis=1
    )

    encoder = _TrainableEncoder(
        input_dim=input_dim + V,
        hidden_dim=HIDDEN_DIM,
        n_states=V,
        seed=SEED,
    )
    final_loss, train_acc = _train_encoder(
        encoder,
        X_conditioned,
        train_ds.y_next,
        epochs=N_TRAIN_EPOCHS,
        lr=5e-3,
        batch_size=64,
        seed=SEED,
    )
    X_t = torch.from_numpy(X_conditioned.astype(np.float32))
    with torch.no_grad():
        hidden = encoder.encode(X_t).cpu().numpy().astype(np.float64)

    # Overcluster: K_range = [V, V+pad] -- BIC picks K_star >= V.
    K_range = list(range(V, V + K_RANGE_PAD + 1))
    extraction = extract_graph(
        hidden, program_ids, K_range=K_range, criterion="bic", seed=SEED
    )
    K_star = int(extraction["K_star"])
    labels_overclustered = extraction["labels"].astype(np.int64)
    over_counts = _build_per_program_transition_counts(
        labels_overclustered, program_ids, K_star
    )
    over_purity = _compute_cluster_purity(labels_overclustered, current_states)

    # Emission counts on observed tokens.
    tokens = [s.observed_token for s in train_ds.samples]
    em_counts, _vocab = _emission_counts_per_cluster(
        labels_overclustered, tokens, K_star
    )

    # Free-K Hamming (only defined if K_star == V; otherwise sentinel).
    over_legality = _legality_from_counts(over_counts)
    free_K_hamming = (
        _hamming_at_K(over_legality, gold_legality)
        if K_star == V
        else 1.0
    )

    # Forced K=V Hamming: redo extraction with K_range=[V].
    extraction_forced = extract_graph(
        hidden, program_ids, K_range=[V], criterion="bic", seed=SEED
    )
    forced_labels = extraction_forced["labels"].astype(np.int64)
    forced_counts = _build_per_program_transition_counts(
        forced_labels, program_ids, V
    )
    forced_legality = _legality_from_counts(forced_counts)
    forced_K_hamming = _hamming_at_K(forced_legality, gold_legality)
    forced_K_purity = _compute_cluster_purity(forced_labels, current_states)

    # Oracle Hamming -- ground-truth state labels.
    oracle_legality = _oracle_legality(current_states, program_ids, V)
    oracle_hamming = _hamming_at_K(oracle_legality, gold_legality)

    # Quotient each criterion down to target_K = V.
    quotient_results: dict[str, dict] = {}
    for crit in MERGE_CRITERIA:
        em_arg = em_counts if crit in ("emission", "full") else None
        qres = quotient_by_bisimulation(
            labels_overclustered,
            over_counts,
            target_K=V,
            criterion=crit,
            emission_counts=em_arg,
            seed=SEED,
        )
        q_legality = _legality_from_counts(qres.transition_counts)
        q_hamming = _hamming_at_K(q_legality, gold_legality)
        q_purity = _compute_cluster_purity(qres.labels, current_states)
        quotient_results[crit] = {
            "hamming": q_hamming,
            "purity": q_purity,
            "n_merges": len(qres.merge_history),
        }

    return {
        "grammar": grammar,
        "V": V,
        "K_star_overclustered": K_star,
        "encoder_train_accuracy": train_acc,
        "over_purity": over_purity,
        "free_K_hamming": free_K_hamming,
        "forced_K_hamming": forced_K_hamming,
        "forced_K_purity": forced_K_purity,
        "oracle_hamming": oracle_hamming,
        "n_total_steps": int(hidden.shape[0]),
        "quotient": quotient_results,
    }


def main() -> int:
    summary: dict = {
        "seed": SEED,
        "n_train_epochs": N_TRAIN_EPOCHS,
        "K_range_pad": K_RANGE_PAD,
        "grammars": {},
    }
    print("=" * 110)
    print(
        "Phase 21 -- overcluster-then-quotient sweep, "
        f"seed={SEED}, n_train_epochs={N_TRAIN_EPOCHS}"
    )
    print("=" * 110)
    crit_cols = "  ".join(f"q[{c[:6]:>6s}]" for c in MERGE_CRITERIA)
    header = (
        f"{'grammar':<18s} {'V':>3s} {'K*':>3s} "
        f"{'enc':>5s} {'forced_H':>9s} {'oracle_H':>9s}  {crit_cols}"
    )
    print(header)
    print("-" * 110)

    t_start = time.perf_counter()
    for grammar in GRAMMARS:
        t0 = time.perf_counter()
        row = _one_grammar(grammar)
        elapsed = time.perf_counter() - t0
        row["elapsed_seconds"] = elapsed
        summary["grammars"][grammar] = row
        q_cells = "  ".join(
            f"{row['quotient'][c]['hamming']:>8.4f}" for c in MERGE_CRITERIA
        )
        print(
            f"{row['grammar']:<18s} {row['V']:>3d} {row['K_star_overclustered']:>3d} "
            f"{row['encoder_train_accuracy']:>5.3f} "
            f"{row['forced_K_hamming']:>9.4f} {row['oracle_hamming']:>9.4f}  "
            f"{q_cells}"
        )
    total_elapsed = time.perf_counter() - t_start

    # Aggregate.
    n_grammars = len(GRAMMARS)
    for criterion in MERGE_CRITERIA:
        mean_h = sum(
            g["quotient"][criterion]["hamming"]
            for g in summary["grammars"].values()
        ) / n_grammars
        summary[f"mean_quotient_{criterion}_hamming"] = mean_h
        n_meets = sum(
            1
            for g in summary["grammars"].values()
            if g["quotient"][criterion]["hamming"] <= 0.05
        )
        summary[f"n_meets_strict_bar_{criterion}"] = n_meets
    summary["mean_forced_K_hamming"] = sum(
        g["forced_K_hamming"] for g in summary["grammars"].values()
    ) / n_grammars
    summary["mean_oracle_hamming"] = sum(
        g["oracle_hamming"] for g in summary["grammars"].values()
    ) / n_grammars
    summary["total_elapsed_seconds"] = total_elapsed

    print("-" * 110)
    print(
        f"mean Hamming  forced_K {summary['mean_forced_K_hamming']:.4f}  "
        f"oracle {summary['mean_oracle_hamming']:.4f}"
    )
    for c in MERGE_CRITERIA:
        n = summary[f"n_meets_strict_bar_{c}"]
        m = summary[f"mean_quotient_{c}_hamming"]
        print(
            f"  quotient[{c:<24s}] mean Hamming {m:.4f}  "
            f"strict bar (<=0.05) met on {n}/{n_grammars} grammars"
        )
    print(f"total sweep wall-clock: {total_elapsed:.2f}s")

    out_path = RUNS_DIR / "phase21_overcluster_quotient_sweep_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
