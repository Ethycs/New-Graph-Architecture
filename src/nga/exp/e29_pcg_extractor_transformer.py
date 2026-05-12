"""E29 - PCG-X over a small causal transformer (Phase 23d).

The decisive test of the PCG-X reframing: the substrate is now a small
causal transformer trained from scratch on the grammar's token
sequences via next-token prediction, with **no current_state injected
into the input** (unlike Phase 22a / 23 / 23b's state-conditioned Wave-C
MLP). The transformer must learn FSM state implicitly from token
context; PCG-X then asks whether the activations at a chosen layer
carry enough state structure that the predictive projection +
argmax-of-next-state partition recovers useful regimes.

Phase 23b predicted that the adversarial token head, which was inert
on the state-conditioned substrate, should be net-positive here: with
no state directly in the input, the transformer's mid-layer
activations have heavy token-level surface variance; stripping it
should expose the slower state signal.

Pipeline
========

1. Generate the grammar's dataset.
2. Group samples by program and build per-program token-id sequences
   (vocab is per-grammar).
3. Train a SmallTransformer (causal LM, ``d_model=64``, ``n_layers=2``,
   ``n_heads=4``) on the token sequences via causal next-token CE.
4. Harvest mid-layer activations (layer 0 by default): one
   ``(d_model,)`` vector per (program, step), aligned with the
   original dataset's sample ordering.
5. Run the PCG-X pipeline on those activations: predictive projection
   ``h -> z`` with (next-state CE + entropy regression + failure BCE +
   optional adversarial token CE with gradient reversal), tropical-lite
   partition by ``argmax(next_state_logits)``, bisimulation quotient
   to ``target_K = V`` as a post-hoc consolidator.
6. Emit a control_graph.json artefact identical in shape to E28's
   plus a per-run record of which transformer layer was harvested.

Comparison
==========

The runner reports the standard PCG-X metrics (purity, failure rate,
entropy, margin, K_argmax, K_after_merge) so the table matches the
Phase 23b sweep's columns. The sweep script flips
``adversarial_token_weight`` between 0.0 and 1.0 to test the
prediction.
"""
from __future__ import annotations

import json
import resource
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from nga.arch.bisimulation_quotient import quotient_by_bisimulation
from nga.arch.forward_backward import (
    bayesian_m_step_beta,
    expected_counts_observed,
)
from nga.arch.graph_fsm import GraphFSM
from nga.arch.posterior_mask import PosteriorMask
from nga.arch.predictive_projection import (
    PredictiveProjection,
    PredictiveProjectionConfig,
    train_predictive_projection,
)
from nga.arch.small_transformer import (
    SmallTransformer,
    train_small_transformer,
)
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH
from nga.exp.e28_pcg_extractor import (
    RegimeEdge,
    RegimeNode,
    _failure_targets,
    _next_state_entropy_targets,
    _peak_memory_kb,
)

__all__ = ["E29Result", "run_e29"]


@dataclass
class E29Result:
    """Headline numbers for one E29 run."""

    grammar: str
    n_total_steps: int
    n_samples: int
    n_tokens: int

    V_ground_truth: int
    n_argmax_cells: int
    n_regimes_after_merge: int

    mean_failure_rate: float
    mean_entropy: float
    mean_margin_to_tie: float
    mean_purity_against_current_state: float

    aligned_hamming_at_target_V: float

    transformer_token_accuracy: float
    projection_next_state_accuracy: float
    projection_failure_accuracy: float
    projection_entropy_mse: float
    projection_token_accuracy: float
    adversarial_token_weight: float
    harvest_layer: int

    phase_1_wall_clock_seconds: float
    phase_2_wall_clock_seconds: float
    phase_3_wall_clock_seconds: float
    phase_4_wall_clock_seconds: float
    total_wall_clock_seconds: float
    peak_memory_kb: float


def _build_token_vocab(
    train_ds,
) -> tuple[dict[str, int], list[str], np.ndarray]:
    """Return ``(tok_to_id, id_to_tok, per_sample_token_ids)``."""
    tokens = [s.observed_token for s in train_ds.samples]
    vocab = sorted(set(tokens))
    tok_to_id = {t: i for i, t in enumerate(vocab)}
    per_sample_ids = np.asarray(
        [tok_to_id[t] for t in tokens], dtype=np.int64
    )
    return tok_to_id, vocab, per_sample_ids


def _per_program_token_sequences(
    train_ds, seq_attr: str, per_sample_token_ids: np.ndarray
) -> tuple[list[np.ndarray], list[list[int]]]:
    """Group sample indices by program id and emit per-program token-id
    arrays plus the original sample-index ordering per program.

    Returns ``(seqs, sample_indices_per_program)``.
    ``seqs[p]`` is the token-id array for program ``p``; ``sample_indices
    _per_program[p][k]`` is the original sample index of step ``k`` in
    program ``p``.
    """
    programs: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(train_ds.samples):
        programs[int(getattr(s, seq_attr))].append(i)
    sorted_pids = sorted(programs.keys())
    seqs: list[np.ndarray] = []
    sample_indices: list[list[int]] = []
    for pid in sorted_pids:
        sample_indices.append(programs[pid])
        seqs.append(
            per_sample_token_ids[np.asarray(programs[pid], dtype=np.int64)]
        )
    return seqs, sample_indices


def run_e29(
    *,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    grammar: str = "python_big",
    n_programs: int | None = None,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    n_transformer_train_epochs: int = 12,
    transformer_lr: float = 3e-3,
    harvest_layer: int = 0,
    projection_z_dim: int = 32,
    projection_hidden_dim: int = 64,
    n_projection_train_epochs: int = 30,
    target_n_regimes: int | None = None,
    adversarial_token_weight: float = 0.0,
) -> E29Result:
    if torch is None:
        raise ImportError("torch is required for E29")
    if grammar not in GRAMMAR_DISPATCH:
        raise ValueError(
            f"unknown grammar {grammar!r}; expected one of "
            f"{sorted(GRAMMAR_DISPATCH.keys())}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    dispatch = GRAMMAR_DISPATCH[grammar]
    V = fsm.vertex_count
    target_K = int(target_n_regimes) if target_n_regimes is not None else V

    t_start = time.perf_counter()

    # ---- Phase 1: dataset + tokenisation + transformer train + harvest.
    t1 = time.perf_counter()
    n = int(n_programs) if n_programs is not None else int(dispatch["default_n"])
    train_ds = dispatch["loader"](fsm, n, seed)
    seq_attr = dispatch["sequence_id_attr"]
    current_states = train_ds.current_states.astype(np.int64)
    n_total = int(train_ds.X.shape[0])

    tok_to_id, vocab, per_sample_token_ids = _build_token_vocab(train_ds)
    n_tokens = len(vocab)
    program_seqs, sample_indices_per_program = _per_program_token_sequences(
        train_ds, seq_attr, per_sample_token_ids
    )
    max_T = max(int(s.size) for s in program_seqs)

    model = SmallTransformer(
        vocab_size=n_tokens,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max(max_T, 16),
        seed=int(seed),
    )
    train_diag = train_small_transformer(
        model,
        program_seqs,
        epochs=n_transformer_train_epochs,
        lr=transformer_lr,
        seed=int(seed),
        pad_token_id=0,
    )

    # Harvest mid-layer activations aligned with the original sample order.
    h_all = np.zeros((n_total, d_model), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for pid_idx, seq in enumerate(program_seqs):
            x = torch.from_numpy(seq.astype(np.int64)).unsqueeze(0)
            _, harvested = model.encode(x, harvest_layer=harvest_layer)
            if harvested is None:
                # Fallback to final hidden if the requested layer is out of range.
                harvested, _ = model.encode(x, harvest_layer=None)
            arr = harvested[0].cpu().numpy().astype(np.float64)
            for k, orig_i in enumerate(sample_indices_per_program[pid_idx]):
                h_all[orig_i] = arr[k]
    phase_1_seconds = time.perf_counter() - t1

    # Per-step program-id assignment (matches the dataset's sample order).
    program_ids = np.asarray(
        [int(getattr(s, seq_attr)) for s in train_ds.samples], dtype=np.int64
    )

    # ---- Phase 2: predictive projection h -> z.
    t2 = time.perf_counter()
    entropy_targets = _next_state_entropy_targets(fsm, current_states)
    failure_targets = _failure_targets(
        train_ds.y_next, fsm, current_states
    )
    proj_cfg = PredictiveProjectionConfig(
        z_dim=projection_z_dim,
        hidden_dim=projection_hidden_dim,
        n_states=V,
        entropy_weight=1.0,
        failure_weight=1.0,
        adversarial_token_weight=float(adversarial_token_weight),
        n_tokens=n_tokens if adversarial_token_weight > 0.0 else 0,
    )
    projection = PredictiveProjection(
        input_dim=d_model, config=proj_cfg
    )
    proj_diag = train_predictive_projection(
        projection,
        h_all,
        next_states=train_ds.y_next,
        entropy_targets=entropy_targets,
        failure_targets=failure_targets,
        token_ids=(
            per_sample_token_ids if adversarial_token_weight > 0.0 else None
        ),
        epochs=n_projection_train_epochs,
        lr=1e-3,
        batch_size=64,
        seed=int(seed),
    )
    with torch.no_grad():
        out = projection(torch.from_numpy(h_all.astype(np.float32)))
    next_logits = out["next_state_logits"].cpu().numpy().astype(np.float64)
    entropy_pred = out["entropy_pred"].cpu().numpy().astype(np.float64)
    failure_pred = out["failure_logit"].cpu().numpy().astype(np.float64)
    phase_2_seconds = time.perf_counter() - t2

    # ---- Phase 3: tropical-lite partition.
    t3 = time.perf_counter()
    argmax_cells = next_logits.argmax(axis=1).astype(np.int64)
    sorted_logits = np.sort(next_logits, axis=1)
    margin_to_tie = (sorted_logits[:, -1] - sorted_logits[:, -2]).astype(
        np.float64
    )
    cells = np.unique(argmax_cells)
    K_argmax = int(cells.size)
    cell_to_idx = {int(c): i for i, c in enumerate(cells)}
    compact = np.asarray(
        [cell_to_idx[int(c)] for c in argmax_cells], dtype=np.int64
    )
    transition_counts = np.zeros((K_argmax, K_argmax), dtype=np.int64)
    n_progs = int(program_ids.max()) + 1
    for p in range(n_progs):
        rows = np.where(program_ids == p)[0]
        if rows.size < 2:
            continue
        transition_counts += expected_counts_observed(compact[rows], K_argmax)
    per_cell_state_counts = np.zeros((K_argmax, V), dtype=np.int64)
    for c_idx in range(K_argmax):
        members = np.where(compact == c_idx)[0]
        if members.size == 0:
            continue
        bc = np.bincount(current_states[members], minlength=V)
        per_cell_state_counts[c_idx] = bc
    phase_3_seconds = time.perf_counter() - t3

    # ---- Phase 4: behavioral merge.
    t4 = time.perf_counter()
    quotient = quotient_by_bisimulation(
        compact,
        transition_counts,
        target_K=min(target_K, K_argmax),
        criterion="full",
        emission_counts=per_cell_state_counts,
        seed=int(seed),
    )
    merged_labels = quotient.labels
    merged_counts = quotient.transition_counts
    n_regimes = merged_counts.shape[0]
    phase_4_seconds = time.perf_counter() - t4

    # Per-regime stats.
    regimes: list[RegimeNode] = []
    for r_id in range(n_regimes):
        members = np.where(merged_labels == r_id)[0]
        if members.size == 0:
            regimes.append(
                RegimeNode(
                    regime_id=r_id,
                    support=0,
                    failure_rate=float("nan"),
                    entropy_mean=float("nan"),
                    mean_margin_to_tie=float("nan"),
                    dominant_current_state="<empty>",
                    purity_against_current_state=0.0,
                )
            )
            continue
        f_prob = 1.0 / (1.0 + np.exp(-failure_pred[members]))
        cs_counts = np.bincount(current_states[members], minlength=V)
        dom = int(cs_counts.argmax())
        purity = float(cs_counts[dom]) / float(members.size)
        regimes.append(
            RegimeNode(
                regime_id=r_id,
                support=int(members.size),
                failure_rate=float(f_prob.mean()),
                entropy_mean=float(entropy_pred[members].mean()),
                mean_margin_to_tie=float(margin_to_tie[members].mean()),
                dominant_current_state=str(fsm.vertex_ids[dom]),
                purity_against_current_state=purity,
            )
        )

    alpha, beta = bayesian_m_step_beta(
        merged_counts.astype(np.float64),
        expected_counts_neg=None,
        prior_alpha=1.0,
        prior_beta=1.0,
    )
    edges: list[RegimeEdge] = []
    row_sums = merged_counts.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    probs = merged_counts.astype(np.float64) / row_sums
    for i in range(n_regimes):
        for j in range(n_regimes):
            count = int(merged_counts[i, j])
            if count == 0:
                continue
            edges.append(
                RegimeEdge(
                    src=i,
                    dst=j,
                    probability=float(probs[i, j]),
                    confidence_alpha=float(alpha[i, j]),
                    confidence_beta=float(beta[i, j]),
                    count=count,
                )
            )

    aligned_hamming = float("nan")
    if n_regimes == V:
        from nga.exp.e24_graph_extraction import _best_permutation_hamming

        posterior_mask = PosteriorMask(V, prior_alpha=1.0, prior_beta=1.0)
        posterior_mask._alpha = alpha
        posterior_mask._beta = beta
        extracted = posterior_mask.legality_matrix(0.5)
        h_norm, _perm = _best_permutation_hamming(
            extracted.astype(np.int64),
            fsm.legality_matrix.astype(np.int64),
        )
        aligned_hamming = float(h_norm)

    total = time.perf_counter() - t_start

    control_graph = {
        "grammar": grammar,
        "substrate": f"small_transformer(d={d_model}, n_layers={n_layers}, n_heads={n_heads})",
        "harvest_layer": int(harvest_layer),
        "n_total_steps": n_total,
        "n_tokens": int(n_tokens),
        "n_argmax_cells_before_merge": int(K_argmax),
        "n_regimes_after_merge": int(n_regimes),
        "regimes": [
            {
                "regime_id": r.regime_id,
                "support": r.support,
                "failure_rate": r.failure_rate,
                "entropy_mean": r.entropy_mean,
                "mean_margin_to_tie": r.mean_margin_to_tie,
                "dominant_current_state": r.dominant_current_state,
                "purity_against_current_state": r.purity_against_current_state,
            }
            for r in regimes
        ],
        "edges": [
            {
                "src": e.src,
                "dst": e.dst,
                "probability": e.probability,
                "confidence_alpha": e.confidence_alpha,
                "confidence_beta": e.confidence_beta,
                "count": e.count,
            }
            for e in edges
        ],
        "diagnostics": {
            "aligned_hamming_at_target_V": aligned_hamming,
            "transformer": train_diag,
            "projection": proj_diag,
            "adversarial_token_weight": float(adversarial_token_weight),
            "n_tokens": int(n_tokens),
        },
    }
    (output_dir / "control_graph.json").write_text(
        json.dumps(control_graph, indent=2)
    )

    mean_failure = float(
        np.mean([r.failure_rate for r in regimes if r.support > 0])
    )
    mean_entropy = float(
        np.mean([r.entropy_mean for r in regimes if r.support > 0])
    )
    mean_margin = float(
        np.mean([r.mean_margin_to_tie for r in regimes if r.support > 0])
    )
    mean_purity = float(
        np.mean(
            [r.purity_against_current_state for r in regimes if r.support > 0]
        )
    )
    return E29Result(
        grammar=grammar,
        n_total_steps=int(n_total),
        n_samples=int(n_progs),
        n_tokens=int(n_tokens),
        V_ground_truth=int(V),
        n_argmax_cells=int(K_argmax),
        n_regimes_after_merge=int(n_regimes),
        mean_failure_rate=mean_failure,
        mean_entropy=mean_entropy,
        mean_margin_to_tie=mean_margin,
        mean_purity_against_current_state=mean_purity,
        aligned_hamming_at_target_V=float(aligned_hamming),
        transformer_token_accuracy=float(
            train_diag.get("train_token_accuracy", float("nan"))
        ),
        projection_next_state_accuracy=float(
            proj_diag.get("next_state_acc", 0.0)
        ),
        projection_failure_accuracy=float(proj_diag.get("failure_acc", 0.0)),
        projection_entropy_mse=float(proj_diag.get("entropy_mse", 0.0)),
        projection_token_accuracy=float(
            proj_diag.get("token_acc", float("nan"))
        ),
        adversarial_token_weight=float(adversarial_token_weight),
        harvest_layer=int(harvest_layer),
        phase_1_wall_clock_seconds=float(phase_1_seconds),
        phase_2_wall_clock_seconds=float(phase_2_seconds),
        phase_3_wall_clock_seconds=float(phase_3_seconds),
        phase_4_wall_clock_seconds=float(phase_4_seconds),
        total_wall_clock_seconds=float(total),
        peak_memory_kb=float(_peak_memory_kb()),
    )
