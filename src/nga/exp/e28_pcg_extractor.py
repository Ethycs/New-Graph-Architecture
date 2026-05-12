"""E28 - Predictive Control Graph Extractor (Phase 23 MVP).

The architectural reframing: stop trying to prove universal graph
extraction; build a practical regime/control-graph extractor from an
arbitrary network's activations. The mantra:

    Partition by prediction, merge by behavior, control by intervention.

This runner implements the partition + merge legs end-to-end. The
control-by-intervention leg is documented as future work.

Pipeline
========

1. **Activation harvest.** Same dataset + Wave-C state-conditioned MLP
   encoder as E26 / Phase 22a; we treat the encoder as a stand-in for
   the "arbitrary network" until we plug in a real off-the-shelf
   model. Harvest the second hidden layer's post-ReLU output as ``h``.
2. **Predictive projection.** A small MLP ``h -> z`` (the
   ``PredictiveProjection`` atom) trained jointly on three losses:
   next-state cross-entropy (partition signal), next-state entropy
   regression (Morse-lite uncertainty signal), and failure prediction
   (risk signal).
3. **Tropical-lite partition.** Partition the corpus by ``argmax`` of
   the projection's next-state head over ``z``. Each (argmax)
   cell is a candidate regime. Record per-cell support, mean entropy,
   mean failure probability, and a margin-to-tie-wall (the gap between
   the top-1 and top-2 logits).
4. **Transition estimation.** Per-program transition counts between
   regime cells; row-normalise into transition probabilities; Beta
   posterior on each edge gives confidence.
5. **Behavioral merge.** Reuse ``bisimulation_quotient`` with a custom
   composite criterion: distance is the average of L2 distance on
   outgoing transition distributions, L2 on per-cell failure-rate
   vectors, and L2 on per-cell mean-output-distribution. Two cells
   merge when they're behaviourally indistinguishable.
6. **Emit control graph JSON.** A self-describing artefact with:
   regimes (cells), edges (transitions with confidence + risk), and
   the merge history.

The output is the artefact the user described in the Phase 23
specification:

    Node 7:
      support: 18240 samples
      failure_rate: 3.1%
      entropy_mean: 0.42
      mean_margin_to_tie: 1.85

    Edges:
      7 -> 9 ...
"""
from __future__ import annotations

import json
import resource
import time
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
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH
from nga.exp.e26_extraction_trained_encoder import (
    _TrainableEncoder,
    _train_encoder,
)

__all__ = ["E28Result", "RegimeNode", "RegimeEdge", "run_e28"]


@dataclass
class RegimeNode:
    """One node in the emitted control graph."""

    regime_id: int
    support: int
    failure_rate: float
    entropy_mean: float
    mean_margin_to_tie: float
    dominant_current_state: str
    purity_against_current_state: float


@dataclass
class RegimeEdge:
    """One directed edge in the emitted control graph."""

    src: int
    dst: int
    probability: float
    confidence_alpha: float
    confidence_beta: float
    count: int


@dataclass
class E28Result:
    """Headline numbers for one PCG-X run."""

    grammar: str
    n_total_steps: int
    n_samples: int

    V_ground_truth: int
    n_argmax_cells: int
    n_regimes_after_merge: int

    mean_failure_rate: float
    mean_entropy: float
    mean_margin_to_tie: float
    mean_purity_against_current_state: float

    # Optional sanity: how well does the merged regime graph align with
    # the gold FSM legality? Reported but not load-bearing -- the PCG-X
    # claim is "useful control graph" not "exact FSM recovery."
    aligned_hamming_at_target_V: float

    encoder_train_accuracy: float
    projection_train_accuracy: float
    projection_failure_accuracy: float
    projection_entropy_mse: float
    # Phase 23b: adversarial token head diagnostics. NaN when the head
    # is disabled (``adversarial_token_weight == 0.0``).
    projection_token_accuracy: float
    adversarial_token_weight: float

    phase_1_wall_clock_seconds: float
    phase_2_wall_clock_seconds: float
    phase_3_wall_clock_seconds: float
    phase_4_wall_clock_seconds: float
    total_wall_clock_seconds: float
    peak_memory_kb: float


def _peak_memory_kb() -> float:
    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:  # pragma: no cover
        return 0.0


def _next_state_entropy_targets(
    fsm: GraphFSM, current_states: np.ndarray
) -> np.ndarray:
    """Per-step entropy target: H[uniform-over-legal-successors] of
    ``current_state``. The simplest cheap target that is well-defined
    without ground-truth next-state distributions."""
    V = fsm.vertex_count
    legal = fsm.legality_matrix.astype(np.float64)
    row_sums = legal.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    p = legal / row_sums  # (V, V) row-stochastic over legal
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0.0, np.log(p), 0.0)
        ent = -(p * log_p).sum(axis=1)  # (V,)
    return ent[current_states.astype(np.int64)]


def _failure_targets(
    y_next: np.ndarray, fsm: GraphFSM, current_states: np.ndarray
) -> np.ndarray:
    """Per-step failure: did the observed next-state break the gold
    legality? 1.0 if illegal, 0.0 if legal. Cheap supervised proxy.
    """
    legal = fsm.legality_matrix
    failures = np.zeros(y_next.shape[0], dtype=np.float64)
    for i in range(y_next.shape[0]):
        s = int(current_states[i])
        n = int(y_next[i])
        failures[i] = 0.0 if legal[s, n] else 1.0
    return failures


def run_e28(
    *,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    grammar: str = "python_big",
    n_programs: int | None = None,
    encoder_hidden_dim: int = 32,
    projection_z_dim: int = 32,
    projection_hidden_dim: int = 64,
    n_encoder_train_epochs: int = 50,
    n_projection_train_epochs: int = 30,
    target_n_regimes: int | None = None,
    adversarial_token_weight: float = 0.0,
) -> E28Result:
    """Predictive Control Graph Extractor MVP run on one grammar."""
    if torch is None:
        raise ImportError("torch is required for E28")
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

    # ---- Phase 1: dataset + base encoder + activation harvest ----
    t1 = time.perf_counter()
    n = int(n_programs) if n_programs is not None else int(dispatch["default_n"])
    train_ds = dispatch["loader"](fsm, n, seed)
    input_dim = int(train_ds.X.shape[1])
    seq_attr = dispatch["sequence_id_attr"]
    program_ids = np.asarray(
        [int(getattr(s, seq_attr)) for s in train_ds.samples], dtype=np.int64
    )
    current_states = train_ds.current_states.astype(np.int64)

    state_onehot = np.zeros((train_ds.X.shape[0], V), dtype=np.float64)
    state_onehot[np.arange(train_ds.X.shape[0]), current_states] = 1.0
    X_conditioned = np.concatenate(
        [train_ds.X.astype(np.float64), state_onehot], axis=1
    )

    base_encoder = _TrainableEncoder(
        input_dim=input_dim + V,
        hidden_dim=encoder_hidden_dim,
        n_states=V,
        seed=int(seed),
    )
    _train_loss, encoder_train_acc = _train_encoder(
        base_encoder,
        X_conditioned,
        train_ds.y_next,
        epochs=n_encoder_train_epochs,
        lr=5e-3,
        batch_size=64,
        seed=int(seed),
    )
    X_t = torch.from_numpy(X_conditioned.astype(np.float32))
    with torch.no_grad():
        h_full = base_encoder.encode(X_t).cpu().numpy().astype(np.float64)
    phase_1_seconds = time.perf_counter() - t1

    # ---- Phase 2: train predictive projection h -> z ----
    t2 = time.perf_counter()
    # Build per-grammar token vocab for the adversarial head. Even if
    # adversarial_token_weight == 0.0 we compute it cheaply so the
    # diagnostic numbers are comparable across runs.
    observed_tokens = [s.observed_token for s in train_ds.samples]
    token_vocab = sorted(set(observed_tokens))
    token_to_id = {t: i for i, t in enumerate(token_vocab)}
    token_ids = np.asarray(
        [token_to_id[t] for t in observed_tokens], dtype=np.int64
    )
    n_tokens = len(token_vocab)

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
        input_dim=h_full.shape[1], config=proj_cfg
    )
    entropy_targets = _next_state_entropy_targets(fsm, current_states)
    failure_targets = _failure_targets(
        train_ds.y_next, fsm, current_states
    )
    proj_diag = train_predictive_projection(
        projection,
        h_full,
        next_states=train_ds.y_next,
        entropy_targets=entropy_targets,
        failure_targets=failure_targets,
        token_ids=(
            token_ids if adversarial_token_weight > 0.0 else None
        ),
        epochs=n_projection_train_epochs,
        lr=1e-3,
        batch_size=64,
        seed=int(seed),
    )
    with torch.no_grad():
        out = projection(torch.from_numpy(h_full.astype(np.float32)))
    z_full = out["z"].cpu().numpy().astype(np.float64)
    next_logits = out["next_state_logits"].cpu().numpy().astype(np.float64)
    entropy_pred = out["entropy_pred"].cpu().numpy().astype(np.float64)
    failure_pred = out["failure_logit"].cpu().numpy().astype(np.float64)
    phase_2_seconds = time.perf_counter() - t2

    # ---- Phase 3: tropical-lite partition by argmax(next-state head) ----
    t3 = time.perf_counter()
    argmax_cells = next_logits.argmax(axis=1).astype(np.int64)
    # Margin-to-tie wall: top1 - top2 over logits.
    sorted_logits = np.sort(next_logits, axis=1)
    margin_to_tie = (sorted_logits[:, -1] - sorted_logits[:, -2]).astype(
        np.float64
    )
    # Per-cell stats.
    cells = np.unique(argmax_cells)
    K_argmax = int(cells.size)

    transition_counts = np.zeros((K_argmax, K_argmax), dtype=np.int64)
    # Map original argmax cell id -> compact contiguous id in [0, K_argmax).
    cell_to_idx = {int(c): i for i, c in enumerate(cells)}
    compact = np.asarray(
        [cell_to_idx[int(c)] for c in argmax_cells], dtype=np.int64
    )
    for p in range(int(program_ids.max()) + 1):
        rows = np.where(program_ids == p)[0]
        if rows.size < 2:
            continue
        seq = compact[rows]
        transition_counts += expected_counts_observed(seq, K_argmax)

    # Per-cell aggregate stats.
    per_cell = []
    for c_idx, _c_id in enumerate(cells):
        members = np.where(compact == c_idx)[0]
        if members.size == 0:
            continue
        # Sigmoid of failure_logit averaged over members = failure prob.
        f_logit = failure_pred[members]
        f_prob = 1.0 / (1.0 + np.exp(-f_logit))
        per_cell.append(
            {
                "support": int(members.size),
                "failure_rate": float(f_prob.mean()),
                "entropy_mean": float(entropy_pred[members].mean()),
                "mean_margin_to_tie": float(margin_to_tie[members].mean()),
                "current_state_distribution": np.bincount(
                    current_states[members], minlength=V
                ).astype(np.int64),
            }
        )
    phase_3_seconds = time.perf_counter() - t3

    # ---- Phase 4: behavioral merge ----
    t4 = time.perf_counter()
    # Emission counts (per-cell, per-current-state). Use as emission_counts
    # input to the bisimulation quotient.
    emission_counts = np.stack(
        [c["current_state_distribution"] for c in per_cell], axis=0
    ).astype(np.int64)
    quotient = quotient_by_bisimulation(
        compact,
        transition_counts,
        target_K=min(target_K, K_argmax),
        criterion="full",
        emission_counts=emission_counts,
        seed=int(seed),
    )
    merged_labels = quotient.labels
    merged_counts = quotient.transition_counts
    n_regimes = merged_counts.shape[0]
    phase_4_seconds = time.perf_counter() - t4

    # Per-regime stats after merge (re-aggregate).
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

    # Edges with Beta confidence.
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

    # Sanity: alignment Hamming if we merged exactly to V regimes.
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

    # Emit the control graph JSON.
    control_graph = {
        "grammar": grammar,
        "n_total_steps": int(h_full.shape[0]),
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
            "encoder_train_accuracy": encoder_train_acc,
            "projection": proj_diag,
            "adversarial_token_weight": float(adversarial_token_weight),
            "n_tokens": int(n_tokens),
        },
    }
    (output_dir / "control_graph.json").write_text(
        json.dumps(control_graph, indent=2)
    )

    # Aggregate result.
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
    return E28Result(
        grammar=grammar,
        n_total_steps=int(h_full.shape[0]),
        n_samples=int(program_ids.max()) + 1,
        V_ground_truth=int(V),
        n_argmax_cells=int(K_argmax),
        n_regimes_after_merge=int(n_regimes),
        mean_failure_rate=mean_failure,
        mean_entropy=mean_entropy,
        mean_margin_to_tie=mean_margin,
        mean_purity_against_current_state=mean_purity,
        aligned_hamming_at_target_V=float(aligned_hamming),
        encoder_train_accuracy=float(encoder_train_acc),
        projection_train_accuracy=float(proj_diag.get("next_state_acc", 0.0)),
        projection_failure_accuracy=float(proj_diag.get("failure_acc", 0.0)),
        projection_entropy_mse=float(proj_diag.get("entropy_mse", 0.0)),
        projection_token_accuracy=float(
            proj_diag.get("token_acc", float("nan"))
        ),
        adversarial_token_weight=float(adversarial_token_weight),
        phase_1_wall_clock_seconds=float(phase_1_seconds),
        phase_2_wall_clock_seconds=float(phase_2_seconds),
        phase_3_wall_clock_seconds=float(phase_3_seconds),
        phase_4_wall_clock_seconds=float(phase_4_seconds),
        total_wall_clock_seconds=float(total),
        peak_memory_kb=float(_peak_memory_kb()),
    )
