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

import datetime as dt
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
from nga.arch.control_policy import ControlPolicy
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
from nga.arch.singularity_detector import SingularityDetector
from nga.drivers.decision_trace_jsonl import (
    ControlAction,
    DecisionTraceRecord,
    MaskAction,
    open_decision_trace_writer,
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
    eval_n_programs: int | None = None,
    eval_seed: int | None = None,
) -> E28Result:
    """Predictive Control Graph Extractor MVP run on one grammar.

    When ``eval_n_programs`` (and optionally ``eval_seed``) is provided,
    the σ + control decision_trace.jsonl is emitted on a fresh dataset
    sampled with ``eval_seed`` (defaulting to ``seed + 1``) — not on the
    training data. The trained projection runs forward over the held-out
    inputs; argmax FSM states that never appeared as a regime cell during
    training map to a sentinel ``regime_unknown`` and fire the illegal
    signal. Existing ``control_graph.json`` artefact is unchanged either
    way (it always reflects the training-set regime extraction).
    """
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

    # ---- σ + control trace on the regime graph ----
    # The σ ensemble and 3-branch control policy operate on whichever graph
    # the system is auditing. Here the audited graph is the PCG-X regime
    # graph (not the typed FSM); the trace records both the model's
    # confidence signals (margin, decision-tie) and the regime-level
    # decisions (illegality, recovery, abstention).
    # FSM-state → regime mapping derived from the bisimulation cluster_map.
    # Reused by both the train- and eval-slice trace branches; FSM states
    # that never appeared as an argmax cell map to _UNKNOWN_REGIME.
    fsm_to_regime = np.full(int(V), _UNKNOWN_REGIME, dtype=np.int64)
    for fsm_state, compact_idx in cell_to_idx.items():
        fsm_to_regime[int(fsm_state)] = int(quotient.cluster_map[compact_idx])

    if eval_n_programs is not None:
        eval_logits, eval_predicted_regimes, eval_program_ids = _harvest_eval_slice(
            fsm=fsm,
            dispatch=dispatch,
            base_encoder=base_encoder,
            projection=projection,
            fsm_to_regime=fsm_to_regime,
            eval_n_programs=int(eval_n_programs),
            eval_seed=(
                int(eval_seed) if eval_seed is not None else int(seed) + 1
            ),
            n_fsm_states=int(V),
        )
        trace_logits = eval_logits
        trace_predicted_regimes = eval_predicted_regimes
        trace_program_ids = eval_program_ids
        trace_label = "eval"
    else:
        # Training-set trace: predicted regime IS merged_labels[i].
        trace_logits = next_logits
        trace_predicted_regimes = merged_labels.astype(np.int64)
        trace_program_ids = program_ids
        trace_label = "train"

    _emit_regime_decision_trace(
        output_dir=output_dir,
        run_id=run_id,
        seed=int(seed),
        next_logits=trace_logits,
        predicted_regimes=trace_predicted_regimes,
        program_ids=trace_program_ids,
        fsm_to_regime=fsm_to_regime,
        regimes=regimes,
        regime_edges=edges,
        n_regimes=n_regimes,
        trace_label=trace_label,
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


# ---------------------------------------------------------------------------
# σ + control trace on the regime graph
# ---------------------------------------------------------------------------


_THETA_NORMAL = 0.3
_THETA_ABSTAIN = 0.7
_GOAL_FAILURE_RATE_THRESHOLD = 0.05
_LOOP_WINDOW = 5
_UNKNOWN_REGIME = -1


def _regime_name(regime_id: int) -> str:
    """``regime_K`` for valid IDs, ``regime_unknown`` for the -1 sentinel."""
    return "regime_unknown" if regime_id == _UNKNOWN_REGIME else f"regime_{regime_id}"


def _harvest_eval_slice(
    *,
    fsm: GraphFSM,
    dispatch: dict,
    base_encoder: "_TrainableEncoder",
    projection: PredictiveProjection,
    fsm_to_regime: np.ndarray,
    eval_n_programs: int,
    eval_seed: int,
    n_fsm_states: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the trained encoder + projection over a held-out dataset.

    Returns
    -------
    eval_logits:
        ``(n_eval_steps, n_fsm_states)`` projection next-state logits.
    predicted_regimes:
        ``(n_eval_steps,)`` regime id per step; argmax FSM states that
        never appeared as a training-set regime cell map to
        ``_UNKNOWN_REGIME``.
    program_ids:
        ``(n_eval_steps,)`` per-step sequence id.
    """
    eval_ds = dispatch["loader"](fsm, eval_n_programs, eval_seed)
    seq_attr = dispatch["sequence_id_attr"]
    program_ids = np.asarray(
        [int(getattr(s, seq_attr)) for s in eval_ds.samples], dtype=np.int64
    )
    current_states = eval_ds.current_states.astype(np.int64)

    V = n_fsm_states
    state_onehot = np.zeros((eval_ds.X.shape[0], V), dtype=np.float64)
    state_onehot[np.arange(eval_ds.X.shape[0]), current_states] = 1.0
    X_conditioned = np.concatenate(
        [eval_ds.X.astype(np.float64), state_onehot], axis=1
    )

    X_t = torch.from_numpy(X_conditioned.astype(np.float32))
    with torch.no_grad():
        h_eval = base_encoder.encode(X_t).cpu().numpy().astype(np.float64)
        out = projection(torch.from_numpy(h_eval.astype(np.float32)))
    eval_logits = out["next_state_logits"].cpu().numpy().astype(np.float64)

    # Map argmax FSM states through the shared fsm_to_regime table. Argmaxes
    # whose FSM state never appeared as a training cell hit the
    # _UNKNOWN_REGIME sentinel already baked into the table.
    argmax_fsm = eval_logits.argmax(axis=1).astype(np.int64)
    predicted_regimes = fsm_to_regime[argmax_fsm]
    return eval_logits, predicted_regimes, program_ids


def _emit_regime_decision_trace(
    *,
    output_dir: Path,
    run_id: str,
    seed: int,
    next_logits: np.ndarray,
    predicted_regimes: np.ndarray,
    program_ids: np.ndarray,
    fsm_to_regime: np.ndarray,
    regimes: list[RegimeNode],
    regime_edges: list[RegimeEdge],
    n_regimes: int,
    trace_label: str = "train",
) -> None:
    """Write one ``DecisionTraceRecord`` per step to ``decision_trace.jsonl``.

    σ is computed on the model's FSM-level prediction (margin, decision-tie
    come from the FSM softmax) and on the regime-graph context (illegality
    against the regime edge set, loop revisits within the program). The
    control verdict operates on the regime graph: recovery searches BFS
    over the regime legality adjacency toward goal regimes (those with
    low failure rate).

    Parameters
    ----------
    predicted_regimes:
        ``(n_steps,)`` per-step regime id. Values ``>= 0`` index into the
        regime graph; the sentinel ``_UNKNOWN_REGIME`` (``-1``) marks an
        argmax FSM state that never appeared as a regime cell during
        training. Unknown predictions automatically fire the illegal
        signal and disable recovery BFS for downstream steps.
    trace_label:
        Free-text tag echoed into ``ablation`` so a downstream reader can
        distinguish train-set vs. held-out-eval rows on the same grammar.

    Schema 1.1 fields beyond v1.0 are left ``None``: PCG-X does not commit
    to a node-tuple identity over multiple axes yet.
    """
    # Build the regime legality adjacency from observed edges. The control
    # policy's BFS uses this to find a path from current_regime to a goal
    # regime.
    regime_legality = np.zeros((n_regimes, n_regimes), dtype=bool)
    for e in regime_edges:
        if e.count > 0:
            regime_legality[e.src, e.dst] = True

    # Goal regimes: those with non-empty support AND failure rate below
    # threshold. Recovery will steer toward any reachable goal.
    goal_regimes = [
        r.regime_id
        for r in regimes
        if r.support > 0 and r.failure_rate < _GOAL_FAILURE_RATE_THRESHOLD
    ]

    detector = SingularityDetector()
    policy = ControlPolicy(
        theta_normal=_THETA_NORMAL,
        theta_abstain=_THETA_ABSTAIN,
        legality_matrix=regime_legality,
        goal_states=goal_regimes if goal_regimes else None,
    )

    # Pre-compute softmax probabilities for the whole run. Numerically
    # stable subtract-max-then-exp.
    shifted = next_logits - next_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    n_steps = int(next_logits.shape[0])
    timestamp = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    ablation_label = f"A0_{trace_label}"

    with open_decision_trace_writer(
        output_dir / "decision_trace.jsonl"
    ) as trace_writer:
        for i in range(n_steps):
            program_id = int(program_ids[i])
            predicted_regime = int(predicted_regimes[i])
            predicted_is_unknown = predicted_regime == _UNKNOWN_REGIME

            # Top-2 FSM states from the model's actual prediction. Argpartition
            # picks the 2 highest without sorting the rest.
            top2_idx = np.argpartition(probs[i], -2)[-2:]
            if probs[i, top2_idx[0]] < probs[i, top2_idx[1]]:
                top2_idx = top2_idx[::-1]
            top1_fsm = int(top2_idx[0])
            top2_fsm = int(top2_idx[1])
            top1_prob = float(probs[i, top1_fsm])
            top2_prob = float(probs[i, top2_fsm])
            margin = top1_prob - top2_prob

            # Current regime = previous step's regime in the same program.
            # First step of a program (or a step whose previous prediction
            # was unknown) has no current regime — recovery cannot fire.
            current_regime: int | None
            if i > 0 and int(program_ids[i - 1]) == program_id:
                prev_regime = int(predicted_regimes[i - 1])
                current_regime = prev_regime if prev_regime != _UNKNOWN_REGIME else None
            else:
                current_regime = None

            # Illegality:
            #   - Unknown predicted regime is always illegal (out of graph).
            #   - Known predicted regime: missing edge (current → predicted)
            #     in the regime graph fires the signal.
            #   - First step of a program (no current_regime) cannot be
            #     illegal: there is no edge to check.
            if predicted_is_unknown:
                is_illegal = True
            elif current_regime is not None:
                is_illegal = not regime_legality[current_regime, predicted_regime]
            else:
                is_illegal = False

            # Loop risk: did predicted_regime appear in the last K steps of
            # this same program? Skip for unknown predictions (no regime to
            # compare against).
            loop_risk = 0.0
            if i > 0 and not predicted_is_unknown:
                window_lo = max(0, i - _LOOP_WINDOW)
                window_program = program_ids[window_lo:i] == program_id
                if window_program.any():
                    recent_regimes = predicted_regimes[window_lo:i][window_program]
                    if predicted_regime in recent_regimes:
                        loop_risk = 1.0

            decision_tie_strength = SingularityDetector.signal_from_decision_tie(
                top1_prob - top2_prob, threshold=0.02
            )

            sigma, contribs = detector.compute(
                margin=margin,
                is_illegal=bool(is_illegal),
                loop_risk=float(loop_risk),
                decision_tie_strength=float(decision_tie_strength),
            )

            policy_result = policy.decide(
                sigma=sigma,
                model_prediction=predicted_regime,
                current_state=current_regime,
            )

            top1_name = _regime_name(predicted_regime)
            top2_regime = int(fsm_to_regime[top2_fsm])
            top2_name = (
                _regime_name(top2_regime) if top2_regime != _UNKNOWN_REGIME else None
            )
            current_name = (
                _regime_name(current_regime) if current_regime is not None else None
            )

            mask_action = MaskAction(
                enabled=False,  # PCG-X does not apply a hard mask to the prediction
                current_state=current_name,
                illegal_indices_zeroed=[],
                pre_mask_argmax=top1_name,
                post_mask_argmax=top1_name,
                mask_changed_prediction=False,
            )
            control_action = ControlAction(
                decision=policy_result.decision,
                reason=policy_result.reason,
                sigma_threshold_used=_THETA_NORMAL,
                sigma_observed=float(sigma),
                abstain_threshold_used=_THETA_ABSTAIN,
            )

            sigma_signals = {
                "margin": float(contribs.margin_signal),
                "decision_tie": float(contribs.decision_tie_signal),
                "illegal": float(contribs.illegal_signal),
                "loop": float(contribs.loop_signal),
                "stabilizer": float(contribs.stabilizer_signal),
                "catastrophe_bias": float(contribs.catastrophe_bias),
            }

            record = DecisionTraceRecord(
                run_id=run_id,
                experiment="E28",
                ablation=ablation_label,
                seed=seed,
                step=i,
                sample_id=f"prog{program_id}_step{i}",
                confidence=top1_prob,
                top1_state=top1_name,
                top2_state=top2_name,
                top1_prob=top1_prob,
                top2_prob=top2_prob,
                margin=float(margin),
                mask=mask_action,
                sigma_total=float(sigma),
                sigma_signals=sigma_signals,
                control=control_action,
                timestamp=timestamp,
            )
            trace_writer.append(record)
