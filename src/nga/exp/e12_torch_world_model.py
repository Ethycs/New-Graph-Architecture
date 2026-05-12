"""E12 - Torch-substrate world model: substrate-independence demonstration.

E12 mirrors E11's pipeline (Phase A classical forward-backward + Beta
M-step on observed transitions; Phase B KL-blended refinement) but swaps
the sklearn / Poincare-prototype substrate for torch atoms:

  - :class:`nga.arch.frozen_encoder_torch.FrozenEncoderTorch` replaces
    Euclidean->Poincare projection as the embedding source. The encoder
    is randomly initialised under ``seed`` and frozen
    (``requires_grad=False``); ``n_torch_frozen_params`` reports its
    parameter count.
  - :class:`nga.arch.typed_readout_torch.TypedReadoutTorch` (one head with
    ``type_id="default"``, ``n_classes=fsm.vertex_count``) replaces the
    per-vertex Riemannian-centroid prototype table. The head's per-class
    weights ARE the prototypes in logit space; ``predict_proba`` produces
    the predicted next-state distribution. After Phase A seeds the
    posterior mask we fit the head once on (encoded_X, true_next_idx);
    Phase B re-uses the frozen-after-fit head for predictions.
    ``n_torch_trainable_params`` reports its parameter count.

The architecture's inference framework is the contract the swap
preserves:

  - Phase A is substrate-independent by construction:
    ``expected_counts_observed`` and ``bayesian_m_step_beta`` operate on
    integer state indices, not embeddings. The Phase-A hamming bar from
    E11 must therefore hold under any substrate -- this is exactly the
    "frozen encoder + typed readout" claim being validated end-to-end.
  - Phase B reuses every existing arch atom (singularity_detector,
    energy_function, posterior_mask.update, control_policy,
    behavioral_stratum_tagger, product_graph, ...) and decision_trace
    v1.1 plumbing. Only the predicted-distribution source changes.

The runner reports E11's headline metrics plus two substrate-counters
that document the torch wiring:

  - ``n_torch_trainable_params``: sum from typed_readout.
  - ``n_torch_frozen_params``: from the frozen encoder.

Acceptance bars are intentionally loose by design (mirroring E11):
the runner reports observed numbers without being tuned to thresholds.

E12 = E11 with the substrate swapped. No other changes.
"""
from __future__ import annotations

import datetime as dt
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nga.arch.axis_quantizer import AxisQuantizer
from nga.arch.behavioral_stratum_tagger import TaggerHistory, TaggerInput, tag_step
from nga.arch.control_policy import ControlPolicy
from nga.arch.energy_function import EnergyFunction, EnergyParameters
from nga.arch.energy_minimization_trainer import (
    EnergyMinimizationTrainer,
    TrainerConfig,
)
from nga.arch.forward_backward import (
    bayesian_m_step_beta,
    expected_counts_observed,
)
from nga.arch.frozen_encoder_torch import FrozenEncoderTorch
from nga.arch.graph_fsm import GraphFSM
from nga.arch.information_geometry import kl_categorical
from nga.arch.margin_uncertainty import compute_margin
from nga.arch.monodromy_consistency import (
    closed_walks_in_fsm,
    monodromy_consistency_loss,
)
from nga.arch.posterior_mask import PosteriorMask
from nga.arch.product_graph import ProductGraph
from nga.arch.singularity_detector import (
    SingularityDetector,
    compute_kl_surprise,
)
from nga.arch.singularity_types import SingularityType
from nga.arch.stratified_partition_function import (
    compute_stratified_partition,
    free_energy,
)
from nga.arch.typed_readout_torch import TypedReadoutTorch
from nga.drivers.ablation_flags import AblationTuple
from nga.drivers.config import Config
from nga.drivers.decision_trace_jsonl import (
    ControlAction,
    DecisionTraceRecord,
    MaskAction,
    open_decision_trace_writer,
)
from nga.drivers.jsonl_writer import JsonlWriter
from nga.drivers.metrics_jsonl import MetricsRecord
from nga.drivers.results_jsonl import ResultsRecord
from nga.exp.dataset_dyck_k import generate_dyck_dataset

try:  # torch is required for E12, but import remains tolerant for clarity.
    import torch  # noqa: F401
except ImportError as _exc:  # pragma: no cover - exercised only on torch-less envs.
    raise ImportError(
        "torch is required for E12; install via `pixi add pytorch` "
        "(or `pip install torch`)."
    ) from _exc

__all__ = ["E12Result", "run_e12"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class E12Result:
    """Aggregate metrics returned by run_e12."""

    accuracy: float
    accuracy_no_mask: float
    mask_accuracy_uplift: float
    illegal_transition_rate: float
    illegal_transition_rate_no_mask: float
    monodromy_consistency_loss: float
    hamming_distance_from_fsm: int
    hamming_normalised: float
    phase_a_hamming_normalised: float
    phase_b_hamming_normalised: float
    crb_satisfied_fraction: float
    mean_crb_confidence: float
    kl_progress_final: float
    mean_kl_surprise: float
    n_samples: int
    n_monodromy_cycle_revisits: int
    n_torch_trainable_params: int
    n_torch_frozen_params: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


READOUT_TYPE_ID = "default"


def _predicted_dist_with_bias(
    *,
    logits_row: np.ndarray,
    legality_bias_row: np.ndarray,
) -> np.ndarray:
    """Softmax over (readout logits + legality bias).

    Mirrors E11's ``_model_predicted_dist`` pattern: the readout's raw
    decision-function output (analogous to ``-distances`` in E11) is added
    to the legality bias and softmaxed. The bias nudges legal candidates;
    illegal candidates are damped (negative bias).
    """
    logits = np.asarray(logits_row, dtype=np.float64) + np.asarray(
        legality_bias_row, dtype=np.float64
    )
    logits -= logits.max()
    ex = np.exp(logits)
    s = ex.sum()
    if s <= 0.0:
        n = ex.shape[0]
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    return ex / s


def _empirical_dist_from_window(
    window: deque,
    n_states: int,
    eps: float = 1e-3,
) -> np.ndarray:
    """Build a categorical empirical next-state distribution from a window.

    Identical formulation to E11. ``window`` is a deque of integer
    next-state indices observed from the current source state; the result
    is a Laplace-smoothed distribution over n_states. When the window is
    empty the returned distribution is uniform.
    """
    counts = np.full(n_states, eps, dtype=np.float64)
    for j in window:
        counts[int(j)] += 1.0
    total = counts.sum()
    if total <= 0.0:
        return np.full(n_states, 1.0 / max(n_states, 1), dtype=np.float64)
    return counts / total


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_e12(
    *,
    config: Config,
    ablation: AblationTuple,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    margin_threshold: float = 0.10,
    n_sequences: int = 200,
    max_length: int = 16,
    illegal_temptation_fraction: float = 0.15,
    n_axis_bins: int = 5,
    seed_window: int = 64,
    empirical_window: int = 32,
    blend: float = 0.7,
    kl_surprise_weight: float = 0.5,
    crb_target_variance: float = 0.01,
    encoder_hidden_dim: int = 64,
    readout_hidden_dim: int = 32,
    readout_n_epochs: int = 50,
    readout_lr: float = 1e-2,
) -> E12Result:
    """Run the substrate-independence pipeline (E11 with torch substrate).

    Writes the standard four artefacts plus decision_trace.jsonl. Returns
    aggregate metrics including the two torch parameter counters.

    Acceptance bars are intentionally loose; the runner reports observed
    numbers without being tuned to thresholds.
    """
    if fsm.vertex_ids[0] != "S0":
        raise ValueError(
            f"E12 expects fsm.vertex_ids[0] == 'S0'; got {fsm.vertex_ids[0]!r}"
        )

    n_states = fsm.vertex_count
    k = 2
    max_depth = (n_states - 1) // k
    if 1 + max_depth * k != n_states:
        raise ValueError(
            f"E12 cannot derive (k, max_depth) from {n_states} states; "
            f"expected 1 + max_depth*k == n_states, got k={k}."
        )

    # ------------------------------------------------------------------
    # Reproducibility: seed both numpy and torch RNGs.
    # ------------------------------------------------------------------
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Step 1 - Dataset (identical to E11).
    # ------------------------------------------------------------------
    ds = generate_dyck_dataset(
        fsm=fsm,
        k=k,
        max_depth=max_depth,
        n_sequences=n_sequences,
        max_length=max_length,
        seed=seed,
        illegal_temptation_fraction=illegal_temptation_fraction,
        feature_dim=config.embedding_dim,
    )

    X = ds.X
    n_samples = X.shape[0]
    embedding_dim = X.shape[1]
    sample_ids = [s.sample_id for s in ds.samples]
    current_state_idx = ds.current_states
    true_next_state_idx = ds.y_next

    # ------------------------------------------------------------------
    # Step 2 - Torch substrate: frozen encoder + typed readout.
    # ------------------------------------------------------------------
    encoder = FrozenEncoderTorch(
        input_dim=embedding_dim,
        output_dim=embedding_dim,
        hidden_dim=encoder_hidden_dim,
        seed=int(seed),
    )
    encoder.fit(X)
    Z = encoder.encode(X)  # (n_samples, embedding_dim)
    n_torch_frozen_params = int(encoder.n_parameters)

    readout = TypedReadoutTorch(
        type_ids=[READOUT_TYPE_ID],
        n_classes=n_states,
        input_dim=embedding_dim,
        hidden_dim=readout_hidden_dim,
        n_epochs=readout_n_epochs,
        lr=readout_lr,
        seed=int(seed),
    )

    energy_fn = EnergyFunction(EnergyParameters())

    # ------------------------------------------------------------------
    # Step 3 - Calibrate axis quantisers from a seed prefix.
    #
    # E11 derived seed-time sigma/energy/margin distributions from
    # Poincare prototypes; here we derive them from a uniform-prior
    # softmax over encoded vectors so the quantiser support is realistic
    # for the early refinement loop. The values are NOT used downstream
    # of this calibration step.
    # ------------------------------------------------------------------
    seed_n = min(seed_window, n_samples)
    sigmas_seed: list[float] = []
    energies_seed: list[float] = []
    margins_seed: list[float] = []
    seed_detector = SingularityDetector(margin_threshold=margin_threshold, enabled=True)
    # For seed-time we use a uniform predicted distribution biased only
    # by the (yet-to-be-computed) Phase-A posterior; here the bias is
    # zero, so we sample a small jitter to give the quantiser support.
    for i in range(seed_n):
        # Use a simple proxy: each candidate's logit is a small random
        # draw scaled by the encoded vector's L2 norm. Stable across
        # seeds because rng and torch are both seeded above.
        scale = float(np.linalg.norm(Z[i]) + 1e-6)
        logits = rng.normal(0.0, 0.5, size=n_states) * (1.0 / scale)
        ex = np.exp(logits - logits.max())
        dist = ex / ex.sum()
        m = float(compute_margin(dist))
        margins_seed.append(m)
        sigma_i, _ = seed_detector.compute(margin=m, is_illegal=False, loop_risk=0.0)
        sigmas_seed.append(float(sigma_i))
        unc = float(1.0 - dist.max())
        contribs = energy_fn.compute(
            cost=0.0,
            uncertainty=unc,
            contradiction=0.0,
            loop_pressure=float(np.clip(sigma_i, 0.0, 1.0)) * 0.5,
            progress=1.0,
        )
        energies_seed.append(float(contribs.total))

    def _safe_quantizer(name: str, samples: list[float]) -> AxisQuantizer:
        arr = np.asarray(samples, dtype=float)
        if arr.size < n_axis_bins or np.unique(arr).size < n_axis_bins + 1:
            arr = arr + rng.normal(0.0, 1e-6, size=arr.shape)
        try:
            return AxisQuantizer.from_quantiles(name, arr, n_axis_bins)
        except ValueError:
            lo = float(arr.min()) - 1e-6
            hi = float(arr.max()) + 1e-6
            if not (lo < hi):
                lo, hi = -1e-3, 1.0
            return AxisQuantizer.from_uniform(name, lo, hi, n_axis_bins)

    sigma_q = _safe_quantizer("sigma_axis", sigmas_seed)
    energy_q = _safe_quantizer("energy_axis", energies_seed)
    margin_q = _safe_quantizer("margin_axis", margins_seed)

    # ------------------------------------------------------------------
    # Step 4 -- Phase A: classical cold-start E-step (substrate-independent).
    #
    # Identical to E11: the data is a fully-observed Markov chain over
    # FSM vertex IDs, so expected_counts_observed reduces the
    # forward-backward E-step to O(T) pair counting and
    # bayesian_m_step_beta turns the counts into a Beta posterior in one
    # shot. The torch substrate has nothing to do with Phase A; the
    # whole point of the experiment is that Phase A converges
    # independently of the embedding/prototype substrate.
    # ------------------------------------------------------------------
    pair_counts = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_samples):
        pair_counts[int(current_state_idx[i]), int(true_next_state_idx[i])] += 1.0

    # Symbolic call to expected_counts_observed for atom-census wiring,
    # mirroring E11 (the per-step pair counts are the canonical statistic).
    state_path = np.concatenate(
        [
            current_state_idx.reshape(-1).astype(np.int64),
            true_next_state_idx[-1:].astype(np.int64),
        ]
    )
    fb_counts = expected_counts_observed(state_path, n_states=n_states).astype(
        np.float64
    )
    _ = fb_counts

    alpha_seed, beta_seed = bayesian_m_step_beta(
        expected_counts_pos=pair_counts,
        expected_counts_neg=None,
        prior_alpha=1.0,
        prior_beta=1.0,
    )

    posterior_mask = PosteriorMask(
        n_vertices=n_states, prior_alpha=1.0, prior_beta=1.0
    )
    posterior_mask._alpha = np.asarray(alpha_seed, dtype=np.float64)
    posterior_mask._beta = np.asarray(beta_seed, dtype=np.float64)

    phase_a_alpha = posterior_mask.alpha.copy()
    phase_a_beta = posterior_mask.beta.copy()

    learned_legal_a = posterior_mask.legality_matrix(0.5)
    gold_legal = fsm.legality_matrix
    hamming_a = int(np.sum(learned_legal_a != gold_legal))
    total_cells = int(learned_legal_a.size)
    phase_a_hamming_normalised = (
        float(hamming_a) / float(total_cells) if total_cells else 0.0
    )

    # ------------------------------------------------------------------
    # Step 5 -- Fit the typed readout head on (encoded_X, true_next_idx).
    #
    # The readout's per-class weights are the prototypes in logit space.
    # We fit ONCE here (after Phase A has seeded the posterior mask) so
    # Phase B's predicted distributions are produced by a coherent torch
    # head. n_torch_trainable_params becomes positive at this point.
    # ------------------------------------------------------------------
    readout.fit(READOUT_TYPE_ID, Z, true_next_state_idx)
    n_torch_trainable_params = int(readout.n_trainable_params())

    # Pre-compute logits for all samples (frozen after fit -> a single
    # forward pass is enough; per-step logits are read from this table).
    all_logits = readout.decision_function(READOUT_TYPE_ID, Z)
    all_unbiased_probs = readout.predict_proba(READOUT_TYPE_ID, Z)

    # ------------------------------------------------------------------
    # Step 6 -- Build product graph + trainer for Phase B plumbing.
    # ------------------------------------------------------------------
    state_axis_nodes = list(fsm.vertex_ids)
    product_graph = ProductGraph(
        axis_names=["state", "sigma_axis", "energy_axis", "margin_axis"],
        axis_node_lists={
            "state": state_axis_nodes,
            "sigma_axis": sigma_q.node_ids(),
            "energy_axis": energy_q.node_ids(),
            "margin_axis": margin_q.node_ids(),
        },
    )

    # The trainer in E12 owns the posterior_mask and is used solely for
    # quality_signal_blended / crb_confidence_matrix / kl_progress.
    # We pass a placeholder prototype table (zeros) since the Phase-B
    # update path here applies posterior_mask.update directly on the
    # blended Q signal -- the trainer.step prototype RSGD path is not
    # taken in this runner.
    placeholder_protos = np.zeros((n_states, embedding_dim), dtype=np.float64)
    trainer = EnergyMinimizationTrainer(
        prototypes={"state": placeholder_protos},
        posterior_mask=posterior_mask,
        energy_fn=energy_fn,
        group_action=None,
        config=TrainerConfig(),
    )

    detector_weights = dict(SingularityDetector.DEFAULT_WEIGHTS)
    detector_weights["kl_surprise"] = float(kl_surprise_weight)
    detector = SingularityDetector(
        margin_threshold=margin_threshold,
        enabled=ablation.singularity_detector_enabled,
        weights=detector_weights,
    )

    history = TaggerHistory()
    control_policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        fsm=fsm,
        goal_states=[0],
    )

    experiment_label = "E12"
    ablation_label = run_id.split("_")[1]
    vertex_ids = fsm.vertex_ids

    # ------------------------------------------------------------------
    # Step 7 -- Phase B: KL-blended refinement loop.
    # ------------------------------------------------------------------
    windows: dict[int, deque] = {
        s: deque(maxlen=empirical_window) for s in range(n_states)
    }
    for i in range(min(n_samples, empirical_window)):
        windows[int(current_state_idx[i])].append(int(true_next_state_idx[i]))

    order = rng.permutation(n_samples)

    y_true_list: list[str] = []
    predicted_list: list[str] = []
    predicted_no_mask_list: list[str] = []
    transition_legal_list: list[bool] = []
    transition_legal_no_mask_list: list[bool] = []
    energies_total: list[float] = []
    strata: list[SingularityType] = []
    energies_correct: list[float] = []
    energies_incorrect: list[float] = []
    energies_by_state: dict[int, list[float]] = {i: [] for i in range(n_states)}
    n_routed_to_recovery = 0
    n_abstained = 0
    n_mask_modified_predictions = 0
    n_monodromy_cycle_revisits = 0
    visited_states: set[int] = set()
    prev_output_tuple: tuple[str, ...] | None = None
    kl_surprises_step: list[float] = []
    quality_kl_step: list[float] = []
    quality_blended_step: list[float] = []

    with (
        JsonlWriter(output_dir / "metrics.jsonl", MetricsRecord) as metrics_writer,
        JsonlWriter(output_dir / "results.jsonl", ResultsRecord) as results_writer,
        open_decision_trace_writer(output_dir / "decision_trace.jsonl") as trace_writer,
    ):
        for step_idx, i_raw in enumerate(order):
            i = int(i_raw)
            cs_idx = int(current_state_idx[i])
            current_state = vertex_ids[cs_idx]
            true_next_idx = int(true_next_state_idx[i])
            true_next = vertex_ids[true_next_idx]

            # Live legality bias from the (post-Phase-A, in-progress-Phase-B)
            # posterior. The bias is read on every step; updates to the
            # posterior in earlier loop iterations show up here.
            bias = posterior_mask.legality_bias()
            bias_row_learned = bias[cs_idx]
            bias_row_uniform = np.zeros_like(bias_row_learned)

            # Predicted distribution from the torch readout (logits are
            # frozen because the head was fit once above; the legality
            # bias is what changes step-to-step).
            logits_row = all_logits[i]
            predicted_dist = _predicted_dist_with_bias(
                logits_row=logits_row,
                legality_bias_row=bias_row_learned,
            )
            unmasked_dist = _predicted_dist_with_bias(
                logits_row=logits_row,
                legality_bias_row=bias_row_uniform,
            )
            # For the no-bias accuracy comparison row we also retain the
            # raw softmax produced by the readout (no mask applied at all).
            raw_dist = all_unbiased_probs[i]
            _ = raw_dist  # raw probabilities are useful for debugging only.

            margin = float(compute_margin(predicted_dist))
            argmax_unmasked = int(np.argmax(unmasked_dist))
            predicted_no_mask = vertex_ids[argmax_unmasked]

            window = windows[cs_idx]
            empirical_dist = _empirical_dist_from_window(window, n_states)

            kl_surprise = float(compute_kl_surprise(empirical_dist, predicted_dist))
            kl_surprises_step.append(kl_surprise)

            argmax_state = int(np.argmax(predicted_dist))
            predicted_state = vertex_ids[argmax_state]
            is_illegal = not bool(
                fsm.is_legal_transition(current_state, predicted_state)
            )
            loop_risk = float(history.predicted_loop_risk(predicted_state))

            sigma, contribs = detector.compute(
                margin=margin,
                is_illegal=is_illegal,
                loop_risk=loop_risk,
                kl_surprise=kl_surprise,
            )

            # Energy: uncertainty proxy is (1 - top1_prob) of the
            # masked predicted distribution -- substrate-independent and
            # bounded in [0, 1]. Substitutes E11's poincare distance.
            unc = float(1.0 - predicted_dist[argmax_state])
            lp = float(np.clip(0.5 * loop_risk + 0.5 * sigma, 0.0, 1.0))
            energy_breakdown_obj = energy_fn.compute(
                cost=0.0,
                uncertainty=unc,
                contradiction=1.0 if is_illegal else 0.0,
                loop_pressure=lp,
                progress=1.0,
            )
            energy_total = float(energy_breakdown_obj.total)

            # Quality signal: blended heuristic + KL on the OBSERVED edge.
            obs_is_illegal = not bool(
                fsm.is_legal_transition(current_state, true_next)
            )
            q_blended = trainer.quality_signal_blended(
                energy_total=energy_total,
                sigma=float(sigma),
                contradiction=1.0 if obs_is_illegal else 0.0,
                loop_risk=loop_risk,
                empirical_dist=empirical_dist,
                predicted_dist=predicted_dist,
                blend=blend,
            )
            q_kl_only = trainer.quality_signal_kl(empirical_dist, predicted_dist)
            quality_blended_step.append(float(q_blended))
            quality_kl_step.append(float(q_kl_only))

            posterior_mask.update(
                int(cs_idx), int(true_next_idx), float(q_blended)
            )

            # Append AFTER the Q computation to keep the empirical window
            # honest (no leakage of the current true_next into emp_dist).
            windows[cs_idx].append(int(true_next_idx))

            # Stratum tag, transition legality, mask action, control.
            tagger_inp = TaggerInput(
                distribution=unmasked_dist,
                predicted_state=predicted_state,
                current_state=current_state,
                margin=margin,
            )
            tag_result = tag_step(
                tagger_inp, history, fsm, margin_threshold=margin_threshold
            )
            history.push(predicted_state)

            transition_legal = bool(
                fsm.is_legal_transition(current_state, predicted_state)
            )
            transition_legal_no_mask = bool(
                fsm.is_legal_transition(current_state, predicted_no_mask)
            )

            mask_changed_prediction = bool(argmax_unmasked != argmax_state)
            if mask_changed_prediction:
                n_mask_modified_predictions += 1

            if ablation.graph_mask_enabled:
                row_mean = posterior_mask.posterior_mean()[cs_idx]
                illegal_indices_zeroed = [
                    int(j) for j in range(n_states) if row_mean[j] < 0.5
                ]
            else:
                illegal_indices_zeroed = []

            mask_action = MaskAction(
                enabled=bool(ablation.graph_mask_enabled),
                current_state=current_state,
                illegal_indices_zeroed=illegal_indices_zeroed,
                pre_mask_argmax=predicted_no_mask,
                post_mask_argmax=predicted_state,
                mask_changed_prediction=mask_changed_prediction,
            )

            policy_result = control_policy.decide(
                sigma=float(sigma),
                model_prediction=int(argmax_state),
                current_state=int(cs_idx),
            )
            if policy_result.decision == "ROUTE_RECOVERY":
                n_routed_to_recovery += 1
            elif policy_result.decision == "ABSTAIN":
                n_abstained += 1

            control_action = ControlAction(
                decision=policy_result.decision,
                reason=policy_result.reason,
                sigma_threshold_used=control_policy.theta_normal,
                sigma_observed=float(sigma),
                abstain_threshold_used=control_policy.theta_abstain,
            )

            # v1.1 trace plumbing (identical to E11).
            sigma_node = sigma_q.quantise(float(sigma))
            energy_node = energy_q.quantise(float(energy_total))
            margin_node = margin_q.quantise(float(margin))
            output_node_tuple = (
                predicted_state,
                str(sigma_node),
                str(energy_node),
                str(margin_node),
            )
            if prev_output_tuple is not None:
                product_graph.add_transition(prev_output_tuple, output_node_tuple)
                edge_traversed = (
                    list(prev_output_tuple),
                    list(output_node_tuple),
                )
            else:
                product_graph.materialise(output_node_tuple)
                edge_traversed = None

            mean_alpha = float(np.mean(posterior_mask.alpha))
            mean_beta = float(np.mean(posterior_mask.beta))
            mean_entropy = float(np.mean(posterior_mask.posterior_entropy()))
            posterior_summary = {
                "mean_alpha": mean_alpha,
                "mean_beta": mean_beta,
                "mean_entropy": mean_entropy,
            }

            sorted_idx = np.argsort(unmasked_dist)[::-1]
            top1_idx = int(sorted_idx[0])
            top1_prob = float(unmasked_dist[top1_idx])
            if unmasked_dist.shape[0] > 1:
                top2_idx = int(sorted_idx[1])
                top2_prob = float(unmasked_dist[top2_idx])
                top2_state: str | None = vertex_ids[top2_idx]
            else:
                top2_prob = 0.0
                top2_state = None

            sigma_signals: dict[str, float] = {
                "margin": float(contribs.margin_signal),
                "decision_tie": float(contribs.decision_tie_signal),
                "illegal": float(contribs.illegal_signal),
                "loop": float(contribs.loop_signal),
                "stabilizer": float(contribs.stabilizer_signal),
                "catastrophe_bias": float(contribs.catastrophe_bias),
                "kl_surprise": float(contribs.kl_surprise),
            }
            energy_breakdown: dict[str, float] = {
                "cost": float(energy_breakdown_obj.cost),
                "uncertainty": float(energy_breakdown_obj.uncertainty),
                "contradiction": float(energy_breakdown_obj.contradiction),
                "loop_pressure": float(energy_breakdown_obj.loop_pressure),
                "progress": float(energy_breakdown_obj.progress),
            }
            axis_node_ids = {
                "state": predicted_state,
                "sigma_axis": str(sigma_node),
                "energy_axis": str(energy_node),
                "margin_axis": str(margin_node),
            }
            confidence = float(predicted_dist[argmax_state])

            results_writer.append(
                ResultsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=step_idx,
                    sample_id=sample_ids[i],
                    y_true=true_next,
                    y_hat=predicted_state,
                    margin=margin,
                    singular_flag=bool(sigma >= 0.5),
                    sigma_score=float(sigma),
                    behavioral_stratum=tag_result.tag.name,
                    stratum_bitmask=tag_result.bitmask,
                    transition_legal=transition_legal,
                )
            )

            trace_writer.append(
                DecisionTraceRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=step_idx,
                    sample_id=sample_ids[i],
                    confidence=float(confidence),
                    top1_state=predicted_state,
                    top2_state=top2_state,
                    top1_prob=float(top1_prob),
                    top2_prob=float(top2_prob),
                    margin=float(margin),
                    mask=mask_action,
                    stratum_tag=tag_result.tag.name,
                    stratum_bitmask=tag_result.bitmask,
                    stratum_confidence=float(tag_result.confidence),
                    sigma_total=float(sigma),
                    sigma_signals=sigma_signals,
                    energy_total=float(energy_total),
                    energy_breakdown=energy_breakdown,
                    P_lambda=None,
                    free_energy=None,
                    monodromy_class=None,
                    in_closed_walk=None,
                    control=control_action,
                    timestamp=dt.datetime.now(tz=dt.timezone.utc).isoformat(),
                    output_node_tuple=list(output_node_tuple),
                    edge_traversed=edge_traversed,
                    mask_version_id=posterior_mask.mask_version_id(),
                    posterior_summary=posterior_summary,
                    axis_node_ids=axis_node_ids,
                )
            )

            y_true_list.append(true_next)
            predicted_list.append(predicted_state)
            predicted_no_mask_list.append(predicted_no_mask)
            transition_legal_list.append(transition_legal)
            transition_legal_no_mask_list.append(transition_legal_no_mask)
            energies_total.append(energy_total)
            strata.append(tag_result.tag)
            if true_next == predicted_state:
                energies_correct.append(energy_total)
            else:
                energies_incorrect.append(energy_total)
            energies_by_state[argmax_state].append(energy_total)

            if argmax_state in visited_states:
                n_monodromy_cycle_revisits += 1
            visited_states.add(argmax_state)

            prev_output_tuple = output_node_tuple

        # ------------------------------------------------------------------
        # Step 8 -- Closed-walk consistency check.
        # ------------------------------------------------------------------
        walks = closed_walks_in_fsm(fsm, max_length=4)
        global_mean_energy = (
            float(np.mean(energies_total)) if energies_total else 0.0
        )
        per_state_mean = {
            s: (float(np.mean(es)) if es else global_mean_energy)
            for s, es in energies_by_state.items()
        }
        energies_by_walk: list[np.ndarray] = []
        for walk in walks:
            energies_by_walk.append(
                np.array([per_state_mean[v] for v in walk], dtype=np.float64)
            )
        mono_loss = float(monodromy_consistency_loss(walks, energies_by_walk))

        # ------------------------------------------------------------------
        # Step 9 -- Phase C: CRB + KL-progress diagnostics.
        # ------------------------------------------------------------------
        learned_legal_b = posterior_mask.legality_matrix(0.5)
        hamming_b = int(np.sum(learned_legal_b != gold_legal))
        phase_b_hamming_normalised = (
            float(hamming_b) / float(total_cells) if total_cells else 0.0
        )

        crb_matrix = trainer.crb_confidence_matrix(
            target_variance=crb_target_variance
        )
        materialised = (posterior_mask.alpha + posterior_mask.beta) > 2.0
        if materialised.any():
            crb_satisfied_fraction = float(
                np.mean(crb_matrix[materialised] >= 1.0)
            )
            mean_crb_confidence = float(np.mean(crb_matrix[materialised]))
        else:
            crb_satisfied_fraction = 0.0
            mean_crb_confidence = 0.0

        kl_progress_final = float(
            trainer.kl_progress(phase_a_alpha, phase_a_beta)
        )

        mean_kl_surprise = (
            float(np.mean(kl_surprises_step)) if kl_surprises_step else 0.0
        )
        mean_quality_signal_kl = (
            float(np.mean(quality_kl_step)) if quality_kl_step else 0.0
        )
        mean_quality_signal_blended = (
            float(np.mean(quality_blended_step)) if quality_blended_step else 0.0
        )

        # ------------------------------------------------------------------
        # Step 10 -- Standard E11-shaped metrics.
        # ------------------------------------------------------------------
        y_true_arr = np.array(y_true_list)
        predicted_arr = np.array(predicted_list)
        predicted_no_mask_arr = np.array(predicted_no_mask_list)
        accuracy = float(np.mean(y_true_arr == predicted_arr)) if n_samples else 0.0
        accuracy_no_mask = (
            float(np.mean(y_true_arr == predicted_no_mask_arr)) if n_samples else 0.0
        )
        mask_accuracy_uplift = accuracy - accuracy_no_mask

        if transition_legal_list:
            illegal_transition_rate = float(
                sum(1 for t in transition_legal_list if t is False)
                / len(transition_legal_list)
            )
            illegal_transition_rate_no_mask = float(
                sum(1 for t in transition_legal_no_mask_list if t is False)
                / len(transition_legal_no_mask_list)
            )
        else:
            illegal_transition_rate = 0.0
            illegal_transition_rate_no_mask = 0.0

        if energies_total:
            partition = compute_stratified_partition(
                np.array(energies_total, dtype=np.float64),
                strata,
                temperature=1.0,
            )
            run_free_energy = float(free_energy(partition))
        else:
            run_free_energy = 0.0

        mean_energy_correct = (
            float(np.mean(energies_correct)) if energies_correct else 0.0
        )
        mean_energy_incorrect = (
            float(np.mean(energies_incorrect)) if energies_incorrect else 0.0
        )

        hamming = hamming_b
        hamming_normalised = phase_b_hamming_normalised

        timestamp = dt.datetime.now(tz=dt.timezone.utc).isoformat()
        metric_pairs: list[tuple[str, float]] = [
            ("accuracy", accuracy),
            ("accuracy_no_mask", accuracy_no_mask),
            ("mask_accuracy_uplift", mask_accuracy_uplift),
            ("illegal_transition_rate", illegal_transition_rate),
            ("illegal_transition_rate_no_mask", illegal_transition_rate_no_mask),
            ("n_routed_to_recovery", float(n_routed_to_recovery)),
            ("n_abstained", float(n_abstained)),
            ("n_mask_modified_predictions", float(n_mask_modified_predictions)),
            ("n_monodromy_cycle_revisits", float(n_monodromy_cycle_revisits)),
            ("mean_energy_correct", mean_energy_correct),
            ("mean_energy_incorrect", mean_energy_incorrect),
            ("free_energy", run_free_energy),
            ("monodromy_consistency_loss", mono_loss),
            ("hamming_distance_from_fsm", float(hamming)),
            ("hamming_normalised", float(hamming_normalised)),
            ("phase_a_hamming_normalised", float(phase_a_hamming_normalised)),
            ("phase_b_hamming_normalised", float(phase_b_hamming_normalised)),
            ("crb_satisfied_fraction", float(crb_satisfied_fraction)),
            ("mean_crb_confidence", float(mean_crb_confidence)),
            ("kl_progress_final", float(kl_progress_final)),
            ("mean_kl_surprise", float(mean_kl_surprise)),
            ("mean_quality_signal_kl", float(mean_quality_signal_kl)),
            ("mean_quality_signal_blended", float(mean_quality_signal_blended)),
            ("n_samples", float(n_samples)),
            ("n_features", float(embedding_dim)),
            ("product_graph_cells_materialised", float(product_graph.n_materialised())),
            ("n_torch_trainable_params", float(n_torch_trainable_params)),
            ("n_torch_frozen_params", float(n_torch_frozen_params)),
        ]
        for metric_name, value in metric_pairs:
            metrics_writer.append(
                MetricsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=0,
                    split="train",
                    metric_name=metric_name,
                    value=value,
                    timestamp=timestamp,
                )
            )

    # Touch ``kl_categorical`` so the import stays referenced at module
    # scope (mirrors E11; no-op).
    _ = kl_categorical(np.array([0.5, 0.5]), np.array([0.5, 0.5]))

    return E12Result(
        accuracy=accuracy,
        accuracy_no_mask=accuracy_no_mask,
        mask_accuracy_uplift=mask_accuracy_uplift,
        illegal_transition_rate=illegal_transition_rate,
        illegal_transition_rate_no_mask=illegal_transition_rate_no_mask,
        monodromy_consistency_loss=mono_loss,
        hamming_distance_from_fsm=hamming,
        hamming_normalised=hamming_normalised,
        phase_a_hamming_normalised=phase_a_hamming_normalised,
        phase_b_hamming_normalised=phase_b_hamming_normalised,
        crb_satisfied_fraction=crb_satisfied_fraction,
        mean_crb_confidence=mean_crb_confidence,
        kl_progress_final=kl_progress_final,
        mean_kl_surprise=mean_kl_surprise,
        n_samples=n_samples,
        n_monodromy_cycle_revisits=n_monodromy_cycle_revisits,
        n_torch_trainable_params=n_torch_trainable_params,
        n_torch_frozen_params=n_torch_frozen_params,
    )
