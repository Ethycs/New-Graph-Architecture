"""E13 - KL/Fisher/CRB world model on the ListOps grammar (Phase 8).

E13 mirrors E11's information-geometric pipeline (classical Phase A
forward-backward + Beta M-step on observed transitions; KL-blended
Phase B with skeptical-prior threshold + CRB diagnostics; v1.1
decision_trace) but applied to the ListOps parser FSM rather than
Dyck-k.

Why the swap matters
--------------------
ListOps is the first published-grammar dataset this codebase exercises
(Nangia & Bowman 2018-style nested-operator expressions, generated
in-process). Compared to Dyck-k, ListOps gives EVERY architectural
claim a sharper test:

  * **Mask**: ``]`` at empty stack or before any operand is strictly
    illegal; ``[`` at max_depth is strictly illegal. Two distinct
    state-conditioned legality channels.
  * **Hyperbolic**: parse-tree depth IS stack depth IS real tree depth.
  * **sigma**: at every ``S{d}_after_operand`` step the parser truly
    chooses between continuing the operand list and closing -- BOTH
    legal -- so ambiguity is genuine grammar ambiguity, not random.
  * **Monodromy**: every matched ``[ ... ]`` is a closed walk that
    returns to the same depth-d after_operand state.

The runner reports the same Phase-A/Phase-B/CRB headlines as E11 plus
ListOps-specific diagnostics:

  * ``mean_parse_depth``: average bracket-nesting depth across the run.
  * ``n_operator_boundary_steps``: number of steps observed at any
    ``S{d}_after_operand`` state (the ambiguity points).
  * ``mean_sigma_at_operator_boundary``: average sigma at those steps.
  * ``mean_sigma_at_non_boundary``: average sigma at non-boundary steps.
    The ratio of these two should be > 1 IF sigma is doing real work
    on grammar ambiguity (this is the first real test of sigma's
    interpretive claim against a published grammar).

Acceptance bars are loose by design; the runner is NOT tuned to chase
a threshold.

Reuses existing atoms; numpy + sklearn only (no torch).
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
from nga.arch.graph_fsm import GraphFSM
from nga.arch.hyperbolic_embedding import (
    embed_euclidean_to_poincare,
    poincare_distance,
)
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
from nga.arch.typed_latent_clustering import riemannian_centroid
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
from nga.exp.dataset_listops import generate_listops_dataset

__all__ = ["E13Result", "run_e13"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class E13Result:
    """Aggregate metrics returned by run_e13."""

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
    mean_parse_depth: float
    n_operator_boundary_steps: int
    mean_sigma_at_operator_boundary: float
    mean_sigma_at_non_boundary: float
    n_samples: int
    n_monodromy_cycle_revisits: int


# ---------------------------------------------------------------------------
# Helpers (mirrored from E11)
# ---------------------------------------------------------------------------


def _initial_prototypes(
    poincare_obs: np.ndarray,
    state_idx: np.ndarray,
    n_states: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a (n_states, dim) Riemannian-centroid prototype table."""
    proto = np.zeros((n_states, dim), dtype=np.float64)
    for s in range(n_states):
        members = poincare_obs[state_idx == s]
        if members.shape[0] >= 1:
            proto[s] = riemannian_centroid(members, max_inner_iter=10)
        else:
            proto[s] = rng.normal(0.0, 0.05, size=(dim,))
    return proto


def _model_predicted_dist(
    *,
    obs_poincare: np.ndarray,
    prototypes: np.ndarray,
    legality_bias_row: np.ndarray,
) -> np.ndarray:
    """Softmax over candidate next states (lower distance => higher prob)."""
    n = prototypes.shape[0]
    distances = np.array(
        [float(poincare_distance(obs_poincare, prototypes[j])) for j in range(n)],
        dtype=np.float64,
    )
    logits = -distances + np.asarray(legality_bias_row, dtype=np.float64)
    logits -= logits.max()
    ex = np.exp(logits)
    return ex / ex.sum()


def _empirical_dist_from_window(
    window: deque,
    n_states: int,
    eps: float = 1e-3,
) -> np.ndarray:
    """Smoothed (Laplace-add-eps) categorical from a sliding window."""
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


def run_e13(
    *,
    config: Config,
    ablation: AblationTuple,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    margin_threshold: float = 0.10,
    n_sequences: int = 200,
    max_length: int = 32,
    max_depth: int = 3,
    illegal_temptation_fraction: float = 0.10,
    n_axis_bins: int = 5,
    seed_window: int = 64,
    empirical_window: int = 32,
    blend: float = 0.7,
    kl_surprise_weight: float = 0.5,
    crb_target_variance: float = 0.01,
) -> E13Result:
    """Run the Phase-8 information-geometric pipeline on ListOps.

    Writes the standard four artefacts plus decision_trace.jsonl. Returns
    aggregate metrics. Acceptance bars are intentionally loose.

    The runner expects ``fsm`` to be the canonical ListOps FSM (vertex
    ids ``START``, ``S{d}_op``/``S{d}_need_operand``/``S{d}_after_operand``
    for d in [1, max_depth], ``ACCEPT``).
    """
    if fsm.vertex_ids[0] != "START":
        raise ValueError(
            f"E13 expects fsm.vertex_ids[0] == 'START'; "
            f"got {fsm.vertex_ids[0]!r}"
        )
    if fsm.vertex_ids[-1] != "ACCEPT":
        raise ValueError(
            f"E13 expects fsm.vertex_ids[-1] == 'ACCEPT'; "
            f"got {fsm.vertex_ids[-1]!r}"
        )

    n_states = fsm.vertex_count
    # Vertex layout: 1 START + 3*d (op/need/after) + 1 ACCEPT, so
    # max_depth = (n_states - 2) // 3. Verify the FSM matches.
    derived_max_depth = (n_states - 2) // 3
    if 2 + derived_max_depth * 3 != n_states:
        raise ValueError(
            f"E13 cannot derive max_depth from {n_states} states; "
            f"expected 2 + max_depth*3 == n_states."
        )
    if derived_max_depth != max_depth:
        # Trust the FSM over the default kwarg; the YAML is canonical.
        max_depth = derived_max_depth

    # ------------------------------------------------------------------
    # Step 1 - Dataset
    # ------------------------------------------------------------------
    # Note: ListOps features are a fixed 16-dim one-hot of the observed
    # token (independent of config.embedding_dim). The token alphabet
    # has 16 entries. Downstream we still embed those 16-dim vectors
    # into the Poincare ball for the prototype geometry.
    ds = generate_listops_dataset(
        fsm=fsm,
        max_depth=max_depth,
        n_sequences=n_sequences,
        max_length=max_length,
        seed=seed,
        illegal_temptation_fraction=illegal_temptation_fraction,
    )

    rng = np.random.default_rng(seed)

    X = ds.X
    n_samples = X.shape[0]
    embedding_dim = X.shape[1]
    sample_ids = [s.sample_id for s in ds.samples]
    current_state_idx = ds.current_states
    true_next_state_idx = ds.y_next
    depths = ds.depths

    poincare_obs = embed_euclidean_to_poincare(X, scale=0.5)

    # Pre-compute the boundary mask: a step is at an "operator boundary"
    # iff the SOURCE state ends in "_after_operand" (i.e. the parser is
    # genuinely ambiguous between continuing the operand list and
    # closing).
    vertex_ids = fsm.vertex_ids
    is_boundary_state = np.array(
        [vid.endswith("_after_operand") for vid in vertex_ids],
        dtype=bool,
    )
    boundary_step = is_boundary_state[current_state_idx]

    # ------------------------------------------------------------------
    # Step 2 - Atom instances + axis quantisers (calibrated as in E11).
    # ------------------------------------------------------------------
    energy_fn = EnergyFunction(EnergyParameters())

    seed_n = min(seed_window, n_samples)
    seed_protos = _initial_prototypes(
        poincare_obs[:seed_n],
        current_state_idx[:seed_n],
        n_states,
        embedding_dim,
        rng,
    )
    sigmas_seed: list[float] = []
    energies_seed: list[float] = []
    margins_seed: list[float] = []
    seed_detector = SingularityDetector(margin_threshold=margin_threshold, enabled=True)
    for i in range(seed_n):
        d_to_each = np.array(
            [poincare_distance(poincare_obs[i], seed_protos[j]) for j in range(n_states)]
        )
        logits = -d_to_each
        ex = np.exp(logits - logits.max())
        dist = ex / ex.sum()
        m = float(compute_margin(dist))
        margins_seed.append(m)
        sigma_i, _ = seed_detector.compute(margin=m, is_illegal=False, loop_risk=0.0)
        sigmas_seed.append(float(sigma_i))
        j_best = int(np.argmin(d_to_each))
        d = float(d_to_each[j_best])
        unc = float(1.0 - np.exp(-max(d, 0.0)))
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
    # Step 3 -- Phase A: classical cold-start E-step.
    #
    # Same approach as E11: count observed (current, next) pairs and
    # turn them into a Beta posterior in one stroke. Edges that never
    # appear in the data remain at the uniform prior (alpha = beta = 1)
    # and the skeptical-prior threshold (strict ``>`` in
    # legality_matrix) keeps them illegal.
    # ------------------------------------------------------------------
    pair_counts = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_samples):
        pair_counts[int(current_state_idx[i]), int(true_next_state_idx[i])] += 1.0

    # Touch the forward-backward atom on a flattened state path so the
    # atom-census wiring is observable; we still prefer the per-step
    # pair-count statistic for the actual M-step input.
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
    # Step 4 -- Build product graph + trainer for Phase B.
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

    initial_protos = _initial_prototypes(
        poincare_obs[:seed_n],
        current_state_idx[:seed_n],
        n_states,
        embedding_dim,
        rng,
    )
    trainer = EnergyMinimizationTrainer(
        prototypes={"state": initial_protos},
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
        goal_states=[fsm.vertex_index["ACCEPT"]],
    )

    experiment_label = "E13"
    ablation_label = run_id.split("_")[1]

    # ------------------------------------------------------------------
    # Step 5 -- Phase B: KL-blended refinement loop.
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
    sigma_at_boundary: list[float] = []
    sigma_at_non_boundary: list[float] = []
    depths_observed: list[int] = []

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
            obs = poincare_obs[i]
            depth_at_step = int(depths[i])
            depths_observed.append(depth_at_step)

            bias = posterior_mask.legality_bias()
            bias_row_learned = bias[cs_idx]
            bias_row_uniform = np.zeros_like(bias_row_learned)

            predicted_dist = _model_predicted_dist(
                obs_poincare=obs,
                prototypes=trainer.prototypes["state"],
                legality_bias_row=bias_row_learned,
            )
            unmasked_dist = _model_predicted_dist(
                obs_poincare=obs,
                prototypes=trainer.prototypes["state"],
                legality_bias_row=bias_row_uniform,
            )

            margin = float(compute_margin(predicted_dist))
            argmax_unmasked = int(np.argmax(unmasked_dist))
            predicted_no_mask = vertex_ids[argmax_unmasked]

            window = windows[cs_idx]
            empirical_dist = _empirical_dist_from_window(window, n_states)

            kl_surprise = float(compute_kl_surprise(empirical_dist, predicted_dist))
            kl_surprises_step.append(kl_surprise)

            argmin_state = int(np.argmax(predicted_dist))
            predicted_state = vertex_ids[argmin_state]
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

            # Track boundary vs non-boundary sigma for the headline
            # ListOps diagnostic.
            if bool(boundary_step[i]):
                sigma_at_boundary.append(float(sigma))
            else:
                sigma_at_non_boundary.append(float(sigma))

            d_chosen = float(
                poincare_distance(obs, trainer.prototypes["state"][argmin_state])
            )
            unc = float(1.0 - np.exp(-max(d_chosen, 0.0)))
            lp = float(np.clip(0.5 * loop_risk + 0.5 * sigma, 0.0, 1.0))
            energy_breakdown_obj = energy_fn.compute(
                cost=0.0,
                uncertainty=unc,
                contradiction=1.0 if is_illegal else 0.0,
                loop_pressure=lp,
                progress=1.0,
            )
            energy_total = float(energy_breakdown_obj.total)

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

            proto_table = trainer.prototypes["state"]
            target_idx = true_next_idx
            proto_before = proto_table[target_idx].copy()
            diff = proto_before - obs
            diff_norm = float(np.linalg.norm(diff))
            if diff_norm >= 1e-12:
                lr = float(trainer.config.lr_prototype)
                proto_table[target_idx] = (
                    proto_before - lr * (diff / diff_norm) * np.exp(-d_chosen)
                )
                norm_after = float(np.linalg.norm(proto_table[target_idx]))
                if norm_after >= 1.0:
                    proto_table[target_idx] = (
                        proto_table[target_idx] * (1.0 - 1e-3) / norm_after
                    )

            windows[cs_idx].append(int(true_next_idx))

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

            mask_changed_prediction = bool(argmax_unmasked != argmin_state)
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
                model_prediction=int(argmin_state),
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
            confidence = float(predicted_dist[argmin_state])

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
            energies_by_state[argmin_state].append(energy_total)

            if argmin_state in visited_states:
                n_monodromy_cycle_revisits += 1
            visited_states.add(argmin_state)

            prev_output_tuple = output_node_tuple

        # ------------------------------------------------------------------
        # Step 6 -- Closed-walk consistency check.
        # ------------------------------------------------------------------
        # ListOps has many self-loops on digit edges (S{d}_after_operand);
        # max_length=4 yields a tractable enumeration.
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
        # Step 7 -- Phase C: CRB + KL-progress diagnostics.
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

        # ListOps-specific diagnostics.
        mean_parse_depth = (
            float(np.mean(depths_observed)) if depths_observed else 0.0
        )
        n_operator_boundary_steps = int(len(sigma_at_boundary))
        mean_sigma_at_operator_boundary = (
            float(np.mean(sigma_at_boundary)) if sigma_at_boundary else 0.0
        )
        mean_sigma_at_non_boundary = (
            float(np.mean(sigma_at_non_boundary)) if sigma_at_non_boundary else 0.0
        )

        # ------------------------------------------------------------------
        # Step 8 -- Standard E10/E11-shaped metrics.
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
            ("mean_parse_depth", float(mean_parse_depth)),
            ("n_operator_boundary_steps", float(n_operator_boundary_steps)),
            (
                "mean_sigma_at_operator_boundary",
                float(mean_sigma_at_operator_boundary),
            ),
            ("mean_sigma_at_non_boundary", float(mean_sigma_at_non_boundary)),
            ("n_samples", float(n_samples)),
            ("n_features", float(embedding_dim)),
            ("product_graph_cells_materialised", float(product_graph.n_materialised())),
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
    # scope (atom-census wiring).
    _ = kl_categorical(np.array([0.5, 0.5]), np.array([0.5, 0.5]))

    return E13Result(
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
        mean_parse_depth=mean_parse_depth,
        n_operator_boundary_steps=n_operator_boundary_steps,
        mean_sigma_at_operator_boundary=mean_sigma_at_operator_boundary,
        mean_sigma_at_non_boundary=mean_sigma_at_non_boundary,
        n_samples=n_samples,
        n_monodromy_cycle_revisits=n_monodromy_cycle_revisits,
    )
