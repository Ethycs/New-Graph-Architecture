"""E10 - Unified world model: every Wave A/B/C atom in one runnable experiment.

Commitment for this run:
  - the system IS a graph;
  - every output is a node;
  - energy IS the loss;
  - the trace IS the explanation.

Pipeline (per cyclic Dyck-k FSM):

  1. Build atom instances at startup.
       - Load FSM, generate Dyck-k dataset.
       - EnergyFunction(EnergyParameters()).
       - AxisQuantizer.from_quantiles for sigma/energy/margin (5 bins each).
       - PosteriorMask cold start (Beta(1,1)).
       - ProductGraph with axes [state, sigma_axis, energy_axis, margin_axis].
       - EnergyMinimizationTrainer (no group_action), one prototype per FSM
         vertex initialised by Riemannian centroid of early observations.

  2. Per-sample training loop.
       - Embed observation into Poincare ball.
       - Compute sigma, margin, energy.
       - Predict next-state by argmin of (energy - legality_bias) over
         legal candidates from the LEARNED posterior mask.
       - Quantise sigma, energy, margin to axis nodes; build the
         output_node_tuple = [state, sigma_axis, energy_axis, margin_axis].
       - Materialise the cell in the product graph; record the edge.
       - Trainer step: prototype + posterior mask update.
       - Append a v1.1 DecisionTraceRecord with output_node_tuple,
         edge_traversed, mask_version_id, posterior_summary, axis_node_ids
         all populated; v1.0 fields populated as before.

  3. Post-training closed-walk consistency check.
       - closed_walks_in_fsm with max_length=4.
       - Per-vertex mean-energy reuse across the walk; compute
         monodromy_consistency_loss; record it as a metric.

  4. Mask convergence diagnostic.
       - Hamming distance between learned mask at threshold 0.5 and the
         FSM's hand-authored legality matrix; record absolute and normalised.

  5. Standard metrics: accuracy, accuracy_no_mask, mask_accuracy_uplift,
     illegal_transition_rate (and _no_mask), n_routed_to_recovery,
     n_abstained, n_mask_modified_predictions, free_energy,
     mean_energy_correct, mean_energy_incorrect, n_monodromy_cycle_revisits.

The runner uses numpy + sklearn only; no torch.
"""
from __future__ import annotations

import datetime as dt
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
from nga.arch.graph_fsm import GraphFSM
from nga.arch.hyperbolic_embedding import (
    embed_euclidean_to_poincare,
    poincare_distance,
)
from nga.arch.margin_uncertainty import compute_margin
from nga.arch.monodromy_consistency import (
    closed_walks_in_fsm,
    monodromy_consistency_loss,
)
from nga.arch.posterior_mask import PosteriorMask
from nga.arch.product_graph import ProductGraph
from nga.arch.singularity_detector import SingularityDetector
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
from nga.exp.dataset_dyck_k import generate_dyck_dataset

__all__ = ["E10Result", "run_e10"]


@dataclass
class E10Result:
    """Aggregate metrics returned by run_e10."""

    accuracy: float
    accuracy_no_mask: float
    mask_accuracy_uplift: float
    illegal_transition_rate: float
    illegal_transition_rate_no_mask: float
    monodromy_consistency_loss: float
    hamming_distance_from_fsm: int
    hamming_normalised: float
    n_samples: int
    n_monodromy_cycle_revisits: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _initial_prototypes(
    poincare_obs: np.ndarray,
    state_idx: np.ndarray,
    n_states: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a (n_states, dim) prototype table.

    For each state with at least one observation in the seed window, the
    prototype is the Riemannian (Karcher) centroid of those points. States
    with no observations get a tiny near-origin random Poincare point.
    """
    proto = np.zeros((n_states, dim), dtype=np.float64)
    for s in range(n_states):
        members = poincare_obs[state_idx == s]
        if members.shape[0] >= 1:
            proto[s] = riemannian_centroid(members, max_inner_iter=10)
        else:
            proto[s] = rng.normal(0.0, 0.05, size=(dim,))
    # Defensive projection happens inside the trainer; nothing more here.
    return proto


def _argmin_energy_legal_candidate(
    *,
    obs_poincare: np.ndarray,
    prototypes: np.ndarray,
    legality_bias_row: np.ndarray,
    energy_fn: EnergyFunction,
    sigma: float,
    contradiction: float,
    loop_risk: float,
) -> int:
    """Pick next-state as argmin over (energy_at(j) - legality_bias[j]).

    Uses uncertainty(d_hyp) = 1 - exp(-d) per the trainer convention. The
    legality_bias is added to the negative of the score (i.e. subtracted from
    energy) so high-bias edges are preferred.
    """
    n = prototypes.shape[0]
    # Hyperbolic distances vectorised by row (compute_distance is per-pair).
    energies = np.empty(n, dtype=np.float64)
    for j in range(n):
        d = float(poincare_distance(obs_poincare, prototypes[j]))
        unc = float(1.0 - np.exp(-max(d, 0.0)))
        # Same loop_pressure folding as EnergyMinimizationTrainer._energy_at.
        lp = float(np.clip(0.5 * loop_risk + 0.5 * sigma, 0.0, 1.0))
        contribs = energy_fn.compute(
            cost=0.0,
            uncertainty=unc,
            contradiction=contradiction,
            loop_pressure=lp,
            progress=1.0,
        )
        energies[j] = float(contribs.total)
    score = energies - legality_bias_row
    return int(np.argmin(score))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_e10(
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
) -> E10Result:
    """Run the unified E10 pipeline. Writes the standard four artefacts plus
    decision_trace.jsonl. Returns aggregate metrics.

    The runner does NOT optimise its hyperparameters to chase a threshold:
    everything reports what it reports. Acceptance bars are loose by design.
    """
    if fsm.vertex_ids[0] != "S0":
        raise ValueError(
            f"E10 expects fsm.vertex_ids[0] == 'S0'; got {fsm.vertex_ids[0]!r}"
        )

    n_states = fsm.vertex_count
    k = 2
    max_depth = (n_states - 1) // k
    if 1 + max_depth * k != n_states:
        raise ValueError(
            f"E10 cannot derive (k, max_depth) from {n_states} states; "
            f"expected 1 + max_depth*k == n_states, got k={k}."
        )

    # ------------------------------------------------------------------
    # Step 1 - Dataset (single, in-order; no train/test split for E10).
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

    rng = np.random.default_rng(seed)

    X = ds.X
    n_samples = X.shape[0]
    embedding_dim = X.shape[1]
    sample_ids = [s.sample_id for s in ds.samples]
    current_state_idx = ds.current_states
    true_next_state_idx = ds.y_next

    # Embed all observations into the Poincare ball once.
    poincare_obs = embed_euclidean_to_poincare(X, scale=0.5)

    # ------------------------------------------------------------------
    # Step 2 - Atom instances.
    # ------------------------------------------------------------------
    energy_fn = EnergyFunction(EnergyParameters())

    # Seed-pass: compute sigma / energy / margin distributions BEFORE any
    # learning so the axis quantisers are calibrated to a real distribution.
    # We use a uniform legality bias here (no learned mask yet).
    seed_n = min(seed_window, n_samples)
    seed_protos = _initial_prototypes(
        poincare_obs[:seed_n], current_state_idx[:seed_n], n_states, embedding_dim, rng
    )
    sigmas_seed: list[float] = []
    energies_seed: list[float] = []
    margins_seed: list[float] = []
    seed_detector = SingularityDetector(margin_threshold=margin_threshold, enabled=True)
    for i in range(seed_n):
        d_to_each = np.array(
            [poincare_distance(poincare_obs[i], seed_protos[j]) for j in range(n_states)]
        )
        # Soft "distribution" over states: softmax(-d).
        logits = -d_to_each
        ex = np.exp(logits - logits.max())
        dist = ex / ex.sum()
        m = float(compute_margin(dist))
        margins_seed.append(m)
        sigma_i, _ = seed_detector.compute(margin=m, is_illegal=False, loop_risk=0.0)
        sigmas_seed.append(float(sigma_i))
        # Energy at the closest prototype.
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

    # AxisQuantizers (5 bins each). from_quantiles raises on degenerate input,
    # which we guard with a small jitter when ties dominate.
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

    # Posterior mask: cold start uniform.
    posterior_mask = PosteriorMask(
        n_vertices=n_states, prior_alpha=1.0, prior_beta=1.0
    )

    # Product graph: state x sigma_axis x energy_axis x margin_axis.
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

    # Trainer: prototypes by FSM vertex, one row each. observation_type "state".
    initial_protos = _initial_prototypes(
        poincare_obs[:seed_n], current_state_idx[:seed_n], n_states, embedding_dim, rng
    )
    trainer = EnergyMinimizationTrainer(
        prototypes={"state": initial_protos},
        posterior_mask=posterior_mask,
        energy_fn=energy_fn,
        group_action=None,
        config=TrainerConfig(),
    )

    # Behavioral stratum + control policy reused from E9 conventions.
    detector = SingularityDetector(
        margin_threshold=margin_threshold,
        enabled=ablation.singularity_detector_enabled,
    )
    history = TaggerHistory()
    control_policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        fsm=fsm,
        goal_states=[0],
    )

    experiment_label = "E10"
    ablation_label = run_id.split("_")[1]

    vertex_ids = fsm.vertex_ids

    # Accumulators.
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

    # ------------------------------------------------------------------
    # Step 3 - Per-sample loop with training + tracing.
    # ------------------------------------------------------------------
    with (
        JsonlWriter(output_dir / "metrics.jsonl", MetricsRecord) as metrics_writer,
        JsonlWriter(output_dir / "results.jsonl", ResultsRecord) as results_writer,
        open_decision_trace_writer(output_dir / "decision_trace.jsonl") as trace_writer,
    ):
        # scores.jsonl is created by CLI.touch and remains intentionally empty.

        for i in range(n_samples):
            cs_idx = int(current_state_idx[i])
            current_state = vertex_ids[cs_idx]
            true_next = vertex_ids[int(true_next_state_idx[i])]
            obs = poincare_obs[i]

            # The legality bias is read from the LIVE posterior on every step.
            bias = posterior_mask.legality_bias()
            # Per-row bias for the current state.
            bias_row_learned = bias[cs_idx]
            bias_row_uniform = np.zeros_like(bias_row_learned)

            # Predict next-state by argmin energy with legality bias subtracted.
            # First pass: a "soft distribution" over candidates from -d so we
            # can compute margin and sigma like the rest of the system.
            d_each = np.array(
                [
                    float(poincare_distance(obs, trainer.prototypes["state"][j]))
                    for j in range(n_states)
                ]
            )
            logits = -d_each
            ex = np.exp(logits - logits.max())
            unmasked_dist = ex / ex.sum()
            margin = float(compute_margin(unmasked_dist))

            argmax_unmasked = int(np.argmax(unmasked_dist))
            predicted_no_mask = vertex_ids[argmax_unmasked]

            # Pre-mask sigma (use unmasked distribution for the margin signal).
            loop_risk = float(history.predicted_loop_risk(predicted_no_mask))
            is_illegal_tmp = not bool(
                fsm.is_legal_transition(current_state, predicted_no_mask)
            )
            sigma_for_pred, contribs = detector.compute(
                margin=margin,
                is_illegal=is_illegal_tmp,
                loop_risk=loop_risk,
            )

            # Argmin energy with legality bias subtracted (LEARNED mask path).
            if ablation.graph_mask_enabled:
                argmin_state = _argmin_energy_legal_candidate(
                    obs_poincare=obs,
                    prototypes=trainer.prototypes["state"],
                    legality_bias_row=bias_row_learned,
                    energy_fn=energy_fn,
                    sigma=float(sigma_for_pred),
                    contradiction=1.0 if is_illegal_tmp else 0.0,
                    loop_risk=loop_risk,
                )
            else:
                argmin_state = _argmin_energy_legal_candidate(
                    obs_poincare=obs,
                    prototypes=trainer.prototypes["state"],
                    legality_bias_row=bias_row_uniform,
                    energy_fn=energy_fn,
                    sigma=float(sigma_for_pred),
                    contradiction=1.0 if is_illegal_tmp else 0.0,
                    loop_risk=loop_risk,
                )
            predicted_state = vertex_ids[argmin_state]

            # Final-form sigma uses the post-mask predicted state's legality.
            is_illegal = not bool(
                fsm.is_legal_transition(current_state, predicted_state)
            )
            sigma, contribs = detector.compute(
                margin=margin,
                is_illegal=is_illegal,
                loop_risk=loop_risk,
            )

            # Final energy at the chosen prototype.
            d_chosen = float(poincare_distance(obs, trainer.prototypes["state"][argmin_state]))
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

            # Quantise sigma, energy, margin into axis nodes.
            sigma_node = sigma_q.quantise(float(sigma))
            energy_node = energy_q.quantise(float(energy_total))
            margin_node = margin_q.quantise(float(margin))

            output_node_tuple = (
                predicted_state,
                str(sigma_node),
                str(energy_node),
                str(margin_node),
            )
            # Materialise + record edge.
            if prev_output_tuple is not None:
                product_graph.add_transition(prev_output_tuple, output_node_tuple)
                edge_traversed = (
                    list(prev_output_tuple),
                    list(output_node_tuple),
                )
            else:
                product_graph.materialise(output_node_tuple)
                edge_traversed = None

            # Trainer step (prototype + posterior update). The trainer updates
            # the posterior on the OBSERVED ground-truth transition -- so the
            # contradiction signal must also evaluate the OBSERVED edge, not
            # the PREDICTED one. Otherwise the cold-start spiral kicks in:
            # bad classifier predicts illegal -> contradiction=1 on every
            # update -> Q drops -> beta dominates -> learned mask inverts the
            # FSM. The honest signal is "is the data's transition itself
            # illegal?" which for valid Dyck-k sequences is always False.
            obs_is_illegal = not bool(
                fsm.is_legal_transition(current_state, vertex_ids[int(true_next_state_idx[i])])
            )
            trainer.step(
                src_state=cs_idx,
                dst_state=int(true_next_state_idx[i]),
                observation_in_poincare=obs,
                observation_type="state",
                prototype_idx=int(true_next_state_idx[i]),
                sigma=float(sigma),
                contradiction=1.0 if obs_is_illegal else 0.0,
                loop_risk=loop_risk,
                cost=0.0,
                progress=1.0,
            )

            # Stratum tag (no-mask softmax is fine for the tagger).
            tagger_inp = TaggerInput(
                distribution=unmasked_dist,
                predicted_state=predicted_state,
                current_state=current_state,
                margin=margin,
            )
            tag_result = tag_step(tagger_inp, history, fsm, margin_threshold=margin_threshold)
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

            # Illegal indices zeroed = states whose learned posterior mean < 0.5.
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

            # Posterior summary across all (V, V) entries.
            mean_alpha = float(np.mean(posterior_mask.alpha))
            mean_beta = float(np.mean(posterior_mask.beta))
            mean_entropy = float(np.mean(posterior_mask.posterior_entropy()))
            posterior_summary = {
                "mean_alpha": mean_alpha,
                "mean_beta": mean_beta,
                "mean_entropy": mean_entropy,
            }

            # Top-2 from the unmasked distribution for trace reporting.
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

            # confidence is the post-softmax probability of the chosen state.
            confidence = float(unmasked_dist[argmin_state])

            results_writer.append(
                ResultsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=i,
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
                    step=i,
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

            # Aggregate.
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
        # Step 4 - Closed-walk consistency check.
        # ------------------------------------------------------------------
        walks = closed_walks_in_fsm(fsm, max_length=4)
        # Per-vertex mean energy as the "energy field" along the walk; if a
        # vertex was never visited we fall back to the global mean to keep
        # the metric defined.
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
        # Step 5 - Mask convergence diagnostic.
        # ------------------------------------------------------------------
        learned_legal = posterior_mask.legality_matrix(0.5)
        gold_legal = fsm.legality_matrix
        hamming = int(np.sum(learned_legal != gold_legal))
        total_cells = int(learned_legal.size)
        hamming_normalised = float(hamming) / float(total_cells) if total_cells else 0.0

        # ------------------------------------------------------------------
        # Step 6 - Standard metrics.
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

        timestamp = dt.datetime.now(tz=dt.timezone.utc).isoformat()
        for metric_name, value in [
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
            ("n_samples", float(n_samples)),
            ("n_features", float(embedding_dim)),
            ("product_graph_cells_materialised", float(product_graph.n_materialised())),
        ]:
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

    return E10Result(
        accuracy=accuracy,
        accuracy_no_mask=accuracy_no_mask,
        mask_accuracy_uplift=mask_accuracy_uplift,
        illegal_transition_rate=illegal_transition_rate,
        illegal_transition_rate_no_mask=illegal_transition_rate_no_mask,
        monodromy_consistency_loss=mono_loss,
        hamming_distance_from_fsm=hamming,
        hamming_normalised=hamming_normalised,
        n_samples=n_samples,
        n_monodromy_cycle_revisits=n_monodromy_cycle_revisits,
    )
