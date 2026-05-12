"""E22 - Torch-native end-to-end gradient training of the Typed Protocol
Network on the diagnostic dataset (Phase 19).

E22 is the FIRST experiment with a real-world (non-synthetic) dataset:
the Kaggle "Disease Prediction from Symptoms" CSV linearised
per-patient as

    [s_1, s_2, ..., s_|S|, d]

where s_i are the patient's TRUE symptoms in canonical column order
and d is the diagnosis. The token alphabet has 173 entries and the
driving FSM has 3 states (OBSERVING_FEW / OBSERVING_ENOUGH /
DIAGNOSED). The structurally interesting site is OBSERVING_ENOUGH,
where BOTH "another symptom" and "commit to a diagnosis" are FSM-legal
continuations.

Pipeline mirrors E21:
  Phase A -- expected_counts_observed + bayesian_m_step_beta seed.
  Phase B -- end-to-end torch training on the train split.
  Phase C -- held-out evaluation with sigma + energy + margin
             diagnostics, plus diagnostic-specific metrics.

Compute-efficiency tracking is baked in (per the Phase 18 Track 1
pattern): per-phase wall-clock, total wall-clock, inference throughput,
peak memory delta.

Diagnostic-specific metrics:
  * diagnostic_accuracy:                 accuracy on the diagnosis
    transition only (OBSERVING_ENOUGH -> DIAGNOSED).
  * mean_symptoms_observed_before_diagnosis: per-patient average.
"""
from __future__ import annotations

import datetime as dt
import resource
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nga.arch.axis_quantizer import AxisQuantizer
from nga.arch.behavioral_stratum_tagger import TaggerHistory, TaggerInput, tag_step
from nga.arch.control_policy import ControlPolicy
from nga.arch.energy_function import EnergyFunction, EnergyParameters
from nga.arch.failure_margin_auroc import (
    margin_auroc,
    sigma_auroc,
    structural_ambiguity_auroc,
)
from nga.arch.forward_backward import (
    bayesian_m_step_beta,
    expected_counts_observed,
)
from nga.arch.frozen_encoder_torch import FrozenEncoderTorch
from nga.arch.graph_fsm import GraphFSM
from nga.arch.information_geometry import kl_categorical
from nga.arch.margin_uncertainty import compute_margin
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
from nga.arch.torch_energy_trainer import TorchEnergyTrainer, TorchTrainerConfig
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
from nga.exp.dataset_diagnostic import (
    generate_diagnostic_dataset,
    train_test_split_by_patient,
)

try:
    import torch
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "torch is required for E22; install via `pixi add pytorch` "
        "(or `pip install torch`)."
    ) from _exc


__all__ = ["E22Result", "run_e22"]


READOUT_TYPE_ID = "default"
DIAGNOSTIC_AMBIGUITY_STATE = "OBSERVING_ENOUGH"


def _is_structural_ambiguity_state(vertex_id: str) -> bool:
    """OBSERVING_ENOUGH is the load-bearing decision: another symptom
    or commit to a diagnosis? Both are FSM-legal."""
    return vertex_id == DIAGNOSTIC_AMBIGUITY_STATE


@dataclass
class E22Result:
    accuracy: float
    accuracy_no_mask: float
    mask_accuracy_uplift: float
    illegal_transition_rate: float
    illegal_transition_rate_no_mask: float
    phase_a_hamming_normalised: float
    phase_b_hamming_normalised: float
    sigma_auroc: float
    margin_auroc: float
    sigma_uplift: float
    sigma_structural_auroc: float
    margin_structural_auroc: float
    sigma_structural_uplift: float
    mean_sigma_at_operator_boundary: float
    mean_sigma_at_non_boundary: float
    sigma_boundary_ratio: float
    diagnostic_accuracy: float
    mean_symptoms_observed_before_diagnosis: float
    n_torch_trainable_params: int
    final_loss: float
    loss_decrease_first_to_last_epoch: float
    n_train: int
    n_test: int
    phase_a_wall_clock_seconds: float
    phase_b_wall_clock_seconds: float
    total_wall_clock_seconds: float
    inference_throughput_samples_per_sec: float
    peak_memory_kb: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empirical_dist_from_window(
    window: deque,
    n_states: int,
    eps: float = 1e-3,
) -> np.ndarray:
    counts = np.full(n_states, eps, dtype=np.float64)
    for j in window:
        counts[int(j)] += 1.0
    total = counts.sum()
    if total <= 0.0:
        return np.full(n_states, 1.0 / max(n_states, 1), dtype=np.float64)
    return counts / total


def _per_class_centroid_init(
    Z: np.ndarray, y: np.ndarray, n_states: int, dim: int
) -> np.ndarray:
    proto = np.zeros((n_states, dim), dtype=np.float32)
    rng = np.random.default_rng(0)
    for s in range(n_states):
        members = Z[y == s]
        if members.shape[0] >= 1:
            mu = np.asarray(members.mean(axis=0), dtype=np.float32)
        else:
            mu = (0.01 * rng.standard_normal(dim)).astype(np.float32)
        norm = float(np.linalg.norm(mu))
        if norm > 0.5:
            mu = mu * (0.5 / max(norm, 1e-6))
        proto[s] = mu
    return proto


def _peak_memory_kb() -> float:
    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:  # pragma: no cover
        return 0.0


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_e22(
    *,
    config: Config,
    ablation: AblationTuple,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    margin_threshold: float = 0.10,
    n_patients: int | None = 1000,
    illegal_temptation_fraction: float = 0.05,
    n_axis_bins: int = 5,
    empirical_window: int = 32,
    n_epochs: int = 5,
    encoder_hidden_dim: int = 32,
    readout_hidden_dim: int = 32,
    test_fraction: float = 0.25,
    kl_surprise_weight: float = 0.5,
    trainer_lr: float = 1e-2,
    grad_clip: float = 5.0,
    csv_path: str | Path | None = None,
) -> E22Result:
    """Run torch-native end-to-end gradient training on the diagnostic
    dataset.

    Writes the standard four artefacts plus decision_trace.jsonl on the
    HELD-OUT test split.
    """
    expected = ["OBSERVING_FEW", "OBSERVING_ENOUGH", "DIAGNOSED"]
    if fsm.vertex_ids != expected:
        raise ValueError(
            f"E22 expects fsm.vertex_ids == {expected!r}; got "
            f"{fsm.vertex_ids!r}"
        )

    if csv_path is None:
        # Default to the repo-checked-in training CSV.
        repo_root = Path(__file__).resolve().parents[3]
        csv_path = (
            repo_root / "tests/fixtures/data/diagnostic_training.csv"
        )

    n_states = fsm.vertex_count

    mem_start_kb = _peak_memory_kb()
    total_start = time.perf_counter()

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Step 1 -- Dataset (diagnostic walker, 176-dim one-hot tokens).
    # ------------------------------------------------------------------
    ds = generate_diagnostic_dataset(
        fsm=fsm,
        csv_path=csv_path,
        n_patients=n_patients,
        seed=seed,
        illegal_temptation_fraction=illegal_temptation_fraction,
    )
    train_ds, test_ds = train_test_split_by_patient(
        ds, seed=seed, test_fraction=test_fraction
    )

    embedding_dim = train_ds.X.shape[1]
    n_train = train_ds.X.shape[0]
    n_test = test_ds.X.shape[0]

    vertex_ids = fsm.vertex_ids
    is_boundary_state = np.array(
        [_is_structural_ambiguity_state(vid) for vid in vertex_ids],
        dtype=bool,
    )

    # mean symptoms observed before diagnosis (over ALL patients in ds).
    if ds.n_patients:
        per_patient_symptom_count: dict[int, int] = {}
        for s in ds.samples:
            if not s.is_diagnosis_step:
                per_patient_symptom_count[s.patient_id] = (
                    per_patient_symptom_count.get(s.patient_id, 0) + 1
                )
        if per_patient_symptom_count:
            mean_symptoms_observed_before_diagnosis = float(
                np.mean(list(per_patient_symptom_count.values()))
            )
        else:
            mean_symptoms_observed_before_diagnosis = 0.0
    else:
        mean_symptoms_observed_before_diagnosis = 0.0

    # ------------------------------------------------------------------
    # Step 2 -- Phase A.
    # ------------------------------------------------------------------
    phase_a_start = time.perf_counter()

    pair_counts = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_train):
        pair_counts[
            int(train_ds.current_states[i]), int(train_ds.y_next[i])
        ] += 1.0

    state_path = np.concatenate(
        [
            train_ds.current_states.reshape(-1).astype(np.int64),
            train_ds.y_next[-1:].astype(np.int64),
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

    posterior_mask_a = PosteriorMask(
        n_vertices=n_states, prior_alpha=1.0, prior_beta=1.0
    )
    posterior_mask_a._alpha = np.asarray(alpha_seed, dtype=np.float64)
    posterior_mask_a._beta = np.asarray(beta_seed, dtype=np.float64)

    learned_legal_a = posterior_mask_a.legality_matrix(0.5)
    gold_legal = fsm.legality_matrix
    hamming_a = int(np.sum(learned_legal_a != gold_legal))
    total_cells = int(learned_legal_a.size)
    phase_a_hamming_normalised = (
        float(hamming_a) / float(total_cells) if total_cells else 0.0
    )

    phase_a_seconds = time.perf_counter() - phase_a_start

    # ------------------------------------------------------------------
    # Step 3 -- Build torch substrate.
    # ------------------------------------------------------------------
    phase_b_start = time.perf_counter()

    encoder = FrozenEncoderTorch(
        input_dim=embedding_dim,
        output_dim=embedding_dim,
        hidden_dim=encoder_hidden_dim,
        seed=int(seed),
    )
    encoder.fit(train_ds.X)

    readout = TypedReadoutTorch(
        type_ids=[READOUT_TYPE_ID],
        n_classes=n_states,
        input_dim=embedding_dim,
        hidden_dim=readout_hidden_dim,
        n_epochs=1,
        lr=1e-3,
        seed=int(seed),
    )

    Z_train = encoder.encode(train_ds.X).astype(np.float32)
    Z_test = encoder.encode(test_ds.X).astype(np.float32)

    proto_init_np = _per_class_centroid_init(
        Z_train, train_ds.y_next, n_states, embedding_dim
    )
    proto_init = torch.from_numpy(proto_init_np).to(torch.float32)

    trainer_config = TorchTrainerConfig(
        lr=trainer_lr,
        lambda_kl=1.0,
        grad_clip=grad_clip,
    )
    trainer = TorchEnergyTrainer(
        n_states=n_states,
        embedding_dim=embedding_dim,
        encoder_output_dim=embedding_dim,
        prototype_init=proto_init,
        encoder=encoder,
        readout=readout,
        posterior_alpha_init=torch.from_numpy(
            np.asarray(alpha_seed, dtype=np.float32)
        ),
        posterior_beta_init=torch.from_numpy(
            np.asarray(beta_seed, dtype=np.float32)
        ),
        config=trainer_config,
        seed=int(seed),
    )
    n_torch_trainable_params = int(
        sum(p.numel() for p in trainer.parameters() if p.requires_grad)
    )

    # ------------------------------------------------------------------
    # Step 4 -- Phase B training.
    # ------------------------------------------------------------------
    Z_train_t = torch.from_numpy(Z_train)
    cur_train_t = torch.from_numpy(train_ds.current_states.astype(np.int64))
    nxt_train_t = torch.from_numpy(train_ds.y_next.astype(np.int64))

    epoch_losses: list[float] = []
    for _epoch in range(int(n_epochs)):
        perm = rng.permutation(n_train)
        idx = torch.from_numpy(perm.astype(np.int64))
        obs_b = Z_train_t[idx]
        cur_b = cur_train_t[idx]
        nxt_b = nxt_train_t[idx]
        out = trainer.step(obs_b, cur_b, nxt_b)
        epoch_losses.append(float(out["loss"].item()))

    final_loss = epoch_losses[-1] if epoch_losses else 0.0
    loss_decrease_first_to_last_epoch = (
        (epoch_losses[0] - epoch_losses[-1]) if len(epoch_losses) >= 2 else 0.0
    )

    phase_b_seconds = time.perf_counter() - phase_b_start

    # ------------------------------------------------------------------
    # Step 5 -- Phase B Hamming.
    # ------------------------------------------------------------------
    posterior_snapshot = trainer.to_classical_posterior_mask()
    learned_legal_b = posterior_snapshot.legality_matrix(0.5)
    hamming_b = int(np.sum(learned_legal_b != gold_legal))
    phase_b_hamming_normalised = (
        float(hamming_b) / float(total_cells) if total_cells else 0.0
    )

    # ------------------------------------------------------------------
    # Step 6 -- Phase C eval.
    # ------------------------------------------------------------------
    energy_fn = EnergyFunction(EnergyParameters())

    detector_weights = dict(SingularityDetector.DEFAULT_WEIGHTS)
    detector_weights["kl_surprise"] = float(kl_surprise_weight)
    detector = SingularityDetector(
        margin_threshold=margin_threshold,
        enabled=ablation.singularity_detector_enabled,
        weights=detector_weights,
    )

    history = TaggerHistory()
    # Goal state for ControlPolicy is the terminal DIAGNOSED.
    control_policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        fsm=fsm,
        goal_states=[fsm.vertex_index["DIAGNOSED"]],
    )

    Z_test_t = torch.from_numpy(Z_test)
    sigmas_calib: list[float] = []
    energies_calib: list[float] = []
    margins_calib: list[float] = []
    with torch.no_grad():
        for i in range(n_test):
            obs_t = Z_test_t[i]
            cs_idx = int(test_ds.current_states[i])
            p = trainer.predicted_distribution(obs_t, current_state=cs_idx)
            p_np = p.detach().cpu().numpy().astype(np.float64)
            m = float(compute_margin(p_np))
            margins_calib.append(m)
            sigma_i, _ = detector.compute(
                margin=m, is_illegal=False, loop_risk=0.0
            )
            sigmas_calib.append(float(sigma_i))
            unc = float(1.0 - p_np.max())
            contribs = energy_fn.compute(
                cost=0.0,
                uncertainty=unc,
                contradiction=0.0,
                loop_pressure=float(np.clip(sigma_i, 0.0, 1.0)) * 0.5,
                progress=1.0,
            )
            energies_calib.append(float(contribs.total))

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

    sigma_q = _safe_quantizer("sigma_axis", sigmas_calib)
    energy_q = _safe_quantizer("energy_axis", energies_calib)
    margin_q = _safe_quantizer("margin_axis", margins_calib)

    state_axis_nodes = list(vertex_ids)
    product_graph = ProductGraph(
        axis_names=["state", "sigma_axis", "energy_axis", "margin_axis"],
        axis_node_lists={
            "state": state_axis_nodes,
            "sigma_axis": sigma_q.node_ids(),
            "energy_axis": energy_q.node_ids(),
            "margin_axis": margin_q.node_ids(),
        },
    )

    windows: dict[int, deque] = {
        s: deque(maxlen=empirical_window) for s in range(n_states)
    }
    for i in range(min(n_train, empirical_window)):
        windows[int(train_ds.current_states[i])].append(
            int(train_ds.y_next[i])
        )

    experiment_label = "E22"
    ablation_label = run_id.split("_")[1]
    sample_ids = [s.sample_id for s in test_ds.samples]

    sigmas_step: list[float] = []
    margins_step: list[float] = []
    errors_step: list[bool] = []
    current_state_names_step: list[str] = []

    sigma_at_boundary: list[float] = []
    sigma_at_non_boundary: list[float] = []

    y_true_list: list[str] = []
    predicted_list: list[str] = []
    predicted_no_mask_list: list[str] = []
    transition_legal_list: list[bool] = []
    transition_legal_no_mask_list: list[bool] = []
    energies_total: list[float] = []
    strata: list[SingularityType] = []
    energies_correct: list[float] = []
    energies_incorrect: list[float] = []
    n_routed_to_recovery = 0
    n_abstained = 0
    n_mask_modified_predictions = 0
    prev_output_tuple: tuple[str, ...] | None = None
    kl_surprises_step: list[float] = []

    # For diagnostic_accuracy: filter to the diagnosis steps.
    diagnostic_correct: list[bool] = []

    boundary_step = is_boundary_state[test_ds.current_states]
    is_diagnosis_step_arr = test_ds.is_diagnosis_step

    inference_start = time.perf_counter()

    with (
        JsonlWriter(output_dir / "metrics.jsonl", MetricsRecord) as metrics_writer,
        JsonlWriter(output_dir / "results.jsonl", ResultsRecord) as results_writer,
        open_decision_trace_writer(
            output_dir / "decision_trace.jsonl"
        ) as trace_writer,
    ):
        with torch.no_grad():
            for step_idx in range(n_test):
                cs_idx = int(test_ds.current_states[step_idx])
                current_state = vertex_ids[cs_idx]
                true_next_idx = int(test_ds.y_next[step_idx])
                true_next = vertex_ids[true_next_idx]
                obs_t = Z_test_t[step_idx]

                bias_torch = trainer.legality_bias().detach().cpu().numpy()
                bias_row_learned = bias_torch[cs_idx].astype(np.float64)
                bias_row_uniform = np.zeros_like(bias_row_learned)

                p_masked = trainer.predicted_distribution(
                    obs_t, current_state=cs_idx
                ).detach().cpu().numpy().astype(np.float64)

                d_all = trainer.poincare_distance_torch(
                    obs_t.unsqueeze(0), trainer.prototypes
                ).squeeze(0).detach().cpu().numpy().astype(np.float64)
                logits_unmasked = -d_all + bias_row_uniform
                logits_unmasked = logits_unmasked - logits_unmasked.max()
                ex = np.exp(logits_unmasked)
                p_unmasked = ex / max(ex.sum(), 1e-12)

                margin = float(compute_margin(p_masked))
                argmax_unmasked = int(np.argmax(p_unmasked))
                predicted_no_mask = vertex_ids[argmax_unmasked]
                argmax_state = int(np.argmax(p_masked))
                predicted_state = vertex_ids[argmax_state]

                window = windows[cs_idx]
                empirical_dist = _empirical_dist_from_window(window, n_states)
                kl_surprise = float(
                    compute_kl_surprise(empirical_dist, p_masked)
                )
                kl_surprises_step.append(kl_surprise)

                is_illegal = not bool(
                    fsm.is_legal_transition(current_state, predicted_state)
                )
                loop_risk = float(
                    history.predicted_loop_risk(predicted_state)
                )

                sigma, contribs = detector.compute(
                    margin=margin,
                    is_illegal=is_illegal,
                    loop_risk=loop_risk,
                    kl_surprise=kl_surprise,
                )

                if bool(boundary_step[step_idx]):
                    sigma_at_boundary.append(float(sigma))
                else:
                    sigma_at_non_boundary.append(float(sigma))

                d_chosen = float(d_all[argmax_state])
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

                windows[cs_idx].append(int(true_next_idx))

                tagger_inp = TaggerInput(
                    distribution=p_unmasked,
                    predicted_state=predicted_state,
                    current_state=current_state,
                    margin=margin,
                )
                tag_result = tag_step(
                    tagger_inp, history, fsm,
                    margin_threshold=margin_threshold,
                )
                history.push(predicted_state)

                transition_legal = bool(
                    fsm.is_legal_transition(current_state, predicted_state)
                )
                transition_legal_no_mask = bool(
                    fsm.is_legal_transition(current_state, predicted_no_mask)
                )

                mask_changed_prediction = bool(
                    argmax_unmasked != argmax_state
                )
                if mask_changed_prediction:
                    n_mask_modified_predictions += 1

                if ablation.graph_mask_enabled:
                    row_mean = posterior_snapshot.posterior_mean()[cs_idx]
                    illegal_indices_zeroed = [
                        int(j)
                        for j in range(n_states)
                        if row_mean[j] < 0.5
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
                    product_graph.add_transition(
                        prev_output_tuple, output_node_tuple
                    )
                    edge_traversed = (
                        list(prev_output_tuple),
                        list(output_node_tuple),
                    )
                else:
                    product_graph.materialise(output_node_tuple)
                    edge_traversed = None

                mean_alpha = float(np.mean(posterior_snapshot.alpha))
                mean_beta = float(np.mean(posterior_snapshot.beta))
                mean_entropy = float(
                    np.mean(posterior_snapshot.posterior_entropy())
                )
                posterior_summary = {
                    "mean_alpha": mean_alpha,
                    "mean_beta": mean_beta,
                    "mean_entropy": mean_entropy,
                }

                sorted_idx = np.argsort(p_unmasked)[::-1]
                top1_idx = int(sorted_idx[0])
                top1_prob = float(p_unmasked[top1_idx])
                if p_unmasked.shape[0] > 1:
                    top2_idx = int(sorted_idx[1])
                    top2_prob = float(p_unmasked[top2_idx])
                    top2_state: str | None = vertex_ids[top2_idx]
                else:
                    top2_prob = 0.0
                    top2_state = None

                sigma_signals = {
                    "margin": float(contribs.margin_signal),
                    "decision_tie": float(contribs.decision_tie_signal),
                    "illegal": float(contribs.illegal_signal),
                    "loop": float(contribs.loop_signal),
                    "stabilizer": float(contribs.stabilizer_signal),
                    "catastrophe_bias": float(contribs.catastrophe_bias),
                    "kl_surprise": float(contribs.kl_surprise),
                }
                energy_breakdown = {
                    "cost": float(energy_breakdown_obj.cost),
                    "uncertainty": float(energy_breakdown_obj.uncertainty),
                    "contradiction": float(
                        energy_breakdown_obj.contradiction
                    ),
                    "loop_pressure": float(
                        energy_breakdown_obj.loop_pressure
                    ),
                    "progress": float(energy_breakdown_obj.progress),
                }
                axis_node_ids = {
                    "state": predicted_state,
                    "sigma_axis": str(sigma_node),
                    "energy_axis": str(energy_node),
                    "margin_axis": str(margin_node),
                }
                confidence = float(p_masked[argmax_state])

                results_writer.append(
                    ResultsRecord(
                        run_id=run_id,
                        experiment=experiment_label,
                        ablation=ablation_label,
                        seed=seed,
                        step=step_idx,
                        sample_id=sample_ids[step_idx],
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
                        sample_id=sample_ids[step_idx],
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
                        timestamp=dt.datetime.now(
                            tz=dt.timezone.utc
                        ).isoformat(),
                        output_node_tuple=list(output_node_tuple),
                        edge_traversed=edge_traversed,
                        mask_version_id=posterior_snapshot.mask_version_id(),
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

                err = bool(true_next != predicted_state)
                sigmas_step.append(float(sigma))
                margins_step.append(float(margin))
                errors_step.append(err)
                current_state_names_step.append(current_state)

                if bool(is_diagnosis_step_arr[step_idx]):
                    diagnostic_correct.append(true_next == predicted_state)

                prev_output_tuple = output_node_tuple

        inference_seconds = time.perf_counter() - inference_start

        # ------------------------------------------------------------------
        # Step 7 -- Aggregate metrics.
        # ------------------------------------------------------------------
        y_true_arr = np.array(y_true_list)
        predicted_arr = np.array(predicted_list)
        predicted_no_mask_arr = np.array(predicted_no_mask_list)
        accuracy = (
            float(np.mean(y_true_arr == predicted_arr)) if n_test else 0.0
        )
        accuracy_no_mask = (
            float(np.mean(y_true_arr == predicted_no_mask_arr))
            if n_test
            else 0.0
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

        sigmas_arr = np.array(sigmas_step, dtype=float)
        margins_arr = np.array(margins_step, dtype=float)
        errors_arr = np.array(errors_step, dtype=bool)
        if errors_arr.size > 0 and 0 < int(errors_arr.sum()) < int(
            errors_arr.size
        ):
            sigma_auroc_value = float(sigma_auroc(sigmas_arr, errors_arr))
            margin_auroc_value = float(margin_auroc(margins_arr, errors_arr))
        else:
            sigma_auroc_value = 0.5
            margin_auroc_value = 0.5
        sigma_uplift = sigma_auroc_value - margin_auroc_value

        ambiguity_labels = np.array(
            [
                1 if _is_structural_ambiguity_state(name) else 0
                for name in current_state_names_step
            ],
            dtype=np.int64,
        )
        sigma_structural_auroc_value = float(
            structural_ambiguity_auroc(sigmas_arr, ambiguity_labels)
        )
        margin_structural_auroc_value = float(
            structural_ambiguity_auroc(1.0 - margins_arr, ambiguity_labels)
        )
        sigma_structural_uplift = (
            sigma_structural_auroc_value - margin_structural_auroc_value
        )

        mean_sigma_at_operator_boundary = (
            float(np.mean(sigma_at_boundary)) if sigma_at_boundary else 0.0
        )
        mean_sigma_at_non_boundary = (
            float(np.mean(sigma_at_non_boundary))
            if sigma_at_non_boundary
            else 0.0
        )
        if mean_sigma_at_non_boundary > 1e-9:
            sigma_boundary_ratio = (
                mean_sigma_at_operator_boundary
                / mean_sigma_at_non_boundary
            )
        else:
            sigma_boundary_ratio = (
                float("inf")
                if mean_sigma_at_operator_boundary > 0.0
                else 1.0
            )

        # Diagnostic-specific.
        diagnostic_accuracy = (
            float(np.mean(diagnostic_correct))
            if diagnostic_correct
            else 0.0
        )

        total_seconds = time.perf_counter() - total_start
        mem_end_kb = _peak_memory_kb()
        peak_memory_kb = max(mem_end_kb - mem_start_kb, 0.0)
        inference_throughput = (
            float(n_test) / inference_seconds
            if inference_seconds > 0
            else 0.0
        )

        timestamp = dt.datetime.now(tz=dt.timezone.utc).isoformat()
        metric_pairs: list[tuple[str, float]] = [
            ("accuracy", accuracy),
            ("accuracy_no_mask", accuracy_no_mask),
            ("mask_accuracy_uplift", mask_accuracy_uplift),
            ("illegal_transition_rate", illegal_transition_rate),
            (
                "illegal_transition_rate_no_mask",
                illegal_transition_rate_no_mask,
            ),
            ("n_routed_to_recovery", float(n_routed_to_recovery)),
            ("n_abstained", float(n_abstained)),
            (
                "n_mask_modified_predictions",
                float(n_mask_modified_predictions),
            ),
            ("mean_energy_correct", mean_energy_correct),
            ("mean_energy_incorrect", mean_energy_incorrect),
            ("free_energy", run_free_energy),
            ("phase_a_hamming_normalised", float(phase_a_hamming_normalised)),
            ("phase_b_hamming_normalised", float(phase_b_hamming_normalised)),
            ("sigma_auroc", float(sigma_auroc_value)),
            ("margin_auroc", float(margin_auroc_value)),
            ("sigma_uplift", float(sigma_uplift)),
            (
                "sigma_structural_auroc",
                float(sigma_structural_auroc_value),
            ),
            (
                "margin_structural_auroc",
                float(margin_structural_auroc_value),
            ),
            (
                "sigma_structural_uplift",
                float(sigma_structural_uplift),
            ),
            (
                "mean_sigma_at_operator_boundary",
                float(mean_sigma_at_operator_boundary),
            ),
            (
                "mean_sigma_at_non_boundary",
                float(mean_sigma_at_non_boundary),
            ),
            (
                "sigma_boundary_ratio",
                float(sigma_boundary_ratio)
                if np.isfinite(sigma_boundary_ratio)
                else 1e9,
            ),
            ("diagnostic_accuracy", float(diagnostic_accuracy)),
            (
                "mean_symptoms_observed_before_diagnosis",
                float(mean_symptoms_observed_before_diagnosis),
            ),
            ("n_torch_trainable_params", float(n_torch_trainable_params)),
            ("final_loss", float(final_loss)),
            (
                "loss_decrease_first_to_last_epoch",
                float(loss_decrease_first_to_last_epoch),
            ),
            (
                "mean_kl_surprise",
                float(np.mean(kl_surprises_step)) if kl_surprises_step else 0.0,
            ),
            ("n_train", float(n_train)),
            ("n_test", float(n_test)),
            (
                "product_graph_cells_materialised",
                float(product_graph.n_materialised()),
            ),
            ("phase_a_wall_clock_seconds", float(phase_a_seconds)),
            ("phase_b_wall_clock_seconds", float(phase_b_seconds)),
            ("total_wall_clock_seconds", float(total_seconds)),
            (
                "inference_throughput_samples_per_sec",
                float(inference_throughput),
            ),
            ("peak_memory_kb", float(peak_memory_kb)),
        ]
        for metric_name, value in metric_pairs:
            metrics_writer.append(
                MetricsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=0,
                    split="test",
                    metric_name=metric_name,
                    value=value,
                    timestamp=timestamp,
                )
            )

    _ = kl_categorical(np.array([0.5, 0.5]), np.array([0.5, 0.5]))

    return E22Result(
        accuracy=accuracy,
        accuracy_no_mask=accuracy_no_mask,
        mask_accuracy_uplift=mask_accuracy_uplift,
        illegal_transition_rate=illegal_transition_rate,
        illegal_transition_rate_no_mask=illegal_transition_rate_no_mask,
        phase_a_hamming_normalised=phase_a_hamming_normalised,
        phase_b_hamming_normalised=phase_b_hamming_normalised,
        sigma_auroc=sigma_auroc_value,
        margin_auroc=margin_auroc_value,
        sigma_uplift=sigma_uplift,
        sigma_structural_auroc=sigma_structural_auroc_value,
        margin_structural_auroc=margin_structural_auroc_value,
        sigma_structural_uplift=sigma_structural_uplift,
        mean_sigma_at_operator_boundary=mean_sigma_at_operator_boundary,
        mean_sigma_at_non_boundary=mean_sigma_at_non_boundary,
        sigma_boundary_ratio=float(sigma_boundary_ratio)
        if np.isfinite(sigma_boundary_ratio)
        else 1e9,
        diagnostic_accuracy=float(diagnostic_accuracy),
        mean_symptoms_observed_before_diagnosis=float(
            mean_symptoms_observed_before_diagnosis
        ),
        n_torch_trainable_params=n_torch_trainable_params,
        final_loss=float(final_loss),
        loss_decrease_first_to_last_epoch=float(
            loss_decrease_first_to_last_epoch
        ),
        n_train=int(n_train),
        n_test=int(n_test),
        phase_a_wall_clock_seconds=float(phase_a_seconds),
        phase_b_wall_clock_seconds=float(phase_b_seconds),
        total_wall_clock_seconds=float(total_seconds),
        inference_throughput_samples_per_sec=float(inference_throughput),
        peak_memory_kb=float(peak_memory_kb),
    )
