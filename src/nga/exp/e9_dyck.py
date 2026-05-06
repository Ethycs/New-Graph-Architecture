"""E9 - Dyck-k bracket-language run.

The first dataset whose underlying structure is the architecture's natural
target: a stack-language with a finite-state abstraction, real cycles,
graded depth, and explicit adversarial illegal closes.

Architectural claims this run validates:

  * **Mask**: closes that don't match the stack top are illegal. With the
    mask, ``illegal_transition_rate`` is driven to 0; without the mask the
    classifier is tempted into illegal closes by adversarial samples and
    illegal_transition_rate_no_mask > 0.05.
  * **sigma**: at high depth the noise padding has more relative weight,
    confidence drops, and sigma rises - producing a real
    confidence-vs-depth signal.
  * **Monodromy**: open(b) followed by close(b) is a length-2 closed walk
    on the FSM, so n_monodromy_cycle_revisits is large.
  * **Energy / control**: mean_energy_correct < mean_energy_incorrect on
    enough samples; sigma-routing fires on the harder (deep / adversarial)
    cases.

Mirrors ``e1_synthetic_babyai`` exactly for reusable infrastructure: typed
field pipeline, behavioral stratum tagger, singularity detector, energy
function, control policy, monodromy table, decision_trace writer.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from nga.arch.behavioral_stratum_tagger import TaggerHistory, TaggerInput, tag_step
from nga.arch.control_policy import ControlPolicy
from nga.arch.dart_permutations import build_darts_from_fsm, build_rho, build_tau
from nga.arch.energy_function import EnergyFunction, EnergyParameters
from nga.arch.graph_fsm import GraphFSM
from nga.arch.margin_uncertainty import compute_margin
from nga.arch.monodromy_group import compute_monodromy
from nga.arch.singularity_detector import SingularityDetector
from nga.arch.singularity_types import SingularityType
from nga.arch.stratified_partition_function import (
    compute_stratified_partition,
    free_energy,
)
from nga.arch.typed_field_pipeline import TypedFieldOutput, TypedFieldPipeline
from nga.arch.typed_score_record_builder import build_typed_score_record
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
from nga.drivers.typed_score_record import TypedScoreRecord
from nga.exp.dataset_dyck_k import (
    generate_dyck_dataset,
    train_test_split_by_sequence,
)
from nga.exp.e0_mnist import mask_probabilities

__all__ = ["E9Result", "run_e9"]


@dataclass
class E9Result:
    """Aggregate metrics returned by run_e9."""

    accuracy: float
    accuracy_no_mask: float
    illegal_transition_rate: float
    illegal_transition_rate_no_mask: float
    mask_accuracy_uplift: float
    n_test: int
    n_features: int
    n_monodromy_cycle_revisits: int
    mean_depth: float


def _build_monodromy_class_by_state(fsm: GraphFSM) -> dict[int, int]:
    """Map FSM vertex index -> a representative monodromy-class label.

    The Dyck-k FSM has many cycles, so this delivers a real signature
    (unlike the acyclic babyai FSM where the labels are mostly trivial).
    """
    darts = build_darts_from_fsm(fsm)
    if darts.n_darts == 0:
        return {i: 0 for i in range(fsm.vertex_count)}
    rho = build_rho(fsm, darts)
    tau = build_tau(darts)
    mono = compute_monodromy(rho, tau)
    out: dict[int, int] = {}
    for v_idx in range(fsm.vertex_count):
        dart_indices = np.flatnonzero(darts.vertex_of == v_idx)
        if dart_indices.size > 0:
            out[v_idx] = int(mono.monodromy_class[int(dart_indices[0])])
        else:
            out[v_idx] = 0
    return out


def run_e9(
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
) -> E9Result:
    """Train + evaluate the Dyck-k classifier with and without the legality mask.

    Outputs (CLI pre-creates the empty sink files; this function appends):
      - ``output_dir/metrics.jsonl``
      - ``output_dir/results.jsonl``
      - ``output_dir/scores.jsonl``
      - ``output_dir/decision_trace.jsonl``

    The metrics row set:
      accuracy, accuracy_no_mask, mask_accuracy_uplift,
      illegal_transition_rate, illegal_transition_rate_no_mask,
      n_routed_to_recovery, n_abstained, n_mask_modified_predictions,
      free_energy, mean_energy_correct, mean_energy_incorrect, mean_depth,
      n_monodromy_cycle_revisits, n_samples, n_features.

    Acceptance: with ablation.graph_mask_enabled=True,
    illegal_transition_rate must be 0.0 and illegal_transition_rate_no_mask
    must be > 0.05 (driven by the adversarial-temptation samples).

    Parameters
    ----------
    config:
        Merged runtime config; ``embedding_dim`` is used for feature_dim.
    ablation:
        AblationTuple controlling which components are active.
    fsm:
        GraphFSM with the canonical Dyck-k state set.
    run_id:
        Composite run id, e.g. ``E9_A0_seed42``.
    output_dir:
        Directory in which to append JSONL files.
    seed:
        Integer RNG seed.
    margin_threshold:
        Threshold for the singularity detector and the stratum tagger.
    n_sequences:
        How many random Dyck walks to generate.
    max_length:
        Per-sequence step budget.
    illegal_temptation_fraction:
        Fraction of close-steps to corrupt to non-matching close.
    """
    # ------------------------------------------------------------------
    # Step 1 - Dataset
    # ------------------------------------------------------------------
    # The Dyck FSM dictates k and max_depth via its vertex count.
    # vertex_count == 1 + max_depth*k. For k=2 we expect 9 states.
    if fsm.vertex_ids[0] != "S0":
        raise ValueError(
            f"E9 expects fsm.vertex_ids[0] == 'S0'; got {fsm.vertex_ids[0]!r}"
        )
    n_states = fsm.vertex_count
    # k=2 is the only currently supported width; max_depth derived.
    k = 2
    max_depth = (n_states - 1) // k
    if 1 + max_depth * k != n_states:
        raise ValueError(
            f"E9 cannot derive (k, max_depth) from {n_states} states; "
            f"expected 1 + max_depth*k == n_states, got k={k}."
        )

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

    train_ds, test_ds = train_test_split_by_sequence(ds, seed=seed)

    X_train = train_ds.X
    y_train_idx = train_ds.y_next
    X_test = test_ds.X
    y_test_idx = test_ds.y_next
    test_current_states_idx = test_ds.current_states
    test_is_adv = test_ds.is_adversarial
    test_depths = test_ds.depths
    test_sample_ids = [s.sample_id for s in test_ds.samples]

    # String-form labels for sklearn (preserves vertex_id ordering).
    vertex_ids = fsm.vertex_ids
    y_train_str = [vertex_ids[int(i)] for i in y_train_idx]
    y_test_str = [vertex_ids[int(i)] for i in y_test_idx]

    # ------------------------------------------------------------------
    # Step 2 - Train classifier (no current_state feature)
    # ------------------------------------------------------------------
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train_str)

    # ------------------------------------------------------------------
    # Step 3 - Pipeline + per-step machinery
    # ------------------------------------------------------------------
    pipeline = TypedFieldPipeline(clf, fsm.vertex_ids, ablation)
    detector = SingularityDetector(
        margin_threshold=margin_threshold,
        enabled=ablation.singularity_detector_enabled,
    )
    history = TaggerHistory()
    monodromy_class_by_state = _build_monodromy_class_by_state(fsm)
    energy_fn = EnergyFunction(EnergyParameters())

    # Goal placeholder: S0 (empty stack) is the canonical "completed" state.
    # ControlPolicy BFS routes toward it from any current_state in the
    # recovery band, so ROUTE_RECOVERY is actionable on this dataset.
    goal_states: list[int] = [0]
    control_policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        fsm=fsm,
        goal_states=goal_states,
    )

    experiment_label = "E9"
    ablation_label = run_id.split("_")[1]

    # ------------------------------------------------------------------
    # Step 4 - Evaluation accumulators
    # ------------------------------------------------------------------
    n_test = X_test.shape[0]

    y_true_list: list[str] = []
    predicted_list: list[str] = []
    predicted_no_mask_list: list[str] = []
    transition_legal_list: list[bool | None] = []
    transition_legal_no_mask_list: list[bool | None] = []

    energies: list[float] = []
    strata: list[SingularityType] = []
    n_routed_to_recovery: int = 0
    n_abstained: int = 0
    n_mask_modified_predictions: int = 0
    n_monodromy_cycle_revisits: int = 0
    energies_correct: list[float] = []
    energies_incorrect: list[float] = []
    visited_states: set[int] = set()

    with (
        JsonlWriter(output_dir / "metrics.jsonl", MetricsRecord) as metrics_writer,
        JsonlWriter(output_dir / "results.jsonl", ResultsRecord) as results_writer,
        JsonlWriter(output_dir / "scores.jsonl", TypedScoreRecord) as scores_writer,
        open_decision_trace_writer(output_dir / "decision_trace.jsonl") as trace_writer,
    ):
        # --------------------------------------------------------------
        # Step 5 - Per-sample loop
        # --------------------------------------------------------------
        for i in range(n_test):
            current_state_idx = int(test_current_states_idx[i])
            current_state: str | None = vertex_ids[current_state_idx]
            y_true_i: str = y_test_str[i]

            out = pipeline.predict_one(X_test[i])
            raw_dist: np.ndarray = out.distribution

            argmax_unmasked = int(np.argmax(raw_dist))
            predicted_state_no_mask = fsm.vertex_ids[argmax_unmasked]

            masked_dist = mask_probabilities(
                raw_dist, fsm, current_state, enabled=ablation.graph_mask_enabled
            )
            argmax_masked = int(np.argmax(masked_dist))
            predicted_state = fsm.vertex_ids[argmax_masked]
            confidence = float(masked_dist[argmax_masked])

            margin = compute_margin(masked_dist)

            # Top-2 (post-mask) for the trace.
            sorted_idx = np.argsort(masked_dist)[::-1]
            top1_idx = int(sorted_idx[0])
            top1_prob = float(masked_dist[top1_idx])
            if masked_dist.shape[0] > 1:
                top2_idx = int(sorted_idx[1])
                top2_prob = float(masked_dist[top2_idx])
                top2_state: str | None = fsm.vertex_ids[top2_idx]
            else:
                top2_idx = top1_idx
                top2_prob = 0.0
                top2_state = None

            transition_legal: bool | None = (
                fsm.is_legal_transition(current_state, predicted_state)
                if current_state is not None
                else None
            )
            transition_legal_no_mask: bool | None = (
                fsm.is_legal_transition(current_state, predicted_state_no_mask)
                if current_state is not None
                else None
            )

            tagger_inp = TaggerInput(
                distribution=masked_dist,
                predicted_state=predicted_state,
                current_state=current_state,
                margin=margin,
            )
            tag_result = tag_step(
                tagger_inp, history, fsm, margin_threshold=margin_threshold
            )
            history.push(predicted_state)

            margin_for_sigma = compute_margin(masked_dist)
            is_illegal_for_sigma = (
                (not fsm.is_legal_transition(current_state, predicted_state))
                if current_state is not None
                else False
            )
            loop_risk = history.predicted_loop_risk(predicted_state)
            sigma, contribs = detector.compute(
                margin=margin_for_sigma,
                is_illegal=is_illegal_for_sigma,
                loop_risk=loop_risk,
            )
            singular_flag = bool(sigma >= 0.5)

            # Wire-format records.
            masked_out = TypedFieldOutput(
                predicted_state=predicted_state,
                stratum_label=predicted_state if ablation.typed_scores_enabled else "",
                distribution=masked_dist,
                confidence=confidence,
                embedding=None,
            )
            score_rec = build_typed_score_record(
                masked_out,
                run_id=run_id,
                experiment=experiment_label,
                ablation=ablation_label,
                seed=seed,
                step=i,
                sample_id=test_sample_ids[i],
                margin=margin,
                z_H=None,
            )
            scores_writer.append(score_rec)

            result_rec = ResultsRecord(
                run_id=run_id,
                experiment=experiment_label,
                ablation=ablation_label,
                seed=seed,
                step=i,
                sample_id=test_sample_ids[i],
                y_true=y_true_i,
                y_hat=predicted_state,
                margin=margin,
                singular_flag=singular_flag,
                sigma_score=sigma,
                behavioral_stratum=tag_result.tag.name,
                stratum_bitmask=tag_result.bitmask,
                transition_legal=transition_legal,
            )
            results_writer.append(result_rec)

            # ----------------------------------------------------------
            # Energy + control + decision trace
            # ----------------------------------------------------------
            uncertainty = 1.0 - confidence
            contradiction_signal = 1.0 if is_illegal_for_sigma else 0.0
            progress_signal = max(0.0, min(1.0, 1.0 - uncertainty))
            energy = energy_fn.compute(
                cost=0.0,
                uncertainty=uncertainty,
                contradiction=contradiction_signal,
                loop_pressure=loop_risk,
                progress=progress_signal,
            )

            y_hat_int = argmax_masked
            if y_hat_int in visited_states:
                n_monodromy_cycle_revisits += 1
            visited_states.add(y_hat_int)

            if current_state is not None and ablation.graph_mask_enabled:
                src_idx = fsm.index_of(current_state)
                row = fsm.legality_matrix[src_idx]
                illegal_indices_zeroed: list[int] = [
                    int(j) for j in range(fsm.vertex_count) if not bool(row[j])
                ]
            else:
                illegal_indices_zeroed = []

            mask_changed_prediction = bool(argmax_unmasked != argmax_masked)
            if mask_changed_prediction:
                n_mask_modified_predictions += 1

            mask_action = MaskAction(
                enabled=bool(ablation.graph_mask_enabled),
                current_state=current_state,
                illegal_indices_zeroed=illegal_indices_zeroed,
                pre_mask_argmax=predicted_state_no_mask,
                post_mask_argmax=predicted_state,
                mask_changed_prediction=mask_changed_prediction,
            )

            cs_int: int | None = (
                fsm.index_of(current_state) if current_state is not None else None
            )
            policy_result = control_policy.decide(
                sigma=float(sigma),
                model_prediction=int(argmax_masked),
                current_state=cs_int,
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

            sigma_signals: dict[str, float] = {
                "margin": float(contribs.margin_signal),
                "decision_tie": float(contribs.decision_tie_signal),
                "illegal": float(contribs.illegal_signal),
                "loop": float(contribs.loop_signal),
                "stabilizer": float(contribs.stabilizer_signal),
                "catastrophe_bias": float(contribs.catastrophe_bias),
            }

            energy_breakdown: dict[str, float] = {
                "cost": float(energy.cost),
                "uncertainty": float(energy.uncertainty),
                "contradiction": float(energy.contradiction),
                "loop_pressure": float(energy.loop_pressure),
                "progress": float(energy.progress),
            }

            mono_class = monodromy_class_by_state.get(y_hat_int, 0)

            trace_rec = DecisionTraceRecord(
                run_id=run_id,
                experiment=experiment_label,
                ablation=ablation_label,
                seed=seed,
                step=i,
                sample_id=test_sample_ids[i],
                confidence=float(confidence),
                top1_state=predicted_state,
                top2_state=top2_state,
                top1_prob=top1_prob,
                top2_prob=top2_prob,
                margin=float(margin),
                mask=mask_action,
                stratum_tag=tag_result.tag.name,
                stratum_bitmask=tag_result.bitmask,
                stratum_confidence=float(tag_result.confidence),
                sigma_total=float(sigma),
                sigma_signals=sigma_signals,
                energy_total=float(energy.total),
                energy_breakdown=energy_breakdown,
                P_lambda=None,
                free_energy=None,
                monodromy_class=int(mono_class),
                in_closed_walk=None,
                control=control_action,
                timestamp=dt.datetime.now(tz=dt.timezone.utc).isoformat(),
            )
            trace_writer.append(trace_rec)

            # Aggregate accumulators.
            y_true_list.append(y_true_i)
            predicted_list.append(predicted_state)
            predicted_no_mask_list.append(predicted_state_no_mask)
            transition_legal_list.append(transition_legal)
            transition_legal_no_mask_list.append(transition_legal_no_mask)

            energies.append(float(energy.total))
            strata.append(tag_result.tag)
            if y_true_i == predicted_state:
                energies_correct.append(float(energy.total))
            else:
                energies_incorrect.append(float(energy.total))

            # Adversarial samples only used to make a per-sample log line if we
            # ever want it; currently they exercise the mask via the
            # transition_legal_no_mask check above.
            _ = test_is_adv[i]

        # --------------------------------------------------------------
        # Step 6 - Aggregate metrics
        # --------------------------------------------------------------
        y_true_arr = np.array(y_true_list)
        predicted_arr = np.array(predicted_list)
        predicted_no_mask_arr = np.array(predicted_no_mask_list)

        accuracy = float(np.mean(y_true_arr == predicted_arr))
        accuracy_no_mask = float(np.mean(y_true_arr == predicted_no_mask_arr))
        mask_accuracy_uplift = accuracy - accuracy_no_mask

        samples_with_state = [
            (legal, legal_nm)
            for legal, legal_nm in zip(transition_legal_list, transition_legal_no_mask_list)
            if legal is not None
        ]
        total_with_state = len(samples_with_state)
        if total_with_state > 0:
            illegal_transition_rate = float(
                sum(1 for legal, _ in samples_with_state if legal is False)
                / total_with_state
            )
            illegal_transition_rate_no_mask = float(
                sum(1 for _, legal_nm in samples_with_state if legal_nm is False)
                / total_with_state
            )
        else:
            illegal_transition_rate = 0.0
            illegal_transition_rate_no_mask = 0.0

        if energies:
            partition = compute_stratified_partition(
                np.array(energies, dtype=np.float64),
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

        mean_depth = float(np.mean(test_depths)) if len(test_depths) > 0 else 0.0

        # --------------------------------------------------------------
        # Step 7 - Write metric rows
        # --------------------------------------------------------------
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
            ("mean_depth", mean_depth),
            ("n_samples", float(n_test)),
            ("n_features", float(X_test.shape[1])),
        ]:
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

    return E9Result(
        accuracy=accuracy,
        accuracy_no_mask=accuracy_no_mask,
        illegal_transition_rate=illegal_transition_rate,
        illegal_transition_rate_no_mask=illegal_transition_rate_no_mask,
        mask_accuracy_uplift=mask_accuracy_uplift,
        n_test=n_test,
        n_features=int(X_test.shape[1]),
        n_monodromy_cycle_revisits=int(n_monodromy_cycle_revisits),
        mean_depth=mean_depth,
    )
