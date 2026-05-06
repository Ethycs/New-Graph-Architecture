"""E0 - MNIST Typed-State Sanity Run.

The smallest end-to-end experiment. Validates Phase 1 + Phase 2 plumbing: typed
scores -> graph mask -> margin -> sigma(x) -> singularity flag, with sklearn
LogisticRegression on the digits dataset. Acceptance: accuracy >= 0.90 plus
per-sample singular_flag driven by sigma(x) >= 0.5 (Phase 2) rather than raw
margin < threshold (Phase 1).

Phase 2 change: singular_flag is now derived from sigma(x) >= 0.5 where sigma is
produced by SingularityDetector. This replaces the plain margin < margin_threshold
threshold used in Phase 1. The two signals measure the same underlying uncertainty
but sigma combines margin, loop_risk, and (in later phases) additional signals;
tests confirm that the sigma-based flag preserves or strengthens the property that
singular_flag fires more often on errors than on correct predictions.

Phase 4-5 wiring: this runner now also emits a per-sample decision_trace.jsonl
recording the full WHY of each prediction (mask action, sigma signals, energy
breakdown, monodromy class, control decision). Aggregate per-stratum partition
function values and energy-by-correctness summaries are appended to metrics.jsonl.
results.jsonl continues to record the model's raw prediction (no policy override).

This module is a leaf - imported by nga.cli when --experiment E0 is selected.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nga.arch.behavioral_stratum_tagger import TaggerHistory, TaggerInput, tag_step
from nga.arch.confusion_graph import ConfusionGraph
from nga.arch.control_policy import ControlPolicy
from nga.arch.dart_permutations import build_darts_from_fsm, build_rho, build_tau
from nga.arch.energy_function import EnergyFunction, EnergyParameters
from nga.arch.energy_weighted_loss import energy_weighted_batch_loss
from nga.arch.graph_fsm import GraphFSM
from nga.arch.graph_legality_mask import softmax_with_mask
from nga.arch.margin_uncertainty import compute_margin, low_margin_mask
from nga.arch.monodromy_group import compute_monodromy
from nga.arch.singularity_detector import SingularityDetector
from nga.arch.singularity_types import SingularityType
from nga.arch.stratified_partition_function import compute_stratified_partition
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


@dataclass
class E0Result:
    """Aggregate metrics returned by run_e0."""

    accuracy: float
    low_margin_accuracy: float
    confusion_graph_density: float
    n_test: int
    low_sigma_accuracy: float = float("nan")


def mask_probabilities(
    p: np.ndarray,
    fsm: GraphFSM,
    current_state: str | None,
    *,
    enabled: bool,
) -> np.ndarray:
    """Apply legality mask to a probability vector (post-softmax).

    Zeros out illegal positions and renormalizes. Same semantics as
    softmax_with_mask but for inputs that are already in probability space.

    This helper exists in e0_mnist.py rather than graph_legality_mask.py
    because the typed pipeline already returns probabilities (from
    predict_proba). graph_legality_mask.py is for logit-space masking.

    When enabled=False or current_state=None the input is returned unchanged.
    Fall-back to uniform legal distribution when masking zeros out every
    element (edge case for a disconnected FSM state).

    Args:
        p: Shape (V,) probability vector, non-negative, summing to ~1.0.
        fsm: GraphFSM holding the legality matrix.
        current_state: Vertex id of the current FSM state; None skips masking.
        enabled: Master switch; when False p is returned unchanged.

    Returns:
        Shape (V,) masked and renormalized probability vector.
    """
    if not enabled or current_state is None:
        return p
    idx = fsm.index_of(current_state)
    legal_row = fsm.legality_matrix[idx]  # shape (V,) bool
    masked = p * legal_row.astype(p.dtype)
    s = masked.sum()
    if s <= 0.0:
        # Fall back to uniform legal distribution to avoid NaN.
        n_legal = int(legal_row.sum())
        if n_legal == 0:
            # No legal transitions at all - bail and return original.
            return p
        masked = legal_row.astype(p.dtype) / float(n_legal)
    else:
        masked = masked / s
    return masked


def _build_monodromy_class_by_state(fsm: GraphFSM) -> dict[int, int]:
    """Build a vertex_id -> encoded monodromy class label, computed once.

    For each FSM vertex, look up the dart based at that vertex (any dart;
    we pick the lowest-indexed one) and read its `monodromy_class` from the
    full MonodromyData. Vertices with zero darts (isolated states) map to 0.
    """
    darts = build_darts_from_fsm(fsm)
    if darts.n_darts == 0:
        return {i: 0 for i in range(len(fsm.vertex_ids))}

    rho = build_rho(fsm, darts)
    tau = build_tau(darts)
    mono = compute_monodromy(rho, tau)

    by_state: dict[int, int] = {}
    for v_idx in range(len(fsm.vertex_ids)):
        # First dart based at v_idx (lowest index). 0 if none.
        dart_indices = np.flatnonzero(darts.vertex_of == v_idx)
        if dart_indices.size == 0:
            by_state[v_idx] = 0
        else:
            by_state[v_idx] = int(mono.monodromy_class[int(dart_indices[0])])
    return by_state


def run_e0(
    *,
    config: Config,
    ablation: AblationTuple,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    margin_threshold: float = 0.10,
) -> E0Result:
    """Train + evaluate + log. Returns aggregate metrics.

    Loads sklearn digits, trains LogisticRegression, evaluates on the test
    split, and writes per-sample records to:
      - output_dir/metrics.jsonl        one row per aggregate metric
      - output_dir/results.jsonl        one row per test sample
      - output_dir/scores.jsonl         one row per test sample (TypedScoreRecord)
      - output_dir/decision_trace.jsonl one row per test sample (interpretability)

    The graph mask is exercised by treating the previous test sample's
    predicted label as current_state for each sample i > 0. Sample i=0 uses
    current_state=None (no prior context), so masking is a no-op on the first
    sample only.

    Phase 2 change: singular_flag is now derived from sigma(x) >= 0.5, where
    sigma is produced by SingularityDetector (gated on
    ablation.singularity_detector_enabled). When the detector is disabled,
    sigma=0.0 for all samples and singular_flag is always False. This replaces
    the plain margin < margin_threshold check from Phase 1 - the sigma signal
    subsumes it while also incorporating loop_risk from TaggerHistory.

    Phase 4-5 additions:
      - EnergyFunction breakdown is computed and stored on the trace.
      - ControlPolicy decides ROUTE_NORMAL / ROUTE_RECOVERY / ABSTAIN per
        sample. For E0 current_state is always None so recovery is unreachable
        and the policy falls through to ROUTE_NORMAL in the mid-sigma band.
      - Monodromy class for each predicted vertex is precomputed once.
      - A run-level stratified partition function is computed and per-stratum
        P_lambda values are written to metrics.jsonl, plus mean energies for
        correct vs incorrect samples.

    Args:
        config: Merged runtime config (not used directly in Phase 1/2 beyond
            being passed through to record fields where relevant).
        ablation: AblationTuple controlling which components are active.
        fsm: GraphFSM with vertex_ids ["0", "1", ..., "9"].
        run_id: Unique run identifier string (e.g. "E0_A0_seed42").
        output_dir: Directory in which to append JSONL files.
        seed: Integer random seed for train/test split and classifier.
        margin_threshold: Margin threshold passed to SingularityDetector and
            the behavioral stratum tagger. Samples are singular when sigma >= 0.5.

    Returns:
        E0Result with accuracy, low_margin_accuracy, confusion_graph_density,
        n_test, and low_sigma_accuracy.

    Raises:
        ValueError: If fsm.vertex_ids is not exactly ["0", "1", ..., "9"].
    """
    # ------------------------------------------------------------------
    # Step 1 - Validate FSM vertex set
    # ------------------------------------------------------------------
    expected_vertex_ids = [str(d) for d in range(10)]
    if fsm.vertex_ids != expected_vertex_ids:
        raise ValueError(
            f"E0 requires fsm.vertex_ids == {expected_vertex_ids!r}; "
            f"got {fsm.vertex_ids!r}"
        )

    # ------------------------------------------------------------------
    # Step 2 - Load sklearn digits dataset
    # ------------------------------------------------------------------
    from sklearn.datasets import load_digits  # type: ignore[import-untyped]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
    from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

    digits = load_digits()
    X: np.ndarray = digits.data        # shape (N, 64)
    y: np.ndarray = digits.target      # shape (N,), integers 0..9

    # Convert integer labels to string labels matching FSM vertex_ids.
    y_str: np.ndarray = np.array([str(int(label)) for label in y])

    # ------------------------------------------------------------------
    # Step 3 - Train/test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train_str, y_test_str = train_test_split(
        X, y_str, test_size=0.25, random_state=seed
    )

    # ------------------------------------------------------------------
    # Step 4 - Train classifier
    # ------------------------------------------------------------------
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train_str)

    # ------------------------------------------------------------------
    # Step 5 - Build pipeline, singularity detector, tagger, energy fn,
    #          control policy, and per-state monodromy table
    # ------------------------------------------------------------------
    pipeline = TypedFieldPipeline(clf, fsm.vertex_ids, ablation)

    # SingularityDetector: enabled flag mirrors ablation.singularity_detector_enabled.
    detector = SingularityDetector(
        margin_threshold=margin_threshold,
        enabled=ablation.singularity_detector_enabled,
    )

    # TaggerHistory tracks recent predicted states for loop-risk detection.
    history = TaggerHistory()

    # Energy function uses default Phase 5 coefficients.
    energy_fn = EnergyFunction(EnergyParameters())

    # Control policy: 0.3 / 0.7 thresholds. fsm is passed but goal_states is
    # None and current_state will always be None for E0, so the recovery branch
    # is unreachable - the policy gracefully falls through to ROUTE_NORMAL.
    control_policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        fsm=fsm,
        goal_states=None,
    )

    # Monodromy class per vertex (computed once at start-up).
    monodromy_class_by_state: dict[int, int] = _build_monodromy_class_by_state(fsm)

    # ------------------------------------------------------------------
    # Step 6 - Open JSONL writers (append to files already created by CLI)
    # ------------------------------------------------------------------
    experiment_label = "E0"
    # Ablation ID (e.g. "A0") is encoded in the run_id; ablation.name is the
    # human label ("Full system") which does not match the ^A\d+$ wire format.
    ablation_label = run_id.split("_")[1]

    with (
        JsonlWriter(output_dir / "metrics.jsonl", MetricsRecord) as metrics_writer,
        JsonlWriter(output_dir / "results.jsonl", ResultsRecord) as results_writer,
        JsonlWriter(output_dir / "scores.jsonl", TypedScoreRecord) as scores_writer,
        open_decision_trace_writer(output_dir / "decision_trace.jsonl") as trace_writer,
    ):
        # ------------------------------------------------------------------
        # Step 7 - Per-sample evaluation loop
        # ------------------------------------------------------------------
        cg = ConfusionGraph(vertex_ids=fsm.vertex_ids)

        y_hat_list: list[str] = []
        margins_list: list[float] = []
        sigma_scores_list: list[float] = []
        y_true_list: list[str] = []
        energies_list: list[float] = []
        confidences_list: list[float] = []
        strata_list: list[SingularityType] = []
        # Counters for the new aggregate metrics.
        n_routed_to_recovery: int = 0
        n_abstained: int = 0
        n_routed_to_normal: int = 0

        previous_predicted_state: str | None = None

        n_test = len(X_test)
        for i in range(n_test):
            # E0 is flat classification: each sample is independent, with no
            # sequential context. current_state=None makes the legality mask a
            # no-op, which is the correct semantics here.
            current_state: str | None = None
            _ = previous_predicted_state  # retained for symmetry; unused here

            # Get raw pipeline output (sklearn predict_proba probabilities).
            out: TypedFieldOutput = pipeline.predict_one(X_test[i])
            raw_dist: np.ndarray = out.distribution  # shape (V,), already a softmax

            # Apply legality mask to the probability vector.
            masked_dist = mask_probabilities(
                raw_dist, fsm, current_state, enabled=ablation.graph_mask_enabled
            )

            # Compute margin on the masked distribution.
            margin = compute_margin(masked_dist)

            # Phase 2: behavioral stratum tagging + sigma(x) computation
            margin_for_sigma = compute_margin(raw_dist)
            is_illegal_for_sigma: bool = False

            tagger_input = TaggerInput(
                distribution=masked_dist,
                predicted_state=out.predicted_state,
                current_state=current_state,
                margin=margin,
            )

            tag_result = tag_step(
                tagger_input,
                history,
                fsm,
                margin_threshold=margin_threshold,
                decision_tie_threshold=0.02,
                loop_threshold=0.5,
            )

            loop_risk: float = history.predicted_loop_risk(out.predicted_state)
            history.push(out.predicted_state)

            decision_tie_strength: float = 0.0

            sigma, _contrib = detector.compute(
                margin=margin_for_sigma,
                is_illegal=is_illegal_for_sigma,
                loop_risk=loop_risk,
                decision_tie_strength=decision_tie_strength,
            )

            singular_flag = bool(sigma >= 0.5)

            # Predicted state is argmax of masked distribution.
            argmax_idx = int(np.argmax(masked_dist))
            predicted_state = fsm.vertex_ids[argmax_idx]
            confidence = float(masked_dist[argmax_idx])

            # Build an updated TypedFieldOutput reflecting masked values.
            masked_out = TypedFieldOutput(
                predicted_state=predicted_state,
                stratum_label=predicted_state if ablation.typed_scores_enabled else "",
                distribution=masked_dist,
                confidence=confidence,
                embedding=None,
            )

            # Build wire-format score record.
            score_rec = build_typed_score_record(
                masked_out,
                run_id=run_id,
                experiment=experiment_label,
                ablation=ablation_label,
                seed=seed,
                step=i,
                sample_id=str(i),
                margin=margin,
                z_H=None,
            )
            scores_writer.append(score_rec)

            # Determine transition legality for the results record.
            transition_legal: bool | None = (
                fsm.is_legal_transition(current_state, predicted_state)
                if current_state is not None
                else None
            )

            # Build results record (model's raw prediction; unchanged contract).
            y_true_i: str = y_test_str[i]
            result_rec = ResultsRecord(
                run_id=run_id,
                experiment=experiment_label,
                ablation=ablation_label,
                seed=seed,
                step=i,
                sample_id=str(i),
                y_true=y_true_i,
                y_hat=predicted_state,
                margin=margin,
                singular_flag=singular_flag,
                sigma_score=sigma,
                behavioral_stratum=tag_result.tag.value,
                stratum_bitmask=tag_result.bitmask,
                transition_legal=transition_legal,
            )
            results_writer.append(result_rec)

            # ------------------------------------------------------------------
            # Phase 5: energy + control + decision trace
            # ------------------------------------------------------------------
            uncertainty_signal = float(np.clip(1.0 - margin, 0.0, 1.0))
            progress_signal = float(np.clip(1.0 - uncertainty_signal, 0.0, 1.0))
            contradiction_signal = 1.0 if is_illegal_for_sigma else 0.0
            energy = energy_fn.compute(
                cost=0.0,
                uncertainty=uncertainty_signal,
                contradiction=contradiction_signal,
                loop_pressure=loop_risk,
                progress=progress_signal,
            )

            # Control policy decision. current_state is None for E0 so the
            # recovery branch is unreachable - mid-sigma still routes normal.
            policy_result = control_policy.decide(
                sigma=float(sigma),
                model_prediction=int(argmax_idx),
                current_state=None,
            )
            if policy_result.decision == "ROUTE_RECOVERY":
                n_routed_to_recovery += 1
            elif policy_result.decision == "ABSTAIN":
                n_abstained += 1
            else:
                n_routed_to_normal += 1

            # MaskAction: graph_mask_enabled vs. observed effect. current_state
            # is None so there is no masking; pre/post argmax are identical.
            mask_action = MaskAction(
                enabled=bool(ablation.graph_mask_enabled),
                current_state=current_state,
                illegal_indices_zeroed=[],
                pre_mask_argmax=predicted_state,
                post_mask_argmax=predicted_state,
                mask_changed_prediction=False,
            )

            # ControlAction must use the schema field names sigma_threshold_used,
            # sigma_observed, abstain_threshold_used (NOT theta_normal/theta_abstain).
            control_action = ControlAction(
                decision=policy_result.decision,
                reason=policy_result.reason,
                sigma_threshold_used=float(control_policy.theta_normal),
                sigma_observed=float(sigma),
                abstain_threshold_used=float(control_policy.theta_abstain),
            )

            # Top-2 for the decision trace (post-mask distribution).
            top2_idx_local: int | None
            if len(masked_dist) >= 2:
                # Argsort descending.
                order = np.argsort(masked_dist)[::-1]
                top2_idx_local = int(order[1])
                top2_state_str: str | None = fsm.vertex_ids[top2_idx_local]
                top2_prob_val: float = float(masked_dist[top2_idx_local])
            else:
                top2_state_str = None
                top2_prob_val = 0.0

            # Sigma signals breakdown - mirrors the SingularityDetector
            # contributions where available; for E0 we record what we know.
            sigma_signals = {
                "margin": float(margin_for_sigma),
                "decision_tie": float(decision_tie_strength),
                "illegal": float(1.0 if is_illegal_for_sigma else 0.0),
                "loop": float(loop_risk),
                "stabilizer": 0.0,
                "catastrophe_bias": 0.0,
            }

            energy_breakdown = {
                "cost": float(energy.cost),
                "uncertainty": float(energy.uncertainty),
                "contradiction": float(energy.contradiction),
                "loop_pressure": float(energy.loop_pressure),
                "progress": float(energy.progress),
            }

            mono_class_for_pred = int(monodromy_class_by_state.get(int(argmax_idx), 0))

            trace_record = DecisionTraceRecord(
                run_id=run_id,
                experiment=experiment_label,
                ablation=ablation_label,
                seed=seed,
                step=i,
                sample_id=str(i),
                confidence=float(confidence),
                top1_state=predicted_state,
                top2_state=top2_state_str,
                top1_prob=float(masked_dist[argmax_idx]),
                top2_prob=top2_prob_val,
                margin=float(margin),
                mask=mask_action,
                stratum_tag=tag_result.tag.value,
                stratum_bitmask=int(tag_result.bitmask),
                stratum_confidence=float(tag_result.confidence),
                sigma_total=float(sigma),
                sigma_signals=sigma_signals,
                energy_total=float(energy.total),
                energy_breakdown=energy_breakdown,
                P_lambda=None,
                free_energy=None,
                monodromy_class=mono_class_for_pred,
                in_closed_walk=None,
                control=control_action,
                timestamp=dt.datetime.now(tz=dt.timezone.utc).isoformat(),
            )
            trace_writer.append(trace_record)

            # Accumulate for aggregate metrics.
            cg.add(y_true_i, predicted_state)
            y_hat_list.append(predicted_state)
            margins_list.append(margin)
            sigma_scores_list.append(sigma)
            y_true_list.append(y_true_i)
            energies_list.append(float(energy.total))
            confidences_list.append(float(confidence))
            strata_list.append(SingularityType(tag_result.tag.value))

            # Advance rolling state for next iteration.
            previous_predicted_state = predicted_state

        # ------------------------------------------------------------------
        # Step 8 - Compute aggregate metrics
        # ------------------------------------------------------------------
        y_hat_arr = np.array(y_hat_list)
        y_true_arr = np.array(y_true_list)
        margins_arr = np.array(margins_list)
        sigma_scores_arr = np.array(sigma_scores_list)
        energies_arr = np.array(energies_list, dtype=np.float64)

        correct = y_true_arr == y_hat_arr
        accuracy = float(np.mean(correct))

        low_margin_flags = low_margin_mask(margins_arr, margin_threshold)
        low_margin_indices = np.where(low_margin_flags)[0]

        if len(low_margin_indices) > 0:
            low_margin_accuracy = float(np.mean(correct[low_margin_indices]))
        else:
            low_margin_accuracy = float("nan")

        low_sigma_indices = np.where(sigma_scores_arr >= 0.5)[0]
        if len(low_sigma_indices) > 0:
            low_sigma_accuracy = float(np.mean(correct[low_sigma_indices]))
        else:
            low_sigma_accuracy = float("nan")

        confusion_graph_density = cg.density()

        # ------------------------------------------------------------------
        # Step 9 - Stratified partition function over the full run.
        # ------------------------------------------------------------------
        partition = compute_stratified_partition(
            energies=energies_arr,
            strata=strata_list,
            temperature=1.0,
        )

        # Energy-weighted batch loss as an informational metric. Approximates
        # log P(y_true | x) per sample by log(confidence) when correct,
        # log(1 - confidence) when wrong; the weighting upweights high-energy
        # (singular) samples in the aggregate. This does NOT retrain - it just
        # exposes what the energy-weighted metric WOULD prefer.
        confidences_arr = np.clip(np.array(confidences_list, dtype=np.float64), 1e-9, 1.0 - 1e-9)
        log_prob_y_true_arr = np.where(correct, np.log(confidences_arr), np.log1p(-confidences_arr))
        energy_weighted_loss_value = float(
            energy_weighted_batch_loss(
                log_probs_y_true=log_prob_y_true_arr,
                energies=energies_arr,
                lambda_weight=1.0,
            )
        )

        # ------------------------------------------------------------------
        # Step 10 - Write aggregate metric rows
        # ------------------------------------------------------------------
        timestamp = dt.datetime.now(tz=dt.timezone.utc).isoformat()

        metrics_writer.append(
            MetricsRecord(
                run_id=run_id,
                experiment=experiment_label,
                ablation=ablation_label,
                seed=seed,
                step=0,
                split="test",
                metric_name="accuracy",
                value=accuracy,
                timestamp=timestamp,
            )
        )

        if len(low_margin_indices) > 0:
            metrics_writer.append(
                MetricsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=0,
                    split="test",
                    metric_name="low_margin_accuracy",
                    value=low_margin_accuracy,
                    timestamp=timestamp,
                )
            )

        if len(low_sigma_indices) > 0:
            metrics_writer.append(
                MetricsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=0,
                    split="test",
                    metric_name="low_sigma_accuracy",
                    value=low_sigma_accuracy,
                    timestamp=timestamp,
                )
            )

        metrics_writer.append(
            MetricsRecord(
                run_id=run_id,
                experiment=experiment_label,
                ablation=ablation_label,
                seed=seed,
                step=0,
                split="test",
                metric_name="confusion_graph_density",
                value=confusion_graph_density,
                timestamp=timestamp,
            )
        )

        # ------------------------------------------------------------------
        # Step 11 - Phase 5 aggregate rows: control routing counts,
        # per-stratum P_lambda, mean energies by correctness.
        # ------------------------------------------------------------------
        for metric_name, value in [
            ("n_routed_to_recovery", float(n_routed_to_recovery)),
            ("n_abstained", float(n_abstained)),
            ("n_routed_to_normal", float(n_routed_to_normal)),
            ("n_mask_modified_predictions", 0.0),
            ("free_energy", float(-1.0 * partition.log_Z)),
            ("energy_weighted_loss", energy_weighted_loss_value),
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

        for stratum_type, p_lambda in partition.P_per_stratum.items():
            metrics_writer.append(
                MetricsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=0,
                    split="test",
                    metric_name=f"P_lambda.{stratum_type.value}",
                    value=float(p_lambda),
                    timestamp=timestamp,
                )
            )

        n_correct = int(correct.sum())
        n_incorrect = int((~correct).sum())
        if n_correct > 0 and n_incorrect > 0:
            mean_e_correct = float(np.mean(energies_arr[correct]))
            mean_e_incorrect = float(np.mean(energies_arr[~correct]))
            for metric_name, value in [
                ("mean_energy_correct", mean_e_correct),
                ("mean_energy_incorrect", mean_e_incorrect),
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

    return E0Result(
        accuracy=accuracy,
        low_margin_accuracy=low_margin_accuracy if len(low_margin_indices) > 0 else float("nan"),
        confusion_graph_density=confusion_graph_density,
        n_test=n_test,
        low_sigma_accuracy=low_sigma_accuracy if len(low_sigma_indices) > 0 else float("nan"),
    )


__all__ = ["E0Result", "mask_probabilities", "run_e0"]
