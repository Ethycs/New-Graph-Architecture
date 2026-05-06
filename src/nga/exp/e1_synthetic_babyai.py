"""E1 - Synthetic BabyAI Grid Run.

Validates Phase 2: typed scores + graph mask + sigma(x) on a synthetic 7-state
grid task. The headline claim from Experiments.md is that the graph mask is
load-bearing when the classifier does NOT see the current_state as a feature -
mask gives roughly +10pt accuracy and eliminates illegal transitions. This
runner reproduces that comparison.

Phase 4/5 wiring (Wave 31a / B1 follow-up): the previously-unused atoms
EnergyFunction, StratifiedPartitionFunction, dart_permutations,
monodromy_group, and ControlPolicy are integrated here so each step emits a
DecisionTraceRecord (the WHY stream). Because synthetic-BabyAI samples carry
a per-sample current_state hint, ROUTE_RECOVERY can fire and produce a real
BFS-based recovery action when sigma is in the [theta_normal, theta_abstain)
band.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from nga.arch.behavioral_stratum_tagger import TaggerHistory, TaggerInput, tag_step
from nga.arch.confusion_graph import ConfusionGraph
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
from nga.exp.dataset_synthetic_babyai_grid import (
    generate_dataset,
    to_features_and_labels,
    train_test_split_by_trajectory,
)
from nga.exp.e0_mnist import mask_probabilities

__all__ = ["E1Result", "run_e1"]

# The canonical 7-vertex set for the synthetic BabyAI task graph.
_EXPECTED_VERTEX_IDS: list[str] = [
    "Parse",
    "Navigate",
    "ResolveDoor",
    "Pickup",
    "Deliver",
    "Interact",
    "Done",
]


@dataclass
class E1Result:
    """Aggregate metrics returned by run_e1."""

    accuracy: float
    accuracy_no_mask: float
    illegal_transition_rate: float
    illegal_transition_rate_no_mask: float
    mask_accuracy_uplift: float          # accuracy - accuracy_no_mask
    confusion_graph_density: float
    n_test: int


def _build_monodromy_class_by_state(fsm: GraphFSM) -> dict[int, int]:
    """Return a dict mapping state index -> a representative monodromy class.

    Monodromy classes live on darts (each vertex has multiple darts based at
    it), so we collapse the per-dart label down to per-vertex by picking the
    first dart at that vertex (lowest dart index). This is sufficient for a
    coarse signature and keeps the per-step trace cheap.

    States with no darts (isolated vertices) map to 0.
    """
    darts = build_darts_from_fsm(fsm)
    rho = build_rho(fsm, darts)
    tau = build_tau(darts)
    mono = compute_monodromy(rho, tau)

    out: dict[int, int] = {}
    for v_idx in range(fsm.vertex_count):
        # First dart whose base vertex is v_idx.
        dart_indices = np.where(darts.vertex_of == v_idx)[0]
        if dart_indices.size > 0:
            out[v_idx] = int(mono.monodromy_class[int(dart_indices[0])])
        else:
            out[v_idx] = 0
    return out


def run_e1(
    *,
    config: Config,
    ablation: AblationTuple,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    margin_threshold: float = 0.10,
    n_trajectories: int = 80,
) -> E1Result:
    """Train + evaluate the synthetic-BabyAI classifier with and without the mask.

    Outputs (already created empty by the CLI; this function appends):
      - output_dir/metrics.jsonl   per-metric rows; includes accuracy, accuracy_no_mask,
                                   illegal_transition_rate, illegal_transition_rate_no_mask,
                                   mask_accuracy_uplift, confusion_graph_density, plus
                                   the new Phase-4/5 metrics: n_routed_to_recovery,
                                   n_abstained, n_mask_modified_predictions,
                                   n_monodromy_cycle_revisits, mean_energy_correct,
                                   mean_energy_incorrect.
      - output_dir/results.jsonl   per-sample rows; populated with margin, sigma_score,
                                   singular_flag, transition_legal, behavioral_stratum.
      - output_dir/scores.jsonl    per-sample TypedScoreRecord.
      - output_dir/decision_trace.jsonl  per-sample DecisionTraceRecord (WHY stream).

    The "no mask" comparison is the load-bearing run for Phase 2's mask claim.
    Acceptance: with ablation.graph_mask_enabled=True, illegal_transition_rate
    must be 0.0 (the mask zeros illegal positions), and mask_accuracy_uplift
    should be > 0 on this dataset (the dataset includes illegal_temptation samples
    designed to exercise the mask).

    Args:
        config: Merged runtime config; embedding_dim is used for feature_dim.
        ablation: AblationTuple controlling which components are active.
        fsm: GraphFSM with vertex_ids matching the 7-vertex BabyAI set.
        run_id: Unique run identifier string (e.g. "E1_A0_seed42").
        output_dir: Directory in which to append JSONL files.
        seed: Integer random seed for dataset generation, split, and classifier.
        margin_threshold: Samples with margin < this are flagged as singular.
        n_trajectories: Number of synthetic trajectories to generate.

    Returns:
        E1Result with aggregate metrics.

    Raises:
        ValueError: If fsm.vertex_ids does not match the expected 7-vertex set.
    """
    # ------------------------------------------------------------------
    # Step 1 - Validate FSM vertex set
    # ------------------------------------------------------------------
    if fsm.vertex_ids != _EXPECTED_VERTEX_IDS:
        raise ValueError(
            f"E1 requires fsm.vertex_ids == {_EXPECTED_VERTEX_IDS!r}; "
            f"got {fsm.vertex_ids!r}"
        )

    # ------------------------------------------------------------------
    # Step 2 - Build the synthetic dataset
    # ------------------------------------------------------------------
    ds = generate_dataset(
        fsm,
        seed=seed,
        n_trajectories=n_trajectories,
        feature_dim=config.embedding_dim,
    )

    # ------------------------------------------------------------------
    # Step 3 - Split by trajectory (no temporal leakage)
    # ------------------------------------------------------------------
    train_ds, test_ds = train_test_split_by_trajectory(ds, seed=seed)

    # ------------------------------------------------------------------
    # Step 4 - Convert splits to arrays
    # Note: the classifier does NOT receive prev_state as a feature.
    # This is the load-bearing condition: the graph mask must compensate
    # for the classifier's lack of state context.
    # ------------------------------------------------------------------
    X_train, y_train_idx, _train_prev, _train_ids = to_features_and_labels(train_ds)
    X_test, y_test_idx, test_prev_states, test_sample_ids = to_features_and_labels(test_ds)

    # Convert integer label indices to string vertex ids.
    y_train_str: list[str] = [fsm.vertex_ids[int(i)] for i in y_train_idx]
    y_test_str: list[str] = [fsm.vertex_ids[int(i)] for i in y_test_idx]

    # ------------------------------------------------------------------
    # Step 5 - Train classifier (no state_id feature; classifier sees
    # only the ambient feature vector)
    # ------------------------------------------------------------------
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train_str)

    # ------------------------------------------------------------------
    # Step 6 - Build pipeline and supporting objects
    # ------------------------------------------------------------------
    pipeline = TypedFieldPipeline(clf, fsm.vertex_ids, ablation)
    detector = SingularityDetector(
        margin_threshold=margin_threshold,
        enabled=ablation.singularity_detector_enabled,
    )
    history = TaggerHistory()

    # --- Phase 4/5 atoms: monodromy lookup, energy fn, control policy ---
    monodromy_class_by_state: dict[int, int] = _build_monodromy_class_by_state(fsm)
    energy_fn = EnergyFunction(EnergyParameters())

    # Goal placeholder: the last vertex (id=Done) is the canonical terminal.
    # ControlPolicy BFS will steer toward it from any current_state in the
    # recovery band. This makes ROUTE_RECOVERY actionable on this dataset.
    goal_states: list[int] = [fsm.vertex_count - 1]
    control_policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        fsm=fsm,
        goal_states=goal_states,
    )

    # ------------------------------------------------------------------
    # Step 7 - Open JSONL writers
    # ------------------------------------------------------------------
    experiment_label = "E1"
    ablation_label = run_id.split("_")[1]

    n_test = len(X_test)

    # Accumulation lists for aggregate metric computation.
    y_true_list: list[str] = []
    predicted_list: list[str] = []          # with mask (or not, depending on ablation)
    predicted_no_mask_list: list[str] = []  # always without mask
    transition_legal_list: list[bool | None] = []
    transition_legal_no_mask_list: list[bool | None] = []

    # Phase 4/5 per-step accumulators.
    energies: list[float] = []
    strata: list[SingularityType] = []
    n_routed_to_recovery: int = 0
    n_abstained: int = 0
    n_mask_modified_predictions: int = 0
    n_monodromy_cycle_revisits: int = 0
    energies_correct: list[float] = []
    energies_incorrect: list[float] = []
    visited_states: set[int] = set()

    cg = ConfusionGraph(vertex_ids=fsm.vertex_ids)

    with (
        JsonlWriter(output_dir / "metrics.jsonl", MetricsRecord) as metrics_writer,
        JsonlWriter(output_dir / "results.jsonl", ResultsRecord) as results_writer,
        JsonlWriter(output_dir / "scores.jsonl", TypedScoreRecord) as scores_writer,
        open_decision_trace_writer(output_dir / "decision_trace.jsonl") as trace_writer,
    ):
        # ------------------------------------------------------------------
        # Step 8 - Per-sample evaluation loop
        # ------------------------------------------------------------------
        for i in range(n_test):
            # The prev_state from the test trajectory provides ground-truth
            # state context for the mask. The classifier itself does NOT see it.
            current_state: str | None = test_prev_states[i]
            y_true_i: str = y_test_str[i]

            # Get raw pipeline output (sklearn predict_proba probabilities).
            out = pipeline.predict_one(X_test[i])
            raw_dist: np.ndarray = out.distribution  # shape (V,)

            # --- Pre-mask argmax (always computed for both legality + trace) ---
            argmax_unmasked = int(np.argmax(raw_dist))
            predicted_state_no_mask = fsm.vertex_ids[argmax_unmasked]

            # --- Masked prediction (controlled by ablation) ---
            masked_dist = mask_probabilities(
                raw_dist, fsm, current_state, enabled=ablation.graph_mask_enabled
            )
            argmax_masked = int(np.argmax(masked_dist))
            predicted_state = fsm.vertex_ids[argmax_masked]
            confidence = float(masked_dist[argmax_masked])

            # --- Margin from masked distribution (used for ResultsRecord) ---
            margin = compute_margin(masked_dist)

            # --- Top-2 (post-mask) for the decision trace ---
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

            # --- Legality of the masked prediction ---
            transition_legal: bool | None = (
                fsm.is_legal_transition(current_state, predicted_state)
                if current_state is not None
                else None
            )

            # --- Legality of the unmasked prediction ---
            transition_legal_no_mask: bool | None = (
                fsm.is_legal_transition(current_state, predicted_state_no_mask)
                if current_state is not None
                else None
            )

            # --- Behavioral stratum tagger ---
            tagger_inp = TaggerInput(
                distribution=masked_dist,
                predicted_state=predicted_state,
                current_state=current_state,
                margin=margin,
            )
            tag_result = tag_step(tagger_inp, history, fsm, margin_threshold=margin_threshold)
            history.push(predicted_state)

            # --- Singularity detector sigma(x) ---
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

            # --- Build masked TypedFieldOutput for the score record ---
            masked_out = TypedFieldOutput(
                predicted_state=predicted_state,
                stratum_label=predicted_state if ablation.typed_scores_enabled else "",
                distribution=masked_dist,
                confidence=confidence,
                embedding=None,
            )

            # --- Wire-format score record ---
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

            # --- Results record ---
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

            # ------------------------------------------------------------------
            # Phase 4/5: energy + control policy + decision trace
            # ------------------------------------------------------------------
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

            # Track cycle revisits (literal post-mask argmax). Window-less for
            # the first integration; future improvement keyed by monodromy
            # class equivalence rather than literal state identity.
            y_hat_int = argmax_masked
            if y_hat_int in visited_states:
                n_monodromy_cycle_revisits += 1
            visited_states.add(y_hat_int)

            # Mask side-effects: list illegal indices zeroed, and whether the
            # mask flipped the argmax.
            if current_state is not None and ablation.graph_mask_enabled:
                src_idx = fsm.index_of(current_state)
                row = fsm.legality_matrix[src_idx]  # shape (V,)
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

            # Control policy decision (uses int state indices).
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

            # Sigma signal breakdown (matches documented keys).
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

            # Monodromy class for the masked argmax state.
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
                P_lambda=None,           # filled post-loop is run-level; left None per-row
                free_energy=None,
                monodromy_class=int(mono_class),
                in_closed_walk=None,
                control=control_action,
                timestamp=dt.datetime.now(tz=dt.timezone.utc).isoformat(),
            )
            trace_writer.append(trace_rec)

            # --- Accumulators for aggregate metrics ---
            cg.add(y_true_i, predicted_state)
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

        # ------------------------------------------------------------------
        # Step 9 - Compute aggregate metrics
        # ------------------------------------------------------------------
        y_true_arr = np.array(y_true_list)
        predicted_arr = np.array(predicted_list)
        predicted_no_mask_arr = np.array(predicted_no_mask_list)

        accuracy = float(np.mean(y_true_arr == predicted_arr))
        accuracy_no_mask = float(np.mean(y_true_arr == predicted_no_mask_arr))
        mask_accuracy_uplift = accuracy - accuracy_no_mask

        # Illegal transition rate: count over samples that have a known
        # current_state (i.e. not the first step of a trajectory).
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

        confusion_graph_density = cg.density()

        # --- Stratified partition (run-level summary) ---
        # Computed but not propagated back into per-row P_lambda for now to
        # keep the trace writer single-pass. The free energy is logged as a
        # metric.
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

        # ------------------------------------------------------------------
        # Step 10 - Write aggregate metric rows
        # ------------------------------------------------------------------
        timestamp = dt.datetime.now(tz=dt.timezone.utc).isoformat()

        for metric_name, value in [
            ("accuracy", accuracy),
            ("accuracy_no_mask", accuracy_no_mask),
            ("illegal_transition_rate", illegal_transition_rate),
            ("illegal_transition_rate_no_mask", illegal_transition_rate_no_mask),
            ("mask_accuracy_uplift", mask_accuracy_uplift),
            ("confusion_graph_density", confusion_graph_density),
            ("n_routed_to_recovery", float(n_routed_to_recovery)),
            ("n_abstained", float(n_abstained)),
            ("n_mask_modified_predictions", float(n_mask_modified_predictions)),
            ("n_monodromy_cycle_revisits", float(n_monodromy_cycle_revisits)),
            ("mean_energy_correct", mean_energy_correct),
            ("mean_energy_incorrect", mean_energy_incorrect),
            ("free_energy", run_free_energy),
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

    return E1Result(
        accuracy=accuracy,
        accuracy_no_mask=accuracy_no_mask,
        illegal_transition_rate=illegal_transition_rate,
        illegal_transition_rate_no_mask=illegal_transition_rate_no_mask,
        mask_accuracy_uplift=mask_accuracy_uplift,
        confusion_graph_density=confusion_graph_density,
        n_test=n_test,
    )
