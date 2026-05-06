"""E2 - Real BabyAI / MiniGrid.

Validates Phase 6: typed scoring + graph mask + sigma(x) on real MiniGrid
trajectories from DoorKey-5x5. The load-bearing question: do sigma > margin
and hyperbolic compression hold on real hierarchical task data, where they
failed on synthetic?

Returns metrics analogous to E1 (mask uplift, illegal rate, accuracy) plus
the per-record schema needed for E4 to compute sigma vs margin AUROC over
E2 outputs.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from nga.arch.behavioral_stratum_tagger import TaggerHistory, TaggerInput, tag_step
from nga.arch.confusion_graph import ConfusionGraph
from nga.arch.graph_fsm import GraphFSM
from nga.arch.margin_uncertainty import compute_margin
from nga.arch.singularity_detector import SingularityDetector
from nga.arch.typed_field_pipeline import TypedFieldOutput, TypedFieldPipeline
from nga.arch.typed_score_record_builder import build_typed_score_record
from nga.drivers.ablation_flags import AblationTuple
from nga.drivers.config import Config
from nga.drivers.jsonl_writer import JsonlWriter
from nga.drivers.metrics_jsonl import MetricsRecord
from nga.drivers.results_jsonl import ResultsRecord
from nga.drivers.typed_score_record import TypedScoreRecord
from nga.exp.dataset_minigrid_wrapper import (
    collect_dataset,
    to_features_and_labels,
    train_test_split_by_episode,
)
from nga.exp.e0_mnist import mask_probabilities

__all__ = ["E2Result", "run_e2"]

# The canonical 7-vertex set for the BabyAI task graph (shared with E1).
_EXPECTED_VERTEX_IDS: list[str] = [
    "Parse",
    "Navigate",
    "ResolveDoor",
    "Pickup",
    "Deliver",
    "Interact",
    "Done",
]


class _AlignedClassifier:
    """Thin wrapper that reorders predict_proba columns to match vertex_ids.

    sklearn classifiers return predict_proba columns in clf.classes_ order
    (alphabetical). TypedFieldPipeline assumes column i maps to vertex_ids[i].
    This wrapper reorders the probability vector so the two orderings agree.

    Parameters
    ----------
    clf:
        Fitted sklearn classifier with .predict_proba and .classes_ attributes.
    vertex_ids:
        Desired column order for the output probability array.
    """

    def __init__(self, clf: object, vertex_ids: list[str]) -> None:
        self._clf = clf
        classes: list[str] = list(clf.classes_)  # type: ignore[attr-defined]
        # Permutation: for each target position i (vertex_ids order),
        # find where that class sits in clf.classes_.
        self._perm: list[int] = [classes.index(v) for v in vertex_ids]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities reordered to vertex_ids column order."""
        raw: np.ndarray = self._clf.predict_proba(X)  # type: ignore[attr-defined]
        return raw[:, self._perm]


_IGNORE_VERTEX_IDS = _EXPECTED_VERTEX_IDS  # silence redefinition below
    "Parse",
    "Navigate",
    "ResolveDoor",
    "Pickup",
    "Deliver",
    "Interact",
    "Done",
]


@dataclass
class E2Result:
    """Aggregate metrics returned by run_e2.

    Fields are identical to E1Result so that downstream tooling (E4,
    evidence tracker) can consume E1 and E2 outputs uniformly.
    """

    accuracy: float
    accuracy_no_mask: float
    illegal_transition_rate: float
    illegal_transition_rate_no_mask: float
    mask_accuracy_uplift: float              # accuracy - accuracy_no_mask
    confusion_graph_density: float
    n_test: int


def run_e2(
    *,
    config: Config,
    ablation: AblationTuple,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    margin_threshold: float = 0.10,
    n_episodes: int = 60,
    env_id: str = "MiniGrid-DoorKey-5x5-v0",
) -> E2Result:
    """Train + evaluate on real MiniGrid trajectories with and without the mask.

    The structure is identical to run_e1 (e1_synthetic_babyai.py) - the only
    substantive difference is that data comes from real MiniGrid episodes
    collected by a heuristic BFS policy, and the train/test split is by
    episode (not trajectory) to prevent temporal leakage.

    Output paths (appended, not overwritten - files are already touched by CLI):
      - output_dir/metrics.jsonl   per-metric rows.
      - output_dir/results.jsonl   per-sample rows with margin, sigma_score,
                                   singular_flag, transition_legal, behavioral_stratum.
      - output_dir/scores.jsonl    per-sample TypedScoreRecord.

    The load-bearing acceptance criteria are:
      - accuracy >= 0.50 (real data is harder than synthetic)
      - mask_accuracy_uplift >= 0.0 (mask should help even slightly)

    Parameters
    ----------
    config:
        Merged runtime config. embedding_dim is unused here since the real
        feature dimension is fixed at 148 by the observation encoding.
    ablation:
        AblationTuple controlling which components are active.
    fsm:
        GraphFSM with vertex_ids matching the 7-vertex BabyAI set.
    run_id:
        Unique run identifier string, e.g. "E2_A0_seed42".
    output_dir:
        Directory in which to append JSONL files.
    seed:
        Integer random seed for dataset collection, split, and classifier.
    margin_threshold:
        Samples with margin below this value are flagged as singular.
    n_episodes:
        Number of MiniGrid episodes to collect.
    env_id:
        Gymnasium environment id for MiniGrid.

    Returns
    -------
    E2Result with aggregate metrics.

    Raises
    ------
    ValueError
        If fsm.vertex_ids does not match the expected 7-vertex BabyAI set,
        or if the collected dataset is missing any FSM state.
    """
    # ------------------------------------------------------------------
    # Step 1 - Validate FSM vertex set
    # ------------------------------------------------------------------
    if fsm.vertex_ids != _EXPECTED_VERTEX_IDS:
        raise ValueError(
            f"E2 requires fsm.vertex_ids == {_EXPECTED_VERTEX_IDS!r}; "
            f"got {fsm.vertex_ids!r}"
        )

    # ------------------------------------------------------------------
    # Step 2 - Collect real MiniGrid trajectories
    # ------------------------------------------------------------------
    ds = collect_dataset(
        fsm,
        seed=seed,
        n_episodes=n_episodes,
        env_id=env_id,
    )

    # ------------------------------------------------------------------
    # Step 3 - Split by episode (no temporal leakage)
    # ------------------------------------------------------------------
    train_ds, test_ds = train_test_split_by_episode(ds, seed=seed)

    # ------------------------------------------------------------------
    # Step 4 - Convert splits to arrays.
    # The classifier does NOT receive prev_state as a feature; the graph
    # mask compensates at inference time.
    # ------------------------------------------------------------------
    X_train, y_train_idx, _train_prev, _train_ids = to_features_and_labels(train_ds)
    X_test, y_test_idx, test_prev_states, test_sample_ids = to_features_and_labels(test_ds)

    # Convert integer label indices to string vertex ids.
    y_train_str: list[str] = [fsm.vertex_ids[int(i)] for i in y_train_idx]
    y_test_str: list[str] = [fsm.vertex_ids[int(i)] for i in y_test_idx]

    # ------------------------------------------------------------------
    # Step 5 - Train classifier on real 148-dim features
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

    # ------------------------------------------------------------------
    # Step 7 - Open JSONL writers
    # ------------------------------------------------------------------
    experiment_label = "E2"
    ablation_label = run_id.split("_")[1]

    n_test = len(X_test)

    # Accumulation lists for aggregate metric computation.
    y_true_list: list[str] = []
    predicted_list: list[str] = []
    predicted_no_mask_list: list[str] = []
    transition_legal_list: list[bool | None] = []
    transition_legal_no_mask_list: list[bool | None] = []

    cg = ConfusionGraph(vertex_ids=fsm.vertex_ids)

    with (
        JsonlWriter(output_dir / "metrics.jsonl", MetricsRecord) as metrics_writer,
        JsonlWriter(output_dir / "results.jsonl", ResultsRecord) as results_writer,
        JsonlWriter(output_dir / "scores.jsonl", TypedScoreRecord) as scores_writer,
    ):
        # ------------------------------------------------------------------
        # Step 8 - Per-sample evaluation loop (identical structure to E1)
        # ------------------------------------------------------------------
        for i in range(n_test):
            current_state: str | None = test_prev_states[i]
            y_true_i: str = y_test_str[i]

            # Get raw pipeline output (predict_proba probabilities).
            out = pipeline.predict_one(X_test[i])
            raw_dist: np.ndarray = out.distribution  # shape (V,)

            # --- Masked prediction (controlled by ablation) ---
            masked_dist = mask_probabilities(
                raw_dist, fsm, current_state, enabled=ablation.graph_mask_enabled
            )
            argmax_masked = int(np.argmax(masked_dist))
            predicted_state = fsm.vertex_ids[argmax_masked]
            confidence = float(masked_dist[argmax_masked])

            # --- Unmasked prediction (always computed for comparison) ---
            argmax_unmasked = int(np.argmax(raw_dist))
            predicted_state_no_mask = fsm.vertex_ids[argmax_unmasked]

            # --- Margin from masked distribution ---
            margin = compute_margin(masked_dist)

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
            sigma, _contribs = detector.compute(
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

            # --- Accumulate for aggregate metrics ---
            cg.add(y_true_i, predicted_state)
            y_true_list.append(y_true_i)
            predicted_list.append(predicted_state)
            predicted_no_mask_list.append(predicted_state_no_mask)
            transition_legal_list.append(transition_legal)
            transition_legal_no_mask_list.append(transition_legal_no_mask)

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
        # current_state (i.e. not the first step of an episode).
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

    return E2Result(
        accuracy=accuracy,
        accuracy_no_mask=accuracy_no_mask,
        illegal_transition_rate=illegal_transition_rate,
        illegal_transition_rate_no_mask=illegal_transition_rate_no_mask,
        mask_accuracy_uplift=mask_accuracy_uplift,
        confusion_graph_density=confusion_graph_density,
        n_test=n_test,
    )
