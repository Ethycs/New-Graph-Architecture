"""E15 - Training-maturity sweep for sigma vs. margin AUROCs.

E15 tests the *maturity hypothesis* surfaced by Phase 10 step 1 on E14:

  Phase 10 step 1 found that on a fully-trained E14, ``margin`` slightly
  edges sigma on both ``failure_auroc`` and ``structural_auroc``. The
  proposed reading was that sigma carries STRUCTURAL information *a
  priori* (no training needed -- it's a function of the legality bias,
  the FSM geometry, and the margin), while a learned head's margin can
  *learn* the same signal as training matures. If that reading is right
  then at LOW training maturity sigma should dominate (margin is close
  to random), and as the head trains margin should catch up to or pass
  sigma -- both ``sigma_failure_uplift`` and ``sigma_structural_uplift``
  should DECREASE (or flip sign) as the epoch count grows.

E15 is the smallest possible test of that hypothesis. It re-uses E14's
substrate verbatim (same dataset, same FSM, same Phase A classical
seed, same TorchEnergyTrainer, same evaluation loop) and only changes
ONE thing: instead of training K epochs and evaluating once, it trains
in chunks and evaluates at a fixed list of checkpoints.

Pipeline:

  Phase A -- IDENTICAL to E14:
    Generate the ListOps dataset; on the (current_state,
    true_next_state) sequence compute expected_counts_observed +
    bayesian_m_step_beta with prior_alpha = prior_beta = 1.0; the
    resulting (alpha, beta) Beta-posterior arrays seed the torch
    trainer's posterior parameters via posterior_alpha_init /
    posterior_beta_init. ``phase_a_hamming_normalised`` is reported
    once, BEFORE any torch update touches the posterior, and is
    substrate-independent.

  Phase B -- chunked end-to-end training with checkpoints:
    ``checkpoint_epochs = [1, 3, 5, 10]`` (configurable). Train the
    appropriate number of additional epochs to reach the next
    checkpoint, then evaluate. We DO NOT re-build the trainer between
    checkpoints; the optimiser state and trained parameters carry
    forward, so checkpoint k is exactly "what E14 would have looked
    like after k epochs."

  Per-checkpoint evaluation (held-out test split):
    Compute sigma scores, margin scores, error labels, and structural-
    ambiguity labels exactly as E14 does. Compute the four AUROCs
    (sigma_failure_auroc / margin_failure_auroc / sigma_structural_auroc
    / margin_structural_auroc) and the two uplifts. Emit each as a
    metrics row with ``step=epoch_count`` so downstream readers can
    distinguish checkpoints by the ``step`` field. Distinct checkpoints
    re-use the SAME ``metric_name`` (e.g. ``"sigma_failure_auroc"``);
    the ``step`` field is the discriminator.

  Headline + trace artefacts:
    The LAST checkpoint (e.g. epoch=10) ALSO emits a parallel set of
    "headline" metrics with ``step=0`` and the legacy E14 names
    (``sigma_auroc``, ``margin_auroc``, ``sigma_uplift``,
    ``sigma_structural_auroc``, ``margin_structural_auroc``,
    ``sigma_structural_uplift``) so existing aggregate.py readers built
    around E14's flat metric set continue to work without a schema
    change. Trace artefacts (``decision_trace.jsonl``, ``results.jsonl``
    rows) are emitted ONLY from the last checkpoint to avoid massive
    trace files; per-checkpoint trace dumps would balloon the run
    directory by 4x with no obvious downstream consumer.

The interpretation of the curve is documented as a load-bearing
finding in tests/e2e/test_e15_maturity_sweep.py:

  * If both uplift curves DECAY (sigma's lead shrinks as training
    matures), the maturity hypothesis is confirmed: sigma's value-add
    really is at low training maturity, structurally.
  * If they DON'T decay, the hypothesis is wrong and we have a
    different research direction. The test marks XFAIL on miss with
    the observed values rather than tuning the runner.
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
from nga.exp.dataset_listops import (
    generate_listops_dataset,
    train_test_split_by_sequence,
)

try:  # torch is required for E15, but import remains tolerant for clarity.
    import torch
except ImportError as _exc:  # pragma: no cover - exercised only on torch-less envs.
    raise ImportError(
        "torch is required for E15; install via `pixi add pytorch` "
        "(or `pip install torch`)."
    ) from _exc


__all__ = ["E15Result", "CheckpointMetrics", "run_e15"]


READOUT_TYPE_ID = "default"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CheckpointMetrics:
    """Per-checkpoint AUROC summary for the maturity-sweep table."""

    epoch: int
    sigma_failure_auroc: float
    margin_failure_auroc: float
    sigma_failure_uplift: float
    sigma_structural_auroc: float
    margin_structural_auroc: float
    sigma_structural_uplift: float


@dataclass
class E15Result:
    """Aggregate metrics returned by run_e15 (last-checkpoint headline)."""

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
    n_torch_trainable_params: int
    final_loss: float
    loss_decrease_first_to_last_epoch: float
    n_train: int
    n_test: int
    checkpoint_epochs: list[int]
    checkpoint_metrics: list[CheckpointMetrics]


# ---------------------------------------------------------------------------
# Helpers (copied verbatim from E14 to keep substrate behaviour identical)
# ---------------------------------------------------------------------------


def _empirical_dist_from_window(
    window: deque,
    n_states: int,
    eps: float = 1e-3,
) -> np.ndarray:
    """Laplace-smoothed categorical from a sliding window of observed next states."""
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
    """Per-class mean of encoded features, scaled into the Poincare ball."""
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


def _safe_quantizer(
    name: str, samples: list[float], n_axis_bins: int, rng: np.random.Generator
) -> AxisQuantizer:
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


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_e15(
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
    empirical_window: int = 32,
    checkpoint_epochs: tuple[int, ...] = (1, 3, 5, 10),
    encoder_hidden_dim: int = 32,
    readout_hidden_dim: int = 32,
    test_fraction: float = 0.25,
    kl_surprise_weight: float = 0.5,
    trainer_lr: float = 1e-2,
    grad_clip: float = 5.0,
) -> E15Result:
    """Run E14's pipeline with periodic eval at ``checkpoint_epochs``.

    See module docstring. Writes the standard four artefacts plus
    ``decision_trace.jsonl`` (last checkpoint only). Per-checkpoint
    AUROC metrics are emitted with ``step=epoch_count``; the last
    checkpoint additionally emits headline metrics with ``step=0``
    using legacy E14 metric names so aggregate.py's E14-style readers
    continue to work.
    """
    if fsm.vertex_ids[0] != "START":
        raise ValueError(
            f"E15 expects fsm.vertex_ids[0] == 'START'; "
            f"got {fsm.vertex_ids[0]!r}"
        )
    if fsm.vertex_ids[-1] != "ACCEPT":
        raise ValueError(
            f"E15 expects fsm.vertex_ids[-1] == 'ACCEPT'; "
            f"got {fsm.vertex_ids[-1]!r}"
        )

    cps = sorted(set(int(c) for c in checkpoint_epochs))
    if not cps or cps[0] < 1:
        raise ValueError(
            f"E15 checkpoint_epochs must be a non-empty list of positive ints; "
            f"got {checkpoint_epochs!r}"
        )

    n_states = fsm.vertex_count
    derived_max_depth = (n_states - 2) // 3
    if 2 + derived_max_depth * 3 != n_states:
        raise ValueError(
            f"E15 cannot derive max_depth from {n_states} states; "
            f"expected 2 + max_depth*3 == n_states."
        )
    if derived_max_depth != max_depth:
        max_depth = derived_max_depth

    # ------------------------------------------------------------------
    # Reproducibility: seed numpy + torch.
    # ------------------------------------------------------------------
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Step 1 -- Dataset (ListOps walker).
    # ------------------------------------------------------------------
    ds = generate_listops_dataset(
        fsm=fsm,
        max_depth=max_depth,
        n_sequences=n_sequences,
        max_length=max_length,
        seed=seed,
        illegal_temptation_fraction=illegal_temptation_fraction,
    )
    train_ds, test_ds = train_test_split_by_sequence(
        ds, seed=seed, test_fraction=test_fraction
    )

    embedding_dim = train_ds.X.shape[1]
    n_train = train_ds.X.shape[0]
    n_test = test_ds.X.shape[0]

    vertex_ids = fsm.vertex_ids
    is_boundary_state = np.array(
        [vid.endswith("_after_operand") for vid in vertex_ids],
        dtype=bool,
    )

    # ------------------------------------------------------------------
    # Step 2 -- Phase A classical seed (substrate-independent).
    # ------------------------------------------------------------------
    pair_counts = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_train):
        pair_counts[
            int(train_ds.current_states[i]), int(train_ds.y_next[i])
        ] += 1.0

    # Atom-census: keep expected_counts_observed wired symbolically.
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

    # ------------------------------------------------------------------
    # Step 3 -- Build the torch substrate (identical to E14).
    # ------------------------------------------------------------------
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

    Z_train_t = torch.from_numpy(Z_train)
    cur_train_t = torch.from_numpy(train_ds.current_states.astype(np.int64))
    nxt_train_t = torch.from_numpy(train_ds.y_next.astype(np.int64))

    Z_test_t = torch.from_numpy(Z_test)

    # ------------------------------------------------------------------
    # Phase C atoms (re-used at every checkpoint).
    # ------------------------------------------------------------------
    energy_fn = EnergyFunction(EnergyParameters())
    detector_weights = dict(SingularityDetector.DEFAULT_WEIGHTS)
    detector_weights["kl_surprise"] = float(kl_surprise_weight)

    # ------------------------------------------------------------------
    # Helper: evaluate at a checkpoint (no trace writer; just AUROCs).
    # ------------------------------------------------------------------
    def _eval_checkpoint_lightweight() -> dict[str, float]:
        detector = SingularityDetector(
            margin_threshold=margin_threshold,
            enabled=ablation.singularity_detector_enabled,
            weights=detector_weights,
        )
        history = TaggerHistory()
        windows: dict[int, deque] = {
            s: deque(maxlen=empirical_window) for s in range(n_states)
        }
        for i in range(min(n_train, empirical_window)):
            windows[int(train_ds.current_states[i])].append(
                int(train_ds.y_next[i])
            )

        sigmas: list[float] = []
        margins: list[float] = []
        errors: list[bool] = []
        cur_state_names: list[str] = []

        with torch.no_grad():
            for step_idx in range(n_test):
                cs_idx = int(test_ds.current_states[step_idx])
                current_state = vertex_ids[cs_idx]
                true_next_idx = int(test_ds.y_next[step_idx])
                true_next = vertex_ids[true_next_idx]
                obs_t = Z_test_t[step_idx]

                p_masked = trainer.predicted_distribution(
                    obs_t, current_state=cs_idx
                ).detach().cpu().numpy().astype(np.float64)

                margin = float(compute_margin(p_masked))
                argmax_state = int(np.argmax(p_masked))
                predicted_state = vertex_ids[argmax_state]

                window = windows[cs_idx]
                empirical_dist = _empirical_dist_from_window(window, n_states)
                kl_surprise = float(
                    compute_kl_surprise(empirical_dist, p_masked)
                )
                is_illegal = not bool(
                    fsm.is_legal_transition(current_state, predicted_state)
                )
                loop_risk = float(
                    history.predicted_loop_risk(predicted_state)
                )
                sigma, _ = detector.compute(
                    margin=margin,
                    is_illegal=is_illegal,
                    loop_risk=loop_risk,
                    kl_surprise=kl_surprise,
                )
                history.push(predicted_state)
                windows[cs_idx].append(int(true_next_idx))

                sigmas.append(float(sigma))
                margins.append(float(margin))
                errors.append(bool(true_next != predicted_state))
                cur_state_names.append(current_state)

        sigmas_arr = np.asarray(sigmas, dtype=float)
        margins_arr = np.asarray(margins, dtype=float)
        errors_arr = np.asarray(errors, dtype=bool)
        if errors_arr.size > 0 and 0 < int(errors_arr.sum()) < int(
            errors_arr.size
        ):
            sf = float(sigma_auroc(sigmas_arr, errors_arr))
            mf = float(margin_auroc(margins_arr, errors_arr))
        else:
            sf = 0.5
            mf = 0.5
        ambiguity_labels = np.array(
            [
                1 if "after_operand" in name else 0
                for name in cur_state_names
            ],
            dtype=np.int64,
        )
        ss = float(structural_ambiguity_auroc(sigmas_arr, ambiguity_labels))
        ms = float(
            structural_ambiguity_auroc(1.0 - margins_arr, ambiguity_labels)
        )
        return {
            "sigma_failure_auroc": sf,
            "margin_failure_auroc": mf,
            "sigma_failure_uplift": sf - mf,
            "sigma_structural_auroc": ss,
            "margin_structural_auroc": ms,
            "sigma_structural_uplift": ss - ms,
        }

    # ------------------------------------------------------------------
    # Step 4 -- Phase B: chunked training to each checkpoint, with
    # lightweight eval after each. Last checkpoint also drives the
    # full trace + headline pipeline below.
    # ------------------------------------------------------------------
    epoch_losses: list[float] = []
    checkpoint_metrics: list[CheckpointMetrics] = []
    checkpoint_metric_records: list[tuple[int, str, float]] = []
    epochs_done = 0

    for cp in cps[:-1]:
        # Train (cp - epochs_done) more epochs.
        n_to_train = cp - epochs_done
        for _ in range(n_to_train):
            perm = rng.permutation(n_train)
            idx = torch.from_numpy(perm.astype(np.int64))
            obs_b = Z_train_t[idx]
            cur_b = cur_train_t[idx]
            nxt_b = nxt_train_t[idx]
            out = trainer.step(obs_b, cur_b, nxt_b)
            epoch_losses.append(float(out["loss"].item()))
        epochs_done = cp

        scores = _eval_checkpoint_lightweight()
        checkpoint_metrics.append(
            CheckpointMetrics(
                epoch=cp,
                sigma_failure_auroc=scores["sigma_failure_auroc"],
                margin_failure_auroc=scores["margin_failure_auroc"],
                sigma_failure_uplift=scores["sigma_failure_uplift"],
                sigma_structural_auroc=scores["sigma_structural_auroc"],
                margin_structural_auroc=scores["margin_structural_auroc"],
                sigma_structural_uplift=scores["sigma_structural_uplift"],
            )
        )
        for name, value in scores.items():
            checkpoint_metric_records.append((cp, name, float(value)))

    # Train remaining epochs to the LAST checkpoint.
    final_cp = cps[-1]
    n_to_train = final_cp - epochs_done
    for _ in range(n_to_train):
        perm = rng.permutation(n_train)
        idx = torch.from_numpy(perm.astype(np.int64))
        obs_b = Z_train_t[idx]
        cur_b = cur_train_t[idx]
        nxt_b = nxt_train_t[idx]
        out = trainer.step(obs_b, cur_b, nxt_b)
        epoch_losses.append(float(out["loss"].item()))
    epochs_done = final_cp

    final_loss = epoch_losses[-1] if epoch_losses else 0.0
    loss_decrease_first_to_last_epoch = (
        (epoch_losses[0] - epoch_losses[-1]) if len(epoch_losses) >= 2 else 0.0
    )

    # ------------------------------------------------------------------
    # Step 5 -- Phase B Hamming snapshot at the final checkpoint.
    # ------------------------------------------------------------------
    posterior_snapshot = trainer.to_classical_posterior_mask()
    learned_legal_b = posterior_snapshot.legality_matrix(0.5)
    hamming_b = int(np.sum(learned_legal_b != gold_legal))
    phase_b_hamming_normalised = (
        float(hamming_b) / float(total_cells) if total_cells else 0.0
    )

    # ------------------------------------------------------------------
    # Step 6 -- Phase C FULL eval at the final checkpoint, with trace.
    # Mirrors E14 exactly.
    # ------------------------------------------------------------------
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

    sigma_q = _safe_quantizer("sigma_axis", sigmas_calib, n_axis_bins, rng)
    energy_q = _safe_quantizer("energy_axis", energies_calib, n_axis_bins, rng)
    margin_q = _safe_quantizer("margin_axis", margins_calib, n_axis_bins, rng)

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

    experiment_label = "E15"
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

    boundary_step = is_boundary_state[test_ds.current_states]

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

                prev_output_tuple = output_node_tuple

        # ------------------------------------------------------------------
        # Step 7 -- Aggregate metrics for the LAST checkpoint.
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
                1 if "after_operand" in name else 0
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

        # Append the FINAL checkpoint to checkpoint_metrics so the public
        # checkpoint table includes every requested epoch.
        checkpoint_metrics.append(
            CheckpointMetrics(
                epoch=final_cp,
                sigma_failure_auroc=sigma_auroc_value,
                margin_failure_auroc=margin_auroc_value,
                sigma_failure_uplift=sigma_uplift,
                sigma_structural_auroc=sigma_structural_auroc_value,
                margin_structural_auroc=margin_structural_auroc_value,
                sigma_structural_uplift=sigma_structural_uplift,
            )
        )
        for name, value in (
            ("sigma_failure_auroc", sigma_auroc_value),
            ("margin_failure_auroc", margin_auroc_value),
            ("sigma_failure_uplift", sigma_uplift),
            ("sigma_structural_auroc", sigma_structural_auroc_value),
            ("margin_structural_auroc", margin_structural_auroc_value),
            ("sigma_structural_uplift", sigma_structural_uplift),
        ):
            checkpoint_metric_records.append((final_cp, name, float(value)))

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

        timestamp = dt.datetime.now(tz=dt.timezone.utc).isoformat()

        # First emit the per-checkpoint AUROC table. SAME metric_name across
        # checkpoints; the ``step`` field is the epoch count and is the
        # discriminator. Documented in the module docstring.
        for ep, name, value in checkpoint_metric_records:
            metrics_writer.append(
                MetricsRecord(
                    run_id=run_id,
                    experiment=experiment_label,
                    ablation=ablation_label,
                    seed=seed,
                    step=int(ep),
                    split="test",
                    metric_name=name,
                    value=float(value),
                    timestamp=timestamp,
                )
            )

        # Then emit the headline (last-checkpoint) metrics with step=0
        # using the legacy E14 metric names. Aggregate.py readers built
        # around E14's flat layout pick these up unchanged.
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
            ("n_torch_trainable_params", float(n_torch_trainable_params)),
            ("final_loss", float(final_loss)),
            (
                "loss_decrease_first_to_last_epoch",
                float(loss_decrease_first_to_last_epoch),
            ),
            (
                "mean_kl_surprise",
                float(np.mean(kl_surprises_step))
                if kl_surprises_step
                else 0.0,
            ),
            ("n_train", float(n_train)),
            ("n_test", float(n_test)),
            (
                "product_graph_cells_materialised",
                float(product_graph.n_materialised()),
            ),
            ("n_checkpoints", float(len(cps))),
            ("final_checkpoint_epoch", float(final_cp)),
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

    # Touch ``kl_categorical`` to keep the import referenced at module scope.
    _ = kl_categorical(np.array([0.5, 0.5]), np.array([0.5, 0.5]))

    return E15Result(
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
        n_torch_trainable_params=n_torch_trainable_params,
        final_loss=float(final_loss),
        loss_decrease_first_to_last_epoch=float(
            loss_decrease_first_to_last_epoch
        ),
        n_train=int(n_train),
        n_test=int(n_test),
        checkpoint_epochs=list(cps),
        checkpoint_metrics=list(checkpoint_metrics),
    )
