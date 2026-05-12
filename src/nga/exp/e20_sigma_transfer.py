"""E20 -- Cross-grammar sigma-weight transfer + compute-efficiency tracking.

Phase 18 Track 1 asks a clean empirical question: are the
:class:`SingularityDetector`'s additive signal weights *grammar-portable*?

Pipeline:

  Phase 0 -- Source training (Python big, E18):
    Re-run E18's torch-native pipeline on the python_big grammar to
    produce a run with per-step (margin, decision_tie, illegal,
    loop_risk, kl_surprise) signal arrays plus error labels. Pass those
    arrays to :func:`nga.arch.sigma_weight_optimizer.optimize_sigma_weights`
    to recover the weight dict that maximises sigma_AUROC on python_big.
    Persist the dict to ``runs/<run_id>/tuned_weights.json``.

  Phase 1 -- Target evaluation (JSON, E19's pipeline):
    Train end-to-end on the JSON FSM, then on the held-out test split
    compute sigma TWICE per step:
      * sigma_default  -- the live SingularityDetector with DEFAULT_WEIGHTS
        (kl_surprise weight bumped exactly as E19 does).
      * sigma_transferred -- a second SingularityDetector instantiated
        with the Phase-0 tuned weight dict.
    Aggregate ``sigma_auroc_default``, ``sigma_auroc_transferred``,
    ``sigma_uplift_default``, ``sigma_uplift_transferred``, and the
    headline ``transfer_lift = sigma_auroc_transferred -
    sigma_auroc_default`` into metrics.jsonl.

Compute-efficiency block (NEW for E20+):
  Track wall-clock seconds for Phase 0 and Phase 1 separately, the
  overall total, inference throughput on Phase 1's eval loop, and peak
  resident memory delta. All five surface as metrics rows so future
  runners inherit the same observability shape.

Acceptance: the runner is observed-only -- it never tunes weights to
make ``transfer_lift`` positive. Whatever the data shows is the answer.
"""
from __future__ import annotations

import datetime as dt
import json
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
from nga.arch.sigma_weight_optimizer import (
    DEFAULT_OPTIMIZER_GRID,
    compute_sigma_from_weights,
    optimize_sigma_weights,
)
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
from nga.drivers.graph_fsm_spec import load as load_graph_fsm_spec
from nga.drivers.jsonl_writer import JsonlWriter
from nga.drivers.metrics_jsonl import MetricsRecord
from nga.drivers.results_jsonl import ResultsRecord
from nga.exp.dataset_json import (
    generate_json_dataset,
    train_test_split_by_document,
)
from nga.exp.dataset_python_big import (
    generate_python_big_dataset,
    train_test_split_by_program,
)

try:  # torch is required for E20.
    import torch
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "torch is required for E20; install via `pixi add pytorch`."
    ) from _exc


__all__ = ["E20Result", "run_e20"]


READOUT_TYPE_ID = "default"

# JSON value-ambiguity helpers (mirrors e19_json).
JSON_TOP_VALUE_STATE = "S0_value"
_OBJ_AFTER_COLON_SUBSTRING = "obj_after_colon"
_ARR_FIRST_VALUE_SUBSTRING = "arr_first_value"
_ARR_AFTER_COMMA_SUBSTRING = "arr_after_comma"


def _is_value_ambiguity_state(vertex_id: str) -> bool:
    if vertex_id == JSON_TOP_VALUE_STATE:
        return True
    if _OBJ_AFTER_COLON_SUBSTRING in vertex_id:
        return True
    if _ARR_FIRST_VALUE_SUBSTRING in vertex_id:
        return True
    if _ARR_AFTER_COMMA_SUBSTRING in vertex_id:
        return True
    return False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class E20Result:
    """Aggregate metrics returned by run_e20."""

    accuracy: float
    sigma_auroc_default: float
    sigma_auroc_transferred: float
    margin_auroc: float
    sigma_uplift_default: float
    sigma_uplift_transferred: float
    transfer_lift: float
    tuned_weights: dict[str, float]
    phase0_wall_clock_seconds: float
    phase1_wall_clock_seconds: float
    total_wall_clock_seconds: float
    inference_throughput_samples_per_sec: float
    peak_memory_kb: float
    phase_a_hamming_normalised: float
    phase_b_hamming_normalised: float
    n_train: int
    n_test: int


# ---------------------------------------------------------------------------
# Helpers (mirrors of e18 / e19)
# ---------------------------------------------------------------------------


def _empirical_dist_from_window(
    window: deque, n_states: int, eps: float = 1e-3
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


def _peak_memory_kb() -> float:
    """Return peak RSS in kilobytes via ``resource``.

    On Linux ``ru_maxrss`` is reported in KB; on macOS the same field is
    in BYTES. We assume Linux (the project's CI target). Returns 0.0 on
    platforms where ``resource`` is unavailable.
    """
    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:  # pragma: no cover - extremely defensive
        return 0.0


# ---------------------------------------------------------------------------
# Phase 0: train on python_big and harvest signals+errors.
# ---------------------------------------------------------------------------


def _phase0_collect_python_big_signals(
    *,
    fsm: GraphFSM,
    seed: int,
    margin_threshold: float,
    n_programs: int,
    max_paren_depth: int,
    p_def: float,
    p_call: float,
    illegal_temptation_fraction: float,
    n_epochs: int,
    encoder_hidden_dim: int,
    readout_hidden_dim: int,
    test_fraction: float,
    kl_surprise_weight: float,
    trainer_lr: float,
    grad_clip: float,
    empirical_window: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run a python_big training+eval pass; return (signals, error_labels).

    Mirrors run_e18's Phase A/B/C path but yields raw per-step signal
    arrays + error labels in memory, without touching disk artefacts.
    """
    n_states = fsm.vertex_count

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)

    ds = generate_python_big_dataset(
        fsm=fsm,
        n_programs=n_programs,
        max_paren_depth=max_paren_depth,
        p_def=p_def,
        p_call=p_call,
        seed=seed,
        illegal_temptation_fraction=illegal_temptation_fraction,
    )
    train_ds, test_ds = train_test_split_by_program(
        ds, seed=seed, test_fraction=test_fraction
    )

    embedding_dim = train_ds.X.shape[1]
    n_train = train_ds.X.shape[0]
    n_test = test_ds.X.shape[0]

    pair_counts = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_train):
        pair_counts[
            int(train_ds.current_states[i]), int(train_ds.y_next[i])
        ] += 1.0

    alpha_seed, beta_seed = bayesian_m_step_beta(
        expected_counts_pos=pair_counts,
        expected_counts_neg=None,
        prior_alpha=1.0,
        prior_beta=1.0,
    )

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
        config=TorchTrainerConfig(lr=trainer_lr, lambda_kl=1.0, grad_clip=grad_clip),
        seed=int(seed),
    )

    Z_train_t = torch.from_numpy(Z_train)
    cur_train_t = torch.from_numpy(train_ds.current_states.astype(np.int64))
    nxt_train_t = torch.from_numpy(train_ds.y_next.astype(np.int64))
    for _epoch in range(int(n_epochs)):
        perm = rng.permutation(n_train)
        idx = torch.from_numpy(perm.astype(np.int64))
        trainer.step(Z_train_t[idx], cur_train_t[idx], nxt_train_t[idx])

    posterior_snapshot = trainer.to_classical_posterior_mask()

    # Build a default-weight detector (with kl_surprise weight bumped) just
    # to harvest per-signal contributions; we do NOT use its sigma value
    # as the optimisation target.
    detector_weights = dict(SingularityDetector.DEFAULT_WEIGHTS)
    detector_weights["kl_surprise"] = float(kl_surprise_weight)
    detector = SingularityDetector(
        margin_threshold=margin_threshold,
        enabled=True,
        weights=detector_weights,
    )

    history = TaggerHistory()
    vertex_ids = fsm.vertex_ids
    Z_test_t = torch.from_numpy(Z_test)

    margin_signals: list[float] = []
    decision_tie_signals: list[float] = []
    illegal_signals: list[float] = []
    loop_signals: list[float] = []
    kl_surprise_signals: list[float] = []
    error_labels: list[int] = []

    windows: dict[int, deque] = {
        s: deque(maxlen=empirical_window) for s in range(n_states)
    }
    for i in range(min(n_train, empirical_window)):
        windows[int(train_ds.current_states[i])].append(int(train_ds.y_next[i]))

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
            kl_s = float(compute_kl_surprise(empirical_dist, p_masked))

            is_illegal = not bool(
                fsm.is_legal_transition(current_state, predicted_state)
            )
            loop_risk = float(history.predicted_loop_risk(predicted_state))

            _, contribs = detector.compute(
                margin=margin,
                is_illegal=is_illegal,
                loop_risk=loop_risk,
                kl_surprise=kl_s,
            )
            history.push(predicted_state)
            windows[cs_idx].append(int(true_next_idx))

            margin_signals.append(float(contribs.margin_signal))
            decision_tie_signals.append(float(contribs.decision_tie_signal))
            illegal_signals.append(float(contribs.illegal_signal))
            loop_signals.append(float(contribs.loop_signal))
            kl_surprise_signals.append(float(contribs.kl_surprise))
            error_labels.append(1 if true_next != predicted_state else 0)

    signals = {
        "margin": np.asarray(margin_signals, dtype=np.float64),
        "decision_tie": np.asarray(decision_tie_signals, dtype=np.float64),
        "illegal": np.asarray(illegal_signals, dtype=np.float64),
        "loop_risk": np.asarray(loop_signals, dtype=np.float64),
        "kl_surprise": np.asarray(kl_surprise_signals, dtype=np.float64),
    }
    labels = np.asarray(error_labels, dtype=np.int64)
    return signals, labels


# ---------------------------------------------------------------------------
# Mapping the optimiser's "loop_risk" key onto SingularityDetector's "loop"
# weight. The optimiser uses "loop_risk" because that is the public
# argument name on ``SingularityDetector.compute`` and on the
# ``SignalContributions.loop_signal`` field. The detector's internal
# weight key is "loop". We translate at the boundary.
# ---------------------------------------------------------------------------


def _detector_weights_from_tuned(
    tuned: dict[str, float],
    *,
    kl_surprise_weight: float,
) -> dict[str, float]:
    """Map optimiser-key dict onto SingularityDetector weight-key dict.

    The optimiser uses ``loop_risk`` (matching the signal name in
    ``SingularityDetector.compute`` and the ``SignalContributions``
    ``loop_signal`` field). The detector's internal weight slot is
    keyed ``loop``. Other keys are 1:1.
    """
    out: dict[str, float] = {}
    for k, v in tuned.items():
        if k == "loop_risk":
            out["loop"] = float(v)
        else:
            out[k] = float(v)
    # Fill keys absent in the tuned dict from defaults so the detector
    # contract (every weight key present) is satisfied. The kl_surprise
    # weight in particular -- if absent from the tuned dict -- falls
    # back to the E18/E19 convention of an explicit positive value.
    if "kl_surprise" not in out:
        out["kl_surprise"] = float(kl_surprise_weight)
    return out


# ---------------------------------------------------------------------------
# Phase 1: JSON evaluation with twin sigma scores.
# ---------------------------------------------------------------------------


def _phase1_run_json(
    *,
    fsm: GraphFSM,
    ablation: AblationTuple,
    run_id: str,
    output_dir: Path,
    seed: int,
    margin_threshold: float,
    n_documents: int,
    max_depth: int,
    p_object: float,
    illegal_temptation_fraction: float,
    n_axis_bins: int,
    empirical_window: int,
    n_epochs: int,
    encoder_hidden_dim: int,
    readout_hidden_dim: int,
    test_fraction: float,
    kl_surprise_weight: float,
    trainer_lr: float,
    grad_clip: float,
    tuned_weights: dict[str, float],
) -> dict[str, float]:
    """Run JSON E2E pipeline; emit standard 4 artefacts; return Phase-1 metrics.

    Computes two sigma scores per step: ``sigma_default`` (live detector
    with DEFAULT_WEIGHTS + kl_surprise bump) and ``sigma_transferred``
    (a second detector built from the Phase-0 tuned weights).
    """
    n_states = fsm.vertex_count

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)

    ds = generate_json_dataset(
        fsm=fsm,
        n_documents=n_documents,
        max_depth=max_depth,
        p_object=p_object,
        seed=seed,
        illegal_temptation_fraction=illegal_temptation_fraction,
    )
    train_ds, test_ds = train_test_split_by_document(
        ds, seed=seed, test_fraction=test_fraction
    )

    embedding_dim = train_ds.X.shape[1]
    n_train = train_ds.X.shape[0]
    n_test = test_ds.X.shape[0]

    vertex_ids = fsm.vertex_ids

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
    _ = expected_counts_observed(state_path, n_states=n_states).astype(np.float64)

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
    phase_a_hamming = (
        float(hamming_a) / float(total_cells) if total_cells else 0.0
    )

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
        config=TorchTrainerConfig(lr=trainer_lr, lambda_kl=1.0, grad_clip=grad_clip),
        seed=int(seed),
    )
    n_torch_trainable_params = int(
        sum(p.numel() for p in trainer.parameters() if p.requires_grad)
    )

    Z_train_t = torch.from_numpy(Z_train)
    cur_train_t = torch.from_numpy(train_ds.current_states.astype(np.int64))
    nxt_train_t = torch.from_numpy(train_ds.y_next.astype(np.int64))
    epoch_losses: list[float] = []
    for _epoch in range(int(n_epochs)):
        perm = rng.permutation(n_train)
        idx = torch.from_numpy(perm.astype(np.int64))
        out = trainer.step(Z_train_t[idx], cur_train_t[idx], nxt_train_t[idx])
        epoch_losses.append(float(out["loss"].item()))
    final_loss = epoch_losses[-1] if epoch_losses else 0.0

    posterior_snapshot = trainer.to_classical_posterior_mask()
    learned_legal_b = posterior_snapshot.legality_matrix(0.5)
    hamming_b = int(np.sum(learned_legal_b != gold_legal))
    phase_b_hamming = float(hamming_b) / float(total_cells) if total_cells else 0.0

    energy_fn = EnergyFunction(EnergyParameters())

    default_detector_weights = dict(SingularityDetector.DEFAULT_WEIGHTS)
    default_detector_weights["kl_surprise"] = float(kl_surprise_weight)
    detector_default = SingularityDetector(
        margin_threshold=margin_threshold,
        enabled=ablation.singularity_detector_enabled,
        weights=default_detector_weights,
    )
    transferred_weights = _detector_weights_from_tuned(
        tuned_weights, kl_surprise_weight=kl_surprise_weight
    )
    detector_transferred = SingularityDetector(
        margin_threshold=margin_threshold,
        enabled=ablation.singularity_detector_enabled,
        weights=transferred_weights,
    )

    history = TaggerHistory()
    control_policy = ControlPolicy(
        theta_normal=0.3,
        theta_abstain=0.7,
        fsm=fsm,
        goal_states=[fsm.vertex_index["ACCEPT"]],
    )

    Z_test_t = torch.from_numpy(Z_test)

    # Calibration pass for axis quantizers (uses sigma_default).
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
            sigma_i, _ = detector_default.compute(
                margin=m, is_illegal=False, loop_risk=0.0
            )
            sigmas_calib.append(float(sigma_i))
            unc = float(1.0 - p_np.max())
            contribs_e = energy_fn.compute(
                cost=0.0,
                uncertainty=unc,
                contradiction=0.0,
                loop_pressure=float(np.clip(sigma_i, 0.0, 1.0)) * 0.5,
                progress=1.0,
            )
            energies_calib.append(float(contribs_e.total))

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
        windows[int(train_ds.current_states[i])].append(int(train_ds.y_next[i]))

    experiment_label = "E20"
    ablation_label = run_id.split("_")[1]
    sample_ids = [s.sample_id for s in test_ds.samples]

    sigmas_default_step: list[float] = []
    sigmas_transferred_step: list[float] = []
    margins_step: list[float] = []
    errors_step: list[bool] = []
    y_true_list: list[str] = []
    predicted_list: list[str] = []
    transition_legal_list: list[bool] = []
    energies_total: list[float] = []
    strata: list[SingularityType] = []
    energies_correct: list[float] = []
    energies_incorrect: list[float] = []
    n_routed_to_recovery = 0
    n_abstained = 0
    n_mask_modified_predictions = 0
    prev_output_tuple: tuple[str, ...] | None = None

    # ---- BEGIN inference loop (timed for throughput) ----
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
                bias_row_uniform = np.zeros_like(bias_torch[cs_idx])

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
                kl_s = float(compute_kl_surprise(empirical_dist, p_masked))

                is_illegal = not bool(
                    fsm.is_legal_transition(current_state, predicted_state)
                )
                loop_risk = float(history.predicted_loop_risk(predicted_state))

                sigma_default, contribs = detector_default.compute(
                    margin=margin,
                    is_illegal=is_illegal,
                    loop_risk=loop_risk,
                    kl_surprise=kl_s,
                )
                sigma_transferred, _ = detector_transferred.compute(
                    margin=margin,
                    is_illegal=is_illegal,
                    loop_risk=loop_risk,
                    kl_surprise=kl_s,
                )

                d_chosen = float(d_all[argmax_state])
                unc = float(1.0 - np.exp(-max(d_chosen, 0.0)))
                lp = float(np.clip(0.5 * loop_risk + 0.5 * sigma_default, 0.0, 1.0))
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
                    tagger_inp, history, fsm, margin_threshold=margin_threshold
                )
                history.push(predicted_state)

                transition_legal = bool(
                    fsm.is_legal_transition(current_state, predicted_state)
                )
                mask_changed_prediction = bool(argmax_unmasked != argmax_state)
                if mask_changed_prediction:
                    n_mask_modified_predictions += 1

                if ablation.graph_mask_enabled:
                    row_mean = posterior_snapshot.posterior_mean()[cs_idx]
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
                    sigma=float(sigma_default),
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
                    sigma_observed=float(sigma_default),
                    abstain_threshold_used=control_policy.theta_abstain,
                )

                sigma_node = sigma_q.quantise(float(sigma_default))
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

                posterior_summary = {
                    "mean_alpha": float(np.mean(posterior_snapshot.alpha)),
                    "mean_beta": float(np.mean(posterior_snapshot.beta)),
                    "mean_entropy": float(
                        np.mean(posterior_snapshot.posterior_entropy())
                    ),
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
                    "sigma_transferred": float(sigma_transferred),
                }
                energy_breakdown = {
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
                        singular_flag=bool(sigma_default >= 0.5),
                        sigma_score=float(sigma_default),
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
                        sigma_total=float(sigma_default),
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
                        mask_version_id=posterior_snapshot.mask_version_id(),
                        posterior_summary=posterior_summary,
                        axis_node_ids=axis_node_ids,
                    )
                )

                y_true_list.append(true_next)
                predicted_list.append(predicted_state)
                transition_legal_list.append(transition_legal)
                energies_total.append(energy_total)
                strata.append(tag_result.tag)
                if true_next == predicted_state:
                    energies_correct.append(energy_total)
                else:
                    energies_incorrect.append(energy_total)

                err = bool(true_next != predicted_state)
                sigmas_default_step.append(float(sigma_default))
                sigmas_transferred_step.append(float(sigma_transferred))
                margins_step.append(float(margin))
                errors_step.append(err)
                prev_output_tuple = output_node_tuple

        inference_end = time.perf_counter()
        inference_seconds = max(inference_end - inference_start, 1e-9)

        # -------- aggregate ----------
        y_true_arr = np.array(y_true_list)
        predicted_arr = np.array(predicted_list)
        accuracy = float(np.mean(y_true_arr == predicted_arr)) if n_test else 0.0

        if energies_total:
            partition = compute_stratified_partition(
                np.array(energies_total, dtype=np.float64),
                strata,
                temperature=1.0,
            )
            run_free_energy = float(free_energy(partition))
        else:
            run_free_energy = 0.0

        sigmas_default_arr = np.array(sigmas_default_step, dtype=float)
        sigmas_transferred_arr = np.array(sigmas_transferred_step, dtype=float)
        margins_arr = np.array(margins_step, dtype=float)
        errors_arr = np.array(errors_step, dtype=bool)

        if errors_arr.size > 0 and 0 < int(errors_arr.sum()) < int(errors_arr.size):
            sigma_auroc_default = float(sigma_auroc(sigmas_default_arr, errors_arr))
            sigma_auroc_transferred = float(
                sigma_auroc(sigmas_transferred_arr, errors_arr)
            )
            margin_auroc_value = float(margin_auroc(margins_arr, errors_arr))
        else:
            sigma_auroc_default = 0.5
            sigma_auroc_transferred = 0.5
            margin_auroc_value = 0.5

        sigma_uplift_default = sigma_auroc_default - margin_auroc_value
        sigma_uplift_transferred = sigma_auroc_transferred - margin_auroc_value
        transfer_lift = sigma_auroc_transferred - sigma_auroc_default

        timestamp = dt.datetime.now(tz=dt.timezone.utc).isoformat()
        metric_pairs: list[tuple[str, float]] = [
            ("accuracy", accuracy),
            ("phase_a_hamming_normalised", float(phase_a_hamming)),
            ("phase_b_hamming_normalised", float(phase_b_hamming)),
            ("sigma_auroc_default", sigma_auroc_default),
            ("sigma_auroc_transferred", sigma_auroc_transferred),
            ("margin_auroc", margin_auroc_value),
            ("sigma_uplift_default", sigma_uplift_default),
            ("sigma_uplift_transferred", sigma_uplift_transferred),
            ("transfer_lift", transfer_lift),
            ("free_energy", run_free_energy),
            ("n_torch_trainable_params", float(n_torch_trainable_params)),
            ("final_loss", float(final_loss)),
            ("n_train", float(n_train)),
            ("n_test", float(n_test)),
            ("n_routed_to_recovery", float(n_routed_to_recovery)),
            ("n_abstained", float(n_abstained)),
            ("n_mask_modified_predictions", float(n_mask_modified_predictions)),
            (
                "product_graph_cells_materialised",
                float(product_graph.n_materialised()),
            ),
        ]
        # Surface tuned weights as metric rows (so they're persisted in the
        # standard-shape metrics.jsonl alongside the AUROCs).
        for k, v in tuned_weights.items():
            metric_pairs.append((f"tuned_weight_{k}", float(v)))

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

    return {
        "accuracy": accuracy,
        "phase_a_hamming_normalised": phase_a_hamming,
        "phase_b_hamming_normalised": phase_b_hamming,
        "sigma_auroc_default": sigma_auroc_default,
        "sigma_auroc_transferred": sigma_auroc_transferred,
        "margin_auroc": margin_auroc_value,
        "sigma_uplift_default": sigma_uplift_default,
        "sigma_uplift_transferred": sigma_uplift_transferred,
        "transfer_lift": transfer_lift,
        "n_train": float(n_train),
        "n_test": float(n_test),
        "inference_seconds": inference_seconds,
        "final_loss": final_loss,
    }


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------


def run_e20(
    *,
    config: Config,
    ablation: AblationTuple,
    fsm: GraphFSM,
    run_id: str,
    output_dir: Path,
    seed: int,
    margin_threshold: float = 0.10,
    # Phase 0 (python_big) hyperparams.
    p0_python_big_fsm_path: str | Path = "tests/fixtures/graphs/python_big.fsm.yaml",
    p0_n_programs: int = 100,
    p0_max_paren_depth: int = 3,
    p0_p_def: float = 0.2,
    p0_p_call: float = 0.3,
    p0_illegal_temptation_fraction: float = 0.10,
    p0_n_epochs: int = 5,
    p0_test_fraction: float = 0.25,
    # Phase 1 (JSON) hyperparams.
    n_documents: int = 200,
    max_depth: int = 3,
    p_object: float = 0.5,
    illegal_temptation_fraction: float = 0.10,
    n_axis_bins: int = 5,
    empirical_window: int = 32,
    n_epochs: int = 10,
    encoder_hidden_dim: int = 32,
    readout_hidden_dim: int = 32,
    test_fraction: float = 0.25,
    kl_surprise_weight: float = 0.5,
    trainer_lr: float = 1e-2,
    grad_clip: float = 5.0,
) -> E20Result:
    """Run E20 (cross-grammar sigma-weight transfer + efficiency tracking).

    The ``fsm`` argument is the JSON FSM (CLI-loaded). The python_big
    FSM used for Phase 0 is loaded from
    ``p0_python_big_fsm_path`` (relative to the cwd; defaults to the
    shipped fixture). Returns aggregate metrics.
    """
    if fsm.vertex_ids[0] != "START":
        raise ValueError(
            f"E20 expects fsm.vertex_ids[0] == 'START'; got {fsm.vertex_ids[0]!r}"
        )
    if "ACCEPT" not in fsm.vertex_ids:
        raise ValueError(
            f"E20 expects fsm to contain 'ACCEPT'; got {fsm.vertex_ids!r}"
        )
    if not any(_is_value_ambiguity_state(vid) for vid in fsm.vertex_ids):
        raise ValueError(
            "E20 expects fsm to contain at least one JSON "
            "value-ambiguity vertex (S0_value / *_obj_after_colon / "
            "*_arr_first_value / *_arr_after_comma)."
        )

    # Compute-efficiency tracking: capture peak RSS at start.
    mem_start_kb = _peak_memory_kb()
    total_start = time.perf_counter()

    # ----- Phase 0: train on python_big and optimise weights -----
    p0_start = time.perf_counter()

    p0_fsm_path = Path(p0_python_big_fsm_path)
    if not p0_fsm_path.is_absolute():
        p0_fsm_path = Path.cwd() / p0_fsm_path
    p0_fsm_spec = load_graph_fsm_spec(p0_fsm_path)
    p0_fsm = GraphFSM(p0_fsm_spec)

    signals, error_labels = _phase0_collect_python_big_signals(
        fsm=p0_fsm,
        seed=seed,
        margin_threshold=margin_threshold,
        n_programs=p0_n_programs,
        max_paren_depth=p0_max_paren_depth,
        p_def=p0_p_def,
        p_call=p0_p_call,
        illegal_temptation_fraction=p0_illegal_temptation_fraction,
        n_epochs=p0_n_epochs,
        encoder_hidden_dim=encoder_hidden_dim,
        readout_hidden_dim=readout_hidden_dim,
        test_fraction=p0_test_fraction,
        kl_surprise_weight=kl_surprise_weight,
        trainer_lr=trainer_lr,
        grad_clip=grad_clip,
        empirical_window=empirical_window,
    )

    tuned_weights = optimize_sigma_weights(
        signals=signals,
        error_labels=error_labels,
        weight_grid=DEFAULT_OPTIMIZER_GRID,
    )

    # Persist the tuned weights so the run is fully reproducible.
    tuned_weights_path = output_dir / "tuned_weights.json"
    with tuned_weights_path.open("w", encoding="utf-8") as fh:
        json.dump(tuned_weights, fh, indent=2, sort_keys=True)

    p0_seconds = time.perf_counter() - p0_start

    # ----- Phase 1: JSON eval with twin sigmas -----
    p1_start = time.perf_counter()
    p1_metrics = _phase1_run_json(
        fsm=fsm,
        ablation=ablation,
        run_id=run_id,
        output_dir=output_dir,
        seed=seed,
        margin_threshold=margin_threshold,
        n_documents=n_documents,
        max_depth=max_depth,
        p_object=p_object,
        illegal_temptation_fraction=illegal_temptation_fraction,
        n_axis_bins=n_axis_bins,
        empirical_window=empirical_window,
        n_epochs=n_epochs,
        encoder_hidden_dim=encoder_hidden_dim,
        readout_hidden_dim=readout_hidden_dim,
        test_fraction=test_fraction,
        kl_surprise_weight=kl_surprise_weight,
        trainer_lr=trainer_lr,
        grad_clip=grad_clip,
        tuned_weights=tuned_weights,
    )
    p1_seconds = time.perf_counter() - p1_start

    total_seconds = time.perf_counter() - total_start
    mem_end_kb = _peak_memory_kb()
    peak_memory_kb = max(mem_end_kb - mem_start_kb, 0.0)

    n_test = int(p1_metrics["n_test"])
    inference_seconds = float(p1_metrics["inference_seconds"])
    inference_throughput = float(n_test) / inference_seconds if inference_seconds > 0 else 0.0

    # Append efficiency rows (and Phase-0 sample-count) to metrics.jsonl.
    timestamp = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    eff_pairs = [
        ("phase0_wall_clock_seconds", float(p0_seconds)),
        ("phase1_wall_clock_seconds", float(p1_seconds)),
        ("total_wall_clock_seconds", float(total_seconds)),
        ("inference_throughput_samples_per_sec", float(inference_throughput)),
        ("peak_memory_kb", float(peak_memory_kb)),
        ("n_phase0_eval_samples", float(error_labels.size)),
    ]
    with JsonlWriter(output_dir / "metrics.jsonl", MetricsRecord) as mw:
        for metric_name, value in eff_pairs:
            mw.append(
                MetricsRecord(
                    run_id=run_id,
                    experiment="E20",
                    ablation=run_id.split("_")[1],
                    seed=seed,
                    step=0,
                    split="test",
                    metric_name=metric_name,
                    value=value,
                    timestamp=timestamp,
                )
            )

    return E20Result(
        accuracy=float(p1_metrics["accuracy"]),
        sigma_auroc_default=float(p1_metrics["sigma_auroc_default"]),
        sigma_auroc_transferred=float(p1_metrics["sigma_auroc_transferred"]),
        margin_auroc=float(p1_metrics["margin_auroc"]),
        sigma_uplift_default=float(p1_metrics["sigma_uplift_default"]),
        sigma_uplift_transferred=float(p1_metrics["sigma_uplift_transferred"]),
        transfer_lift=float(p1_metrics["transfer_lift"]),
        tuned_weights=dict(tuned_weights),
        phase0_wall_clock_seconds=float(p0_seconds),
        phase1_wall_clock_seconds=float(p1_seconds),
        total_wall_clock_seconds=float(total_seconds),
        inference_throughput_samples_per_sec=float(inference_throughput),
        peak_memory_kb=float(peak_memory_kb),
        phase_a_hamming_normalised=float(p1_metrics["phase_a_hamming_normalised"]),
        phase_b_hamming_normalised=float(p1_metrics["phase_b_hamming_normalised"]),
        n_train=int(p1_metrics["n_train"]),
        n_test=int(p1_metrics["n_test"]),
    )
