"""Single-loss trainer that minimises observed-trajectory energy.

This atom is the unifier for inference and training. Instead of optimising a
classification cross-entropy, the trainer's loss IS the energy at the observed
(input -> predicted node-tuple) point, and a single ``step`` performs three
intertwined updates:

1. Riemannian SGD on the hyperbolic prototype that the observation is bound to.
   The Euclidean gradient of energy w.r.t. prototype position comes through the
   ``uncertainty`` channel of :class:`EnergyFunction` -- ``uncertainty`` is taken
   to be a monotone proxy for hyperbolic distance to the prototype, so

       d/d(prototype) E  =  beta_uncertainty * d/d(prototype) ||obs - proto||^2
                         =  -2 * beta_uncertainty * (obs - proto)

   in the ambient (Euclidean) tangent picture. ``riemannian_gradient`` then
   converts that to the Poincare-ball gradient and ``rsgd_step`` retracts.
   Net effect: each step *pulls the prototype toward the observation*, with
   strength scaled by the energy weight on uncertainty.

2. Beta-posterior update on the (src, dst) edge in :class:`PosteriorMask`. The
   ``quality`` signal is ``Q = exp(-(gamma E + delta sigma + eps c + zeta L))``,
   so zero-failure observations push alpha by ~1 (full positive evidence) and
   high-energy/high-sigma/contradictory transitions push the posterior toward
   "illegal" with Q -> 0. Using exp(-z) rather than sigmoid(-z) keeps the
   "no failure observed" pole at Q=1 (not 0.5), which is what makes the
   cold-start posterior converge to the underlying FSM legality structure.

3. (Optional) Orbit-equivariance enforcement. When a :class:`GroupAction` is
   supplied, prototypes whose vertex IDs lie in the same orbit are averaged
   in the tangent space at zero (via ``log_map_zero``) and redistributed
   (via ``exp_map_zero``). This keeps prototypes related-by-symmetry related
   after gradient updates.

The closed-walk monodromy loss is exposed as a regulariser hook -- the trainer
itself does not modify prototypes from monodromy drift; callers can fold it
into a higher-level objective if they wish.

Phase 7 -- Information geometry
-------------------------------
Phase 7 adds principled, additive information-geometry tooling that sits
alongside (never replacing) the heuristic ``quality_signal``/``step`` API:

* :meth:`quality_signal_kl` computes ``Q = exp(-KL(empirical || predicted))``
  between an empirically-observed conditional next-state distribution and the
  model's softmax prediction. This is the principled "Baum-Welch-shaped"
  quality: zero KL <=> distributions match <=> Q = 1, and Q decays
  monotonically as the model's predicted distribution moves away from the
  empirical one. :meth:`quality_signal_blended` is a convex combination of
  the heuristic and KL forms so callers can run a curriculum (e.g. blend=1.0
  cold-start while the empirical histogram is unreliable, decaying to 0.0 as
  the trainer converges and the KL signal becomes the canonical M-step
  objective). When either distribution is missing the blended call falls back
  to pure heuristic so callers can drop it in unconditionally.

* :meth:`natural_gradient_update` rescales a Euclidean gradient on the
  posterior mask (alpha, beta) by the diagonal Fisher information per edge
  (alpha + beta, the Beta concentration). Edges with little evidence get
  amplified updates and well-determined edges get damped updates. This is
  a parameter-space-geometry-correct alternative to a flat learning rate.

* :meth:`crb_confidence` (and :meth:`crb_confidence_matrix`) reads the
  Cramer-Rao bound on the variance of an unbiased estimator of an edge's
  Bernoulli mean and reports n / n_required, where n is the effective
  sample count alpha + beta - 2. Values >= 1.0 mean the CRB is met for the
  requested target variance and the edge is "well-determined"; smaller
  values are a sample-efficiency progress indicator.

* :meth:`kl_progress` returns the mean per-edge KL between a snapshot of the
  posterior mask and the current one, suitable as a convergence diagnostic
  (training has converged when this drops below threshold).

These methods compose: a typical training loop updates with the existing
``step``, monitors :meth:`crb_confidence_matrix` to know which edges still
need data, and stops when :meth:`kl_progress` between consecutive snapshots
drops below threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

import numpy as np

from nga.arch.energy_function import EnergyFunction
from nga.arch.group_action_on_graph import GroupAction
from nga.arch.hyperbolic_embedding import (
    exp_map_zero,
    log_map_zero,
    poincare_distance,
    project,
    riemannian_gradient,
    rsgd_step,
)
from nga.arch.information_geometry import (
    crb_confidence_for_observations,
    fisher_information_bernoulli,
    fisher_information_diagonal_mask,
    kl_beta,
    kl_categorical,
    natural_gradient_diagonal,
)
from nga.arch.monodromy_consistency import monodromy_consistency_loss
from nga.arch.orbit_quotient_space import decompose_orbits
from nga.arch.posterior_mask import PosteriorMask

__all__ = [
    "TrainerConfig",
    "TrainerStepResult",
    "EnergyMinimizationTrainer",
]


# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Coefficients controlling the trainer's update rules.

    Fields
    ------
    lr_prototype:
        Riemannian-SGD step size on the prototype tower.
    posterior_quality_gamma:
        Weight on energy in the quality signal.
    posterior_quality_delta:
        Weight on sigma in the quality signal.
    posterior_quality_eps:
        Weight on the contradiction flag in the quality signal.
    posterior_quality_zeta:
        Weight on loop-risk in the quality signal.
    lambda_monodromy:
        Coefficient on the closed-walk consistency penalty when callers fold
        it into a higher-level objective. The trainer itself stores this for
        reporting only.
    enforce_orbit_sharing:
        If True, ``enforce_orbit_sharing()`` averages prototypes within each
        orbit. Has no effect when no GroupAction is attached.
    """

    lr_prototype: float = 0.05
    posterior_quality_gamma: float = 1.0
    posterior_quality_delta: float = 1.0
    posterior_quality_eps: float = 1.0
    posterior_quality_zeta: float = 0.5
    lambda_monodromy: float = 0.1
    enforce_orbit_sharing: bool = True


@dataclass
class TrainerStepResult:
    """Per-step diagnostics emitted by :meth:`EnergyMinimizationTrainer.step`.

    Fields
    ------
    energy_observed:
        Total energy at the observation BEFORE the prototype update.
    energy_after_update:
        Total energy at the observation AFTER the prototype shift. Should be
        non-greater than ``energy_observed`` on a well-conditioned step.
    quality:
        Q in [0, 1] used for the Beta-posterior update.
    posterior_alpha_beta_delta:
        How much (alpha, beta) grew on the (src, dst) entry. The pair sums
        to 1 by construction (one observation, fractionally split).
    prototype_shift_magnitude:
        Hyperbolic distance the active prototype moved this step.
    monodromy_loss:
        Last computed closed-walk consistency loss; informational only,
        populated only when ``closed_walk_loss`` has been called this step.
    """

    energy_observed: float
    energy_after_update: float
    quality: float
    posterior_alpha_beta_delta: tuple[float, float]
    prototype_shift_magnitude: float
    monodromy_loss: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sigmoid(z: float) -> float:
    """Numerically stable scalar sigmoid."""
    if z >= 0.0:
        ez = np.exp(-z)
        return float(1.0 / (1.0 + ez))
    ez = np.exp(z)
    return float(ez / (1.0 + ez))


def _uncertainty_from_distance(d: float) -> float:
    """Bounded monotone map from hyperbolic distance to ``uncertainty`` in [0, 1].

    We use ``1 - exp(-d)`` so that d=0 -> 0 (perfect confidence) and d -> inf
    -> 1 (maximally uncertain). This is monotone, smooth, bounded, and lets the
    energy function consume it without clipping artefacts.
    """
    return float(1.0 - np.exp(-max(d, 0.0)))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class EnergyMinimizationTrainer:
    """One-step trainer that unifies inference and training under energy.

    The trainer owns three pieces of state:

    - ``prototypes``: per-type dictionary mapping ``type_id`` -> ``np.ndarray``
      of shape ``(K_type, dim)`` in the Poincare ball.
    - ``posterior_mask``: a :class:`PosteriorMask` over the FSM vertex set.
    - ``energy_fn``: the prescribed :class:`EnergyFunction` whose ``beta``
      weight on the uncertainty channel is read to scale the prototype
      gradient.

    The optional :class:`GroupAction` lets ``enforce_orbit_sharing`` tie
    prototypes that should be related by symmetry.
    """

    def __init__(
        self,
        prototypes: dict[Hashable, np.ndarray],
        posterior_mask: PosteriorMask,
        energy_fn: EnergyFunction,
        group_action: GroupAction | None = None,
        config: TrainerConfig | None = None,
    ) -> None:
        if not isinstance(prototypes, dict):
            raise TypeError(
                f"prototypes must be a dict[type_id, np.ndarray]; got {type(prototypes)}"
            )
        # Validate prototype tensors and store float64 copies.
        self._prototypes: dict[Hashable, np.ndarray] = {}
        for type_id, arr in prototypes.items():
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim != 2:
                raise ValueError(
                    f"prototypes[{type_id!r}] must be 2-D (K, dim); got shape {a.shape}"
                )
            self._prototypes[type_id] = project(a)
        self._posterior = posterior_mask
        self._energy_fn = energy_fn
        self._group_action = group_action
        self._config = config if config is not None else TrainerConfig()
        self._last_monodromy_loss: float = 0.0

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def prototypes(self) -> dict[Hashable, np.ndarray]:
        """Mutable view of the current prototype dictionary."""
        return self._prototypes

    @property
    def posterior_mask(self) -> PosteriorMask:
        """The attached :class:`PosteriorMask`."""
        return self._posterior

    @property
    def energy_fn(self) -> EnergyFunction:
        """The attached :class:`EnergyFunction`."""
        return self._energy_fn

    @property
    def config(self) -> TrainerConfig:
        """The attached :class:`TrainerConfig`."""
        return self._config

    @property
    def group_action(self) -> GroupAction | None:
        """The attached :class:`GroupAction`, or ``None`` if not set."""
        return self._group_action

    # ------------------------------------------------------------------
    # Quality signal
    # ------------------------------------------------------------------

    def quality_signal(
        self,
        energy_total: float,
        sigma: float,
        contradiction: float,
        loop_risk: float,
    ) -> float:
        """Compute Q = exp(-(gamma E + delta sigma + eps c + zeta L)).

        Q -> 1 means the observation is high-quality evidence FOR the
        observed transition (zero failure signals); Q -> 0 means it is
        evidence AGAINST (high energy / sigma / contradiction / loop_risk).
        Q is bounded in [0, 1] by construction (the exponent is non-positive
        because all components are non-negative).

        Why exp(-z), not sigmoid(-z): the failure signals (E, sigma,
        contradiction, loop_risk) are all non-negative, so z >= 0 always.
        sigmoid(-z) asymptotes to 0.5 at z=0, which means a "perfect" zero-
        failure observation contributes only 0.5 to alpha (and 0.5 to beta) -
        no informational gain over the prior. exp(-z) instead gives Q=1 at
        z=0 (full positive evidence), which is the correct Bernoulli shape
        for the cold-start posterior to converge.
        """
        cfg = self._config
        z = (
            cfg.posterior_quality_gamma * float(energy_total)
            + cfg.posterior_quality_delta * float(sigma)
            + cfg.posterior_quality_eps * float(contradiction)
            + cfg.posterior_quality_zeta * float(loop_risk)
        )
        # Energy can go negative when progress dominates; clamp z>=0 so that
        # any "better than baseline" observation saturates at Q=1 (full
        # positive evidence) without violating the Bernoulli [0,1] contract.
        return float(np.exp(-max(z, 0.0)))

    # ------------------------------------------------------------------
    # Phase 7 -- Information-geometry quality signals
    # ------------------------------------------------------------------

    def quality_signal_kl(
        self,
        empirical_next_state_dist: np.ndarray,
        predicted_next_state_dist: np.ndarray,
    ) -> float:
        """Quality based on KL(empirical || predicted). Lower KL -> higher Q.

        Parameters
        ----------
        empirical_next_state_dist:
            ``(n_states,)`` empirical conditional distribution over next
            states given the current source state, from observed history.
            Must be non-negative; we re-normalise defensively.
        predicted_next_state_dist:
            ``(n_states,)`` model's softmax prediction over next states.
            Must be non-negative; we re-normalise defensively.

        Returns
        -------
        float
            ``Q = exp(-KL(empirical || predicted))``, in (0, 1]. ``Q = 1``
            iff the two distributions are identical (up to floating-point
            rounding).

        Notes
        -----
        This is the principled replacement for the heuristic
        :meth:`quality_signal`. Baum-Welch's M-step is exactly minimisation
        of this KL: maximising :meth:`quality_signal_kl` is equivalent to
        the maximum-likelihood update of the predicted distribution.
        """
        emp = np.asarray(empirical_next_state_dist, dtype=np.float64)
        pred = np.asarray(predicted_next_state_dist, dtype=np.float64)
        if emp.ndim != 1 or pred.ndim != 1:
            raise ValueError(
                f"empirical and predicted distributions must be 1-D; got "
                f"shapes {emp.shape} and {pred.shape}"
            )
        if emp.shape[0] != pred.shape[0]:
            raise ValueError(
                f"empirical and predicted distributions must have the same "
                f"length; got {emp.shape[0]} and {pred.shape[0]}"
            )
        if np.any(emp < 0.0) or np.any(pred < 0.0):
            raise ValueError(
                "empirical and predicted distributions must be non-negative"
            )
        emp_sum = float(emp.sum())
        pred_sum = float(pred.sum())
        if emp_sum <= 0.0 or pred_sum <= 0.0:
            raise ValueError(
                "empirical and predicted distributions must have positive mass"
            )
        emp_n = emp / emp_sum
        pred_n = pred / pred_sum
        kl = float(kl_categorical(emp_n, pred_n))
        return float(np.exp(-max(kl, 0.0)))

    def quality_signal_blended(
        self,
        energy_total: float,
        sigma: float,
        contradiction: float,
        loop_risk: float,
        empirical_dist: np.ndarray | None = None,
        predicted_dist: np.ndarray | None = None,
        blend: float = 0.5,
    ) -> float:
        """Convex blend of :meth:`quality_signal` and :meth:`quality_signal_kl`.

        Parameters
        ----------
        energy_total, sigma, contradiction, loop_risk:
            Inputs to the heuristic :meth:`quality_signal`.
        empirical_dist, predicted_dist:
            Inputs to :meth:`quality_signal_kl`. Either may be ``None``;
            in that case the blended call falls back to pure heuristic
            quality regardless of ``blend`` (this is the curriculum-friendly
            fallback so callers can drop it in unconditionally before the
            empirical histogram is reliable).
        blend:
            Convex weight on the KL term in [0, 1]. ``0.0`` => pure
            heuristic; ``1.0`` => pure KL. A typical curriculum decays
            ``blend`` from 1.0 (cold-start: rely on heuristic) toward 0.0 as
            the empirical histogram becomes reliable -- but here we expose
            it as a free parameter so the schedule can be set externally.

        Returns
        -------
        float
            ``(1 - blend) * Q_heuristic + blend * Q_kl`` when both
            distributions are provided; otherwise ``Q_heuristic``.
        """
        if not (0.0 <= float(blend) <= 1.0):
            raise ValueError(f"blend must be in [0, 1], got {blend}")
        q_heuristic = self.quality_signal(
            energy_total, sigma, contradiction, loop_risk
        )
        # Curriculum-friendly fallback: when either distribution is missing,
        # the KL term is undefined, so we silently degrade to pure heuristic
        # regardless of the requested blend. This lets callers set blend on a
        # schedule without having to gate the call site on data availability.
        if empirical_dist is None or predicted_dist is None:
            return float(q_heuristic)
        q_kl = self.quality_signal_kl(empirical_dist, predicted_dist)
        b = float(blend)
        return float((1.0 - b) * q_heuristic + b * q_kl)

    # ------------------------------------------------------------------
    # Phase 7 -- Fisher-natural gradient
    # ------------------------------------------------------------------

    def natural_gradient_update(
        self,
        gradient_alpha: np.ndarray,
        gradient_beta: np.ndarray,
        damping: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fisher-natural gradient on the posterior mask.

        Parameters
        ----------
        gradient_alpha:
            ``(V, V)`` Euclidean gradient on alpha, matching the shape of
            ``self._posterior.alpha``.
        gradient_beta:
            ``(V, V)`` Euclidean gradient on beta.
        damping:
            Tikhonov damping in the natural-gradient denominator. Must be
            non-negative; 1e-6 is a robust default.

        Returns
        -------
        (nat_grad_alpha, nat_grad_beta)
            Each is the input gradient divided by ``F + damping`` where
            ``F = alpha + beta`` is the diagonal Fisher information per
            edge (Beta concentration). Edges with little data have small
            ``F`` and therefore receive amplified updates; well-determined
            edges receive damped updates.

        Notes
        -----
        The natural-gradient learning rate is geometry-derived: it is a
        per-edge auto-adaptive step from the local information-manifold
        curvature. No explicit learning-rate schedule is needed.
        """
        ga = np.asarray(gradient_alpha, dtype=np.float64)
        gb = np.asarray(gradient_beta, dtype=np.float64)
        target_shape = self._posterior.alpha.shape
        if ga.shape != target_shape or gb.shape != target_shape:
            raise ValueError(
                f"gradient_alpha and gradient_beta must have shape "
                f"{target_shape}; got {ga.shape} and {gb.shape}"
            )
        f_diag = fisher_information_diagonal_mask(
            self._posterior.alpha, self._posterior.beta
        )
        nat_a = natural_gradient_diagonal(ga, f_diag, damping=damping)
        nat_b = natural_gradient_diagonal(gb, f_diag, damping=damping)
        return nat_a, nat_b

    # ------------------------------------------------------------------
    # Phase 7 -- Cramer-Rao confidence
    # ------------------------------------------------------------------

    def crb_confidence(
        self,
        edge_src: int,
        edge_dst: int,
        target_variance: float = 0.01,
    ) -> float:
        """Cramer-Rao confidence for one edge.

        Parameters
        ----------
        edge_src, edge_dst:
            Source and destination vertex indices.
        target_variance:
            Variance ceiling we want any unbiased estimator of the edge
            probability to clear. Smaller targets demand more data.

        Returns
        -------
        float
            ``n / n_required``, where ``n = alpha + beta - 2`` is the
            effective sample count (subtracting the Beta(1, 1) cold-start
            prior so a never-observed edge has n = 0) and ``n_required``
            is the CRB-derived sample count for the requested
            ``target_variance``. Values >= 1.0 mean the CRB target is
            satisfied; values near 0 mean the edge is essentially
            undetermined.
        """
        v = self._posterior.n_vertices
        if not (0 <= edge_src < v and 0 <= edge_dst < v):
            raise IndexError(
                f"(edge_src, edge_dst) = ({edge_src}, {edge_dst}) out of "
                f"range for V={v}"
            )
        a = float(self._posterior.alpha[edge_src, edge_dst])
        b = float(self._posterior.beta[edge_src, edge_dst])
        # Effective sample count: subtract Beta(1, 1) cold-start prior so a
        # never-observed edge has n = 0 and crb_confidence returns 0.
        n_eff = max(a + b - 2.0, 0.0)
        if n_eff <= 0.0:
            return 0.0
        p_bar = a / (a + b)
        fisher = float(fisher_information_bernoulli(p_bar))
        # crb_confidence_for_observations expects integer n; we pass the
        # effective sample count as a float by computing the ratio directly.
        n_req = 1.0 / (float(target_variance) * fisher)
        return float(n_eff / n_req)

    def crb_confidence_matrix(
        self,
        target_variance: float = 0.01,
    ) -> np.ndarray:
        """Per-edge CRB confidence over the entire posterior mask.

        Parameters
        ----------
        target_variance:
            Variance ceiling per edge.

        Returns
        -------
        np.ndarray
            ``(V, V)`` array of CRB confidences with the same shape as
            ``self._posterior.alpha``. Entry [i, j] is :meth:`crb_confidence`
            applied to edge (i, j).

        Notes
        -----
        Use as a diagnostic / stopping rule: training is "done" for edge
        (i, j) when ``crb_confidence_matrix[i, j] >= 1.0``. The minimum
        across edges (or some quantile of it) is a global progress signal.
        """
        if target_variance <= 0.0:
            raise ValueError(
                f"target_variance must be positive, got {target_variance}"
            )
        a = self._posterior.alpha
        b = self._posterior.beta
        n_eff = np.maximum(a + b - 2.0, 0.0)
        p_bar = a / (a + b)
        # Bernoulli Fisher info clipped at the boundary by the helper.
        fisher = fisher_information_bernoulli(p_bar)
        n_req = 1.0 / (float(target_variance) * fisher)
        out = np.where(n_eff > 0.0, n_eff / n_req, 0.0)
        return np.asarray(out, dtype=np.float64)

    # ------------------------------------------------------------------
    # Phase 7 -- KL convergence diagnostic
    # ------------------------------------------------------------------

    def kl_progress(
        self,
        previous_alpha: np.ndarray,
        previous_beta: np.ndarray,
    ) -> float:
        """Mean per-edge KL(previous Beta || current Beta).

        Parameters
        ----------
        previous_alpha, previous_beta:
            Snapshots of the posterior alpha and beta tensors at the start
            of the window. Must be the same shape as the current posterior.

        Returns
        -------
        float
            Mean across all edges of
            ``KL(Beta(prev_alpha[i, j], prev_beta[i, j]) ||
                Beta(curr_alpha[i, j], curr_beta[i, j]))``.

        Notes
        -----
        Use as a convergence diagnostic: when the mean KL between
        consecutive posterior snapshots drops below threshold, training
        has converged in the Fisher-Rao sense.
        """
        prev_a = np.asarray(previous_alpha, dtype=np.float64)
        prev_b = np.asarray(previous_beta, dtype=np.float64)
        target_shape = self._posterior.alpha.shape
        if prev_a.shape != target_shape or prev_b.shape != target_shape:
            raise ValueError(
                f"previous_alpha and previous_beta must have shape "
                f"{target_shape}; got {prev_a.shape} and {prev_b.shape}"
            )
        curr_a = self._posterior.alpha
        curr_b = self._posterior.beta
        total = 0.0
        n = 0
        # Per-edge closed-form Beta KL; no broadcasting available in
        # information_geometry.kl_beta, so loop. Posterior mask sizes are
        # small in practice (V x V).
        flat_pa = prev_a.ravel()
        flat_pb = prev_b.ravel()
        flat_ca = curr_a.ravel()
        flat_cb = curr_b.ravel()
        for i in range(flat_pa.shape[0]):
            total += kl_beta(
                float(flat_pa[i]),
                float(flat_pb[i]),
                float(flat_ca[i]),
                float(flat_cb[i]),
            )
            n += 1
        if n == 0:
            return 0.0
        return float(total / n)

    # ------------------------------------------------------------------
    # Energy at observation
    # ------------------------------------------------------------------

    def _energy_at(
        self,
        observation: np.ndarray,
        prototype: np.ndarray,
        sigma: float,
        contradiction: float,
        loop_risk: float,
        cost: float,
        progress: float,
    ) -> float:
        """Total energy E(obs, proto, ...) using the shared EnergyFunction.

        The hyperbolic distance to the prototype is converted to an
        ``uncertainty`` value in [0, 1] via :func:`_uncertainty_from_distance`.
        ``sigma`` is folded into ``loop_pressure`` as an additive proxy --
        the EnergyFunction has no first-class sigma channel and uncertainty
        is reserved for the prototype-distance term.
        """
        d = float(poincare_distance(observation, prototype))
        unc = _uncertainty_from_distance(d)
        # Feed loop_risk and sigma both into loop_pressure (clipped in compute).
        # sigma is a confidence-residual signal in [0, 1].
        lp = float(np.clip(0.5 * loop_risk + 0.5 * sigma, 0.0, 1.0))
        contribs = self._energy_fn.compute(
            cost=cost,
            uncertainty=unc,
            contradiction=contradiction,
            loop_pressure=lp,
            progress=progress,
        )
        return float(contribs.total)

    # ------------------------------------------------------------------
    # Single-step update
    # ------------------------------------------------------------------

    def step(
        self,
        src_state: int,
        dst_state: int,
        observation_in_poincare: np.ndarray,
        observation_type: Hashable,
        prototype_idx: int,
        sigma: float,
        contradiction: float,
        loop_risk: float,
        cost: float = 0.0,
        progress: float = 1.0,
    ) -> TrainerStepResult:
        """Perform one Riemannian-SGD + Beta-update + (optional) orbit step.

        Parameters
        ----------
        src_state, dst_state:
            Source and destination vertex indices for the posterior update.
        observation_in_poincare:
            (dim,) observation already embedded into the Poincare ball.
        observation_type:
            Key into the ``prototypes`` dict selecting which type/orbit owns
            this observation.
        prototype_idx:
            Row index into ``prototypes[observation_type]`` selecting the
            prototype the observation is bound to.
        sigma:
            Singularity-detector residual signal in [0, 1].
        contradiction:
            0 or 1 -- whether the transition contradicts the legality oracle.
        loop_risk:
            Loop-pressure score from TaggerHistory in [0, 1].
        cost, progress:
            Compute-cost and task-progress proxies in [0, 1].

        Returns
        -------
        TrainerStepResult
            Diagnostics for the step.
        """
        if observation_type not in self._prototypes:
            raise KeyError(
                f"observation_type {observation_type!r} not in prototypes "
                f"(have {list(self._prototypes.keys())})"
            )
        proto_table = self._prototypes[observation_type]
        if not (0 <= prototype_idx < proto_table.shape[0]):
            raise IndexError(
                f"prototype_idx {prototype_idx} out of range for type "
                f"{observation_type!r} (K={proto_table.shape[0]})"
            )

        obs = np.asarray(observation_in_poincare, dtype=np.float64)
        if obs.ndim != 1 or obs.shape[0] != proto_table.shape[1]:
            raise ValueError(
                f"observation_in_poincare must have shape (dim,) matching "
                f"prototypes[{observation_type!r}].shape[1]={proto_table.shape[1]}; "
                f"got {obs.shape}"
            )
        # Project the observation defensively in case it skirts the boundary.
        obs = project(obs)

        proto_before = proto_table[prototype_idx].copy()

        # ------------------------------------------------------------------
        # 1. Energy at the observation BEFORE the update.
        # ------------------------------------------------------------------
        energy_observed = self._energy_at(
            obs, proto_before, sigma, contradiction, loop_risk, cost, progress
        )

        # ------------------------------------------------------------------
        # 2. Riemannian SGD step on the prototype.
        #
        # The energy increases with hyperbolic distance ||obs - proto|| through
        # the uncertainty channel. With uncertainty(d) = 1 - exp(-d), the
        # ambient gradient of E w.r.t. the prototype position p, holding obs
        # fixed, is
        #
        #     dE/dp  =  beta_uncertainty * d/dp [1 - exp(-d(obs, p))]
        #            =  -beta_uncertainty * exp(-d) * d/dp [d(obs, p)].
        #
        # We approximate d/dp d(obs, p) by the (Euclidean) direction
        # (p - obs) / ||p - obs||_2, normalised by the local conformal scale.
        # ``riemannian_gradient`` then converts to the Poincare metric. The
        # factored result is "pull the prototype toward the observation,
        # with magnitude proportional to beta * exp(-d)".
        # ------------------------------------------------------------------
        beta_unc = float(self._energy_fn._params.beta)
        d_hyp = float(poincare_distance(obs, proto_before))
        diff = proto_before - obs
        diff_norm = float(np.linalg.norm(diff))
        if diff_norm < 1e-12:
            euclidean_grad = np.zeros_like(proto_before)
        else:
            direction = diff / diff_norm
            # exp(-d) gives a saturating magnitude; we scale by beta_unc.
            euclidean_grad = beta_unc * np.exp(-d_hyp) * direction

        proto_after = rsgd_step(
            proto_before, euclidean_grad, lr=self._config.lr_prototype
        )
        proto_table[prototype_idx] = proto_after
        prototype_shift = float(poincare_distance(proto_before, proto_after))

        # ------------------------------------------------------------------
        # 3. Energy at the observation AFTER the prototype shift.
        # ------------------------------------------------------------------
        energy_after = self._energy_at(
            obs, proto_after, sigma, contradiction, loop_risk, cost, progress
        )

        # ------------------------------------------------------------------
        # 4. Beta-posterior update from a single Q in [0, 1].
        # ------------------------------------------------------------------
        q = self.quality_signal(energy_after, sigma, contradiction, loop_risk)
        # We update with the post-shift quality so the trainer's two
        # subsystems are coherent: the prototype just moved, and the
        # posterior should be informed by the post-move evidence.
        self._posterior.update(int(src_state), int(dst_state), q)
        alpha_delta = q
        beta_delta = 1.0 - q

        # ------------------------------------------------------------------
        # 5. Orbit sharing: optional, no-op when no group action is attached.
        # ------------------------------------------------------------------
        if self._config.enforce_orbit_sharing and self._group_action is not None:
            self.enforce_orbit_sharing()

        return TrainerStepResult(
            energy_observed=float(energy_observed),
            energy_after_update=float(energy_after),
            quality=float(q),
            posterior_alpha_beta_delta=(float(alpha_delta), float(beta_delta)),
            prototype_shift_magnitude=float(prototype_shift),
            monodromy_loss=float(self._last_monodromy_loss),
        )

    # ------------------------------------------------------------------
    # Orbit-equivariance enforcement
    # ------------------------------------------------------------------

    def enforce_orbit_sharing(self) -> None:
        """Average prototypes within each orbit, redistribute, project.

        Implementation: for each orbit produced by ``decompose_orbits`` on
        the attached :class:`GroupAction`, walk the prototype tables looking
        for entries whose ROW index matches an orbit member; average them in
        the tangent space at the origin (``log_map_zero``), then write the
        averaged tangent vector back through ``exp_map_zero`` to every member
        of the orbit. Prototype tables whose rows do not span the orbit
        member set are left untouched.

        No-op when ``self._group_action is None``.
        """
        if self._group_action is None:
            return

        decomposition = decompose_orbits(self._group_action)

        for type_id, table in self._prototypes.items():
            n_rows = table.shape[0]
            for orbit in decomposition.orbits:
                if len(orbit) <= 1:
                    continue
                # Only act on orbits whose member indices all index this
                # prototype table. This keeps the operation a no-op for
                # tables that don't represent vertex-aligned prototypes.
                if any(v < 0 or v >= n_rows for v in orbit):
                    continue
                rows = table[orbit, :]  # (k, dim)
                tangents = log_map_zero(rows)  # (k, dim)
                mean_tangent = np.mean(tangents, axis=0, keepdims=True)
                shared = exp_map_zero(mean_tangent)  # (1, dim)
                shared = project(shared)
                for v in orbit:
                    table[v, :] = shared[0]

    # ------------------------------------------------------------------
    # Closed-walk consistency
    # ------------------------------------------------------------------

    def closed_walk_loss(
        self,
        walks: list[list[int]],
        energies_by_walk: list[np.ndarray],
    ) -> float:
        """Wrap :func:`monodromy_consistency_loss` with internal bookkeeping.

        The trainer caches the most recently computed value so it can be
        emitted in :class:`TrainerStepResult` without recomputation.
        """
        loss = monodromy_consistency_loss(walks, energies_by_walk)
        self._last_monodromy_loss = float(loss)
        return float(loss)
