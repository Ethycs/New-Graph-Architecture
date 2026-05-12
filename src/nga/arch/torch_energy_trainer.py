"""Torch-backed end-to-end energy-minimisation trainer (Phase 9).

This atom is the gradient-trained sibling of
:class:`nga.arch.energy_minimization_trainer.EnergyMinimizationTrainer`.

Where the classical trainer interleaves Riemannian-SGD on numpy prototype
arrays with closed-form Beta-posterior updates on the
:class:`PosteriorMask`, this atom lifts every learnable quantity into a
single ``nn.Module`` whose ``forward`` IS the energy at the observed
(input -> true_next_state) transition. Backprop through that scalar walks
the gradient back through

* the **prototypes** (``nn.Parameter`` of shape ``(n_states,
  embedding_dim)`` living in the open Poincare ball),
* the posterior **legality bias** in the form of two ``nn.Parameter``
  tensors ``alpha_log`` and ``beta_log`` (shape ``(n_states, n_states)``;
  log-space for stability so the parameters live in R rather than R_+),
* the typed_readout heads' weights (passed in as a
  :class:`nga.arch.typed_readout_torch.TypedReadoutTorch`; its heads'
  parameters become trainable members of this module),

while the encoder
(:class:`nga.arch.frozen_encoder_torch.FrozenEncoderTorch`) is preserved
as frozen-by-construction.

The energy formula is the differentiable mirror of
``nga.arch.energy_function.EnergyFunction``:

    energy = alpha_cost * cost
           + beta_uncertainty  * poincare_distance(obs, prototype_chosen)
           + gamma_contradiction * contradiction
           + eta_loop          * loop_pressure
           - kappa_progress    * progress

The training loss is the mean energy at the *true* next-state prototype
plus an optional KL term measuring the discrepancy between the model's
softmax-over-prototype-distance distribution and the one-hot true
distribution.

Differentiability subtleties
----------------------------
* The Poincare distance is implemented in torch via the arccosh form:

      d(x, y) = arccosh(1 + 2 ||x - y||^2 / ((1 - ||x||^2) (1 - ||y||^2)))

  Three numerical guards are critical for finite gradients:

  - The squared-norms of x and y are clipped to <= ``1 - eps`` so the
    denominator stays bounded away from zero.
  - The arccosh argument is clipped to >= ``1 + eps`` so the derivative
    ``1 / sqrt(arg^2 - 1)`` does not blow up at the diagonal.
  - The point coordinates themselves are softly projected via a
    norm-rescaling that keeps ``||p|| <= 1 - eps`` *with gradients*
    flowing through the rescale (the rescale uses
    ``torch.where(||p|| > 1 - eps, (1 - eps)/||p||, 1.0)``, which is
    differentiable everywhere except at the boundary, and we never
    actually hit the boundary).

* The legality bias is the differentiable substitute for the
  closed-form Beta-mean log-odds: ``bias[i, j] = alpha_log[i, j] -
  beta_log[i, j]`` is the model's idea of ``log(alpha) - log(beta)``,
  which is the log-odds of the posterior mean. Gradients on this bias
  flow back into both ``alpha_log`` and ``beta_log`` symmetrically.
  ``posterior_mean()`` recovers the Bernoulli mean as
  ``sigmoid(alpha_log - beta_log)``.

The classical trainer's snapshot is one-way: a torch trainer can emit a
:class:`PosteriorMask` for downstream analysis tools, but those tools'
results do not feed back into the gradient graph. The constructor accepts
a ``posterior_alpha_init``/``posterior_beta_init`` pair so a numpy
PosteriorMask can warm-start the torch trainer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

try:  # Lazy / optional torch dependency.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only on torch-less envs.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from nga.arch.posterior_mask import PosteriorMask

if TYPE_CHECKING:  # pragma: no cover - import-only typing.
    from nga.arch.frozen_encoder_torch import FrozenEncoderTorch
    from nga.arch.typed_readout_torch import TypedReadoutTorch

__all__ = ["TorchTrainerConfig", "TorchEnergyTrainer"]

_TORCH_MISSING_MSG = (
    "torch is required for TorchEnergyTrainer; install via "
    "`pixi add pytorch` (or `pip install torch`)."
)

# Boundary epsilons. Tighter than numpy hyperbolic_embedding's because we
# also need the gradient through arccosh' to be finite at arg = 1 + eps.
_EPS = 1e-5
_MAX_NORM = 1.0 - 1e-4


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TorchTrainerConfig:
    """Hyperparameters for :class:`TorchEnergyTrainer`.

    Fields mirror :class:`nga.arch.energy_function.EnergyParameters` for
    the per-term energy weights; the ``lambda_*`` and ``grad_clip`` fields
    are trainer-side knobs that the classical trainer did not need.
    """

    lr: float = 1e-3
    energy_alpha_cost: float = 1.0
    energy_beta_uncertainty: float = 1.0
    energy_gamma_contradiction: float = 1.0
    energy_eta_loop: float = 1.0
    energy_kappa_progress: float = 1.0
    lambda_kl: float = 1.0
    lambda_monodromy: float = 0.1
    grad_clip: float = 5.0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TorchEnergyTrainer(nn.Module if nn is not None else object):
    """End-to-end gradient-trained energy minimiser.

    Owns one ``nn.Module``-shaped graph whose forward is the energy at the
    (observation -> true_next_state) point and whose backward updates
    prototypes, posterior log-counts, and any readout-head weights through
    a single Adam optimiser.
    """

    def __init__(
        self,
        n_states: int,
        embedding_dim: int = 16,
        encoder_output_dim: int = 16,
        prototype_init: "torch.Tensor | None" = None,
        encoder: "FrozenEncoderTorch | None" = None,
        readout: "TypedReadoutTorch | None" = None,
        posterior_alpha_init: "torch.Tensor | None" = None,
        posterior_beta_init: "torch.Tensor | None" = None,
        config: TorchTrainerConfig | None = None,
        seed: int = 0,
    ) -> None:
        if torch is None or nn is None:
            raise ImportError(_TORCH_MISSING_MSG)
        super().__init__()
        if n_states <= 0:
            raise ValueError(f"n_states must be positive, got {n_states}")
        if embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {embedding_dim}"
            )
        if encoder_output_dim <= 0:
            raise ValueError(
                f"encoder_output_dim must be positive, got {encoder_output_dim}"
            )

        self.n_states: int = int(n_states)
        self.embedding_dim: int = int(embedding_dim)
        self.encoder_output_dim: int = int(encoder_output_dim)
        self._config: TorchTrainerConfig = (
            config if config is not None else TorchTrainerConfig()
        )
        # Seed BEFORE prototype init so the random fallback is reproducible.
        torch.manual_seed(int(seed))

        # ------------------------------------------------------------------
        # Prototypes -- nn.Parameter on the Poincare ball.
        # ------------------------------------------------------------------
        if prototype_init is None:
            # Small random init so the projection in poincare_distance is a
            # no-op and gradients are well-defined from step 1.
            init = 0.01 * torch.randn(
                self.n_states, self.embedding_dim, dtype=torch.float32
            )
        else:
            init = torch.as_tensor(prototype_init, dtype=torch.float32)
            if init.shape != (self.n_states, self.embedding_dim):
                raise ValueError(
                    f"prototype_init must have shape "
                    f"({self.n_states}, {self.embedding_dim}); got {tuple(init.shape)}"
                )
        # Soft-project at construction time so the parameter starts inside
        # the open ball; we do NOT clamp at every forward (the
        # poincare_distance helper handles that with autograd-friendly ops).
        init = self._poincare_project(init).detach().clone()
        self.prototypes = nn.Parameter(init)

        # ------------------------------------------------------------------
        # Posterior log-space parameters.
        # ------------------------------------------------------------------
        shape = (self.n_states, self.n_states)
        if posterior_alpha_init is None:
            alpha0 = torch.zeros(shape, dtype=torch.float32)
        else:
            alpha0 = torch.as_tensor(posterior_alpha_init, dtype=torch.float32)
            if alpha0.shape != shape:
                raise ValueError(
                    f"posterior_alpha_init must have shape {shape}; got "
                    f"{tuple(alpha0.shape)}"
                )
            # Convert numpy alpha (>0) to log-space; if user passes log-space
            # already (e.g. zeros) we still need positivity, so we accept any
            # finite tensor and treat negatives as already-log.
            if torch.all(alpha0 > 0):
                alpha0 = torch.log(alpha0)
        if posterior_beta_init is None:
            beta0 = torch.zeros(shape, dtype=torch.float32)
        else:
            beta0 = torch.as_tensor(posterior_beta_init, dtype=torch.float32)
            if beta0.shape != shape:
                raise ValueError(
                    f"posterior_beta_init must have shape {shape}; got "
                    f"{tuple(beta0.shape)}"
                )
            if torch.all(beta0 > 0):
                beta0 = torch.log(beta0)
        self.alpha_log = nn.Parameter(alpha0.clone())
        self.beta_log = nn.Parameter(beta0.clone())

        # ------------------------------------------------------------------
        # Encoder (frozen) and readout (heads contribute parameters).
        # ------------------------------------------------------------------
        # We don't import these modules at module-load time to keep the
        # sklearn-only import path lean; resolve them lazily here.
        self._encoder = encoder
        self._readout = readout

        # Register frozen-encoder parameters as a submodule so a parent
        # module sees them (with requires_grad=False) but Adam ignores them.
        if encoder is not None:
            from nga.arch.frozen_encoder_torch import FrozenEncoderTorch

            if not isinstance(encoder, FrozenEncoderTorch):
                raise TypeError(
                    f"encoder must be FrozenEncoderTorch, got {type(encoder)}"
                )
            # The underlying nn.Sequential already has requires_grad=False.
            self._encoder_net = encoder._net
            for p in self._encoder_net.parameters():
                p.requires_grad_(False)
        else:
            self._encoder_net = None

        if readout is not None:
            from nga.arch.typed_readout_torch import TypedReadoutTorch

            if not isinstance(readout, TypedReadoutTorch):
                raise TypeError(
                    f"readout must be TypedReadoutTorch, got {type(readout)}"
                )
            # Register every head as a child module so Adam picks them up.
            self._readout_heads = nn.ModuleDict(
                {str(t): readout._heads[t] for t in readout.type_ids}
            )
        else:
            self._readout_heads = None

        # ------------------------------------------------------------------
        # Optimiser. Built AFTER parameters so it sees them all.
        # ------------------------------------------------------------------
        self._optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self._config.lr,
        )

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def config(self) -> TorchTrainerConfig:
        return self._config

    @property
    def optimizer(self) -> "torch.optim.Optimizer":
        return self._optimizer

    @property
    def encoder(self) -> "FrozenEncoderTorch | None":
        return self._encoder

    @property
    def readout(self) -> "TypedReadoutTorch | None":
        return self._readout

    # ------------------------------------------------------------------
    # Differentiable Poincare ops
    # ------------------------------------------------------------------

    @staticmethod
    def _poincare_project(p: "torch.Tensor") -> "torch.Tensor":
        """Soft-project ``p`` to ``||p|| <= 1 - eps`` with gradients flowing.

        Uses ``where(norm > max_norm, max_norm/norm, 1.0)`` so the rescale
        factor is differentiable in the interior (no-op) and the boundary
        case has a well-defined direction. Critically, we never *clamp* the
        norm with ``.clamp_`` -- that would cut gradients.
        """
        norms = torch.linalg.norm(p, dim=-1, keepdim=True)
        # Guard against zero norms before division.
        safe_norms = torch.clamp(norms, min=_EPS)
        factor = torch.where(
            norms > _MAX_NORM,
            torch.full_like(norms, _MAX_NORM) / safe_norms,
            torch.ones_like(norms),
        )
        return p * factor

    @classmethod
    def poincare_distance_torch(
        cls, x: "torch.Tensor", y: "torch.Tensor"
    ) -> "torch.Tensor":
        """Differentiable Poincare-ball distance via the arccosh form.

        Supports
        - ``x: (d,), y: (d,)`` -> scalar.
        - ``x: (n, d), y: (n, d)`` -> ``(n,)`` element-wise.
        - ``x: (n, d), y: (m, d)`` -> ``(n, m)`` all-pairs.

        Numerically guarded so gradients stay finite:

        * point norms clamped to ``[0, 1 - eps]`` BEFORE the
          ``1 - ||p||^2`` denominator so the denominator stays bounded
          away from zero;
        * the arccosh argument floored at ``1 + eps`` so the
          ``1/sqrt(arg^2 - 1)`` factor in arccosh' is finite at the
          diagonal where ``x == y`` and ``diff == 0``.
        """
        x1d = x.dim() == 1
        y1d = y.dim() == 1
        if x1d:
            x = x.unsqueeze(0)
        if y1d:
            y = y.unsqueeze(0)
        # If shapes match we want elementwise; otherwise broadcast (n, m).
        elementwise = x.shape == y.shape and not (x1d ^ y1d)
        if elementwise:
            diff = x - y
            diff_sq = (diff * diff).sum(dim=-1)
            x_sq = (x * x).sum(dim=-1).clamp(0.0, 1.0 - _EPS)
            y_sq = (y * y).sum(dim=-1).clamp(0.0, 1.0 - _EPS)
            denom = (1.0 - x_sq) * (1.0 - y_sq)
            denom = denom.clamp(min=_EPS)
            arg = 1.0 + 2.0 * diff_sq / denom
            arg = arg.clamp(min=1.0 + _EPS)
            d = torch.acosh(arg)
            if x1d and y1d:
                return d.squeeze(0)
            return d

        # Cross case: (n, d) vs (m, d) -> (n, m).
        x_exp = x.unsqueeze(1)  # (n, 1, d)
        y_exp = y.unsqueeze(0)  # (1, m, d)
        diff = x_exp - y_exp
        diff_sq = (diff * diff).sum(dim=-1)  # (n, m)
        x_sq = (x * x).sum(dim=-1).clamp(0.0, 1.0 - _EPS)  # (n,)
        y_sq = (y * y).sum(dim=-1).clamp(0.0, 1.0 - _EPS)  # (m,)
        denom = (1.0 - x_sq).unsqueeze(1) * (1.0 - y_sq).unsqueeze(0)
        denom = denom.clamp(min=_EPS)
        arg = 1.0 + 2.0 * diff_sq / denom
        arg = arg.clamp(min=1.0 + _EPS)
        d = torch.acosh(arg)
        if x1d:
            d = d.squeeze(0)
        if y1d:
            d = d.squeeze(-1)
        return d

    # ------------------------------------------------------------------
    # Posterior summaries
    # ------------------------------------------------------------------

    def posterior_mean(self) -> "torch.Tensor":
        """Bernoulli posterior mean ``sigmoid(alpha_log - beta_log)``.

        This is the differentiable analogue of
        :meth:`PosteriorMask.posterior_mean`. At cold-start
        ``alpha_log = beta_log = 0`` it is uniform 0.5 everywhere.
        """
        return torch.sigmoid(self.alpha_log - self.beta_log)

    def legality_bias(self) -> "torch.Tensor":
        """Pre-softmax additive bias in log-odds form: ``alpha_log - beta_log``.

        This is the gradient-trained substitute for
        :meth:`PosteriorMask.legality_bias`. No clipping (the underlying
        log-counts can drift); the model learns the magnitude.
        """
        return self.alpha_log - self.beta_log

    def legality_matrix(self, threshold: float = 0.5) -> "torch.Tensor":
        """Boolean mask, ``posterior_mean > threshold``.

        Strict inequality matches :meth:`PosteriorMask.legality_matrix`'s
        skeptical-prior default: at cold-start every entry is exactly
        0.5 and therefore returns False.
        """
        return self.posterior_mean() > float(threshold)

    # ------------------------------------------------------------------
    # Predictive distribution
    # ------------------------------------------------------------------

    def predicted_distribution(
        self,
        observation: "torch.Tensor",
        current_state: int | "torch.Tensor",
    ) -> "torch.Tensor":
        """Softmax over distance-based logits with legality-bias offset.

        ``logit_j = -d_poincare(observation, prototype_j) +
        legality_bias[current_state, j]``.

        Differentiable in **prototypes** (via the distance) AND in
        **alpha_log/beta_log** (via the legality bias).

        Parameters
        ----------
        observation:
            ``(embedding_dim,)`` Poincare-ball point.
        current_state:
            Int row index into ``alpha_log`` / ``beta_log``.

        Returns
        -------
        torch.Tensor
            ``(n_states,)`` probability distribution.
        """
        if observation.dim() != 1:
            raise ValueError(
                f"observation must be 1-D (embedding_dim,); got shape "
                f"{tuple(observation.shape)}"
            )
        if observation.shape[0] != self.embedding_dim:
            raise ValueError(
                f"observation has {observation.shape[0]} dims but trainer "
                f"expects {self.embedding_dim}"
            )
        d = self.poincare_distance_torch(
            observation.unsqueeze(0), self.prototypes
        ).squeeze(0)  # (n_states,)
        bias = self.legality_bias()  # (n_states, n_states)
        cs = int(current_state) if not isinstance(current_state, torch.Tensor) else int(current_state.item())
        if not (0 <= cs < self.n_states):
            raise IndexError(
                f"current_state {cs} out of range for n_states={self.n_states}"
            )
        logits = -d + bias[cs]
        return torch.softmax(logits, dim=0)

    # ------------------------------------------------------------------
    # Energy at a (state, observation) point
    # ------------------------------------------------------------------

    def _energy_at_states(
        self,
        observation: "torch.Tensor",
        states: "torch.Tensor",
        cost: "torch.Tensor",
        progress: "torch.Tensor",
        contradiction: "torch.Tensor",
        loop_pressure: "torch.Tensor",
    ) -> "torch.Tensor":
        """Differentiable energy at the prototype indexed by ``states``.

        Parameters
        ----------
        observation:
            ``(B, embedding_dim)``.
        states:
            ``(B,)`` int.
        cost, progress, contradiction, loop_pressure:
            ``(B,)`` float in [0, 1] (clamped defensively).

        Returns
        -------
        torch.Tensor
            ``(B,)`` total energy.
        """
        cfg = self._config
        # Index the prototype tower by state; this is differentiable in
        # `prototypes` because torch index_select / advanced indexing keeps
        # the parameter in the autograd graph.
        protos = self.prototypes[states]  # (B, embedding_dim)
        # Element-wise distance over the batch.
        unc = self.poincare_distance_torch(observation, protos)  # (B,)
        cost_c = cost.clamp(0.0, 1.0)
        contradiction_c = contradiction.clamp(0.0, 1.0)
        loop_c = loop_pressure.clamp(0.0, 1.0)
        progress_c = progress.clamp(0.0, 1.0)
        energy = (
            cfg.energy_alpha_cost * cost_c
            + cfg.energy_beta_uncertainty * unc
            + cfg.energy_gamma_contradiction * contradiction_c
            + cfg.energy_eta_loop * loop_c
            - cfg.energy_kappa_progress * progress_c
        )
        return energy

    # ------------------------------------------------------------------
    # Forward / step
    # ------------------------------------------------------------------

    def forward(
        self,
        observation: "torch.Tensor",
        current_state: "torch.Tensor",
        true_next_state: "torch.Tensor",
        cost: "torch.Tensor | None" = None,
        progress: "torch.Tensor | None" = None,
        contradiction: "torch.Tensor | None" = None,
        loop_pressure: "torch.Tensor | None" = None,
    ) -> dict[str, "torch.Tensor"]:
        """Compute energy, predicted distribution, KL term, and total loss.

        Returns a dict (also returned by ``step`` after backward+update)
        with the following entries:

        * ``loss`` -- scalar trainer objective.
        * ``energy_total`` -- ``(B,)`` energy at the truth prototype.
        * ``kl_term`` -- scalar KL(predicted || onehot(truth)).
        * ``quality`` -- ``(B,)`` ``exp(-energy_total)`` clamped to [0, 1].
        * ``predicted`` -- ``(B, n_states)`` softmax distribution.
        """
        if observation.dim() != 2:
            raise ValueError(
                f"observation must be 2-D (B, embedding_dim); got shape "
                f"{tuple(observation.shape)}"
            )
        if observation.shape[1] != self.embedding_dim:
            raise ValueError(
                f"observation has {observation.shape[1]} dims but trainer "
                f"expects {self.embedding_dim}"
            )
        b = observation.shape[0]
        if current_state.shape != (b,):
            raise ValueError(
                f"current_state must have shape ({b},); got "
                f"{tuple(current_state.shape)}"
            )
        if true_next_state.shape != (b,):
            raise ValueError(
                f"true_next_state must have shape ({b},); got "
                f"{tuple(true_next_state.shape)}"
            )

        device = observation.device
        zeros_b = torch.zeros(b, device=device, dtype=observation.dtype)
        ones_b = torch.ones(b, device=device, dtype=observation.dtype)
        cost_b = zeros_b if cost is None else cost.to(observation.dtype)
        progress_b = ones_b if progress is None else progress.to(observation.dtype)
        contradiction_b = zeros_b if contradiction is None else contradiction.to(observation.dtype)
        loop_b = zeros_b if loop_pressure is None else loop_pressure.to(observation.dtype)

        # --- Energy at the TRUTH prototype: this IS the loss kernel. ---
        states_long = true_next_state.long()
        energy_total = self._energy_at_states(
            observation, states_long, cost_b, progress_b, contradiction_b, loop_b
        )

        # --- Predicted distribution (per-row over the prototype tower). ---
        # Logits: -d(obs, all prototypes) + legality_bias[current_state].
        d_all = self.poincare_distance_torch(
            observation, self.prototypes
        )  # (B, n_states)
        bias = self.legality_bias()  # (n_states, n_states)
        cur_long = current_state.long()
        logits = -d_all + bias[cur_long]  # (B, n_states)
        log_probs = torch.log_softmax(logits, dim=1)
        predicted = torch.softmax(logits, dim=1)

        # KL(onehot(truth) || predicted) = -log p(truth). Differentiable
        # in prototypes (via d_all) AND in alpha_log/beta_log (via bias).
        # Using log_softmax + gather avoids an explicit one-hot construction
        # and is more numerically stable than log(predicted).
        truth_log_p = log_probs.gather(
            1, states_long.unsqueeze(1)
        ).squeeze(1)
        kl_term = -truth_log_p.mean()

        # Quality is a diagnostic, not a gradient target.
        with torch.no_grad():
            quality = torch.exp(-energy_total.clamp(min=0.0)).clamp(0.0, 1.0)

        loss = energy_total.mean() + self._config.lambda_kl * kl_term

        return {
            "loss": loss,
            "energy_total": energy_total,
            "kl_term": kl_term,
            "quality": quality,
            "predicted": predicted,
        }

    def step(
        self,
        observation: "torch.Tensor",
        current_state: "torch.Tensor",
        true_next_state: "torch.Tensor",
        cost: "torch.Tensor | None" = None,
        progress: "torch.Tensor | None" = None,
        contradiction: "torch.Tensor | None" = None,
        loop_pressure: "torch.Tensor | None" = None,
    ) -> dict[str, "torch.Tensor"]:
        """One forward + backward + optimiser step.

        Returns the same dict as :meth:`forward`, augmented with
        ``prototype_grad_norm`` and ``posterior_grad_norm`` (post-clip).
        """
        self._optimizer.zero_grad()
        out = self.forward(
            observation,
            current_state,
            true_next_state,
            cost=cost,
            progress=progress,
            contradiction=contradiction,
            loop_pressure=loop_pressure,
        )
        out["loss"].backward()
        # Grad clipping AFTER backward, BEFORE step. Single-shot total-norm
        # clip across every trainable parameter.
        if self._config.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.parameters() if p.requires_grad],
                max_norm=self._config.grad_clip,
            )
        # Snapshot grad norms for the diagnostic dict (post-clip view).
        proto_grad_norm = (
            self.prototypes.grad.detach().norm()
            if self.prototypes.grad is not None
            else torch.tensor(0.0)
        )
        post_grad_norm = (
            torch.sqrt(
                (self.alpha_log.grad.detach() ** 2).sum()
                + (self.beta_log.grad.detach() ** 2).sum()
            )
            if (self.alpha_log.grad is not None and self.beta_log.grad is not None)
            else torch.tensor(0.0)
        )
        self._optimizer.step()

        # Soft-reproject prototypes back into the open ball (in-place but
        # under no_grad so the parameter is a fresh tensor with the same
        # storage; this preserves Adam's running stats).
        with torch.no_grad():
            self.prototypes.data.copy_(self._poincare_project(self.prototypes.data))

        out["prototype_grad_norm"] = proto_grad_norm
        out["posterior_grad_norm"] = post_grad_norm
        return out

    # ------------------------------------------------------------------
    # Snapshot helpers (one-way: torch -> numpy)
    # ------------------------------------------------------------------

    def to_classical_posterior_mask(self) -> PosteriorMask:
        """Snapshot the torch parameters into a numpy :class:`PosteriorMask`.

        ``alpha = exp(alpha_log)``, ``beta = exp(beta_log)``. The numpy
        mask is independent of the torch parameters: subsequent training
        does not affect the snapshot, and edits to the snapshot do not
        affect the trainer. Round-trip via the constructor's
        ``posterior_alpha_init``/``posterior_beta_init`` arguments
        recovers the same posterior mean.
        """
        with torch.no_grad():
            alpha = torch.exp(self.alpha_log).detach().cpu().numpy()
            beta = torch.exp(self.beta_log).detach().cpu().numpy()
        mask = PosteriorMask(self.n_states, prior_alpha=1.0, prior_beta=1.0)
        # Override the priors with the snapshot. We touch the protected
        # storage directly (PosteriorMask exposes alpha/beta as read-only
        # properties intentionally; the snapshot path is the one supported
        # mutator).
        mask._alpha = np.asarray(alpha, dtype=np.float64)
        mask._beta = np.asarray(beta, dtype=np.float64)
        return mask
