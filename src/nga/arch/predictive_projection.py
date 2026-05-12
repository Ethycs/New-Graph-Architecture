"""Predictive projection: arbitrary-network activation -> regime-space z.

Phase 23 / PCG-Extractor atom. The reframing for Phase 23 (see
``research_log2.md``):

    activation trajectories
      -> predictive probe-space
      -> certified-ish cells
      -> transition graph
      -> behavioral quotient
      -> intervention-conditioned control graph

This atom owns the *first* step: turn arbitrary hidden states ``h`` into
a regime-space representation ``z`` whose argmax decision cells are
"the regimes the network actually visits." Three small probe heads sit
on top of ``z``:

* a **future / next-state head** predicting where the trajectory goes
  next (the partition signal),
* an **entropy regression head** predicting the entropy of the
  next-state distribution at this step (the Morse-lite uncertainty
  signal),
* a **failure head** predicting whether the next step is wrong /
  illegal / out-of-distribution (the risk signal).

Training combines the three losses; the projection ``z`` is the shared
representation. Partitioning by ``argmax`` of the next-state head over
the cells gives polyhedral cells (tropical-lite). Per-cell entropy and
failure-probability are emitted as regime annotations.

Optional adversarial token-prediction head with gradient reversal:
disabled by default (the natural Wave-C / E26 trainable encoder we
have today does not need it because the encoder's input already
includes ``current_state``; a token head with reversed gradient on
``z`` is the structurally correct add-on for arbitrary frozen
base networks, and is exposed as a flag for Phase 23+ work).

The whole atom is intentionally cheap: a single-hidden-layer MLP
projection with three small linear heads. Speed matters more than
expressiveness because the regime-extraction pipeline runs probes on
top of every layer of any base network the user wants to analyse.
"""
from __future__ import annotations

import resource
from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

__all__ = [
    "PredictiveProjection",
    "PredictiveProjectionConfig",
    "train_predictive_projection",
]


@dataclass(frozen=True)
class PredictiveProjectionConfig:
    """Hyperparameters and switches for :class:`PredictiveProjection`.

    Attributes
    ----------
    z_dim:
        Width of the regime-space projection. Defaults to 32.
    hidden_dim:
        Hidden width inside the projection MLP. Defaults to 64.
    n_states:
        Cardinality of the future / next-state head's output.
    entropy_weight:
        Loss weight on the entropy-regression head. 0.0 disables the head.
    failure_weight:
        Loss weight on the failure head. 0.0 disables the head.
    adversarial_token_weight:
        Loss weight on the adversarial token-prediction head's gradient-
        reversed signal. 0.0 disables the adversarial head entirely.
        Non-zero requires ``n_tokens`` to be supplied.
    n_tokens:
        Vocab size for the adversarial token head; only used when
        ``adversarial_token_weight > 0``.
    """

    z_dim: int = 32
    hidden_dim: int = 64
    n_states: int = 1
    entropy_weight: float = 1.0
    failure_weight: float = 1.0
    adversarial_token_weight: float = 0.0
    n_tokens: int = 0


class _GradientReversal(torch.autograd.Function if torch is not None else object):
    """Standard gradient reversal layer (Ganin & Lempitsky 2015)."""

    @staticmethod
    def forward(ctx, x, lambda_):  # type: ignore[override]
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return grad_output.neg() * ctx.lambda_, None


def _grad_reverse(x, lambda_: float):
    return _GradientReversal.apply(x, lambda_)  # type: ignore[arg-type]


class PredictiveProjection(nn.Module if nn is not None else object):
    """Projection ``h -> z`` plus three (optional four) probe heads.

    The projection itself is ``Linear(input_dim, hidden_dim) -> ReLU ->
    Linear(hidden_dim, z_dim)`` -- one hidden layer. Probe heads are
    single linears off ``z``:

    * ``next_state_head`` -- logits over ``n_states``.
    * ``entropy_head``    -- scalar regression target (next-state entropy).
    * ``failure_head``    -- scalar logit (sigmoid -> failure probability).
    * ``token_head``      -- logits over ``n_tokens`` (adversarial, gradient-
      reversed before reaching ``z``).
    """

    def __init__(self, input_dim: int, config: PredictiveProjectionConfig) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for PredictiveProjection")
        super().__init__()
        self.cfg = config
        self.l1 = nn.Linear(input_dim, config.hidden_dim)
        self.l2 = nn.Linear(config.hidden_dim, config.z_dim)
        self.next_state_head = nn.Linear(config.z_dim, config.n_states)
        self.entropy_head = nn.Linear(config.z_dim, 1)
        self.failure_head = nn.Linear(config.z_dim, 1)
        if config.adversarial_token_weight > 0.0:
            if config.n_tokens <= 0:
                raise ValueError(
                    "adversarial_token_weight > 0 requires n_tokens > 0"
                )
            self.token_head: nn.Linear | None = nn.Linear(
                config.z_dim, config.n_tokens
            )
        else:
            self.token_head = None

    def project(self, h: "torch.Tensor") -> "torch.Tensor":
        """Compute ``z = projection(h)``."""
        return self.l2(torch.relu(self.l1(h)))

    def forward(
        self, h: "torch.Tensor"
    ) -> dict[str, "torch.Tensor"]:
        z = self.project(h)
        out: dict[str, "torch.Tensor"] = {
            "z": z,
            "next_state_logits": self.next_state_head(z),
            "entropy_pred": self.entropy_head(z).squeeze(-1),
            "failure_logit": self.failure_head(z).squeeze(-1),
        }
        if self.token_head is not None:
            z_rev = _grad_reverse(z, self.cfg.adversarial_token_weight)
            out["token_logits"] = self.token_head(z_rev)
        return out


def _peak_memory_kb() -> float:
    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:  # pragma: no cover
        return 0.0


def train_predictive_projection(
    projection: PredictiveProjection,
    h: np.ndarray,
    *,
    next_states: np.ndarray,
    entropy_targets: np.ndarray | None,
    failure_targets: np.ndarray | None,
    token_ids: np.ndarray | None = None,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 0,
) -> dict[str, float]:
    """Train the projection + heads jointly.

    Returns a dict of training-time diagnostics: ``final_loss``,
    ``next_state_acc``, ``entropy_mse``, ``failure_acc``, ``token_acc``
    (last three only present when the corresponding head was enabled).
    """
    if torch is None or nn is None:  # pragma: no cover
        raise ImportError("torch is required")
    cfg = projection.cfg
    n = h.shape[0]
    h_t = torch.from_numpy(h.astype(np.float32))
    y_t = torch.from_numpy(next_states.astype(np.int64))
    e_t = (
        torch.from_numpy(entropy_targets.astype(np.float32))
        if entropy_targets is not None and cfg.entropy_weight > 0.0
        else None
    )
    f_t = (
        torch.from_numpy(failure_targets.astype(np.float32))
        if failure_targets is not None and cfg.failure_weight > 0.0
        else None
    )
    tok_t = (
        torch.from_numpy(token_ids.astype(np.int64))
        if token_ids is not None and cfg.adversarial_token_weight > 0.0
        else None
    )

    opt = torch.optim.Adam(projection.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(int(seed))
    final = {"final_loss": float("nan")}
    for _epoch in range(int(epochs)):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            idx = torch.from_numpy(
                perm[start:stop].astype(np.int64)
            )
            opt.zero_grad()
            out = projection(h_t[idx])
            loss = ce(out["next_state_logits"], y_t[idx])
            if e_t is not None:
                loss = loss + cfg.entropy_weight * mse(
                    out["entropy_pred"], e_t[idx]
                )
            if f_t is not None:
                loss = loss + cfg.failure_weight * bce(
                    out["failure_logit"], f_t[idx]
                )
            if tok_t is not None and "token_logits" in out:
                loss = loss + ce(out["token_logits"], tok_t[idx])
            loss.backward()
            opt.step()
            final["final_loss"] = float(loss.item())
    projection.eval()
    with torch.no_grad():
        out = projection(h_t)
        ns_preds = out["next_state_logits"].argmax(dim=1)
        final["next_state_acc"] = float((ns_preds == y_t).float().mean().item())
        if e_t is not None:
            final["entropy_mse"] = float(
                ((out["entropy_pred"] - e_t) ** 2).mean().item()
            )
        if f_t is not None:
            f_pred = (out["failure_logit"] >= 0.0).float()
            final["failure_acc"] = float((f_pred == f_t).float().mean().item())
        if tok_t is not None and "token_logits" in out:
            tok_preds = out["token_logits"].argmax(dim=1)
            final["token_acc"] = float((tok_preds == tok_t).float().mean().item())
    return final
