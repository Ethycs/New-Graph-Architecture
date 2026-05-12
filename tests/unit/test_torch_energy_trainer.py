"""Unit tests for TorchEnergyTrainer (Phase 9 gradient-trained sibling).

Mirrors the contract assertions from
:mod:`tests.unit.test_energy_minimization_trainer` but in the torch
end-to-end gradient regime: every `nn.Parameter` is exercised by Adam
through a single ``loss.backward()`` call, and the test suite asserts
gradients flow to prototypes / posterior log-counts, encoder remains
frozen, posterior_mean stays a Bernoulli probability, the
classical-PosteriorMask round-trip preserves the mean, and seed
determinism + grad clipping both hold.

All tests are gated by ``pytest.importorskip("torch")``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from nga.arch.frozen_encoder_torch import FrozenEncoderTorch  # noqa: E402
from nga.arch.posterior_mask import PosteriorMask  # noqa: E402
from nga.arch.torch_energy_trainer import (  # noqa: E402
    TorchEnergyTrainer,
    TorchTrainerConfig,
)
from nga.arch.typed_readout_torch import TypedReadoutTorch  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------


def _make_trainer(
    n_states: int = 4,
    embedding_dim: int = 4,
    seed: int = 0,
    config: TorchTrainerConfig | None = None,
    with_encoder: bool = False,
    with_readout: bool = False,
) -> TorchEnergyTrainer:
    encoder = (
        FrozenEncoderTorch(input_dim=8, output_dim=embedding_dim, seed=seed)
        if with_encoder
        else None
    )
    readout = (
        TypedReadoutTorch(
            type_ids=["A", "B"],
            n_classes=n_states,
            input_dim=embedding_dim,
            hidden_dim=8,
            n_epochs=1,
            seed=seed,
        )
        if with_readout
        else None
    )
    return TorchEnergyTrainer(
        n_states=n_states,
        embedding_dim=embedding_dim,
        encoder_output_dim=embedding_dim,
        encoder=encoder,
        readout=readout,
        config=config,
        seed=seed,
    )


def _synthetic_batch(
    n_states: int,
    embedding_dim: int,
    batch_size: int = 8,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a deterministic 4-state synthetic task.

    Each true_next_state has a fixed cluster center at a small radius
    inside the Poincare ball; observations are jittered around it. The
    optimal solution is "prototype j == cluster center j", so a healthy
    trainer should drive distance(obs, prototype) toward zero.
    """
    rng = np.random.default_rng(seed)
    centers = 0.4 * np.eye(n_states, embedding_dim, dtype=np.float32)
    true_next = rng.integers(0, n_states, size=batch_size).astype(np.int64)
    current = rng.integers(0, n_states, size=batch_size).astype(np.int64)
    obs = centers[true_next] + 0.02 * rng.standard_normal(
        (batch_size, embedding_dim)
    ).astype(np.float32)
    return (
        torch.from_numpy(obs),
        torch.from_numpy(current),
        torch.from_numpy(true_next),
    )


# ----------------------------------------------------------------------
# Construction / shape contract
# ----------------------------------------------------------------------


def test_trainer_constructs():
    trainer = _make_trainer(n_states=4, embedding_dim=8)
    # Required nn.Parameters.
    assert isinstance(trainer.prototypes, torch.nn.Parameter)
    assert trainer.prototypes.shape == (4, 8)
    assert trainer.prototypes.requires_grad
    assert isinstance(trainer.alpha_log, torch.nn.Parameter)
    assert trainer.alpha_log.shape == (4, 4)
    assert trainer.alpha_log.requires_grad
    assert isinstance(trainer.beta_log, torch.nn.Parameter)
    assert trainer.beta_log.shape == (4, 4)
    assert trainer.beta_log.requires_grad


def test_trainer_with_encoder_and_readout_constructs():
    trainer = _make_trainer(with_encoder=True, with_readout=True)
    # Encoder parameters MUST be frozen.
    assert trainer._encoder_net is not None
    for p in trainer._encoder_net.parameters():
        assert not p.requires_grad
    # Readout heads contribute trainable parameters.
    assert trainer._readout_heads is not None
    for head in trainer._readout_heads.values():
        for p in head.parameters():
            assert p.requires_grad


# ----------------------------------------------------------------------
# Optimisation behaviour
# ----------------------------------------------------------------------


def test_step_decreases_loss_on_simple_task():
    n_states, dim = 4, 4
    config = TorchTrainerConfig(lr=5e-2, lambda_kl=0.5, grad_clip=5.0)
    trainer = _make_trainer(
        n_states=n_states, embedding_dim=dim, config=config, seed=1
    )
    obs, cur, nxt = _synthetic_batch(n_states, dim, batch_size=16, seed=2)

    losses: list[float] = []
    for _ in range(60):
        out = trainer.step(obs, cur, nxt)
        losses.append(float(out["loss"].item()))

    # The 4-state synthetic is well-conditioned: rolling-mean of the last
    # third must be strictly below rolling-mean of the first third.
    third = len(losses) // 3
    early = float(np.mean(losses[:third]))
    late = float(np.mean(losses[-third:]))
    assert late < early, f"loss did not decrease: early={early}, late={late}"


# ----------------------------------------------------------------------
# Posterior contract
# ----------------------------------------------------------------------


def test_posterior_mean_in_unit_interval():
    trainer = _make_trainer()
    pm = trainer.posterior_mean()
    assert torch.all(pm > 0.0)
    assert torch.all(pm < 1.0)


def test_legality_matrix_skeptical_default():
    trainer = _make_trainer()
    # Cold-start: alpha_log == beta_log == 0 -> posterior_mean == 0.5.
    pm = trainer.posterior_mean()
    assert torch.allclose(pm, torch.full_like(pm, 0.5))
    # legality_matrix(0.5) is strict >, so all entries are False.
    legal = trainer.legality_matrix(threshold=0.5)
    assert legal.dtype == torch.bool
    assert not legal.any().item()


# ----------------------------------------------------------------------
# Snapshot round-trip
# ----------------------------------------------------------------------


def test_to_classical_posterior_mask_round_trip():
    config = TorchTrainerConfig(lr=5e-2, grad_clip=5.0)
    trainer = _make_trainer(config=config, seed=3)
    obs, cur, nxt = _synthetic_batch(4, 4, batch_size=8, seed=4)
    for _ in range(5):
        trainer.step(obs, cur, nxt)

    snapshot = trainer.to_classical_posterior_mask()
    assert isinstance(snapshot, PosteriorMask)
    assert snapshot.n_vertices == trainer.n_states
    snap_mean = snapshot.posterior_mean()

    # Re-instantiate a fresh trainer warm-started from the snapshot.
    trainer2 = TorchEnergyTrainer(
        n_states=trainer.n_states,
        embedding_dim=trainer.embedding_dim,
        posterior_alpha_init=torch.from_numpy(
            snapshot.alpha.astype(np.float32)
        ),
        posterior_beta_init=torch.from_numpy(
            snapshot.beta.astype(np.float32)
        ),
        seed=99,  # different seed; only posterior should match
    )
    pm2 = trainer2.posterior_mean().detach().cpu().numpy()
    np.testing.assert_allclose(pm2, snap_mean, atol=1e-5)


# ----------------------------------------------------------------------
# Gradient flow
# ----------------------------------------------------------------------


def test_gradients_flow_to_prototypes():
    trainer = _make_trainer(seed=5)
    obs, cur, nxt = _synthetic_batch(4, 4, batch_size=4, seed=6)
    out = trainer.forward(obs, cur, nxt)
    out["loss"].backward()
    g = trainer.prototypes.grad
    assert g is not None
    assert torch.isfinite(g).all()
    assert g.norm().item() > 0.0


def test_gradients_flow_to_posterior():
    trainer = _make_trainer(seed=7)
    obs, cur, nxt = _synthetic_batch(4, 4, batch_size=4, seed=8)
    out = trainer.forward(obs, cur, nxt)
    out["loss"].backward()
    ga = trainer.alpha_log.grad
    gb = trainer.beta_log.grad
    assert ga is not None and gb is not None
    assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
    # KL term gives both non-zero gradient contributions.
    assert ga.abs().sum().item() > 0.0
    assert gb.abs().sum().item() > 0.0


def test_encoder_remains_frozen():
    trainer = _make_trainer(with_encoder=True, seed=9)
    obs, cur, nxt = _synthetic_batch(4, 4, batch_size=4, seed=10)
    out = trainer.step(obs, cur, nxt)
    # Encoder parameters: requires_grad False AND grad is None or zero.
    assert trainer._encoder_net is not None
    for p in trainer._encoder_net.parameters():
        assert not p.requires_grad
        assert p.grad is None or float(p.grad.abs().sum().item()) == 0.0
    _ = out  # silence unused


# ----------------------------------------------------------------------
# Determinism & clipping
# ----------------------------------------------------------------------


def test_deterministic_under_seed():
    config = TorchTrainerConfig(lr=5e-2, grad_clip=5.0)
    obs, cur, nxt = _synthetic_batch(4, 4, batch_size=4, seed=11)

    trainer_a = _make_trainer(config=config, seed=42)
    trainer_b = _make_trainer(config=config, seed=42)

    losses_a: list[float] = []
    losses_b: list[float] = []
    for _ in range(10):
        losses_a.append(float(trainer_a.step(obs, cur, nxt)["loss"].item()))
        losses_b.append(float(trainer_b.step(obs, cur, nxt)["loss"].item()))

    np.testing.assert_allclose(np.array(losses_a), np.array(losses_b), atol=1e-6)


def test_grad_clip_active():
    """With grad_clip very small the actual parameter delta is bounded."""
    config_clipped = TorchTrainerConfig(lr=1.0, grad_clip=1e-3)
    trainer = _make_trainer(config=config_clipped, seed=13)
    obs, cur, nxt = _synthetic_batch(4, 4, batch_size=4, seed=14)
    proto_before = trainer.prototypes.detach().clone()
    trainer.step(obs, cur, nxt)
    delta = (trainer.prototypes.detach() - proto_before).norm().item()
    # Without clipping, lr=1.0 with a small batch of moderately-different
    # observations would move the prototypes by O(1e-2..1e-1). With the
    # 1e-3 clip the post-clip global grad norm is at most 1e-3 and Adam's
    # first step with eps=1e-8 produces a delta of comparable magnitude to
    # the lr -- but we only assert the delta is far below the unclipped
    # baseline (taken from lr=1.0, grad_clip=5.0 below).
    config_unclipped = TorchTrainerConfig(lr=1.0, grad_clip=5.0)
    trainer_u = _make_trainer(config=config_unclipped, seed=13)
    proto_before_u = trainer_u.prototypes.detach().clone()
    trainer_u.step(obs, cur, nxt)
    delta_u = (trainer_u.prototypes.detach() - proto_before_u).norm().item()
    assert delta < delta_u, (
        f"clipping not active: delta_clipped={delta:.6f} >= "
        f"delta_unclipped={delta_u:.6f}"
    )


# ----------------------------------------------------------------------
# Predictive distribution sanity
# ----------------------------------------------------------------------


def test_predicted_distribution_is_simplex():
    trainer = _make_trainer(seed=15)
    obs = torch.randn(trainer.embedding_dim) * 0.05
    p = trainer.predicted_distribution(obs, current_state=0)
    assert p.shape == (trainer.n_states,)
    assert torch.all(p >= 0.0)
    assert torch.all(p <= 1.0)
    assert abs(float(p.sum().item()) - 1.0) < 1e-5
