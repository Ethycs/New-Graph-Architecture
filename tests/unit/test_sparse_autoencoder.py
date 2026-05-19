"""Unit tests for ``SparseAutoencoder`` (Phase 28 SAE training)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nga.arch.sae_adapter import PretrainedSAEAdapter
from nga.arch.sparse_autoencoder import (
    SparseAutoencoder,
    SparseAutoencoderConfig,
    train_sparse_autoencoder,
)


def test_sae_forward_returns_z_and_recon() -> None:
    cfg = SparseAutoencoderConfig(d_in=8, n_features=16, sparsity_coef=1e-3)
    sae = SparseAutoencoder(cfg)
    x = torch.randn(4, 8)
    z, h_hat = sae(x)
    assert z.shape == (4, 16)
    assert h_hat.shape == (4, 8)
    # z is post-ReLU; must be non-negative.
    assert (z >= 0).all()


def test_sae_loss_components_are_finite() -> None:
    cfg = SparseAutoencoderConfig(d_in=8, n_features=16, sparsity_coef=1e-3)
    sae = SparseAutoencoder(cfg)
    x = torch.randn(4, 8)
    total, comp = sae.loss(x)
    assert torch.isfinite(total)
    assert np.isfinite(comp["loss_total"])
    assert np.isfinite(comp["loss_recon"])
    assert np.isfinite(comp["loss_sparsity"])
    assert comp["n_active_per_sample"] >= 0


def test_train_reduces_reconstruction_loss() -> None:
    """A short training run should reduce reconstruction loss vs the initial state."""
    cfg = SparseAutoencoderConfig(d_in=16, n_features=64, sparsity_coef=1e-3)
    torch.manual_seed(0)
    sae = SparseAutoencoder(cfg)
    # Synthetic activations: a low-dim subspace plus noise (compressible).
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((4, 16)).astype(np.float32)
    coeffs = rng.standard_normal((200, 4)).astype(np.float32)
    H = coeffs @ basis + 0.1 * rng.standard_normal((200, 16)).astype(np.float32)

    # Initial loss.
    init_total, init_comp = sae.loss(torch.from_numpy(H))
    init_recon = init_comp["loss_recon"]

    final = train_sparse_autoencoder(sae, H, epochs=50, batch_size=32, lr=1e-2, seed=0)
    assert final["loss_recon"] < init_recon, (
        f"recon loss did not decrease: {init_recon} -> {final['loss_recon']}"
    )


def test_save_npz_round_trip(tmp_path: Path) -> None:
    """save_npz -> PretrainedSAEAdapter.from_checkpoint preserves encode behavior."""
    cfg = SparseAutoencoderConfig(d_in=8, n_features=16, sparsity_coef=1e-3)
    torch.manual_seed(1)
    sae = SparseAutoencoder(cfg)

    ckpt_path = tmp_path / "sae.npz"
    sae.save_npz(ckpt_path)
    assert ckpt_path.exists()

    adapter = PretrainedSAEAdapter.from_checkpoint(ckpt_path)
    assert adapter.d_in == 8
    assert adapter.n_features == 16

    # Encoder-only forward parity: torch's encode(h) ~= adapter's encode(h).
    x = np.random.default_rng(42).standard_normal(8).astype(np.float32)
    with torch.no_grad():
        z_torch = sae.encode(torch.from_numpy(x).unsqueeze(0))[0].numpy()
    code = adapter.encode(x)
    # Active-feature set should match: post-ReLU strictly positive entries.
    expected_active = np.where(z_torch > 0)[0].tolist()
    assert code.active_features == expected_active
    np.testing.assert_array_almost_equal(
        code.activations, z_torch[expected_active], decimal=5
    )
