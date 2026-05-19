"""Small sparse autoencoder for Phase 28 (SAE plug-in).

A minimal SAE implementation in PyTorch -- one linear encoder + ReLU +
one linear decoder, with L1-on-activations sparsity penalty. Trained on
harvested transformer activations; the encoder weights become the
``PretrainedSAEAdapter`` checkpoint that fills the labelled hypergraph's
``named``/``residual`` field.

Forward pass:

    x = h - b_dec                  # pre-encoder centering
    z = relu(W_enc @ x + b_enc)    # sparse code
    h_hat = W_dec @ z + b_dec      # reconstruction

Loss:

    L = ||h - h_hat||^2 + sparsity_coef * ||z||_1

This is the canonical Anthropic-style SAE training objective
(Bricken et al. 2023). The architectural choices are intentionally
minimal -- this file is meant to ship the *machinery*, not to compete
with curated SAE training infrastructure.

Saving:

    SparseAutoencoder.save_npz(path)
        Saves W_enc, b_enc, W_dec, b_dec to a numpy .npz file. This is
        the format ``PretrainedSAEAdapter.from_checkpoint`` reads.

Dependencies: torch (only at training / forward time; the .npz file is
torch-free and can be loaded by PretrainedSAEAdapter with numpy alone).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SparseAutoencoderConfig:
    """Hyperparameters."""

    d_in: int
    n_features: int
    sparsity_coef: float = 1e-3
    init_decoder_orthogonal: bool = True


class SparseAutoencoder(nn.Module if nn is not None else object):
    """Small SAE: 1 linear encoder, 1 linear decoder, L1 sparsity loss."""

    def __init__(self, config: SparseAutoencoderConfig) -> None:
        if torch is None or nn is None:  # pragma: no cover
            raise ImportError("torch is required for SparseAutoencoder")
        super().__init__()
        self.cfg = config
        self.W_enc = nn.Parameter(torch.empty(config.n_features, config.d_in))
        self.b_enc = nn.Parameter(torch.zeros(config.n_features))
        self.W_dec = nn.Parameter(torch.empty(config.d_in, config.n_features))
        self.b_dec = nn.Parameter(torch.zeros(config.d_in))
        # Anthropic-style init: encoder small-random; decoder columns are
        # the *features in input-space*, so we initialize them to unit
        # norm (rotational structure preserved across features).
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        if config.init_decoder_orthogonal:
            nn.init.kaiming_uniform_(self.W_dec, a=5 ** 0.5)
            with torch.no_grad():
                self.W_dec.data.div_(self.W_dec.data.norm(dim=0, keepdim=True).clamp_min(1e-8))
                # Tie encoder to decoder transpose for a stable start
                # (a common SAE initialization trick).
                self.W_enc.data.copy_(self.W_dec.data.t())
        else:
            nn.init.kaiming_uniform_(self.W_dec, a=5 ** 0.5)

    def encode(self, h: "torch.Tensor") -> "torch.Tensor":
        """Return the post-ReLU sparse code z."""
        x = h - self.b_dec
        pre = x @ self.W_enc.t() + self.b_enc
        return torch.relu(pre)

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        return z @ self.W_dec.t() + self.b_dec

    def forward(self, h: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        z = self.encode(h)
        h_hat = self.decode(z)
        return z, h_hat

    def loss(self, h: "torch.Tensor") -> tuple["torch.Tensor", dict[str, float]]:
        z, h_hat = self(h)
        recon = ((h - h_hat) ** 2).mean()
        sparsity = z.abs().sum(dim=-1).mean()
        total = recon + self.cfg.sparsity_coef * sparsity
        return total, {
            "loss_total": float(total.detach().cpu().item()),
            "loss_recon": float(recon.detach().cpu().item()),
            "loss_sparsity": float(sparsity.detach().cpu().item()),
            "n_active_per_sample": float((z > 0).float().sum(dim=-1).mean().detach().cpu().item()),
        }

    def save_npz(self, path: str | Path) -> None:
        """Save weights as numpy .npz (the PretrainedSAEAdapter format)."""
        np.savez_compressed(
            Path(path),
            W_enc=self.W_enc.detach().cpu().numpy().astype(np.float32),
            b_enc=self.b_enc.detach().cpu().numpy().astype(np.float32),
            W_dec=self.W_dec.detach().cpu().numpy().astype(np.float32),
            b_dec=self.b_dec.detach().cpu().numpy().astype(np.float32),
            sparsity_coef=np.float32(self.cfg.sparsity_coef),
            n_features=np.int64(self.cfg.n_features),
            d_in=np.int64(self.cfg.d_in),
        )


def train_sparse_autoencoder(
    sae: SparseAutoencoder,
    activations: np.ndarray,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    verbose: bool = False,
) -> dict[str, float]:
    """Train an SAE on a stack of activations.

    Parameters
    ----------
    sae:
        The model to train (modified in place).
    activations:
        ``(N, d_in)`` float array of activations to fit.
    epochs:
        Number of full passes over the data.
    batch_size:
        SGD batch size.
    lr:
        Adam learning rate.
    seed:
        Shuffle / init seed.
    verbose:
        If True, print per-epoch loss summaries.

    Returns
    -------
    dict
        Final training stats (loss components + sparsity at last epoch).
    """
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required")
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    h_all = torch.from_numpy(activations.astype(np.float32))
    n = h_all.shape[0]
    rng = np.random.default_rng(int(seed))
    stats: dict[str, float] = {}
    for epoch in range(int(epochs)):
        perm = rng.permutation(n)
        epoch_total = 0.0
        epoch_n_batches = 0
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            idx = torch.from_numpy(perm[start:stop].astype(np.int64))
            batch = h_all[idx]
            total, comp = sae.loss(batch)
            opt.zero_grad()
            total.backward()
            opt.step()
            epoch_total += comp["loss_total"]
            epoch_n_batches += 1
            stats = comp
        avg = epoch_total / max(1, epoch_n_batches)
        if verbose and (epoch < 5 or (epoch + 1) % 20 == 0 or epoch == epochs - 1):
            print(
                f"  epoch {epoch + 1:>3}/{epochs}  loss_avg={avg:.5f}  "
                f"recon={stats['loss_recon']:.5f}  L1={stats['loss_sparsity']:.3f}  "
                f"n_active={stats['n_active_per_sample']:.1f}"
            )
    return stats


__all__ = (
    "SparseAutoencoderConfig",
    "SparseAutoencoder",
    "train_sparse_autoencoder",
)
