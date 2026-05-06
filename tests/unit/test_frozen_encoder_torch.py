"""Unit tests for FrozenEncoderTorch (torch-backed sibling of FrozenEncoderBackbone).

Mirrors the sklearn sibling's contract tests -- freeze invariants,
encode purity, output shape, determinism under seed, parameter
accounting -- adjusted for the two contract differences vs the sklearn
sibling:

  * The torch encoder's ``fit`` is **idempotent** (params are random and
    frozen at ``__init__``; ``fit`` only flips ``is_fitted=True``), so a
    double-fit does NOT raise.
  * ``n_parameters`` is positive at construction time (the MLP weights
    exist before ``fit``); the sklearn sibling lazily creates its
    projection matrix inside ``fit``.

All tests are gated by ``pytest.importorskip("torch")`` so the file is a
no-op in environments that lack torch.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the entire module if torch is unavailable. Must come before the
# arch import too -- the arch module imports torch lazily but the tests
# below assume torch tensors actually work.
pytest.importorskip("torch")

from nga.arch.frozen_encoder_torch import FrozenEncoderTorch  # noqa: E402


def _make_data(n: int = 64, d: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def test_frozen_encoder_torch_fit_then_encode_pure() -> None:
    X = _make_data()
    enc = FrozenEncoderTorch(input_dim=16, output_dim=8, seed=123).fit(X)

    h1 = enc.encode(X)
    h2 = enc.encode(X)

    np.testing.assert_array_equal(h1, h2)
    assert enc.is_frozen is True
    assert enc.is_fitted is True


def test_frozen_encoder_torch_double_fit_idempotent() -> None:
    """Torch sibling's fit is a no-op flag flip; double-fit is allowed."""
    X = _make_data()
    enc = FrozenEncoderTorch(input_dim=16, output_dim=8, seed=1).fit(X)
    h_before = enc.encode(X)

    # Second fit must not raise and must not change the encoding (params
    # are frozen at construction; fit does not touch them).
    enc.fit(X)
    h_after = enc.encode(X)

    np.testing.assert_array_equal(h_before, h_after)
    assert enc.is_fitted is True


def test_frozen_encoder_torch_update_raises() -> None:
    X = _make_data()
    enc = FrozenEncoderTorch(input_dim=16, output_dim=8, seed=1).fit(X)

    with pytest.raises(RuntimeError, match="immutable"):
        enc.update()

    # Also raises before fit -- immutability is unconditional.
    enc2 = FrozenEncoderTorch(input_dim=16, output_dim=8, seed=1)
    with pytest.raises(RuntimeError, match="immutable"):
        enc2.update(grad=np.zeros(3))


def test_frozen_encoder_torch_output_shape() -> None:
    X = _make_data(n=37, d=12)
    enc = FrozenEncoderTorch(input_dim=12, output_dim=5, seed=7).fit(X)

    h = enc.encode(X)
    assert h.shape == (37, 5)

    X2 = _make_data(n=11, d=12, seed=99)
    h2 = enc.encode(X2)
    assert h2.shape == (11, 5)


def test_frozen_encoder_torch_deterministic_under_seed() -> None:
    X = _make_data(seed=42)
    enc_a = FrozenEncoderTorch(input_dim=16, output_dim=8, seed=2026).fit(X)
    enc_b = FrozenEncoderTorch(input_dim=16, output_dim=8, seed=2026).fit(X)

    np.testing.assert_array_equal(enc_a.encode(X), enc_b.encode(X))

    # Different seed -> different projection -> different output.
    enc_c = FrozenEncoderTorch(input_dim=16, output_dim=8, seed=2027).fit(X)
    assert not np.allclose(enc_a.encode(X), enc_c.encode(X))


def test_frozen_encoder_torch_n_parameters_reported() -> None:
    enc = FrozenEncoderTorch(input_dim=10, output_dim=4, hidden_dim=64, seed=0)

    # MLP params exist at construction time (Linear+Linear with biases):
    # (10*64 + 64) + (64*4 + 4) = 704 + 260 = 964
    assert enc.n_parameters == (10 * 64 + 64) + (64 * 4 + 4)
    assert enc.n_parameters > 0

    enc.fit(_make_data(n=20, d=10))
    # fit does not change the parameter count.
    assert enc.n_parameters == (10 * 64 + 64) + (64 * 4 + 4)


def test_frozen_encoder_torch_requires_grad_false() -> None:
    """Every parameter must be frozen (requires_grad=False) post-init."""
    import torch  # safe: importorskip at module top.

    enc = FrozenEncoderTorch(input_dim=8, output_dim=4, seed=11)
    params = list(enc._net.parameters())
    assert len(params) > 0
    assert all(isinstance(p, torch.Tensor) for p in params)
    assert all(p.requires_grad is False for p in params)

    # Same after fit (which is a no-op for the params).
    enc.fit(_make_data(n=12, d=8))
    assert all(p.requires_grad is False for p in enc._net.parameters())
