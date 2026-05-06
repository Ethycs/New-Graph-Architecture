"""Unit tests for FrozenEncoderBackbone.

Covers the freeze contract (single-fit, immutable update), purity of
``encode``, output shape, determinism under seed, and parameter accounting.
"""

from __future__ import annotations

import numpy as np
import pytest

from nga.arch.frozen_encoder_backbone import FrozenEncoderBackbone


def _make_data(n: int = 64, d: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def test_frozen_encoder_fit_then_encode_pure() -> None:
    X = _make_data()
    enc = FrozenEncoderBackbone(output_dim=8, seed=123).fit(X)

    h1 = enc.encode(X)
    h2 = enc.encode(X)

    np.testing.assert_array_equal(h1, h2)
    assert enc.is_frozen is True


def test_frozen_encoder_double_fit_raises() -> None:
    X = _make_data()
    enc = FrozenEncoderBackbone(output_dim=8, seed=1).fit(X)

    with pytest.raises(RuntimeError, match="already fit"):
        enc.fit(X)


def test_frozen_encoder_update_raises() -> None:
    X = _make_data()
    enc = FrozenEncoderBackbone(output_dim=8, seed=1).fit(X)

    with pytest.raises(RuntimeError, match="immutable"):
        enc.update()

    # Also raises before fit - immutability is unconditional.
    enc2 = FrozenEncoderBackbone(output_dim=8, seed=1)
    with pytest.raises(RuntimeError, match="immutable"):
        enc2.update(grad=np.zeros(3))


def test_frozen_encoder_output_shape() -> None:
    X = _make_data(n=37, d=12)
    enc = FrozenEncoderBackbone(output_dim=5, seed=7).fit(X)

    h = enc.encode(X)
    assert h.shape == (37, 5)

    # Encoding a different-size batch (same feature dim) preserves N rows.
    X2 = _make_data(n=11, d=12, seed=99)
    h2 = enc.encode(X2)
    assert h2.shape == (11, 5)


def test_frozen_encoder_deterministic_under_seed() -> None:
    X = _make_data(seed=42)
    enc_a = FrozenEncoderBackbone(output_dim=16, seed=2026).fit(X)
    enc_b = FrozenEncoderBackbone(output_dim=16, seed=2026).fit(X)

    np.testing.assert_array_equal(enc_a.encode(X), enc_b.encode(X))

    # Different seed -> different projection -> different output.
    enc_c = FrozenEncoderBackbone(output_dim=16, seed=2027).fit(X)
    assert not np.allclose(enc_a.encode(X), enc_c.encode(X))


def test_frozen_encoder_n_parameters_reported() -> None:
    X = _make_data(n=20, d=10)
    enc = FrozenEncoderBackbone(output_dim=4, seed=0)

    # Before fit: nothing stored.
    assert enc.n_parameters == 0

    enc.fit(X)
    # 10*4 (W) + 2*10 (scaler mean_ and scale_) = 60.
    assert enc.n_parameters > 0
    assert enc.n_parameters == 10 * 4 + 2 * 10
