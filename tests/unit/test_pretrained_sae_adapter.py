"""Unit tests for ``PretrainedSAEAdapter`` (Phase 28)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nga.arch.sae_adapter import (
    PretrainedSAEAdapter,
    SparseFeatureCode,
)


def _make_ckpt(tmp_path: Path, d_in: int = 4, n_features: int = 8) -> Path:
    """Write a minimal SAE checkpoint to ``tmp_path`` and return the path."""
    rng = np.random.default_rng(0)
    ckpt = tmp_path / "sae.npz"
    np.savez_compressed(
        ckpt,
        W_enc=rng.standard_normal((n_features, d_in)).astype(np.float32),
        b_enc=np.zeros(n_features, dtype=np.float32),
        b_dec=np.zeros(d_in, dtype=np.float32),
    )
    return ckpt


def test_from_checkpoint_loads_shapes(tmp_path: Path) -> None:
    ckpt = _make_ckpt(tmp_path, d_in=4, n_features=8)
    sae = PretrainedSAEAdapter.from_checkpoint(ckpt)
    assert sae.d_in == 4
    assert sae.n_features == 8
    assert sae.feature_labels() == {}


def test_from_checkpoint_with_labels(tmp_path: Path) -> None:
    ckpt = _make_ckpt(tmp_path, d_in=4, n_features=8)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"0": "joy", "3": "sadness"}))
    sae = PretrainedSAEAdapter.from_checkpoint(ckpt, labels_path=labels_path)
    assert sae.feature_labels() == {0: "joy", 3: "sadness"}


def test_from_checkpoint_missing_W_enc_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, b_enc=np.zeros(4))
    with pytest.raises(ValueError, match="missing W_enc"):
        PretrainedSAEAdapter.from_checkpoint(bad)


def test_from_checkpoint_labels_out_of_range_raises(tmp_path: Path) -> None:
    ckpt = _make_ckpt(tmp_path, d_in=4, n_features=8)
    bad_labels = tmp_path / "labels.json"
    bad_labels.write_text(json.dumps({"99": "out-of-range"}))
    with pytest.raises(ValueError, match=r"outside \[0, 8\)"):
        PretrainedSAEAdapter.from_checkpoint(ckpt, labels_path=bad_labels)


def test_encode_returns_sparse_feature_code() -> None:
    # Construct an SAE where feature 0 fires on positive input[0],
    # feature 1 fires on positive input[1], etc. (identity-like encoder).
    d_in = 4
    n_features = 4
    sae = PretrainedSAEAdapter(
        W_enc=np.eye(d_in),
        b_enc=np.zeros(n_features),
        b_dec=np.zeros(d_in),
        labels={0: "first", 2: "third"},
    )
    code = sae.encode(np.array([1.0, -0.5, 2.0, 0.0]))
    assert isinstance(code, SparseFeatureCode)
    # Feature 0 (act 1.0) and feature 2 (act 2.0) should be active; feature 1
    # is negative (-> relu = 0), feature 3 is zero.
    assert code.active_features == [0, 2]
    np.testing.assert_array_almost_equal(code.activations, [1.0, 2.0])
    assert code.total_features == 4
    # 0 is labelled "first"; 2 is labelled "third"; both go to `named`.
    assert sorted(code.named) == ["0", "2"]
    assert code.residual == []
    assert code.feature_labels == {"0": "first", "2": "third"}


def test_encode_with_unlabelled_features_populates_residual() -> None:
    sae = PretrainedSAEAdapter(
        W_enc=np.eye(4),
        b_enc=np.zeros(4),
        b_dec=np.zeros(4),
        labels={0: "labelled"},  # only feature 0 is labelled
    )
    code = sae.encode(np.array([1.0, 1.0, 1.0, 0.0]))
    # Features 0, 1, 2 are active; only 0 is named, 1 and 2 are residual.
    assert code.active_features == [0, 1, 2]
    assert code.named == ["0"]
    assert sorted(code.residual) == ["1", "2"]


def test_encode_b_dec_centering() -> None:
    # If b_dec subtracts the input, the pre-encoder activation goes through zero.
    sae = PretrainedSAEAdapter(
        W_enc=np.eye(4),
        b_enc=np.zeros(4),
        b_dec=np.array([0.5, 0.5, 0.5, 0.5]),
    )
    code = sae.encode(np.array([1.0, 0.3, 0.0, 1.0]))
    # x = h - b_dec = [0.5, -0.2, -0.5, 0.5]
    # z = relu(x) = [0.5, 0.0, 0.0, 0.5]
    assert code.active_features == [0, 3]
    np.testing.assert_array_almost_equal(code.activations, [0.5, 0.5])


def test_activation_threshold_filter() -> None:
    sae = PretrainedSAEAdapter(
        W_enc=np.eye(4),
        b_enc=np.zeros(4),
        b_dec=np.zeros(4),
        activation_threshold=1.0,  # require z > 1.0
    )
    code = sae.encode(np.array([0.5, 2.0, 1.0, 3.0]))
    # Post-ReLU = [0.5, 2.0, 1.0, 3.0]. After threshold > 1.0: feat 1 and 3 only.
    assert code.active_features == [1, 3]


def test_top_k_caps_active_features() -> None:
    sae = PretrainedSAEAdapter(
        W_enc=np.eye(8),
        b_enc=np.zeros(8),
        b_dec=np.zeros(8),
        top_k=3,
    )
    code = sae.encode(np.array([0.1, 0.2, 0.5, 0.8, 0.3, 0.05, 0.6, 0.4]))
    # Top-3 by magnitude: features 3 (0.8), 6 (0.6), 2 (0.5).
    assert sorted(code.active_features) == [2, 3, 6]


def test_encode_dim_mismatch_raises() -> None:
    sae = PretrainedSAEAdapter(W_enc=np.eye(4), b_enc=np.zeros(4), b_dec=np.zeros(4))
    with pytest.raises(ValueError, match="length 4"):
        sae.encode(np.array([1.0, 2.0]))  # wrong length


def test_init_invalid_shapes_raises() -> None:
    with pytest.raises(ValueError, match="W_enc must be 2D"):
        PretrainedSAEAdapter(W_enc=np.zeros(4), b_enc=np.zeros(4), b_dec=np.zeros(4))
    with pytest.raises(ValueError, match="b_enc shape"):
        PretrainedSAEAdapter(W_enc=np.eye(4), b_enc=np.zeros(3), b_dec=np.zeros(4))
    with pytest.raises(ValueError, match="b_dec shape"):
        PretrainedSAEAdapter(W_enc=np.eye(4), b_enc=np.zeros(4), b_dec=np.zeros(3))
    with pytest.raises(ValueError, match="top_k must be positive"):
        PretrainedSAEAdapter(W_enc=np.eye(4), b_enc=np.zeros(4), b_dec=np.zeros(4), top_k=0)


def test_round_trip_via_checkpoint(tmp_path: Path) -> None:
    """Save -> load preserves encoder behavior bit-identically."""
    d_in, n_features = 4, 8
    rng = np.random.default_rng(7)
    W_enc = rng.standard_normal((n_features, d_in)).astype(np.float32)
    b_enc = rng.standard_normal(n_features).astype(np.float32)
    b_dec = rng.standard_normal(d_in).astype(np.float32)

    ckpt = tmp_path / "sae.npz"
    np.savez_compressed(ckpt, W_enc=W_enc, b_enc=b_enc, b_dec=b_dec)
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"0": "alpha", "5": "beta"}))

    sae = PretrainedSAEAdapter.from_checkpoint(ckpt, labels_path=labels_path)

    # Re-create the same encoder by hand and check encode parity.
    direct = PretrainedSAEAdapter(W_enc=W_enc, b_enc=b_enc, b_dec=b_dec,
                                    labels={0: "alpha", 5: "beta"})

    x = rng.standard_normal(d_in)
    a = sae.encode(x)
    b = direct.encode(x)
    assert a.active_features == b.active_features
    np.testing.assert_array_almost_equal(a.activations, b.activations, decimal=5)
    assert a.named == b.named
    assert a.residual == b.residual
