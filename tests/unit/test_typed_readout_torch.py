"""Unit tests for TypedReadoutTorch (torch-backed sibling of TypedReadoutLayer).

Covers the per-type isolation contract -- independent fit state per
type, strict dispatch on type_id, parameter accounting, decision_function
shape, and seed determinism. All tests are gated by
``pytest.importorskip("torch")`` so the file is a no-op in environments
that lack torch. ``n_epochs`` is kept small (10-20) to keep the suite
fast.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from nga.arch.typed_readout_torch import TypedReadoutTorch  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------


def _make_classification_xy(
    n_samples: int = 60,
    n_features: int = 8,
    n_classes: int = 3,
    seed: int = 0,
    label_shift: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n_samples)
    means = rng.normal(0.0, 3.0, size=(n_classes, n_features))
    X = means[y] + rng.normal(0.0, 0.5, size=(n_samples, n_features))
    if label_shift:
        y = (y + label_shift) % n_classes
    return X.astype(np.float32), y


def _make_layer(
    type_ids: list[str | int] | None = None,
    n_classes: int = 3,
    input_dim: int = 8,
    hidden_dim: int = 16,
    n_epochs: int = 15,
    seed: int = 0,
) -> TypedReadoutTorch:
    if type_ids is None:
        type_ids = ["A", "B"]
    return TypedReadoutTorch(
        type_ids=type_ids,
        n_classes=n_classes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_epochs=n_epochs,
        lr=1e-2,
        seed=seed,
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_typed_readout_torch_per_type_isolation() -> None:
    layer = _make_layer(["A", "B"], n_classes=3)
    X, y = _make_classification_xy(seed=1)

    layer.fit("A", X, y)

    assert layer.is_fitted("A")
    assert not layer.is_fitted("B")

    with pytest.raises(RuntimeError):
        layer.predict("B", X)
    with pytest.raises(RuntimeError):
        layer.predict_proba("B", X)
    with pytest.raises(RuntimeError):
        layer.decision_function("B", X)

    preds_a = layer.predict("A", X)
    assert preds_a.shape == (X.shape[0],)


def test_typed_readout_torch_n_params_zero_before_fit_positive_after() -> None:
    layer = _make_layer(["A", "B"], n_classes=3)
    assert layer.n_trainable_params() == 0
    assert layer.n_trainable_params_per_type() == {"A": 0, "B": 0}

    X, y = _make_classification_xy(seed=2)
    layer.fit("A", X, y)

    total_after = layer.n_trainable_params()
    assert total_after > 0
    per_type = layer.n_trainable_params_per_type()
    assert per_type["A"] == total_after
    assert per_type["B"] == 0


def test_typed_readout_torch_dispatch_correct() -> None:
    """Heads A and B fit on different y disagree on the same X."""
    layer = _make_layer(["A", "B"], n_classes=3, n_epochs=20)

    X, y_a = _make_classification_xy(seed=3, label_shift=0)
    _, y_b = _make_classification_xy(seed=3, label_shift=1)
    assert not np.array_equal(y_a, y_b)

    layer.fit("A", X, y_a)
    layer.fit("B", X, y_b)

    preds_a = layer.predict("A", X)
    preds_b = layer.predict("B", X)

    assert np.any(preds_a != preds_b), (
        "Heads A and B produced identical predictions; dispatch may be broken."
    )


def test_typed_readout_torch_unknown_type_raises_KeyError() -> None:
    layer = _make_layer(["A", "B"], n_classes=3)
    X, y = _make_classification_xy(seed=4)
    layer.fit("A", X, y)

    with pytest.raises(KeyError):
        layer.predict("Z", X)
    with pytest.raises(KeyError):
        layer.predict_proba("Z", X)
    with pytest.raises(KeyError):
        layer.decision_function("Z", X)
    with pytest.raises(KeyError):
        layer.fit("Z", X, y)
    with pytest.raises(KeyError):
        layer.is_fitted("Z")


def test_typed_readout_torch_decision_function_shape() -> None:
    """Torch sibling: decision_function is always (N, n_classes) -- raw logits."""
    multi = _make_layer(["M"], n_classes=3, n_epochs=10)
    X_m, y_m = _make_classification_xy(seed=5, n_classes=3)
    multi.fit("M", X_m, y_m)
    df_m = multi.decision_function("M", X_m)
    assert df_m.ndim == 2
    assert df_m.shape == (X_m.shape[0], 3)

    # Binary head -- the torch sibling outputs (N, 2) logits (it does NOT
    # follow sklearn's (N,)-for-binary convention; logits stay rectangular).
    binary = _make_layer(["Bn"], n_classes=2, n_epochs=10)
    X_b, y_b = _make_classification_xy(seed=6, n_classes=2)
    binary.fit("Bn", X_b, y_b)
    df_b = binary.decision_function("Bn", X_b)
    assert df_b.ndim == 2
    assert df_b.shape == (X_b.shape[0], 2)

    # predict_proba shape parity: (N, n_classes), rows sum to 1.
    proba = binary.predict_proba("Bn", X_b)
    assert proba.shape == (X_b.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_typed_readout_torch_n_params_per_type_sums_to_total() -> None:
    layer = _make_layer(["A", "B", "C"], n_classes=3, n_epochs=10)

    per = layer.n_trainable_params_per_type()
    assert sum(per.values()) == layer.n_trainable_params() == 0

    X, y = _make_classification_xy(seed=7)
    layer.fit("A", X, y)
    per1 = layer.n_trainable_params_per_type()
    assert sum(per1.values()) == layer.n_trainable_params()
    assert per1["A"] > 0
    assert per1["B"] == 0
    assert per1["C"] == 0

    X2, y2 = _make_classification_xy(seed=8)
    layer.fit("B", X2, y2)
    layer.fit("C", X2, y2)
    per2 = layer.n_trainable_params_per_type()
    total2 = layer.n_trainable_params()
    assert sum(per2.values()) == total2
    assert all(v > 0 for v in per2.values())


def test_typed_readout_torch_deterministic_under_seed() -> None:
    """Same seed -> same predictions across fresh instances."""
    X, y = _make_classification_xy(seed=11, n_classes=3)

    layer_a = _make_layer(["A"], n_classes=3, n_epochs=15, seed=2026)
    layer_b = _make_layer(["A"], n_classes=3, n_epochs=15, seed=2026)
    layer_a.fit("A", X, y)
    layer_b.fit("A", X, y)

    df_a = layer_a.decision_function("A", X)
    df_b = layer_b.decision_function("A", X)
    np.testing.assert_allclose(df_a, df_b, atol=1e-6)

    np.testing.assert_array_equal(
        layer_a.predict("A", X), layer_b.predict("A", X)
    )

    # Different seed -> different logits (with very high probability).
    layer_c = _make_layer(["A"], n_classes=3, n_epochs=15, seed=2027)
    layer_c.fit("A", X, y)
    df_c = layer_c.decision_function("A", X)
    assert not np.allclose(df_a, df_c, atol=1e-3)
