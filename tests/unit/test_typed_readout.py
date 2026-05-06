"""Unit tests for nga.arch.typed_readout_layer.TypedReadoutLayer.

Covers the per-type isolation contract: independent fit state per type,
strict dispatch on type_id, parameter accounting, and decision_function
shape conventions matching sklearn.
"""

from __future__ import annotations

import numpy as np
import pytest

from nga.arch.typed_readout_layer import TypedReadoutLayer


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
    """Generate a simple linearly-separable-ish (X, y) pair.

    ``label_shift`` rotates the per-feature class assignment so two heads
    fed the same X but different y end up with different decision rules.
    """
    rng = np.random.default_rng(seed)
    # Sample y first, then build X clustered around y-dependent means.
    y = rng.integers(0, n_classes, size=n_samples)
    means = rng.normal(0.0, 3.0, size=(n_classes, n_features))
    X = means[y] + rng.normal(0.0, 0.5, size=(n_samples, n_features))
    # Optionally permute the labels deterministically.
    if label_shift:
        y = (y + label_shift) % n_classes
    return X, y


def _make_layer(
    type_ids: list[str | int] | None = None,
    n_classes: int = 3,
) -> TypedReadoutLayer:
    if type_ids is None:
        type_ids = ["A", "B"]
    return TypedReadoutLayer(
        type_ids=type_ids,
        n_classes=n_classes,
        max_iter=500,
        random_state=0,
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_typed_readout_per_type_isolation() -> None:
    """Fitting head A must not fit head B; predict on B raises RuntimeError."""
    layer = _make_layer(["A", "B"], n_classes=3)
    X, y = _make_classification_xy(seed=1)

    layer.fit("A", X, y)

    assert layer.is_fitted("A")
    assert not layer.is_fitted("B")

    # Predicting on the unfitted head must raise.
    with pytest.raises(RuntimeError):
        layer.predict("B", X)
    with pytest.raises(RuntimeError):
        layer.predict_proba("B", X)
    with pytest.raises(RuntimeError):
        layer.decision_function("B", X)

    # Head A still works post-isolation check.
    preds_a = layer.predict("A", X)
    assert preds_a.shape == (X.shape[0],)


def test_typed_readout_n_params_zero_before_fit_positive_after() -> None:
    """Total params is 0 before any fit and strictly positive after fitting."""
    layer = _make_layer(["A", "B"], n_classes=3)
    assert layer.n_trainable_params() == 0
    assert layer.n_trainable_params_per_type() == {"A": 0, "B": 0}

    X, y = _make_classification_xy(seed=2)
    layer.fit("A", X, y)

    total_after = layer.n_trainable_params()
    assert total_after > 0
    # B is still unfitted, so it stays at 0.
    per_type = layer.n_trainable_params_per_type()
    assert per_type["A"] == total_after
    assert per_type["B"] == 0


def test_typed_readout_dispatch_correct() -> None:
    """Heads A and B fit on different y distributions disagree on the same X."""
    layer = _make_layer(["A", "B"], n_classes=3)

    # Same X, different label assignments -> different fitted decision rules.
    X, y_a = _make_classification_xy(seed=3, label_shift=0)
    _, y_b = _make_classification_xy(seed=3, label_shift=1)
    # Sanity: label vectors actually differ.
    assert not np.array_equal(y_a, y_b)

    layer.fit("A", X, y_a)
    layer.fit("B", X, y_b)

    preds_a = layer.predict("A", X)
    preds_b = layer.predict("B", X)

    # The two heads should not agree on every sample -- if they did the
    # dispatch would be a no-op. We only require a non-trivial disagreement.
    assert np.any(preds_a != preds_b), (
        "Heads A and B produced identical predictions; dispatch may be broken."
    )


def test_typed_readout_unknown_type_raises() -> None:
    """A type_id not registered at construction time must raise KeyError."""
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


def test_typed_readout_decision_function_shape() -> None:
    """decision_function: (N, n_classes) for multiclass, (N,) for binary."""
    # Multiclass head
    multi = _make_layer(["M"], n_classes=3)
    X_m, y_m = _make_classification_xy(seed=5, n_classes=3)
    multi.fit("M", X_m, y_m)
    df_m = multi.decision_function("M", X_m)
    assert df_m.ndim == 2
    assert df_m.shape[0] == X_m.shape[0]
    # sklearn returns one column per fitted class.
    n_classes_seen_m = len(np.unique(y_m))
    assert df_m.shape[1] == n_classes_seen_m

    # Binary head -- sklearn convention is shape (N,).
    binary = _make_layer(["Bn"], n_classes=2)
    X_b, y_b = _make_classification_xy(seed=6, n_classes=2)
    binary.fit("Bn", X_b, y_b)
    df_b = binary.decision_function("Bn", X_b)
    assert df_b.ndim == 1
    assert df_b.shape == (X_b.shape[0],)


def test_typed_readout_n_params_per_type_sums_to_total() -> None:
    """sum(per_type) == total, both before and after partial / full fits."""
    layer = _make_layer(["A", "B", "C"], n_classes=3)

    # Before any fits.
    per = layer.n_trainable_params_per_type()
    assert sum(per.values()) == layer.n_trainable_params() == 0

    # After fitting A only.
    X, y = _make_classification_xy(seed=7)
    layer.fit("A", X, y)
    per1 = layer.n_trainable_params_per_type()
    assert sum(per1.values()) == layer.n_trainable_params()
    assert per1["A"] > 0
    assert per1["B"] == 0
    assert per1["C"] == 0

    # After fitting B and C as well.
    X2, y2 = _make_classification_xy(seed=8)
    layer.fit("B", X2, y2)
    layer.fit("C", X2, y2)
    per2 = layer.n_trainable_params_per_type()
    total2 = layer.n_trainable_params()
    assert sum(per2.values()) == total2
    assert all(v > 0 for v in per2.values())
