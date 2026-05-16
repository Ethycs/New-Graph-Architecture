"""Unit tests for ``nga.arch.sae_adapter``.

Acceptance bar A5 from ``docs/proposals/labelled-hypergraph.md``: a partial
label dictionary correctly splits an activation into named features (those
in the dictionary) and residual features (those not).
"""
from __future__ import annotations

import numpy as np
import pytest

from nga.arch.sae_adapter import (
    IdentitySAEAdapter,
    MockLabelledSAEAdapter,
    SparseFeatureCode,
)


# ---------------------------------------------------------------------------
# SparseFeatureCode
# ---------------------------------------------------------------------------


def test_sparse_feature_code_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        SparseFeatureCode(
            active_features=[1, 2, 3],
            activations=np.array([0.1, 0.2]),
            total_features=10,
        )


def test_sparse_feature_code_sparsity() -> None:
    code = SparseFeatureCode(
        active_features=[1, 5, 8],
        activations=np.array([0.1, 0.2, 0.3]),
        total_features=10,
    )
    assert code.sparsity() == pytest.approx(0.3)


def test_sparse_feature_code_zero_total_gives_zero_sparsity() -> None:
    code = SparseFeatureCode(
        active_features=[],
        activations=np.array([]),
        total_features=0,
    )
    assert code.sparsity() == 0.0


# ---------------------------------------------------------------------------
# IdentitySAEAdapter
# ---------------------------------------------------------------------------


def test_identity_adapter_no_top_k_uses_all_features() -> None:
    adapter = IdentitySAEAdapter(dim=4)
    code = adapter.encode(np.array([0.1, -0.5, 0.0, 0.3]))
    assert code.active_features == [0, 1, 2, 3]
    assert code.total_features == 4
    assert code.named == []  # No labels in identity.
    assert code.residual == ["0", "1", "2", "3"]
    assert code.feature_labels == {}


def test_identity_adapter_top_k_selects_largest_magnitude() -> None:
    adapter = IdentitySAEAdapter(dim=5, top_k=2)
    # |activations| = [0.1, 0.5, 0.0, 0.3, 0.7]; top-2 = indices 1, 4.
    code = adapter.encode(np.array([0.1, -0.5, 0.0, 0.3, 0.7]))
    assert sorted(code.active_features) == [1, 4]
    assert set(code.residual) == {"1", "4"}


def test_identity_adapter_rejects_wrong_dim() -> None:
    adapter = IdentitySAEAdapter(dim=4)
    with pytest.raises(ValueError, match="length 4"):
        adapter.encode(np.array([0.1, 0.2, 0.3]))


def test_identity_adapter_rejects_bad_construction() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        IdentitySAEAdapter(dim=0)
    with pytest.raises(ValueError, match="top_k must be positive"):
        IdentitySAEAdapter(dim=4, top_k=0)


def test_identity_adapter_feature_labels_is_empty() -> None:
    adapter = IdentitySAEAdapter(dim=4)
    assert adapter.feature_labels() == {}


# ---------------------------------------------------------------------------
# A5: MockLabelledSAEAdapter splits named vs residual
# ---------------------------------------------------------------------------


def test_mock_labelled_splits_active_into_named_and_residual() -> None:
    labels = {1: "eiffel_tower", 4: "subordinate_clause_opener"}
    adapter = MockLabelledSAEAdapter(dim=5, labels=labels)
    # All features active by default (no top_k).
    code = adapter.encode(np.array([0.1, 0.5, 0.0, 0.3, 0.7]))
    assert set(code.named) == {"1", "4"}
    assert set(code.residual) == {"0", "2", "3"}
    assert code.feature_labels == {"1": "eiffel_tower", "4": "subordinate_clause_opener"}


def test_mock_labelled_top_k_filters_active_first() -> None:
    labels = {0: "tower", 4: "clause"}
    adapter = MockLabelledSAEAdapter(dim=5, labels=labels, top_k=2)
    # |a| = [0.1, 0.5, 0.0, 0.3, 0.7]; top-2 = indices 1, 4.
    # Only feature 4 has a label; feature 1 does not.
    code = adapter.encode(np.array([0.1, -0.5, 0.0, 0.3, 0.7]))
    assert sorted(code.active_features) == [1, 4]
    assert code.named == ["4"]
    assert code.residual == ["1"]
    assert code.feature_labels == {"4": "clause"}


def test_mock_labelled_feature_labels_returned() -> None:
    labels = {0: "a", 2: "b"}
    adapter = MockLabelledSAEAdapter(dim=4, labels=labels)
    assert adapter.feature_labels() == labels


def test_mock_labelled_rejects_out_of_range_labels() -> None:
    with pytest.raises(ValueError, match="outside"):
        MockLabelledSAEAdapter(dim=4, labels={99: "bad"})


def test_mock_labelled_rejects_bad_construction() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        MockLabelledSAEAdapter(dim=0, labels={})
    with pytest.raises(ValueError, match="top_k must be positive"):
        MockLabelledSAEAdapter(dim=4, labels={}, top_k=0)


def test_mock_labelled_no_labels_behaves_like_identity() -> None:
    adapter = MockLabelledSAEAdapter(dim=4, labels={})
    code = adapter.encode(np.array([0.1, 0.5, 0.3, 0.7]))
    assert code.named == []
    assert sorted(code.residual) == ["0", "1", "2", "3"]
