"""Unit tests for nga.arch.hidden_state_harvester.HiddenStateHarvester."""
from __future__ import annotations

import numpy as np
import pytest

from nga.arch.hidden_state_harvester import HarvestResult, HiddenStateHarvester


def test_from_callable_per_sample_vector() -> None:
    """A callable emitting one vector per sample produces N_total = N_samples."""
    def encode(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) * 2.0

    harvester = HiddenStateHarvester.from_callable(encode)
    corpus = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    result = harvester.harvest(corpus)

    assert isinstance(result, HarvestResult)
    assert result.n_samples == 2
    assert result.n_steps == 2
    assert result.hidden_dim == 3
    np.testing.assert_allclose(
        result.hidden_states,
        np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]),
    )
    np.testing.assert_array_equal(result.sample_index, [0, 1])


def test_from_callable_per_sample_sequence() -> None:
    """A callable emitting a (T, d) sequence per sample concatenates rows
    and labels them with the sample index."""
    def encode(x: np.ndarray) -> np.ndarray:
        # x is a (T, d) array; return it scaled by 0.5.
        return np.asarray(x, dtype=np.float64) * 0.5

    harvester = HiddenStateHarvester.from_callable(
        encode, per_sample_emits_sequence=True
    )
    corpus = [
        np.array([[1.0, 0.0], [2.0, 0.0]]),
        np.array([[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]),
    ]
    result = harvester.harvest(corpus)

    assert result.n_samples == 2
    assert result.n_steps == 5
    assert result.hidden_dim == 2
    np.testing.assert_allclose(
        result.hidden_states,
        np.array(
            [
                [0.5, 0.0],
                [1.0, 0.0],
                [1.5, 0.0],
                [2.0, 0.0],
                [2.5, 0.0],
            ]
        ),
    )
    np.testing.assert_array_equal(result.sample_index, [0, 0, 1, 1, 1])


def test_from_sequences_consumes_iterable_in_order() -> None:
    """from_sequences yields the iterable's arrays in order, indexing samples
    by enumeration order."""
    seq_a = np.array([[1.0], [2.0]])
    seq_b = np.array([[3.0]])
    seq_c = np.array([[4.0], [5.0], [6.0]])

    harvester = HiddenStateHarvester.from_sequences(iter([seq_a, seq_b, seq_c]))
    result = harvester.harvest([None, None, None])  # corpus elements unused

    assert result.n_samples == 3
    assert result.n_steps == 6
    np.testing.assert_allclose(
        result.hidden_states,
        np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
    )
    np.testing.assert_array_equal(result.sample_index, [0, 0, 1, 2, 2, 2])


def test_empty_corpus_returns_empty_result() -> None:
    """An empty corpus is legal; the harvester returns an empty result."""
    harvester = HiddenStateHarvester.from_callable(lambda x: np.zeros(3))
    result = harvester.harvest([])
    assert result.n_steps == 0
    assert result.n_samples == 0
    assert result.hidden_dim == 0


def test_callable_emits_2d_with_leading_1_is_squeezed() -> None:
    """A callable emitting (1, d) for a per-sample-vector adapter is
    squeezed to (d,)."""
    def encode(_: np.ndarray) -> np.ndarray:
        return np.array([[7.0, 8.0, 9.0]])

    harvester = HiddenStateHarvester.from_callable(encode)
    result = harvester.harvest([None, None])
    assert result.n_steps == 2
    assert result.hidden_dim == 3
    np.testing.assert_allclose(
        result.hidden_states,
        np.array([[7.0, 8.0, 9.0], [7.0, 8.0, 9.0]]),
    )


def test_wrong_shape_raises_value_error() -> None:
    """If the encoder emits a shape inconsistent with the adapter mode,
    the harvester raises ValueError."""
    # Per-sample-vector adapter, but encoder emits a (T, d) sequence.
    def encode(_: np.ndarray) -> np.ndarray:
        return np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2) -- not (1, d)

    harvester = HiddenStateHarvester.from_callable(encode)
    with pytest.raises(ValueError):
        harvester.harvest([None])

    # Per-sample-sequence adapter, but encoder emits a 3-D tensor.
    def encode_3d(_: np.ndarray) -> np.ndarray:
        return np.zeros((2, 3, 4))

    harvester_seq = HiddenStateHarvester.from_callable(
        encode_3d, per_sample_emits_sequence=True
    )
    with pytest.raises(ValueError):
        harvester_seq.harvest([None])


def test_harvest_result_validates_shapes() -> None:
    """HarvestResult constructor rejects mismatched arrays."""
    with pytest.raises(ValueError):
        HarvestResult(
            hidden_states=np.zeros((3, 2)),
            sample_index=np.zeros((4,), dtype=np.int64),
        )
    with pytest.raises(ValueError):
        HarvestResult(
            hidden_states=np.zeros((3,)),
            sample_index=np.zeros((3,), dtype=np.int64),
        )


def test_context_manager_calls_close() -> None:
    """The harvester is usable as a context manager; exiting calls close."""
    closed: list[bool] = []

    harvester = HiddenStateHarvester.from_callable(lambda x: np.zeros(2))
    # No-op cleanup for the plain-callable adapter, but the protocol holds.
    harvester._cleanup_fn = lambda: closed.append(True)
    with harvester as h:
        assert h is harvester
    assert closed == [True]


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="torch not available") is None,
    reason="torch not available",
)
def test_from_torch_module_captures_layer_output() -> None:
    """The torch adapter captures the output of a named submodule via hook."""
    import torch
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 3)
            self.fc2 = nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.fc1(x)
            return self.fc2(torch.relu(h))

    torch.manual_seed(42)
    model = Tiny()
    harvester = HiddenStateHarvester.from_torch_module(model, layer_name="fc1")
    corpus = [np.zeros(4), np.ones(4)]
    result = harvester.harvest(corpus)
    harvester.close()

    assert result.n_steps == 2
    assert result.hidden_dim == 3  # fc1 has 3 output units
    # First sample fed all zeros, fc1 output equals fc1.bias.
    np.testing.assert_allclose(
        result.hidden_states[0],
        model.fc1.bias.detach().numpy(),
        atol=1e-6,
    )
