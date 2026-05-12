"""Unit tests for nga.arch.axis_quantizer.

The axis quantiser is a deterministic projection from R to a totally-ordered
vertex set. These tests pin down idempotence, monotonicity, clamping, the
two convenience constructors, and validation of malformed bin edges.
"""

from __future__ import annotations

import numpy as np
import pytest

from nga.arch.axis_quantizer import AxisQuantizer


def _uniform_axis(n_bins: int = 5) -> AxisQuantizer:
    return AxisQuantizer.from_uniform("sigma", low=0.0, high=1.0, n_bins=n_bins)


def test_axis_quantizer_idempotent() -> None:
    q = _uniform_axis()

    # Stable on repeated calls for the same float.
    nid1 = q.quantise(0.37)
    nid2 = q.quantise(0.37)
    assert nid1 == nid2

    # String IDs that already belong to the axis pass through unchanged.
    for nid in q.node_ids():
        assert q.quantise(nid) == nid

    # And quantise(quantise(x)) == quantise(x).
    nid = q.quantise(0.62)
    assert q.quantise(nid) == nid


def test_axis_quantizer_monotone() -> None:
    q = _uniform_axis(n_bins=5)
    xs = np.linspace(0.05, 0.95, 25)
    ids = [q.quantise(float(x)) for x in xs]
    indices = [int(nid.split("_")[-1]) for nid in ids]
    # Sorted inputs => non-decreasing bin index.
    assert all(b >= a for a, b in zip(indices, indices[1:]))


def test_axis_quantizer_clamping() -> None:
    q = _uniform_axis(n_bins=5)
    nodes = q.node_ids()

    # Below the minimum edge clamps to the first bin.
    assert q.quantise(-100.0) == nodes[0]
    assert q.quantise(0.0) == nodes[0]

    # Above the maximum edge clamps to the last bin.
    assert q.quantise(100.0) == nodes[-1]
    # The right-most edge value also lands in the last bin.
    assert q.quantise(1.0) == nodes[-1]


def test_axis_quantizer_bin_count() -> None:
    edges = np.linspace(0.0, 1.0, 6)  # length 6 => 5 bins
    q = AxisQuantizer("energy", edges)
    assert q.n_nodes() == 5
    ids = q.node_ids()
    assert len(ids) == 5
    assert len(set(ids)) == 5


def test_axis_quantizer_from_quantiles() -> None:
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=0.0, scale=1.0, size=2000)
    q = AxisQuantizer.from_quantiles("sigma", samples, n_bins=5)

    # Each bin should hold roughly 20% of the samples.
    counts = {nid: 0 for nid in q.node_ids()}
    for v in samples:
        counts[q.quantise(float(v))] += 1
    fractions = np.array([c / len(samples) for c in counts.values()])
    assert np.all(np.abs(fractions - 0.2) < 0.05)


def test_axis_quantizer_from_uniform() -> None:
    q = AxisQuantizer.from_uniform("margin", low=-1.0, high=1.0, n_bins=4)
    widths = np.diff(q.bin_edges)
    # All widths must be equal up to float precision.
    assert np.allclose(widths, widths[0])
    assert q.n_nodes() == 4


def test_axis_quantizer_rejects_non_monotone_edges() -> None:
    with pytest.raises(ValueError):
        AxisQuantizer("bad", np.array([0.0, 0.5, 0.3]))

    # Equal adjacent edges are also rejected (not strictly increasing).
    with pytest.raises(ValueError):
        AxisQuantizer("bad", np.array([0.0, 0.5, 0.5, 1.0]))

    # Length < 2 is rejected.
    with pytest.raises(ValueError):
        AxisQuantizer("bad", np.array([0.0]))


def test_axis_quantizer_array_input() -> None:
    q = _uniform_axis(n_bins=5)
    out = q.quantise(np.array([0.1, 0.5, 0.9]))
    assert isinstance(out, list)
    assert len(out) == 3
    for nid in out:
        assert nid in q.node_ids()


def test_axis_quantizer_node_ids_unique() -> None:
    q = AxisQuantizer.from_uniform("sigma", 0.0, 1.0, n_bins=10)
    ids = q.node_ids()
    assert len(ids) == 10
    assert len(set(ids)) == 10

    # bin_for returns the half-open interval matching the edges.
    for i, nid in enumerate(ids):
        lo, hi = q.bin_for(nid)
        assert lo == pytest.approx(float(q.bin_edges[i]))
        assert hi == pytest.approx(float(q.bin_edges[i + 1]))
