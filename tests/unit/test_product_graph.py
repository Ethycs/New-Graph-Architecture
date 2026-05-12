"""Unit tests for nga.arch.product_graph.

The product graph is the Cartesian product of typed axes, materialised
lazily. These tests pin down: laziness at construction, idempotent
materialisation, validation of arity and axis membership, transition
bookkeeping, the cell-id <-> node-tuple round-trip, axis-neighbour
expansion semantics (one component changes per neighbour, only listed
axes participate when filtered), and the storage invariant
``n_materialised <= n_theoretical`` even after many traversals.
"""

from __future__ import annotations

import pytest

from nga.arch.product_graph import ProductGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _four_axis_product() -> ProductGraph:
    """4 axes named a/b/c/d, each with 5 nodes => theoretical = 5**4 = 625."""
    axis_names = ["a", "b", "c", "d"]
    axis_node_lists = {
        a: [f"{a}_{i}" for i in range(5)] for a in axis_names
    }
    return ProductGraph(axis_names, axis_node_lists)


def _small_product() -> ProductGraph:
    """3 axes of 4 nodes each for tighter neighbour tests."""
    axis_names = ["x", "y", "z"]
    axis_node_lists = {
        a: [f"{a}_{i}" for i in range(4)] for a in axis_names
    }
    return ProductGraph(axis_names, axis_node_lists)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_product_graph_init_does_not_materialise() -> None:
    pg = _four_axis_product()
    assert pg.n_materialised() == 0
    # 4 axes x 5 nodes => 5 * 5 * 5 * 5 = 625.
    assert pg.n_theoretical() == 5 ** 4


def test_materialise_idempotent() -> None:
    pg = _four_axis_product()
    t = ("a_0", "b_1", "c_2", "d_3")
    cid1 = pg.materialise(t)
    cid2 = pg.materialise(t)
    assert cid1 == cid2
    assert pg.n_materialised() == 1
    # Same tuple identity after re-materialisation.
    assert pg.cell_id(t) == cid1


def test_materialise_validates_node_ids() -> None:
    pg = _four_axis_product()
    with pytest.raises(KeyError):
        pg.materialise(("a_0", "b_1", "c_2", "NOT_A_NODE"))


def test_materialise_validates_arity() -> None:
    pg = _four_axis_product()
    # Too few components.
    with pytest.raises(ValueError):
        pg.materialise(("a_0", "b_1", "c_2"))
    # Too many components.
    with pytest.raises(ValueError):
        pg.materialise(("a_0", "b_1", "c_2", "d_3", "e_0"))


def test_add_transition_materialises_both_endpoints() -> None:
    pg = _four_axis_product()
    src = ("a_0", "b_1", "c_2", "d_3")
    dst = ("a_1", "b_1", "c_2", "d_3")
    assert not pg.is_materialised(src)
    assert not pg.is_materialised(dst)

    src_id, dst_id = pg.add_transition(src, dst)

    assert pg.is_materialised(src)
    assert pg.is_materialised(dst)
    assert pg.cell_id(src) == src_id
    assert pg.cell_id(dst) == dst_id
    assert pg.n_materialised() == 2


def test_transitions_count() -> None:
    pg = _four_axis_product()
    src = ("a_0", "b_0", "c_0", "d_0")
    dst = ("a_1", "b_0", "c_0", "d_0")
    for _ in range(3):
        pg.add_transition(src, dst)

    pairs = list(pg.transitions())
    assert len(pairs) == 1
    s, d, c = pairs[0]
    assert s == src
    assert d == dst
    assert c == 3


def test_node_tuple_for_inverse_of_cell_id() -> None:
    pg = _four_axis_product()
    tuples = [
        ("a_0", "b_0", "c_0", "d_0"),
        ("a_1", "b_2", "c_3", "d_4"),
        ("a_4", "b_3", "c_2", "d_1"),
    ]
    cids = [pg.materialise(t) for t in tuples]
    # Round-trip: cell_id -> node_tuple -> cell_id is identity.
    for t, cid in zip(tuples, cids):
        assert pg.node_tuple_for(cid) == t
        assert pg.cell_id(t) == cid


def test_n_materialised_strictly_le_theoretical() -> None:
    pg = _four_axis_product()
    # Materialise a handful of tuples; never approach the theoretical bound.
    for i in range(10):
        pg.materialise((f"a_{i % 5}", f"b_{i % 5}", f"c_{i % 5}", f"d_{i % 5}"))
    assert pg.n_materialised() <= pg.n_theoretical()
    assert pg.n_materialised() < pg.n_theoretical()


def test_expand_neighbours_one_axis_per_call() -> None:
    pg = _small_product()
    t = ("x_1", "y_2", "z_1")
    neighbours = pg.expand_neighbours(t)
    # Each neighbour differs from t in exactly one component.
    for nb in neighbours:
        diffs = sum(1 for a, b in zip(t, nb) if a != b)
        assert diffs == 1
    # Interior tuple => 2 neighbours per axis * 3 axes = 6.
    assert len(neighbours) == 6
    # Distinct neighbours.
    assert len(set(neighbours)) == len(neighbours)


def test_expand_neighbours_respects_axis_filter() -> None:
    pg = _small_product()
    t = ("x_1", "y_2", "z_1")
    only_y = pg.expand_neighbours(t, axes=["y"])
    # Only the y component changes.
    assert len(only_y) == 2
    for nb in only_y:
        assert nb[0] == t[0]
        assert nb[2] == t[2]
        assert nb[1] != t[1]

    # Filtering with two axes contributes their neighbours only.
    xz = pg.expand_neighbours(t, axes=["x", "z"])
    assert len(xz) == 4
    for nb in xz:
        assert nb[1] == t[1]  # y untouched

    # Boundary handling: corner tuple has only +1 neighbours per axis.
    corner = ("x_0", "y_0", "z_0")
    nbs = pg.expand_neighbours(corner)
    assert len(nbs) == 3
    for nb in nbs:
        diffs = sum(1 for a, b in zip(corner, nb) if a != b)
        assert diffs == 1

    # Unknown axis name in filter is rejected.
    with pytest.raises(KeyError):
        pg.expand_neighbours(t, axes=["nope"])


def test_visited_cells_only_grow_with_observations() -> None:
    pg = _four_axis_product()
    # Construct 12 distinct (src, dst) pairs walking through the product.
    pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for i in range(12):
        src = (f"a_{i % 5}", f"b_{(i + 1) % 5}", f"c_{(i + 2) % 5}", "d_0")
        dst = (f"a_{(i + 1) % 5}", f"b_{(i + 1) % 5}", f"c_{(i + 2) % 5}", "d_0")
        pairs.append((src, dst))

    seen: set[tuple[str, ...]] = set()
    for src, dst in pairs:
        pg.add_transition(src, dst)
        seen.add(src)
        seen.add(dst)

    # n_materialised matches the set of distinct endpoints.
    assert pg.n_materialised() == len(seen)
    # Bound: at most 24 endpoints across 12 pairs.
    assert pg.n_materialised() <= 24
    # And nothing close to the theoretical product.
    assert pg.n_materialised() < pg.n_theoretical()
    assert pg.n_theoretical() == 5 ** 4
