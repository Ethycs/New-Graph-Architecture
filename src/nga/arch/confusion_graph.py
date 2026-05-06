"""Confusion graph: directed multigraph of (y_true, y_hat) classification errors.

Aggregates per-sample results into a sparse adjacency-with-weights structure.
The "edges" carry counts of misclassifications; the "nodes" are class labels.
"""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

__all__ = ["ConfusionGraph", "build_from_records"]


@dataclass
class ConfusionGraph:
    """Directed weighted graph accumulating classification confusion counts.

    Vertices are class labels (strings). Edge (u, v) has a count equal to the
    number of times class u was the true label and v was the predicted label.

    Note on density: self-loops (y_true == y_hat, i.e. correct predictions)
    ARE counted as edges and DO contribute to density. The density measure is
    a structural one: it describes how many of the V*V possible directed edges
    (including self-loops) carry at least one observation. It does not
    distinguish correct from incorrect predictions.
    """

    vertex_ids: list[str]
    edge_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        """Number of vertices in the graph."""
        return len(self.vertex_ids)

    @property
    def edge_count(self) -> int:
        """Number of edges with a positive observation count."""
        return sum(1 for c in self.edge_counts.values() if c > 0)

    @property
    def total_observations(self) -> int:
        """Total number of (y_true, y_hat) pairs recorded."""
        return sum(self.edge_counts.values())

    def add(self, y_true: str, y_hat: str) -> None:
        """Record one (y_true, y_hat) observation.

        Args:
            y_true: Ground-truth class label.
            y_hat: Predicted class label.

        Raises:
            ValueError: If either label is not in vertex_ids.
        """
        if y_true not in self.vertex_ids:
            raise ValueError(
                f"y_true={y_true!r} is not in vertex_ids; "
                f"known labels: {self.vertex_ids}"
            )
        if y_hat not in self.vertex_ids:
            raise ValueError(
                f"y_hat={y_hat!r} is not in vertex_ids; "
                f"known labels: {self.vertex_ids}"
            )
        key = (y_true, y_hat)
        self.edge_counts[key] = self.edge_counts.get(key, 0) + 1

    def density(self) -> float:
        """Edges with positive count / max possible edges (V * V).

        Returns 0.0 for V <= 1. Self-loops (y_true == y_hat) ARE counted as
        edges and DO contribute to density. The density measure is a structural
        one: it reflects what fraction of all V*V directed pairs (including
        self-loops) have been observed at least once.
        """
        v = self.vertex_count
        if v <= 1:
            return 0.0
        max_edges = v * v
        return self.edge_count / max_edges

    def to_matrix(self) -> np.ndarray:
        """Return shape (V, V) numpy int matrix with vertex_ids ordering.

        Entry [i, j] holds the count for (vertex_ids[i], vertex_ids[j]).
        Unobserved pairs are zero.
        """
        v = self.vertex_count
        index = {label: i for i, label in enumerate(self.vertex_ids)}
        mat = np.zeros((v, v), dtype=int)
        for (y_true, y_hat), count in self.edge_counts.items():
            i = index[y_true]
            j = index[y_hat]
            mat[i, j] = count
        return mat


def build_from_records(
    records: list,         # list of ResultsRecord
    vertex_ids: list[str],
) -> ConfusionGraph:
    """Build a ConfusionGraph from a list of ResultsRecord.

    Skips any record whose y_true or y_hat is not in vertex_ids and emits a
    single UserWarning summary at the end reporting how many records were
    skipped and which unseen labels were encountered.

    Args:
        records: Sequence of ResultsRecord instances (from drivers/results_jsonl.py).
        vertex_ids: The known class labels used to initialise the graph.

    Returns:
        A populated ConfusionGraph.
    """
    cg = ConfusionGraph(vertex_ids=list(vertex_ids))
    vertex_set = set(vertex_ids)
    skipped = 0
    unknown_labels: Counter[str] = Counter()

    for rec in records:
        y_true: str = rec.y_true
        y_hat: str = rec.y_hat
        missing = False
        if y_true not in vertex_set:
            unknown_labels[y_true] += 1
            missing = True
        if y_hat not in vertex_set:
            unknown_labels[y_hat] += 1
            missing = True
        if missing:
            skipped += 1
            continue
        cg.add(y_true, y_hat)

    if skipped > 0:
        label_summary = ", ".join(
            f"{lbl!r} (x{cnt})" for lbl, cnt in unknown_labels.most_common()
        )
        warnings.warn(
            f"build_from_records: skipped {skipped} record(s) with labels not in "
            f"vertex_ids. Unknown labels encountered: {label_summary}",
            UserWarning,
            stacklevel=2,
        )

    return cg
