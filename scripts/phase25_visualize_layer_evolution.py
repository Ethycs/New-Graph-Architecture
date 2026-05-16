"""Phase 25 -- visualise how the regime graph evolves through GPT-2's layers.

For one grammar (default ``python_control``, V=37 — the most structurally
rich grammar in the suite), render a 1×6 strip:

    gold FSM  |  L0  |  L2  |  L6  |  L10  |  L12

Reads from ``runs/E30_phase25_<grammar>_L<k>/control_graph.json`` (the
artefacts produced by ``scripts/phase25_layer_ablation_sweep.py``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402

from nga.arch.graph_fsm import GraphFSM  # noqa: E402
from nga.drivers import graph_fsm_spec as graph_fsm_spec_mod  # noqa: E402
from nga.exp.e25_extraction_torch import GRAMMAR_DISPATCH  # noqa: E402

LAYERS = [0, 2, 6, 10, 12]
DEFAULT_GRAMMAR = "python_control"


def _load_gold_fsm(grammar: str) -> GraphFSM:
    dispatch = GRAMMAR_DISPATCH[grammar]
    return GraphFSM(graph_fsm_spec_mod.load(REPO_ROOT / dispatch["fsm_yaml_path"]))


def _build_gold_graph(fsm: GraphFSM) -> nx.DiGraph:
    g = nx.DiGraph()
    for v_id in fsm.vertex_ids:
        g.add_node(v_id)
    L = fsm.legality_matrix
    for i, src in enumerate(fsm.vertex_ids):
        for j, dst in enumerate(fsm.vertex_ids):
            if bool(L[i, j]):
                g.add_edge(src, dst)
    return g


def _build_regime_graph(cg_path: Path) -> tuple[nx.DiGraph, dict]:
    data = json.loads(cg_path.read_text())
    g = nx.DiGraph()
    for r in data["regimes"]:
        g.add_node(
            r["regime_id"],
            support=int(r["support"]),
            purity=float(r["purity_against_current_state"]),
            label_state=str(r["dominant_current_state"]),
        )
    for e in data["edges"]:
        g.add_edge(int(e["src"]), int(e["dst"]), probability=float(e["probability"]))
    return g, data


def _draw_gold(ax, fsm: GraphFSM) -> None:
    g = _build_gold_graph(fsm)
    pos = nx.kamada_kawai_layout(g) if len(g) <= 30 else nx.spring_layout(g, seed=0, k=1.2)
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color="#cfd8dc", node_size=180, edgecolors="#37474f"
    )
    nx.draw_networkx_edges(
        g, pos, ax=ax, arrows=True, edge_color="#90a4ae", width=0.5, arrowsize=4
    )
    ax.set_title(f"gold FSM (V={fsm.vertex_count})", fontsize=9)
    ax.set_axis_off()


def _draw_regime(ax, g: nx.DiGraph, data: dict, layer: int) -> None:
    pos = nx.kamada_kawai_layout(g) if len(g) <= 30 else nx.spring_layout(g, seed=0, k=1.2)
    cmap = colormaps["RdYlGn"]
    supports = np.array([g.nodes[n]["support"] for n in g.nodes])
    purities = np.array([g.nodes[n]["purity"] for n in g.nodes])
    sup_norm = supports / supports.max() if supports.max() > 0 else supports
    node_colors = [cmap(p) for p in purities]
    node_sizes = (80 + 220 * sup_norm).tolist()

    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
        edgecolors="#263238", linewidths=0.4,
    )
    edge_weights = np.array([g.edges[u, v]["probability"] for u, v in g.edges])
    widths = 0.3 + 1.4 * edge_weights
    nx.draw_networkx_edges(
        g, pos, ax=ax, arrows=True, edge_color="#455a64",
        width=widths.tolist(), arrowsize=4,
        connectionstyle="arc3,rad=0.05",
    )
    mean_purity = float(purities.mean())
    K = data["n_regimes_after_merge"]
    ax.set_title(f"L{layer}  K={K}  mean purity {mean_purity:.3f}", fontsize=9)
    ax.set_axis_off()


def main() -> int:
    grammar = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GRAMMAR
    out_dir = REPO_ROOT / "runs" / "phase25_layer_evolution"
    out_dir.mkdir(parents=True, exist_ok=True)

    fsm = _load_gold_fsm(grammar)

    fig, axes = plt.subplots(1, 1 + len(LAYERS), figsize=(4 * (1 + len(LAYERS)), 5))
    _draw_gold(axes[0], fsm)

    for col, layer in enumerate(LAYERS, start=1):
        cg_path = REPO_ROOT / "runs" / f"E30_phase25_{grammar}_L{layer}" / "control_graph.json"
        if not cg_path.exists():
            axes[col].set_title(f"L{layer} (missing)", fontsize=9)
            axes[col].set_axis_off()
            continue
        g, data = _build_regime_graph(cg_path)
        _draw_regime(axes[col], g, data, layer)

    fig.suptitle(
        f"Phase 25 — regime graph evolution through GPT-2 layers ({grammar}, V={fsm.vertex_count})",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / f"{grammar}_layer_evolution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
